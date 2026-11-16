//! The worker threads of the parallel loops: started once and reused, so a
//! loop pays for waking them, not for creating them.
//!
//! [`run`] is a parallel region: `f(0)` on the calling thread and `f(1)`,
//! ..., `f(n - 1)` on workers, returning when all have returned.  The loops
//! decide themselves what each of the *n* threads does (most take pieces of
//! the work from a shared counter).

use std::any::Any;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, TryLockError};

/// The threads of the parallel loops: the available parallelism (at most
/// 8), or `MPLRS_THREADS` (1: one thread).
pub fn threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("MPLRS_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&t| t >= 1)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |t| t.get()).min(8))
    })
}

/// The closure of the region being run (valid until `run` returns: it
/// waits for every worker that took it).
#[derive(Clone, Copy)]
struct Job(*const (dyn Fn(usize) + Sync));

// The closure is `Sync`, and `run` keeps it alive while workers call it.
unsafe impl Send for Job {}

struct State {
    /// incremented for every region
    generation: u64,
    job: Option<Job>,
    /// the workers 1..participants run the job
    participants: usize,
    /// the workers that have not finished it
    running: usize,
    /// the first panic of a worker in the region
    panic: Option<Box<dyn Any + Send>>,
}

struct Shared {
    state: Mutex<State>,
    start: Condvar,
    done: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        // (never poisoned: jobs run outside the lock)
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }
}

struct Pool {
    /// the process that started the workers (a forked child has none)
    pid: u32,
    shared: Arc<Shared>,
    workers: usize,
}

impl Pool {
    fn new() -> Pool {
        let state = State { generation: 0, job: None, participants: 0, running: 0, panic: None };
        let shared = Arc::new(Shared { state: Mutex::new(state), start: Condvar::new(), done: Condvar::new() });
        Pool { pid: std::process::id(), shared, workers: 0 }
    }

    /// Start workers until there are *count*.
    fn grow(&mut self, count: usize) {
        while self.workers < count {
            let index = self.workers + 1;
            let shared = self.shared.clone();
            let seen = shared.lock().generation;
            std::thread::Builder::new()
                .name(format!("mplrs-{index}"))
                .spawn(move || worker(&shared, index, seen))
                .expect("failed to start a matplotlib_rs worker thread");
            self.workers += 1;
        }
    }
}

fn worker(shared: &Shared, index: usize, mut seen: u64) {
    loop {
        let job = {
            let mut st = shared.lock();
            loop {
                if st.generation != seen {
                    seen = st.generation;
                    if index < st.participants {
                        break st.job.expect("a region without a job");
                    }
                }
                st = shared.start.wait(st).unwrap_or_else(|p| p.into_inner());
            }
        };
        // SAFETY: `run` does not return before `running` reaches 0.
        let result = catch_unwind(AssertUnwindSafe(|| unsafe { (*job.0)(index) }));
        let mut st = shared.lock();
        if let Err(p) = result {
            st.panic.get_or_insert(p);
        }
        st.running -= 1;
        if st.running == 0 {
            shared.done.notify_all();
        }
    }
}

static POOL: Mutex<Option<Pool>> = Mutex::new(None);

/// `f(0)` on this thread and `f(1)`, ..., `f(n - 1)` on other threads, at
/// the same time; returns when all have returned.  A panic of one of them
/// is resumed here (after the others have returned).
///
/// The other threads are the pool's workers.  A region started while
/// another one runs (from a worker, or from another thread) uses new
/// threads instead, as does a forked child whose parent was in a region.
pub fn run<F: Fn(usize) + Sync>(n: usize, f: F) {
    if n <= 1 {
        if n == 1 {
            f(0);
        }
        return;
    }
    let mut guard = match POOL.try_lock() {
        Ok(g) => g,
        Err(TryLockError::Poisoned(p)) => p.into_inner(),
        Err(TryLockError::WouldBlock) => return run_scoped(n, &f),
    };
    let pid = std::process::id();
    if guard.as_ref().is_some_and(|p| p.pid != pid) {
        // After a fork the workers are gone, and what they shared may be
        // locked for ever: leave it as it is.
        std::mem::forget(guard.take());
    }
    let pool = guard.get_or_insert_with(Pool::new);
    pool.grow(n - 1);
    let shared = pool.shared.clone();
    {
        let job: &(dyn Fn(usize) + Sync) = &f;
        // SAFETY: only the lifetime changes; see `Job`.
        let job: &'static (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(job) };
        let mut st = shared.lock();
        st.job = Some(Job(job));
        st.participants = n;
        st.running = n - 1;
        st.generation += 1;
    }
    shared.start.notify_all();
    let mine = catch_unwind(AssertUnwindSafe(|| f(0)));
    let theirs = {
        let mut st = shared.lock();
        while st.running > 0 {
            st = shared.done.wait(st).unwrap_or_else(|p| p.into_inner());
        }
        st.job = None;
        st.participants = 0;
        st.panic.take()
    };
    drop(guard);
    if let Err(p) = mine {
        resume_unwind(p);
    }
    if let Some(p) = theirs {
        resume_unwind(p);
    }
}

/// `run` on threads started for this region.
fn run_scoped(n: usize, f: &(dyn Fn(usize) + Sync)) {
    std::thread::scope(|s| {
        for t in 1..n {
            s.spawn(move || f(t));
        }
        f(0);
    });
}

/// `f(piece)` for each of *pieces*, on at most *threads* threads; the
/// threads take the pieces one after the other, so pieces of unequal cost
/// are shared between them.
pub fn for_each<T: Send, F: Fn(T) + Sync>(threads: usize, pieces: Vec<T>, f: F) {
    let n = threads.min(pieces.len());
    if n <= 1 {
        pieces.into_iter().for_each(f);
        return;
    }
    let queue = Mutex::new(pieces.into_iter());
    run(n, |_| loop {
        let piece = queue.lock().unwrap_or_else(|p| p.into_inner()).next();
        match piece {
            Some(p) => f(p),
            None => break,
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn regions_run_every_thread_once() {
        for n in [1, 2, 3, 5, 2, 8, 1, 4] {
            let seen: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            run(n, |t| {
                seen[t].fetch_add(1, Ordering::Relaxed);
            });
            assert!(seen.iter().all(|s| s.load(Ordering::Relaxed) == 1), "n = {n}");
        }
    }

    #[test]
    fn threads_run_at_the_same_time() {
        // (a barrier returns only when all the threads wait on it)
        let barrier = std::sync::Barrier::new(4);
        run(4, |_| {
            barrier.wait();
        });
    }

    #[test]
    fn nested_and_concurrent_regions() {
        let total = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..3 {
                s.spawn(|| {
                    for _ in 0..20 {
                        run(3, |_| {
                            run(2, |_| {
                                total.fetch_add(1, Ordering::Relaxed);
                            });
                        });
                    }
                });
            }
        });
        assert_eq!(total.load(Ordering::Relaxed), 3 * 20 * 3 * 2);
    }

    #[test]
    fn panics_are_resumed_and_the_pool_survives() {
        for bad in [0, 2] {
            let r = catch_unwind(|| {
                run(3, |t| {
                    if t == bad {
                        panic!("thread {t}");
                    }
                })
            });
            assert!(r.is_err());
        }
        let count = AtomicUsize::new(0);
        run(3, |_| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn for_each_gives_every_piece_once() {
        let mut data = vec![0u32; 1000];
        let pieces: Vec<&mut [u32]> = data.chunks_mut(7).collect();
        for_each(4, pieces, |p| p.iter_mut().for_each(|v| *v += 1));
        assert!(data.iter().all(|&v| v == 1));
    }
}
