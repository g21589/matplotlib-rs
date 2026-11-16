//! The vertices of a path, in a buffer that paths can share.

use std::ops::{Deref, DerefMut, Range};
use std::sync::Arc;

/// The vertices of a path: a buffer or a part of one.  Clones and slices
/// share the buffer (a line's data and its path, the visible part of a long
/// line), so neither copies vertices; writing to shared vertices copies them
/// first.
pub struct Verts {
    /// what keeps the vertices alive (none: no vertices)
    owner: Option<Arc<Vec<[f64; 2]>>>,
    /// the vertices, within `owner`'s buffer
    ptr: *const [f64; 2],
    len: usize,
}

// The vertices are read-only while shared, and owned through an `Arc`.
unsafe impl Send for Verts {}
unsafe impl Sync for Verts {}

impl Verts {
    pub const fn new() -> Verts {
        Verts { owner: None, ptr: std::ptr::NonNull::dangling().as_ptr(), len: 0 }
    }

    /// The vertices *range* of these, sharing their buffer.
    pub fn slice(&self, range: Range<usize>) -> Verts {
        assert!(range.start <= range.end && range.end <= self.len, "vertex range out of bounds");
        if range.is_empty() {
            return Verts::new();
        }
        // SAFETY: the range is within the vertices.
        Verts { owner: self.owner.clone(), ptr: unsafe { self.ptr.add(range.start) }, len: range.len() }
    }

    /// The vertices as a vector (not copied when nothing shares them).
    pub fn into_vec(self) -> Vec<[f64; 2]> {
        let whole = self.owner.as_ref().is_some_and(|o| o.as_ptr() == self.ptr && o.len() == self.len);
        if !whole {
            return self.to_vec();
        }
        let Verts { owner, .. } = self;
        Arc::try_unwrap(owner.unwrap()).unwrap_or_else(|shared| (*shared).clone())
    }
}

impl Default for Verts {
    fn default() -> Verts {
        Verts::new()
    }
}

impl Clone for Verts {
    fn clone(&self) -> Verts {
        Verts { owner: self.owner.clone(), ptr: self.ptr, len: self.len }
    }
}

impl From<Vec<[f64; 2]>> for Verts {
    fn from(v: Vec<[f64; 2]>) -> Verts {
        if v.is_empty() {
            return Verts::new();
        }
        let owner = Arc::new(v);
        Verts { ptr: owner.as_ptr(), len: owner.len(), owner: Some(owner) }
    }
}

impl FromIterator<[f64; 2]> for Verts {
    fn from_iter<I: IntoIterator<Item = [f64; 2]>>(iter: I) -> Verts {
        iter.into_iter().collect::<Vec<_>>().into()
    }
}

impl Deref for Verts {
    type Target = [[f64; 2]];
    #[inline(always)]
    fn deref(&self) -> &[[f64; 2]] {
        // SAFETY: `ptr..ptr + len` is within `owner`'s buffer, which is not
        // written while shared (or dangling with len 0).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl DerefMut for Verts {
    /// The vertices, copied first if they are shared.
    fn deref_mut(&mut self) -> &mut [[f64; 2]] {
        if self.len == 0 {
            return &mut [];
        }
        let unique = self.owner.as_mut().and_then(Arc::get_mut).is_some();
        if !unique {
            *self = self.to_vec().into();
            // (a new buffer, which nothing else shares)
        }
        // SAFETY: as `deref`, and nothing else refers to the buffer.
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut [f64; 2], self.len) }
    }
}

impl PartialEq for Verts {
    fn eq(&self, other: &Verts) -> bool {
        **self == **other
    }
}

impl std::fmt::Debug for Verts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

impl<'a> IntoIterator for &'a Verts {
    type Item = &'a [f64; 2];
    type IntoIter = std::slice::Iter<'a, [f64; 2]>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::Verts;

    fn points(n: usize) -> Vec<[f64; 2]> {
        (0..n).map(|i| [i as f64, -(i as f64)]).collect()
    }

    #[test]
    fn slices_share_and_writes_copy() {
        let all: Verts = points(10).into();
        let part = all.slice(2..7);
        assert_eq!(&*part, &points(10)[2..7]);
        assert_eq!(part.as_ptr(), all[2..].as_ptr());
        assert_eq!(&*part.slice(1..3), &points(10)[3..5]);
        assert!(all.slice(4..4).is_empty());

        // writing to shared vertices leaves the others as they were
        let mut copy = part.clone();
        copy[0] = [100.0, 100.0];
        assert_eq!(part[0], [2.0, -2.0]);
        assert_eq!(all[2], [2.0, -2.0]);
        assert_eq!(copy[0], [100.0, 100.0]);
        assert_eq!(&copy[1..], &points(10)[3..7]);

        // unshared vertices are written in place
        let mut own: Verts = points(4).into();
        let before = own.as_ptr();
        own[3] = [7.0, 7.0];
        assert_eq!(own.as_ptr(), before);
        assert_eq!(own[3], [7.0, 7.0]);
    }

    #[test]
    fn into_vec_and_empty() {
        let v = points(5);
        let ptr = v.as_ptr();
        let verts: Verts = v.into();
        let back = verts.into_vec();
        assert_eq!(back.as_ptr(), ptr);
        assert_eq!(back, points(5));

        let verts: Verts = points(5).into();
        let other = verts.clone();
        assert_eq!(verts.into_vec(), points(5));
        assert_eq!(other.slice(1..3).into_vec(), points(5)[1..3].to_vec());

        let mut empty = Verts::default();
        assert!(empty.is_empty() && empty.iter_mut().next().is_none());
        assert_eq!(Verts::from(Vec::new()), empty);
        assert!(empty.into_vec().is_empty());
    }
}
