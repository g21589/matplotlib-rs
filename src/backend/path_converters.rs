//! Port of matplotlib's `src/path_converters.h`: PathNanRemover,
//! PathClipper, PathSnapper, PathSimplifier and Sketch.

use crate::agg::basics::*;
use crate::agg::clip::clip_line_segment;

#[derive(Clone, Copy, Default)]
struct Item {
    cmd: u32,
    x: f64,
    y: f64,
}

/// Fixed-capacity queue (`EmbeddedQueue<N>`).
struct EmbeddedQueue<const N: usize> {
    read: usize,
    write: usize,
    q: [Item; N],
}

impl<const N: usize> EmbeddedQueue<N> {
    fn new() -> Self {
        EmbeddedQueue { read: 0, write: 0, q: [Item::default(); N] }
    }
    #[inline(always)]
    fn push(&mut self, cmd: u32, x: f64, y: f64) {
        self.q[self.write] = Item { cmd, x, y };
        self.write += 1;
    }
    #[inline(always)]
    fn nonempty(&self) -> bool {
        self.read < self.write
    }
    #[inline(always)]
    fn pop(&mut self, cmd: &mut u32, x: &mut f64, y: &mut f64) -> bool {
        if self.nonempty() {
            let f = self.q[self.read];
            self.read += 1;
            *cmd = f.cmd;
            *x = f.x;
            *y = f.y;
            return true;
        }
        self.read = 0;
        self.write = 0;
        false
    }
    #[inline(always)]
    fn clear(&mut self) {
        self.read = 0;
        self.write = 0;
    }
}

const NUM_EXTRA_POINTS_MAP: [usize; 16] = [0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

#[inline(always)]
fn finite(x: f64, y: f64) -> bool {
    x.is_finite() && y.is_finite()
}

// ---------------------------------------------------------------------------

pub struct PathNanRemover<S: VertexSource> {
    pub source: S,
    remove_nans: bool,
    has_codes: bool,
    valid_segment_exists: bool,
    last_segment_valid: bool,
    was_broken: bool,
    init_x: f64,
    init_y: f64,
    queue: EmbeddedQueue<4>,
}

impl<S: VertexSource> PathNanRemover<S> {
    pub fn new(source: S, remove_nans: bool, has_codes: bool) -> Self {
        PathNanRemover {
            source,
            remove_nans,
            has_codes,
            valid_segment_exists: false,
            last_segment_valid: false,
            was_broken: false,
            init_x: f64::NAN,
            init_y: f64::NAN,
            queue: EmbeddedQueue::new(),
        }
    }
}

impl<S: VertexSource> VertexSource for PathNanRemover<S> {
    fn rewind(&mut self, path_id: u32) {
        self.queue.clear();
        self.source.rewind(path_id);
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let mut code: u32 = 0;

        if !self.remove_nans {
            return self.source.vertex(x, y);
        }

        if self.has_codes {
            // This is the slow method for when there might be curves or closed loops.
            if self.queue.pop(&mut code, x, y) {
                return code;
            }

            let mut needs_move_to = false;
            loop {
                code = self.source.vertex(x, y);
                if code == PATH_CMD_STOP {
                    return code;
                } else if code == (PATH_CMD_END_POLY | PATH_FLAGS_CLOSE) && self.valid_segment_exists {
                    if self.was_broken {
                        if self.last_segment_valid && finite(self.init_x, self.init_y) {
                            // Join to start if both ends are valid.
                            self.queue.push(PATH_CMD_LINE_TO, self.init_x, self.init_y);
                            break;
                        } else {
                            // Skip the close, in case there are additional subpaths.
                            continue;
                        }
                    } else {
                        return code;
                    }
                } else if code == PATH_CMD_MOVE_TO {
                    // Save the initial point in order to produce the last
                    // segment closing a loop, *if* we broke the loop.
                    self.init_x = *x;
                    self.init_y = *y;
                    self.was_broken = false;
                }

                if needs_move_to {
                    self.queue.push(PATH_CMD_MOVE_TO, *x, *y);
                }

                let num_extra_points = NUM_EXTRA_POINTS_MAP[(code & 0xF) as usize];
                self.last_segment_valid = finite(*x, *y);
                self.queue.push(code, *x, *y);

                // Note: this test cannot be short-circuited, since we need to
                // advance through the entire curve no matter what.
                for _ in 0..num_extra_points {
                    self.source.vertex(x, y);
                    self.last_segment_valid = self.last_segment_valid && finite(*x, *y);
                    self.queue.push(code, *x, *y);
                }

                if self.last_segment_valid {
                    self.valid_segment_exists = true;
                    break;
                }

                self.was_broken = true;
                self.queue.clear();

                // If the last point is finite, we use that for the moveto,
                // otherwise, we'll use the first vertex of the next curve.
                if finite(*x, *y) {
                    self.queue.push(PATH_CMD_MOVE_TO, *x, *y);
                    needs_move_to = false;
                } else {
                    needs_move_to = true;
                }
            }

            if self.queue.pop(&mut code, x, y) {
                code
            } else {
                PATH_CMD_STOP
            }
        } else {
            // This is the fast path for when we know we have no codes.
            code = self.source.vertex(x, y);

            if code == PATH_CMD_STOP || (code == (PATH_CMD_END_POLY | PATH_FLAGS_CLOSE) && self.valid_segment_exists) {
                return code;
            }

            if !finite(*x, *y) {
                loop {
                    code = self.source.vertex(x, y);
                    if code == PATH_CMD_STOP
                        || (code == (PATH_CMD_END_POLY | PATH_FLAGS_CLOSE) && self.valid_segment_exists)
                    {
                        return code;
                    }
                    if finite(*x, *y) {
                        break;
                    }
                }
                return PATH_CMD_MOVE_TO;
            }
            self.valid_segment_exists = true;
            code
        }
    }
}

// ---------------------------------------------------------------------------

pub struct PathClipper<S: VertexSource> {
    pub source: S,
    do_clipping: bool,
    cliprect: RectD,
    last_x: f64,
    last_y: f64,
    moveto: bool,
    init_x: f64,
    init_y: f64,
    has_init: bool,
    was_clipped: bool,
    queue: EmbeddedQueue<3>,
}

impl<S: VertexSource> PathClipper<S> {
    pub fn new(source: S, do_clipping: bool, width: f64, height: f64) -> Self {
        Self::with_rect(source, do_clipping, RectD::new(-1.0, -1.0, width + 1.0, height + 1.0))
    }

    /// The `(source, do_clipping, rect)` constructor, which pads by 1 pixel.
    pub fn from_rect(source: S, do_clipping: bool, rect: RectD) -> Self {
        let r = RectD::new(rect.x1 - 1.0, rect.y1 - 1.0, rect.x2 + 1.0, rect.y2 + 1.0);
        Self::with_rect(source, do_clipping, r)
    }

    fn with_rect(source: S, do_clipping: bool, cliprect: RectD) -> Self {
        PathClipper {
            source,
            do_clipping,
            cliprect,
            last_x: f64::NAN,
            last_y: f64::NAN,
            moveto: true,
            init_x: f64::NAN,
            init_y: f64::NAN,
            has_init: false,
            was_clipped: false,
            queue: EmbeddedQueue::new(),
        }
    }

    fn draw_clipped_line(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, closed: bool) -> bool {
        let (mut x0, mut y0, mut x1, mut y1) = (x0, y0, x1, y1);
        let moved = clip_line_segment(&mut x0, &mut y0, &mut x1, &mut y1, &self.cliprect);
        // moved >= 4 - Fully clipped
        // moved & 1 != 0 - First point has been moved
        // moved & 2 != 0 - Second point has been moved
        self.was_clipped = self.was_clipped || (moved != 0);
        if moved < 4 {
            if moved & 1 != 0 || self.moveto {
                self.queue.push(PATH_CMD_MOVE_TO, x0, y0);
            }
            self.queue.push(PATH_CMD_LINE_TO, x1, y1);
            if closed && !self.was_clipped {
                // Close the path only if the end point hasn't moved.
                self.queue.push(PATH_CMD_END_POLY | PATH_FLAGS_CLOSE, x1, y1);
            }
            self.moveto = false;
            return true;
        }
        false
    }

    #[inline(always)]
    fn last_in_rect(&self) -> bool {
        self.last_x >= self.cliprect.x1
            && self.last_x <= self.cliprect.x2
            && self.last_y >= self.cliprect.y1
            && self.last_y <= self.cliprect.y2
    }
}

impl<S: VertexSource> VertexSource for PathClipper<S> {
    fn rewind(&mut self, path_id: u32) {
        self.has_init = false;
        self.was_clipped = false;
        self.moveto = true;
        self.source.rewind(path_id);
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let mut code: u32 = 0;

        if !self.do_clipping {
            // If not doing any clipping, just pass along the vertices verbatim
            return self.source.vertex(x, y);
        }

        // This is the slow path where we actually do clipping
        if self.queue.pop(&mut code, x, y) {
            return code;
        }

        loop {
            code = self.source.vertex(x, y);
            if code == PATH_CMD_STOP {
                break;
            }
            let mut emit_moveto = false;
            match code {
                c if c == (PATH_CMD_END_POLY | PATH_FLAGS_CLOSE) => {
                    if self.has_init {
                        // Queue the line from last point to the initial point, and
                        // if never clipped, add a close code.
                        let (lx, ly, ix, iy) = (self.last_x, self.last_y, self.init_x, self.init_y);
                        self.draw_clipped_line(lx, ly, ix, iy, true);
                    } else {
                        // An empty path that is immediately closed.
                        self.queue.push(PATH_CMD_END_POLY | PATH_FLAGS_CLOSE, self.last_x, self.last_y);
                    }
                    // If paths were not clipped, then the above code queued
                    // something, and we should exit the loop. Otherwise, continue
                    // to the next point, as there may be a new subpath.
                    if self.queue.nonempty() {
                        break;
                    }
                }
                PATH_CMD_MOVE_TO => {
                    // was the last command a moveto (and we have
                    // seen at least one command ?
                    // if so, shove it in the queue if in clip box
                    if self.moveto && self.has_init && self.last_in_rect() {
                        // push the last moveto onto the queue
                        self.queue.push(PATH_CMD_MOVE_TO, self.last_x, self.last_y);
                        // flag that we need to emit it
                        emit_moveto = true;
                    }
                    // update the internal state for this moveto
                    self.init_x = *x;
                    self.last_x = *x;
                    self.init_y = *y;
                    self.last_y = *y;
                    self.has_init = true;
                    self.moveto = true;
                    self.was_clipped = false;
                    // if the last command was moveto exit the loop to emit the code
                    if emit_moveto {
                        break;
                    }
                }
                PATH_CMD_LINE_TO => {
                    let (lx, ly) = (self.last_x, self.last_y);
                    if self.draw_clipped_line(lx, ly, *x, *y, false) {
                        self.last_x = *x;
                        self.last_y = *y;
                        break;
                    }
                    self.last_x = *x;
                    self.last_y = *y;
                }
                _ => {
                    if self.moveto {
                        self.queue.push(PATH_CMD_MOVE_TO, self.last_x, self.last_y);
                        self.moveto = false;
                    }
                    self.queue.push(code, *x, *y);
                    self.last_x = *x;
                    self.last_y = *y;
                    break;
                }
            }
        }

        // exit_loop:
        if self.queue.pop(&mut code, x, y) {
            return code;
        }

        if self.moveto && self.has_init && self.last_in_rect() {
            *x = self.last_x;
            *y = self.last_y;
            self.moveto = false;
            return PATH_CMD_MOVE_TO;
        }

        PATH_CMD_STOP
    }
}

// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SnapMode {
    Auto,
    False,
    True,
}

pub struct PathSnapper<S: VertexSource> {
    pub source: S,
    snap: bool,
    snap_value: f64,
}

impl<S: VertexSource> PathSnapper<S> {
    fn should_snap(path: &mut S, snap_mode: SnapMode, total_vertices: usize) -> bool {
        // If this contains only straight horizontal or vertical lines, it should be
        // snapped to the nearest pixels
        let (mut x0, mut y0, mut x1, mut y1) = (0.0, 0.0, 0.0, 0.0);
        match snap_mode {
            SnapMode::Auto => {
                if total_vertices > 1024 {
                    return false;
                }
                let code = path.vertex(&mut x0, &mut y0);
                if code == PATH_CMD_STOP {
                    return false;
                }
                loop {
                    let code = path.vertex(&mut x1, &mut y1);
                    if code == PATH_CMD_STOP {
                        break;
                    }
                    match code {
                        PATH_CMD_CURVE3 | PATH_CMD_CURVE4 => return false,
                        PATH_CMD_LINE_TO => {
                            if (x0 - x1).abs() >= 1e-4 && (y0 - y1).abs() >= 1e-4 {
                                return false;
                            }
                        }
                        _ => {}
                    }
                    x0 = x1;
                    y0 = y1;
                }
                true
            }
            SnapMode::False => false,
            SnapMode::True => true,
        }
    }

    pub fn new(mut source: S, snap_mode: SnapMode, total_vertices: usize, stroke_width: f64) -> Self {
        let snap = Self::should_snap(&mut source, snap_mode, total_vertices);
        let mut snap_value = 0.0;
        if snap {
            let is_odd = mpl_round_to_int(stroke_width) % 2;
            snap_value = if is_odd != 0 { 0.5 } else { 0.0 };
        }
        source.rewind(0);
        PathSnapper { source, snap, snap_value }
    }

    pub fn is_snapping(&self) -> bool {
        self.snap
    }
}

impl<S: VertexSource> VertexSource for PathSnapper<S> {
    #[inline]
    fn rewind(&mut self, path_id: u32) {
        self.source.rewind(path_id);
    }

    #[inline]
    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let code = self.source.vertex(x, y);
        if self.snap && is_vertex(code) {
            *x = (*x + 0.5).floor() + self.snap_value;
            *y = (*y + 0.5).floor() + self.snap_value;
        }
        code
    }
}

// ---------------------------------------------------------------------------

pub struct PathSimplifier<S: VertexSource> {
    pub source: S,
    simplify: bool,
    simplify_threshold: f64,
    moveto: bool,
    after_moveto: bool,
    clipped: bool,
    has_init: bool,
    init_x: f64,
    init_y: f64,
    lastx: f64,
    lasty: f64,
    origdx: f64,
    origdy: f64,
    orig_dnorm2: f64,
    dnorm2_forward_max: f64,
    dnorm2_backward_max: f64,
    last_forward_max: bool,
    last_backward_max: bool,
    next_x: f64,
    next_y: f64,
    next_backward_x: f64,
    next_backward_y: f64,
    curr_vec_start_x: f64,
    curr_vec_start_y: f64,
    queue: EmbeddedQueue<9>,
}

impl<S: VertexSource> PathSimplifier<S> {
    pub fn new(source: S, do_simplify: bool, simplify_threshold: f64) -> Self {
        PathSimplifier {
            source,
            simplify: do_simplify,
            simplify_threshold: simplify_threshold * simplify_threshold,
            moveto: true,
            after_moveto: false,
            clipped: false,
            has_init: false,
            init_x: 0.0,
            init_y: 0.0,
            lastx: 0.0,
            lasty: 0.0,
            origdx: 0.0,
            origdy: 0.0,
            orig_dnorm2: 0.0,
            dnorm2_forward_max: 0.0,
            dnorm2_backward_max: 0.0,
            last_forward_max: false,
            last_backward_max: false,
            next_x: 0.0,
            next_y: 0.0,
            next_backward_x: 0.0,
            next_backward_y: 0.0,
            curr_vec_start_x: 0.0,
            curr_vec_start_y: 0.0,
            queue: EmbeddedQueue::new(),
        }
    }

    fn push_(&mut self, x: f64, y: f64) {
        let need_to_push_back = self.dnorm2_backward_max > 0.0;

        // If we observed any backward (anti-parallel) vectors, then
        // we need to push both forward and backward vectors.
        if need_to_push_back {
            // If the last vector seen was the maximum in the forward direction,
            // then we need to push the forward after the backward. Otherwise,
            // the last vector seen was the maximum in the backward direction,
            // or somewhere in between, either way we are safe pushing forward
            // before backward.
            if self.last_forward_max {
                self.queue.push(PATH_CMD_LINE_TO, self.next_backward_x, self.next_backward_y);
                self.queue.push(PATH_CMD_LINE_TO, self.next_x, self.next_y);
            } else {
                self.queue.push(PATH_CMD_LINE_TO, self.next_x, self.next_y);
                self.queue.push(PATH_CMD_LINE_TO, self.next_backward_x, self.next_backward_y);
            }
        } else {
            // If we did not observe any backwards vectors, just push forward.
            self.queue.push(PATH_CMD_LINE_TO, self.next_x, self.next_y);
        }

        // If we clipped some segments between this line and the next line
        // we are starting, we also need to move to the last point.
        if self.clipped {
            self.queue.push(PATH_CMD_MOVE_TO, self.lastx, self.lasty);
        } else if !self.last_forward_max && !self.last_backward_max {
            // If the last line was not the longest line, then move
            // back to the end point of the last line in the
            // sequence. Only do this if not clipped, since in that
            // case lastx,lasty is not part of the line just drawn.

            // Would be move_to if not for the artifacts
            self.queue.push(PATH_CMD_LINE_TO, self.lastx, self.lasty);
        }

        // Now reset all the variables to get ready for the next line
        self.origdx = x - self.lastx;
        self.origdy = y - self.lasty;
        self.orig_dnorm2 = self.origdx * self.origdx + self.origdy * self.origdy;

        self.dnorm2_forward_max = self.orig_dnorm2;
        self.last_forward_max = true;
        let last = self.queue.q[self.queue.write - 1];
        self.curr_vec_start_x = last.x;
        self.curr_vec_start_y = last.y;
        self.lastx = x;
        self.next_x = x;
        self.lasty = y;
        self.next_y = y;
        self.dnorm2_backward_max = 0.0;
        self.last_backward_max = false;

        self.clipped = false;
    }
}

impl<S: VertexSource> VertexSource for PathSimplifier<S> {
    fn rewind(&mut self, path_id: u32) {
        self.queue.clear();
        self.moveto = true;
        self.source.rewind(path_id);
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let mut cmd: u32 = 0;

        // The simplification algorithm doesn't support curves or compound paths
        // so we just don't do it at all in that case...
        if !self.simplify {
            return self.source.vertex(x, y);
        }

        if self.queue.pop(&mut cmd, x, y) {
            return cmd;
        }

        loop {
            cmd = self.source.vertex(x, y);
            if cmd == PATH_CMD_STOP {
                break;
            }
            // if we are starting a new path segment, move to the first point + init
            if self.moveto || cmd == PATH_CMD_MOVE_TO {
                // m_moveto check is not generally needed because
                // m_source generates an initial moveto; but it is
                // retained for safety in case circumstances arise
                // where this is not true.
                if self.orig_dnorm2 != 0.0 && !self.after_moveto {
                    // m_origdNorm2 is nonzero only if we have a
                    // vector; the m_after_moveto check ensures we
                    // push this vector to the queue only once.
                    self.push_(*x, *y);
                }
                self.after_moveto = true;

                if finite(*x, *y) {
                    self.has_init = true;
                    self.init_x = *x;
                    self.init_y = *y;
                } else {
                    self.has_init = false;
                }

                self.lastx = *x;
                self.lasty = *y;
                self.moveto = false;
                self.orig_dnorm2 = 0.0;
                self.dnorm2_backward_max = 0.0;
                self.clipped = true;
                if self.queue.nonempty() {
                    // If we did a push, empty the queue now.
                    break;
                }
                continue;
            }
            self.after_moveto = false;

            if is_close(cmd) {
                if self.has_init {
                    // If we have a valid initial vertex, then
                    // replace the current vertex with the initial vertex
                    *x = self.init_x;
                    *y = self.init_y;
                } else {
                    // If we don't have a valid initial vertex, then
                    // we can't close the path, so we skip the vertex
                    continue;
                }
            }

            // if we have no orig vector, set it to this vector and
            // continue.  this orig vector is the reference vector we
            // will build up the line to
            if self.orig_dnorm2 == 0.0 {
                if self.clipped {
                    self.queue.push(PATH_CMD_MOVE_TO, self.lastx, self.lasty);
                    self.clipped = false;
                }

                self.origdx = *x - self.lastx;
                self.origdy = *y - self.lasty;
                self.orig_dnorm2 = self.origdx * self.origdx + self.origdy * self.origdy;

                // set all the variables to reflect this new orig vector
                self.dnorm2_forward_max = self.orig_dnorm2;
                self.dnorm2_backward_max = 0.0;
                self.last_forward_max = true;
                self.last_backward_max = false;

                self.curr_vec_start_x = self.lastx;
                self.curr_vec_start_y = self.lasty;
                self.lastx = *x;
                self.next_x = *x;
                self.lasty = *y;
                self.next_y = *y;
                continue;
            }

            // If got to here, then we have an orig vector and we just got
            // a vector in the sequence.

            // get the v vector
            let totdx = *x - self.curr_vec_start_x;
            let totdy = *y - self.curr_vec_start_y;

            // get the dot product o.v
            let totdot = self.origdx * totdx + self.origdy * totdy;

            // get the para vector ( = (o.v)o/(o.o))
            let paradx = totdot * self.origdx / self.orig_dnorm2;
            let parady = totdot * self.origdy / self.orig_dnorm2;

            // get the perp vector ( = v - para)
            let perpdx = totdx - paradx;
            let perpdy = totdy - parady;

            // get the squared norm of perp vector ( = p.p)
            let perpd_norm2 = perpdx * perpdx + perpdy * perpdy;

            // If the perpendicular vector is less than
            // m_simplify_threshold pixels in size, then merge
            // current x,y with the current vector
            if perpd_norm2 < self.simplify_threshold {
                // check if the current vector is parallel or
                // anti-parallel to the orig vector. In either case,
                // test if it is the longest of the vectors
                // we are merging in that direction. If it is, then
                // update the current vector in that direction.
                let parad_norm2 = paradx * paradx + parady * parady;

                self.last_forward_max = false;
                self.last_backward_max = false;
                if totdot > 0.0 {
                    if parad_norm2 > self.dnorm2_forward_max {
                        self.last_forward_max = true;
                        self.dnorm2_forward_max = parad_norm2;
                        self.next_x = *x;
                        self.next_y = *y;
                    }
                } else if parad_norm2 > self.dnorm2_backward_max {
                    self.last_backward_max = true;
                    self.dnorm2_backward_max = parad_norm2;
                    self.next_backward_x = *x;
                    self.next_backward_y = *y;
                }

                self.lastx = *x;
                self.lasty = *y;
                continue;
            }

            // If we get here, then this vector was not similar enough to the
            // line we are building, so we need to draw that line and start the
            // next one.

            // If the line needs to extend in the opposite direction from the
            // direction we are drawing in, move back to we start drawing from
            // back there.
            self.push_(*x, *y);

            break;
        }

        // Fill the queue with the remaining vertices if we've finished the
        // path in the above loop.
        if cmd == PATH_CMD_STOP {
            if self.orig_dnorm2 != 0.0 {
                let c = if self.moveto || self.after_moveto { PATH_CMD_MOVE_TO } else { PATH_CMD_LINE_TO };
                self.queue.push(c, self.next_x, self.next_y);
                if self.dnorm2_backward_max > 0.0 {
                    let c = if self.moveto || self.after_moveto { PATH_CMD_MOVE_TO } else { PATH_CMD_LINE_TO };
                    self.queue.push(c, self.next_backward_x, self.next_backward_y);
                }
                self.moveto = false;
            }
            let c = if self.moveto || self.after_moveto { PATH_CMD_MOVE_TO } else { PATH_CMD_LINE_TO };
            self.queue.push(c, self.lastx, self.lasty);
            self.moveto = false;
            self.queue.push(PATH_CMD_STOP, 0.0, 0.0);
        }

        // Return the first item in the queue, if any, otherwise
        // indicate that we're done.
        if self.queue.pop(&mut cmd, x, y) {
            cmd
        } else {
            PATH_CMD_STOP
        }
    }
}

// ---------------------------------------------------------------------------

/// The linear-congruential RNG used by `Sketch`.
struct RandomNumberGenerator {
    seed: u32,
}

impl RandomNumberGenerator {
    fn get_double(&mut self) -> f64 {
        self.seed = 214013u32.wrapping_mul(self.seed).wrapping_add(2531011);
        self.seed as f64 / (1u64 << 32) as f64
    }
}

/// `agg::conv_segmentator` (`vpgen_segmentator` with approximation scale 1).
struct Segmentator<S: VertexSource> {
    source: S,
    // vpgen state
    x1: f64,
    y1: f64,
    dx: f64,
    dy: f64,
    dl: f64,
    ddl: f64,
    cmd: u32,
    // conv_adaptor_vpgen state
    start_x: f64,
    start_y: f64,
    poly_flags: u32,
    vertices: i32,
}

impl<S: VertexSource> Segmentator<S> {
    fn new(source: S) -> Self {
        Segmentator {
            source,
            x1: 0.0,
            y1: 0.0,
            dx: 0.0,
            dy: 0.0,
            dl: 2.0,
            ddl: 2.0,
            cmd: PATH_CMD_STOP,
            start_x: 0.0,
            start_y: 0.0,
            poly_flags: 0,
            vertices: 0,
        }
    }

    fn vp_reset(&mut self) {
        self.cmd = PATH_CMD_STOP;
    }
    fn vp_move_to(&mut self, x: f64, y: f64) {
        self.x1 = x;
        self.y1 = y;
        self.dx = 0.0;
        self.dy = 0.0;
        self.dl = 2.0;
        self.ddl = 2.0;
        self.cmd = PATH_CMD_MOVE_TO;
    }
    fn vp_line_to(&mut self, x: f64, y: f64) {
        self.x1 += self.dx;
        self.y1 += self.dy;
        self.dx = x - self.x1;
        self.dy = y - self.y1;
        let approximation_scale = 1.0;
        let mut len = (self.dx * self.dx + self.dy * self.dy).sqrt() * approximation_scale;
        if len < 1e-30 {
            len = 1e-30;
        }
        self.ddl = 1.0 / len;
        self.dl = if self.cmd == PATH_CMD_MOVE_TO { 0.0 } else { self.ddl };
        if self.cmd == PATH_CMD_STOP {
            self.cmd = PATH_CMD_LINE_TO;
        }
    }
    fn vp_vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        if self.cmd == PATH_CMD_STOP {
            return PATH_CMD_STOP;
        }
        let cmd = self.cmd;
        self.cmd = PATH_CMD_LINE_TO;
        if self.dl >= 1.0 - self.ddl {
            self.dl = 1.0;
            self.cmd = PATH_CMD_STOP;
            *x = self.x1 + self.dx;
            *y = self.y1 + self.dy;
            return cmd;
        }
        *x = self.x1 + self.dx * self.dl;
        *y = self.y1 + self.dy * self.dl;
        self.dl += self.ddl;
        cmd
    }
}

impl<S: VertexSource> VertexSource for Segmentator<S> {
    fn rewind(&mut self, path_id: u32) {
        self.source.rewind(path_id);
        self.vp_reset();
        self.start_x = 0.0;
        self.start_y = 0.0;
        self.poly_flags = 0;
        self.vertices = 0;
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        // conv_adaptor_vpgen::vertex; vpgen_segmentator neither auto-closes
        // nor auto-uncloses.
        let mut cmd;
        loop {
            cmd = self.vp_vertex(x, y);
            if !is_stop(cmd) {
                break;
            }

            if self.poly_flags != 0 {
                *x = 0.0;
                *y = 0.0;
                cmd = self.poly_flags;
                self.poly_flags = 0;
                break;
            }

            if self.vertices < 0 {
                if self.vertices < -1 {
                    self.vertices = 0;
                    return PATH_CMD_STOP;
                }
                let (sx, sy) = (self.start_x, self.start_y);
                self.vp_move_to(sx, sy);
                self.vertices = 1;
                continue;
            }

            let (mut tx, mut ty) = (0.0, 0.0);
            cmd = self.source.vertex(&mut tx, &mut ty);
            if is_vertex(cmd) {
                if is_move_to(cmd) {
                    self.vp_move_to(tx, ty);
                    self.start_x = tx;
                    self.start_y = ty;
                    self.vertices = 1;
                } else {
                    self.vp_line_to(tx, ty);
                    self.vertices += 1;
                }
            } else if is_end_poly(cmd) {
                self.poly_flags = cmd;
                if is_closed(cmd) {
                    if self.vertices > 2 {
                        let (sx, sy) = (self.start_x, self.start_y);
                        self.vp_line_to(sx, sy);
                    }
                    self.vertices = 0;
                }
            } else {
                // path_cmd_stop
                break;
            }
        }
        cmd
    }
}

#[inline(always)]
fn is_closed(c: u32) -> bool {
    (c & PATH_FLAGS_CLOSE) != 0
}

pub struct Sketch<S: VertexSource> {
    scale: f64,
    segmented: Segmentator<S>,
    last_x: f64,
    last_y: f64,
    has_last: bool,
    p: f64,
    rand: RandomNumberGenerator,
    p_scale: f64,
    log_randomness: f64,
}

impl<S: VertexSource> Sketch<S> {
    pub fn new(source: S, scale: f64, length: f64, randomness: f64) -> Self {
        let p_scale = if length <= f64::EPSILON || randomness <= f64::EPSILON {
            0.0
        } else {
            (2.0 * std::f64::consts::PI) / (length * randomness)
        };
        let log_randomness = if randomness <= f64::EPSILON { 0.0 } else { 2.0 * randomness.ln() };
        let mut s = Sketch {
            scale,
            segmented: Segmentator::new(source),
            last_x: 0.0,
            last_y: 0.0,
            has_last: false,
            p: 0.0,
            rand: RandomNumberGenerator { seed: 0 },
            p_scale,
            log_randomness,
        };
        s.rewind(0);
        s
    }
}

impl<S: VertexSource> VertexSource for Sketch<S> {
    fn rewind(&mut self, path_id: u32) {
        self.has_last = false;
        self.p = 0.0;
        if self.scale != 0.0 {
            self.rand.seed = 0;
            self.segmented.rewind(path_id);
        } else {
            self.segmented.source.rewind(path_id);
        }
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        if self.scale == 0.0 {
            return self.segmented.source.vertex(x, y);
        }

        let code = self.segmented.vertex(x, y);

        if code == PATH_CMD_MOVE_TO {
            self.has_last = false;
            self.p = 0.0;
        }

        if self.has_last {
            // We want the "cursor" along the sine wave to move at a
            // random rate.
            let d_rand = self.rand.get_double();
            self.p += (d_rand * self.log_randomness).exp();
            let den = self.last_x - *x;
            let num = self.last_y - *y;
            let mut len = num * num + den * den;
            self.last_x = *x;
            self.last_y = *y;
            if len != 0.0 {
                len = len.sqrt();
                let r = (self.p * self.p_scale).sin() * self.scale;
                let roverlen = r / len;
                *x += roverlen * num;
                *y -= roverlen * den;
            }
        } else {
            self.last_x = *x;
            self.last_y = *y;
        }

        self.has_last = true;

        code
    }
}
