//! Many artists drawn at once.
//!
//! An `ArtistBatch` keeps the lines and patches of one plotting call (the
//! components of a boxplot) as arrays instead of one handle per artist; it is
//! one child of its Axes until an artist is changed, when `explode` replaces
//! it by the artists' handles.
//!
//! `draw_run` draws consecutive lines, patches and batches of an Axes.  When
//! the renderer is the Agg one and their `draw` only makes plain `draw_path`
//! and `draw_markers` calls, these are handed over together
//! (`RendererAgg::draw_artists`) to be rasterized on several threads; the
//! image is the same as when each artist draws itself.

use std::sync::{Arc, MutexGuard, TryLockError};

use super::artist::{shared, ClipCtx, GcState, Lock, Shared, TSpec, TransCtx};
use super::axes::Child;
use super::lines::{simplify_params, Line2D};
use super::patches::{Patch, PatchKind};
use super::path::{new_path, Path, SimplifyParams, LINETO};
use super::renderer::Renderer;
use super::transforms::Mat;
use crate::agg::pixfmt::Rgba;
use crate::agg::TransAffine;
use crate::backend::renderer_agg::{ArtistOp, ArtistOps, Gc, MarkerStamp, MarkersOp, PathOp, RendererAgg};

/// The artists of a slot: artist i of a batch of n has the points
/// `i * m .. (i + 1) * m` (`Lines`, `Patches`) or `offsets[i] .. offsets[i + 1]`
/// (`Ragged`) of the slot.
#[derive(Clone, Debug)]
pub enum SlotKind {
    Lines { style: Line2D, m: usize },
    Ragged { style: Line2D, offsets: Vec<usize> },
    /// Path patches with the path codes `codes`, created when `simplify`
    /// were the simplification settings.
    Patches { style: Patch, m: usize, codes: Vec<u8>, simplify: SimplifyParams },
}

/// One artist per box of a batch: e.g. the lower whiskers.
#[derive(Clone, Debug)]
pub struct Slot {
    pub kind: SlotKind,
    pub xy: Vec<[f64; 2]>,
    /// lines without label are labelled `_child{index among the children}`
    pub auto_label: bool,
    /// the artists as children, once they are needed
    handles: Option<Vec<Child>>,
}

impl Slot {
    pub fn new(kind: SlotKind, xy: Vec<[f64; 2]>, auto_label: bool) -> Slot {
        Slot { kind, xy, auto_label, handles: None }
    }

    fn zorder(&self) -> f64 {
        match &self.kind {
            SlotKind::Lines { style, .. } | SlotKind::Ragged { style, .. } => style.base.zorder,
            SlotKind::Patches { style, .. } => style.base.zorder,
        }
    }

    fn points(&self, i: usize) -> &[[f64; 2]] {
        match &self.kind {
            SlotKind::Lines { m, .. } | SlotKind::Patches { m, .. } => &self.xy[i * m..(i + 1) * m],
            SlotKind::Ragged { offsets, .. } => &self.xy[offsets[i]..offsets[i + 1]],
        }
    }

    /// The most points an artist has.
    fn max_points(&self, n: usize) -> usize {
        match &self.kind {
            SlotKind::Lines { m, .. } | SlotKind::Patches { m, .. } => *m,
            SlotKind::Ragged { offsets, .. } => (0..n).map(|i| offsets[i + 1] - offsets[i]).max().unwrap_or(0),
        }
    }
}

/// See the module documentation.  Artist *i* of slot *s* is child
/// `base + i * slots.len() + s` of the Axes.
#[derive(Debug)]
pub struct ArtistBatch {
    pub n: usize,
    pub base: usize,
    pub slots: Vec<Slot>,
    /// the Axes has the artists' handles instead of the batch
    pub exploded: bool,
}

impl ArtistBatch {
    pub fn len(&self) -> usize {
        self.n * self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The distinct zorders of the artists, in slot order.
    pub fn zorders(&self) -> Vec<f64> {
        let mut out: Vec<f64> = Vec::new();
        for s in &self.slots {
            let z = s.zorder();
            if !out.iter().any(|v| v.to_bits() == z.to_bits()) {
                out.push(z);
            }
        }
        out
    }

    /// The artists of slot *s*, as the handles `Line2D.copy_many` and
    /// `Patch.copy_many_paths` make.
    pub fn handles(&mut self, s: usize) -> &[Child] {
        if self.slots[s].handles.is_none() {
            let nslots = self.slots.len();
            let (n, base) = (self.n, self.base);
            let slot = &self.slots[s];
            let handles = (0..n)
                .map(|i| {
                    let pts = slot.points(i);
                    match &slot.kind {
                        SlotKind::Lines { style, .. } | SlotKind::Ragged { style, .. } => {
                            let mut line = style.clone();
                            line.set_data(pts.iter().map(|p| p[0]).collect(), pts.iter().map(|p| p[1]).collect());
                            if slot.auto_label {
                                line.base.label = format!("_child{}", base + i * nslots + s);
                            }
                            Child::Line(shared(line))
                        }
                        SlotKind::Patches { style, codes, simplify, .. } => {
                            let mut patch = style.clone();
                            patch.kind = PatchKind::Generic {
                                path: Arc::new(new_path(pts.to_vec(), Some(codes.clone()), *simplify)),
                                patch_transform: None,
                            };
                            Child::Patch(shared(patch))
                        }
                    }
                })
                .collect();
            self.slots[s].handles = Some(handles);
        }
        self.slots[s].handles.as_deref().unwrap()
    }

    /// The handles of all artists in child order; the batch draws nothing
    /// from now on (its arrays are dropped).
    pub fn explode(&mut self) -> Vec<Child> {
        let nslots = self.slots.len();
        for s in 0..nslots {
            self.handles(s);
        }
        let mut out = Vec::with_capacity(self.len());
        for i in 0..self.n {
            for slot in &self.slots {
                out.push(slot.handles.as_ref().unwrap()[i].clone());
            }
        }
        if !self.exploded {
            self.exploded = true;
            for slot in &mut self.slots {
                slot.xy = Vec::new();
            }
        }
        out
    }

    /// The vertices that count for the data limits: (points, transform) of
    /// each slot of visible (if *visible_only*) lines.
    pub fn limit_points(&self, visible_only: bool) -> Vec<(&[[f64; 2]], &TSpec)> {
        self.slots
            .iter()
            .filter_map(|slot| match &slot.kind {
                SlotKind::Lines { style, .. } | SlotKind::Ragged { style, .. } if !visible_only || style.base.visible => {
                    Some((&slot.xy[..], &style.base.transform))
                }
                _ => None,
            })
            .collect()
    }

    /// Each artist of the layer *z* drawing itself, in child order.
    fn draw_layer(
        &self,
        z: f64,
        renderer: &mut Renderer,
        tc: &TransCtx,
        clip: &ClipCtx,
        xbound: Option<(f64, f64)>,
        subslice_ok: bool,
    ) {
        // (with drawing disabled, for layouts, lines and patches do nothing)
        if self.exploded || renderer.disabled {
            return;
        }
        // one artist per slot, given the data of each box in turn
        enum Scratch<'a> {
            Line(Line2D, &'a Slot),
            Patch(Patch, &'a Slot),
        }
        let mut scratch: Vec<Scratch> = self
            .slots
            .iter()
            .filter(|s| s.zorder().to_bits() == z.to_bits())
            .map(|s| match &s.kind {
                SlotKind::Lines { style, .. } | SlotKind::Ragged { style, .. } => Scratch::Line(style.clone(), s),
                SlotKind::Patches { style, .. } => Scratch::Patch(style.clone(), s),
            })
            .collect();
        for i in 0..self.n {
            for sc in &mut scratch {
                match sc {
                    Scratch::Line(line, slot) => {
                        let pts = slot.points(i);
                        line.set_data(pts.iter().map(|p| p[0]).collect(), pts.iter().map(|p| p[1]).collect());
                        line.base.clip_ident = next_clip_ident();
                        let ok = subslice_ok && line.base.transform == TSpec::Data;
                        line.draw(renderer, tc, clip, xbound, ok);
                    }
                    Scratch::Patch(patch, slot) => {
                        let SlotKind::Patches { codes, simplify, .. } = &slot.kind else { unreachable!() };
                        patch.kind = PatchKind::Generic {
                            path: Arc::new(new_path(slot.points(i).to_vec(), Some(codes.clone()), *simplify)),
                            patch_transform: None,
                        };
                        patch.base.clip_ident = next_clip_ident();
                        patch.draw(renderer, tc, clip);
                    }
                }
            }
        }
    }
}

/// A clip identity (see `ArtistBase::clip_ident`) no artist has: artists
/// are identified by their (even) addresses.
fn next_clip_ident() -> usize {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
    NEXT.fetch_add(2, std::sync::atomic::Ordering::Relaxed)
}

/// An item of a run of `draw_run`: a child drawn at its zorder (for a
/// batch: its artists of that zorder).
pub type RunItem = (Child, f64);

/// Whether `draw_run` draws this child.
pub fn in_run(c: &Child) -> bool {
    matches!(c, Child::Line(_) | Child::Patch(_) | Child::Batch(_))
}

/// The number of artists of a run.
pub fn run_len(items: &[RunItem]) -> usize {
    items
        .iter()
        .map(|(c, z)| match c {
            Child::Batch(b) => {
                let b = b.lk();
                b.n * b.slots.iter().filter(|s| s.zorder().to_bits() == z.to_bits()).count()
            }
            _ => 1,
        })
        .sum()
}

/// Draw the items one after the other: `Line2D::draw`, `Patch::draw` and
/// the same for each artist of a batch.
pub fn draw_items(
    items: &[RunItem],
    renderer: &mut Renderer,
    tc: &TransCtx,
    clip: &ClipCtx,
    xbound: Option<(f64, f64)>,
    subslice_ok: bool,
) {
    for (c, z) in items {
        match c {
            Child::Line(l) => {
                let mut l = l.lk();
                let ok = subslice_ok && l.base.transform == TSpec::Data;
                l.draw(renderer, tc, clip, xbound, ok);
            }
            Child::Patch(p) => p.lk().draw(renderer, tc, clip),
            Child::Batch(b) => b.lk().draw_layer(*z, renderer, tc, clip, xbound, subslice_ok),
            _ => unreachable!("not an item of a run"),
        }
    }
}

/// Below this many artists a run is drawn artist by artist.
const MIN_RUN: usize = 256;

/// The most items planned (and locked) at a time: a longer run is drawn in
/// parts, so that the plans of very many artists do not all exist at once.
const MAX_PLANNED: usize = 16384;

/// Draw a run of items (see `in_run`), with the image of `draw_items`.
pub fn draw_run(
    items: &[RunItem],
    renderer: &mut Renderer,
    tc: &TransCtx,
    clip: &ClipCtx,
    xbound: Option<(f64, f64)>,
    subslice_ok: bool,
) {
    if renderer.foreign.is_some() || renderer.disabled || run_len(items) < MIN_RUN {
        return draw_items(items, renderer, tc, clip, xbound, subslice_ok);
    }
    let mut start = 0;
    while start < items.len() {
        // as many items as have a plan, then the one that has none
        let mut locked = Locked::default();
        let mut planned: Vec<Planned> = Vec::new();
        let mut end = start;
        while end < items.len() && planned.len() < MAX_PLANNED {
            match locked.plan(&items[end], renderer, tc, clip, subslice_ok) {
                Some(p) => planned.push(p),
                None => break,
            }
            end += 1;
        }
        if !planned.is_empty() {
            let run = Run::new(&locked, &planned);
            renderer.agg.draw_artists(&run);
        }
        let full = planned.len() == MAX_PLANNED;
        drop(planned);
        drop(locked);
        // (the item that has no plan draws itself)
        if end < items.len() && !full {
            draw_items(&items[end..end + 1], renderer, tc, clip, xbound, subslice_ok);
            end += 1;
        }
        start = end;
    }
}

/// The locks held while a run is drawn.
#[derive(Default)]
struct Locked<'a> {
    lines: Vec<MutexGuard<'a, Line2D>>,
    patches: Vec<MutexGuard<'a, Patch>>,
    /// (address of the batch, its lock)
    batches: Vec<(usize, MutexGuard<'a, ArtistBatch>)>,
    stamps: Stamps,
}

/// The marker stamps of a run.
#[derive(Default)]
struct Stamps {
    stamps: Vec<MarkerStamp>,
    /// (stamp, what it was made from) of the last line with markers
    last: Option<(usize, StampKey)>,
}

/// What a marker stamp depends on.
#[derive(PartialEq)]
struct StampKey {
    gc: (f64, Rgba, crate::agg::stroke::LineCap, crate::agg::stroke::LineJoin, crate::backend::path_converters::SnapMode),
    path: Path,
    trans: Mat,
    face: Option<Rgba>,
}

/// The `draw_path` call of an artist, except its path.
struct PathPlan {
    gc: Gc,
    face: Option<Rgba>,
    trans: TransAffine,
}

/// The `draw_markers` call of a line, except its points.
struct MarkersPlan {
    stamp: usize,
    cliprect: Option<[f64; 4]>,
    trans: TransAffine,
}

/// A slot of a batch layer.
struct SlotPlan {
    slot: usize,
    path: Option<PathPlan>,
    markers: Option<MarkersPlan>,
    simplify: SimplifyParams,
    /// whether the paths have only MOVETO and LINETO codes
    plain_codes: bool,
}

/// How an item of a run is drawn (the indices are those of its lock in `Locked`).
enum Planned {
    Line { guard: usize, path: Option<PathPlan>, markers: Option<MarkersPlan> },
    Patch { guard: usize, path: Option<PathPlan> },
    Layer { guard: usize, slots: Vec<SlotPlan> },
}

fn try_lock<T>(m: &Shared<T>) -> Option<MutexGuard<'_, T>> {
    match m.try_lock() {
        Ok(g) => Some(g),
        Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
        // (locked for this run already: the artist is a child twice)
        Err(TryLockError::WouldBlock) => None,
    }
}

/// The `draw_path` call with the context *gc* and the face *face*, if
/// `RendererAgg::draw_artists` can make it.
fn path_plan(gc: &GcState, face: Option<Rgba>, affine: &Mat) -> Option<PathPlan> {
    let agg = gc.to_agg();
    RendererAgg::can_batch(&agg).then(|| PathPlan { face: gc.face(face, false), gc: agg, trans: TransAffine::from_matrix(affine) })
}

impl Stamps {
    /// The `draw_markers` call of *line* (see `Line2D::plan`), if
    /// `RendererAgg::draw_artists` can make it.
    fn markers_plan(&mut self, line: &Line2D, m: &super::lines::MarkerPlan, affine: &Mat, renderer: &Renderer) -> Option<MarkersPlan> {
        let agg = m.gc.to_agg();
        if !RendererAgg::can_batch(&agg) {
            return None;
        }
        let face = m.gc.face(Some(m.face), false);
        let gc = (agg.linewidth, agg.color, agg.cap, agg.join, agg.snap_mode);
        let stamp = match &self.last {
            Some((i, k)) if k.gc == gc && k.trans == m.trans && k.face == face && k.path == *line.marker_path() => *i,
            _ => {
                let stamp = renderer.agg.prepare_markers(&agg, line.marker_path(), &TransAffine::from_matrix(&m.trans), face);
                self.stamps.push(stamp);
                let key = StampKey { gc, path: line.marker_path().clone(), trans: m.trans, face };
                self.last = Some((self.stamps.len() - 1, key));
                self.stamps.len() - 1
            }
        };
        Some(MarkersPlan { stamp, cliprect: agg.cliprect, trans: TransAffine::from_matrix(affine) })
    }

    /// (path, markers) of *line* with the plan *plan*.
    fn line_plans(&mut self, line: &Line2D, plan: &super::lines::LinePlan, renderer: &Renderer) -> Option<(Option<PathPlan>, Option<MarkersPlan>)> {
        let path = match &plan.line {
            Some(gc) => Some(path_plan(gc, None, &plan.affine)?),
            None => None,
        };
        let markers = match &plan.markers {
            Some(m) => Some(self.markers_plan(line, m, &plan.affine, renderer)?),
            None => None,
        };
        Some((path, markers))
    }

    /// The plans of the slots of the layer *z* of *batch*.
    fn layer_plans(&mut self, batch: &ArtistBatch, z: f64, renderer: &Renderer, tc: &TransCtx, clip: &ClipCtx) -> Option<Vec<SlotPlan>> {
        if batch.exploded {
            return Some(Vec::new());
        }
        let params = simplify_params();
        let mut out = Vec::new();
        for (s, slot) in batch.slots.iter().enumerate() {
            if slot.zorder().to_bits() != z.to_bits() {
                continue;
            }
            match &slot.kind {
                SlotKind::Lines { style, .. } | SlotKind::Ragged { style, .. } => {
                    // (`Line2D::draw` uses all the points of lines it does
                    // not subslice: those of at most 1000 points)
                    if !style.has_default_drawstyle() || slot.max_points(batch.n) > 1000 {
                        return None;
                    }
                    let plan = style.plan(renderer, tc, clip)?;
                    let (path, markers) = self.line_plans(style, &plan, renderer)?;
                    out.push(SlotPlan { slot: s, path, markers, simplify: params, plain_codes: true });
                }
                SlotKind::Patches { style, codes, simplify, .. } => {
                    let plan = style.plan(tc, clip)?;
                    let path = match &plan.gc {
                        Some(gc) => Some(path_plan(gc, plan.face, &plan.affine)?),
                        None => None,
                    };
                    let plain_codes = codes.iter().all(|&c| c <= LINETO);
                    out.push(SlotPlan { slot: s, path, markers: None, simplify: *simplify, plain_codes });
                }
            }
        }
        Some(out)
    }
}

impl<'a> Locked<'a> {
    /// Lock *item* and plan its drawing; None if it draws itself (a line or
    /// patch is not locked then).
    fn plan(&mut self, item: &'a RunItem, renderer: &Renderer, tc: &TransCtx, clip: &ClipCtx, subslice_ok: bool) -> Option<Planned> {
        match &item.0 {
            Child::Line(l) => {
                let mut g = try_lock(l)?;
                let ok = subslice_ok && g.base.transform == TSpec::Data;
                if g.base.visible {
                    g.build_cache(ok);
                    g.cached_path()?;
                }
                let plan = g.plan(renderer, tc, clip)?;
                let (path, markers) = self.stamps.line_plans(&g, &plan, renderer)?;
                self.lines.push(g);
                Some(Planned::Line { guard: self.lines.len() - 1, path, markers })
            }
            Child::Patch(p) => {
                let g = try_lock(p)?;
                g.stored_path()?;
                let plan = g.plan(tc, clip)?;
                let path = match &plan.gc {
                    Some(gc) => Some(path_plan(gc, plan.face, &plan.affine)?),
                    None => None,
                };
                self.patches.push(g);
                Some(Planned::Patch { guard: self.patches.len() - 1, path })
            }
            Child::Batch(b) => {
                // (both layers of a batch may be in the run; a batch that has
                // no plan stays locked until the run so far is drawn)
                let key = Arc::as_ptr(b) as usize;
                let guard = match self.batches.iter().position(|(k, _)| *k == key) {
                    Some(i) => i,
                    None => {
                        self.batches.push((key, try_lock(b)?));
                        self.batches.len() - 1
                    }
                };
                let slots = self.stamps.layer_plans(&self.batches[guard].1, item.1, renderer, tc, clip)?;
                Some(Planned::Layer { guard, slots })
            }
            _ => None,
        }
    }
}

/// The artists of a planned run, for `RendererAgg::draw_artists`.
struct Run<'a> {
    locked: &'a Locked<'a>,
    planned: &'a [Planned],
    /// the index of the first artist of each planned item, and the total
    starts: Vec<usize>,
}

impl<'a> Run<'a> {
    fn new(locked: &'a Locked<'a>, planned: &'a [Planned]) -> Run<'a> {
        let mut starts = Vec::with_capacity(planned.len() + 1);
        let mut total = 0;
        for p in planned {
            starts.push(total);
            total += match p {
                Planned::Layer { guard, slots } => locked.batches[*guard].1.n * slots.len(),
                _ => 1,
            };
        }
        starts.push(total);
        Run { locked, planned, starts }
    }

    fn path_op<'b>(plan: &'b PathPlan, verts: &'b [[f64; 2]], codes: Option<&'b [u8]>, should_simplify: bool, threshold: f64) -> PathOp<'b> {
        PathOp {
            gc: &plan.gc,
            verts,
            codes,
            should_simplify,
            simplify_threshold: threshold,
            trans: &plan.trans,
            face: plan.face,
        }
    }

    fn markers_op<'b>(&'b self, plan: &'b MarkersPlan, points: &'b [[f64; 2]]) -> MarkersOp<'b> {
        MarkersOp { stamp: &self.locked.stamps.stamps[plan.stamp], cliprect: plan.cliprect, points, trans: &plan.trans }
    }
}

impl ArtistOps for Run<'_> {
    fn len(&self) -> usize {
        *self.starts.last().unwrap()
    }

    fn get(&self, i: usize) -> ArtistOp<'_> {
        let k = self.starts.partition_point(|&s| s <= i) - 1;
        match &self.planned[k] {
            Planned::Line { guard, path, markers } => {
                let line = &self.locked.lines[*guard];
                // (an invisible line has no plans, and maybe no cache)
                let Some((p, xy)) = line.cached_path() else { return ArtistOp::default() };
                ArtistOp {
                    // (`Line2D::draw` draws no empty path, nor markers at no point)
                    path: path.as_ref().filter(|_| !p.vertices.is_empty()).map(|plan| {
                        Self::path_op(plan, &p.vertices, p.codes.as_deref(), p.should_simplify, p.simplify_threshold)
                    }),
                    markers: markers.as_ref().filter(|_| !xy.is_empty()).map(|plan| self.markers_op(plan, xy)),
                }
            }
            Planned::Patch { guard, path } => {
                let p = self.locked.patches[*guard].stored_path().unwrap();
                ArtistOp {
                    path: path.as_ref().map(|plan| {
                        Self::path_op(plan, &p.vertices, p.codes.as_deref(), p.should_simplify, p.simplify_threshold)
                    }),
                    markers: None,
                }
            }
            Planned::Layer { guard, slots } => {
                let batch = &self.locked.batches[*guard].1;
                let j = i - self.starts[k];
                let plan = &slots[j % slots.len()];
                let slot = &batch.slots[plan.slot];
                let pts = slot.points(j / slots.len());
                let codes = match &slot.kind {
                    SlotKind::Patches { codes, .. } => Some(&codes[..]),
                    _ => None,
                };
                // (`new_path`)
                let sp = plan.simplify;
                let should_simplify = sp.threshold > 0.0 && sp.simplify && pts.len() >= 128 && plan.plain_codes;
                ArtistOp {
                    path: plan
                        .path
                        .as_ref()
                        .filter(|_| codes.is_some() || !pts.is_empty())
                        .map(|p| Self::path_op(p, pts, codes, should_simplify, sp.threshold)),
                    markers: plan.markers.as_ref().filter(|_| !pts.is_empty()).map(|m| self.markers_op(m, pts)),
                }
            }
        }
    }
}
