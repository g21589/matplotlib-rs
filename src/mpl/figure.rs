//! Port of `matplotlib.figure.Figure` (drawing, subplots, savefig).

use super::artist::{shared, ClipCtx, ExtArtist, Lock, Shared, TSpec, TransCtx};
use super::axes::{Axes, FigCtx, GridSpec, SharedLims, SubplotParams, SubplotSpec};
use super::colors::ColorSpec;
use super::legend::Legend;
use super::lines::Line2D;
use super::patches::Patch;
use super::rc::rc;
use super::renderer::Renderer;
use super::text::Text;
use super::transforms::{Affine2D, Bbox, Transform};

#[derive(Clone, Debug, PartialEq)]
pub enum LayoutEngine {
    None,
    Tight { pad: f64, h_pad: Option<f64>, w_pad: Option<f64>, rect: Option<(f64, f64, f64, f64)> },
    Constrained,
}

#[derive(Clone, Debug)]
pub enum FigChild {
    Text(Shared<Text>),
    Line(Shared<Line2D>),
    Patch(Shared<Patch>),
    Legend(Shared<Legend>),
    /// An external artist and its `get_children` category (0 artists,
    /// 2 lines, 3 patches, 4 texts, 5 images, 6 legends; axes are 1).
    Ext(std::sync::Arc<dyn ExtArtist>, u8),
}

#[derive(Debug)]
pub struct Figure {
    pub figsize: (f64, f64),
    pub dpi: f64,
    pub patch: Patch,
    pub axes: Vec<Axes>,
    /// axes created but not (yet) added to the figure
    pub detached: Vec<Axes>,
    /// axes uids in "current axes" order (last is current)
    pub axstack: Vec<u64>,
    pub children: Vec<FigChild>,
    pub suptitle: Option<Shared<Text>>,
    pub supxlabel: Option<Shared<Text>>,
    pub supylabel: Option<Shared<Text>>,
    pub subplotpars: SubplotParams,
    pub gridspecs: Vec<Shared<GridSpec>>,
    pub layout: LayoutEngine,
    pub visible: bool,
    /// `transFigure._boxout` override (bbox_inches adjustment)
    pub boxout: Option<Bbox>,
    /// identity of the owning handle (set by the Python layer; 0 if none)
    pub key: usize,
}

impl Figure {
    /// Whether any external (Python-drawn) artist is in the figure.
    pub fn has_ext(&self) -> bool {
        self.children.iter().any(|c| matches!(c, FigChild::Ext(..)))
            || self.axes.iter().any(|a| a.children.iter().any(|c| matches!(c, super::axes::Child::Ext(_))))
    }

    pub fn new(figsize: Option<(f64, f64)>, dpi: Option<f64>, facecolor: Option<ColorSpec>, edgecolor: Option<ColorSpec>, linewidth: f64, frameon: Option<bool>) -> Figure {
        let r = rc();
        let fs = r.num_list("figure.figsize");
        let figsize = figsize.unwrap_or((fs[0], fs[1]));
        let dpi = dpi.unwrap_or_else(|| r.num("figure.dpi"));
        let facecolor = facecolor.unwrap_or_else(|| ColorSpec::Rgba(r.color("figure.facecolor")));
        let edgecolor = edgecolor.unwrap_or_else(|| ColorSpec::Rgba(r.color("figure.edgecolor")));
        let frameon = frameon.unwrap_or_else(|| r.boolean("figure.frameon"));
        let mut patch = Patch::rectangle(0.0, 0.0, 1.0, 1.0);
        patch.base.visible = frameon;
        patch.set_facecolor(Some(facecolor));
        patch.set_edgecolor(Some(edgecolor));
        patch.set_linewidth(Some(linewidth));
        patch.base.in_layout = false;
        patch.base.transform = TSpec::Figure;
        patch.antialiased = false;
        let layout = if r.boolean("figure.autolayout") {
            LayoutEngine::Tight { pad: 1.08, h_pad: None, w_pad: None, rect: None }
        } else if r.boolean("figure.constrained_layout.use") {
            LayoutEngine::Constrained
        } else {
            LayoutEngine::None
        };
        Figure {
            figsize,
            dpi,
            patch,
            axes: Vec::new(),
            detached: Vec::new(),
            axstack: Vec::new(),
            children: Vec::new(),
            suptitle: None,
            supxlabel: None,
            supylabel: None,
            subplotpars: SubplotParams::from_rc(),
            gridspecs: Vec::new(),
            layout,
            visible: true,
            boxout: None,
            key: 0,
        }
    }

    pub fn fig_bbox(&self) -> Bbox {
        let t = Transform::Affine(Affine2D::identity().scale(self.dpi, self.dpi).0);
        Bbox::from_bounds(0.0, 0.0, self.figsize.0, self.figsize.1).transformed_bbox(&t)
    }

    pub fn ctx(&self) -> FigCtx {
        let dpi_scale = Transform::Affine(Affine2D::identity().scale(self.dpi, self.dpi).0);
        let fig_bbox = self.fig_bbox();
        let tf = Transform::bbox_to(self.boxout.as_ref().unwrap_or(&fig_bbox));
        FigCtx { dpi: self.dpi, trans_figure: tf, dpi_scale, fig_bbox }
    }

    pub fn trans_ctx(&self) -> TransCtx {
        let c = self.ctx();
        let mut t = TransCtx::figure_only(self.dpi, c.trans_figure.clone());
        t.dpi_scale = c.dpi_scale;
        t
    }

    /// Canvas size in pixels (`get_width_height(physical=True)`).
    pub fn canvas_size(&self) -> (u32, u32) {
        let b = self.fig_bbox();
        (b.xmax() as u32, b.ymax() as u32)
    }

    pub fn axes_index(&self, uid: u64) -> Option<usize> {
        self.axes.iter().position(|a| a.uid == uid)
    }

    pub fn ax(&mut self, uid: u64) -> Option<&mut Axes> {
        if let Some(i) = self.axes.iter().position(|a| a.uid == uid) {
            return Some(&mut self.axes[i]);
        }
        self.detached.iter_mut().find(|a| a.uid == uid)
    }

    /// Create an axes that is not yet part of the figure.
    pub fn new_detached(&mut self, rect: (f64, f64, f64, f64)) -> u64 {
        let ax = Axes::new(Bbox::from_bounds(rect.0, rect.1, rect.2, rect.3), None, &self.subplotpars);
        let uid = ax.uid;
        self.detached.push(ax);
        uid
    }

    /// Move a detached axes into the figure (draw order = attach order).
    pub fn attach(&mut self, uid: u64) -> bool {
        if let Some(i) = self.detached.iter().position(|a| a.uid == uid) {
            let ax = self.detached.remove(i);
            self.add_axes_internal(ax);
            true
        } else {
            self.axes_index(uid).is_some()
        }
    }

    /// Remove an axes from the figure, keeping it detached.
    pub fn detach(&mut self, uid: u64) {
        if let Some(i) = self.axes.iter().position(|a| a.uid == uid) {
            let ax = self.axes.remove(i);
            self.detached.push(ax);
        }
        self.axstack.retain(|&u| u != uid);
    }

    /// `_add_axes_internal`.
    pub fn add_axes_internal(&mut self, ax: Axes) -> u64 {
        let uid = ax.uid;
        self.axstack.retain(|&u| u != uid);
        self.axstack.push(uid);
        self.axes.push(ax);
        uid
    }

    pub fn sca(&mut self, uid: u64) {
        if self.axes_index(uid).is_some() {
            self.axstack.retain(|&u| u != uid);
            self.axstack.push(uid);
        }
    }

    pub fn gca_uid(&self) -> Option<u64> {
        self.axstack.last().copied()
    }

    /// `GridSpec._check_gridspec_exists` + new gridspec.
    pub fn gridspec_for(&mut self, nrows: usize, ncols: usize) -> Shared<GridSpec> {
        for ax in &self.axes {
            if let Some(ss) = &ax.subplotspec {
                let gs = ss.gridspec.lk();
                if gs.nrows == nrows && gs.ncols == ncols {
                    drop(gs);
                    return ss.gridspec.clone();
                }
            }
        }
        let gs = shared(GridSpec::new(nrows, ncols));
        self.gridspecs.push(gs.clone());
        gs
    }

    /// `add_subplot(nrows, ncols, index)` (index 1-based, or a range).
    pub fn add_subplot(&mut self, gs: Shared<GridSpec>, num1: usize, num2: usize) -> u64 {
        let (nrows, ncols) = {
            let g = gs.lk();
            (g.nrows, g.ncols)
        };
        let ss = SubplotSpec { nrows, ncols, num1, num2, gridspec: gs };
        let ax = Axes::new(Bbox::unit(), Some(ss), &self.subplotpars);
        self.add_axes_internal(ax)
    }

    pub fn add_axes_rect(&mut self, rect: (f64, f64, f64, f64)) -> u64 {
        let ax = Axes::new(Bbox::from_bounds(rect.0, rect.1, rect.2, rect.3), None, &self.subplotpars);
        self.add_axes_internal(ax)
    }

    pub fn delaxes(&mut self, uid: u64) {
        self.axes.retain(|a| a.uid != uid);
        self.detached.retain(|a| a.uid != uid);
        self.axstack.retain(|&u| u != uid);
        for a in self.axes.iter_mut() {
            a.shared_x.retain(|&u| u != uid);
            a.shared_y.retain(|&u| u != uid);
            a.twinned.retain(|&u| u != uid);
        }
    }

    /// `subplots_adjust`.
    pub fn subplots_adjust(&mut self, left: Option<f64>, bottom: Option<f64>, right: Option<f64>, top: Option<f64>, wspace: Option<f64>, hspace: Option<f64>) {
        let p = &mut self.subplotpars;
        if let Some(v) = left {
            p.left = v;
        }
        if let Some(v) = right {
            p.right = v;
        }
        if let Some(v) = bottom {
            p.bottom = v;
        }
        if let Some(v) = top {
            p.top = v;
        }
        if let Some(v) = wspace {
            p.wspace = v;
        }
        if let Some(v) = hspace {
            p.hspace = v;
        }
        let sp = self.subplotpars;
        for ax in self.axes.iter_mut() {
            if let Some(ss) = &ax.subplotspec {
                let pos = ss.get_position(&sp);
                ax.position = pos;
                ax.original_position = pos;
            }
        }
    }

    /// Shared data limits for autoscaling axes `i`.
    fn shared_lims(&self, i: usize) -> Option<SharedLims> {
        let ax = &self.axes[i];
        if ax.shared_x.is_empty() && ax.shared_y.is_empty() {
            return None;
        }
        let mut s = SharedLims::default();
        let collect = |uids: &[u64], x: bool, out: &mut Vec<((f64, f64), f64)>, st: &mut Vec<f64>| {
            for &u in uids {
                if let Some(o) = self.axes.iter().find(|a| a.uid == u) {
                    let iv = if x { o.data_lim.bbox.intervalx() } else { o.data_lim.bbox.intervaly() };
                    let mp = if x { o.data_lim.minpos[0] } else { o.data_lim.minpos[1] };
                    out.push((iv, mp));
                    if o.uid != ax.uid {
                        for c in &o.children {
                            let (sx, sy) = c.sticky_edges();
                            st.extend(if x { sx } else { sy });
                        }
                    }
                }
            }
        };
        if !ax.shared_x.is_empty() {
            collect(&ax.shared_x, true, &mut s.x, &mut s.x_stickies);
        }
        if !ax.shared_y.is_empty() {
            collect(&ax.shared_y, false, &mut s.y, &mut s.y_stickies);
        }
        Some(s)
    }

    /// `_unstale_viewLim` for axes `i` including shared siblings.
    pub fn unstale_viewlim(&mut self, i: usize) {
        let fig = self.ctx();
        let ax = &self.axes[i];
        let sib = |uids: &Vec<u64>, idx: usize| -> bool {
            if uids.is_empty() {
                ax.stale_viewlims[idx]
            } else {
                uids.iter().any(|&u| self.axes.iter().find(|a| a.uid == u).map(|a| a.stale_viewlims[idx]).unwrap_or(false))
            }
        };
        let need = [sib(&ax.shared_x, 0), sib(&ax.shared_y, 1)];
        if need[0] || need[1] {
            let sx = ax.shared_x.clone();
            let sy = ax.shared_y.clone();
            let me = ax.uid;
            for a in self.axes.iter_mut() {
                if sx.contains(&a.uid) || a.uid == me {
                    a.stale_viewlims[0] = false;
                }
            }
            for a in self.axes.iter_mut() {
                if sy.contains(&a.uid) {
                    a.stale_viewlims[1] = false;
                }
            }
            self.axes[i].stale_viewlims = [false, false];
            let sl = self.shared_lims(i);
            self.axes[i].autoscale_view(None, need[0], need[1], &fig, sl.as_ref());
            self.sync_shared(i);
        }
    }

    /// Propagate view limits of axes `i` to its shared siblings.
    pub fn sync_shared(&mut self, i: usize) {
        let (uid, sx, sy, vl) = {
            let a = &self.axes[i];
            (a.uid, a.shared_x.clone(), a.shared_y.clone(), a.view_lim)
        };
        let ax_auto = (self.axes[i].xaxis.autoscale_on, self.axes[i].yaxis.autoscale_on);
        for a in self.axes.iter_mut() {
            if a.uid == uid {
                continue;
            }
            if sx.contains(&a.uid) {
                a.view_lim.x0 = vl.x0;
                a.view_lim.x1 = vl.x1;
                a.xaxis.autoscale_on = ax_auto.0;
            }
            if sy.contains(&a.uid) {
                a.view_lim.y0 = vl.y0;
                a.view_lim.y1 = vl.y1;
                a.yaxis.autoscale_on = ax_auto.1;
            }
        }
    }

    /// `Figure.draw(renderer)`.
    pub fn draw(&mut self, renderer: &mut Renderer) {
        if !self.visible {
            return;
        }
        for i in 0..self.axes.len() {
            self.unstale_viewlim(i);
        }
        let fig = self.ctx();
        for ax in self.axes.iter_mut() {
            ax.apply_aspect(None, &fig);
        }
        if self.layout != LayoutEngine::None && !self.axes.is_empty() {
            self.execute_layout(renderer);
        }
        let fig = self.ctx();
        let tc = self.trans_ctx();
        let _snap = self.publish_snapshot();
        let clip = ClipCtx { axes_bbox: None, trans: &tc };
        self.patch.draw(renderer, &tc, &clip);
        // artists sorted by zorder, in `get_children` order:
        // [*artists, *axes, *lines, *patches, *texts, *images, *legends]
        enum Item {
            Axes(usize),
            Child(usize),
        }
        let order = |c: &FigChild| match c {
            FigChild::Line(_) => 2,
            FigChild::Patch(_) => 3,
            FigChild::Text(_) => 4,
            FigChild::Legend(_) => 6,
            FigChild::Ext(_, cat) => *cat,
        };
        let mut keyed: Vec<(u8, usize, Item)> = Vec::new();
        for (i, c) in self.children.iter().enumerate() {
            keyed.push((order(c), i, Item::Child(i)));
        }
        for i in 0..self.axes.len() {
            keyed.push((1, i, Item::Axes(i)));
        }
        keyed.sort_by_key(|k| (k.0, k.1));
        let mut items: Vec<(f64, usize, Item)> = Vec::new();
        for (seq, (_, _, it)) in keyed.into_iter().enumerate() {
            let z = match &it {
                Item::Axes(i) => self.axes[*i].base.zorder,
                Item::Child(i) => match &self.children[*i] {
                    FigChild::Text(t) => t.lk().base.zorder,
                    FigChild::Line(l) => l.lk().base.zorder,
                    FigChild::Patch(p) => p.lk().base.zorder,
                    FigChild::Legend(l) => l.lk().zorder,
                    FigChild::Ext(e, _) => e.zorder(),
                },
            };
            items.push((z, seq, it));
        }
        items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        for (_, _, it) in items {
            match it {
                Item::Axes(i) => {
                    self.axes[i].draw(renderer, &fig);
                }
                Item::Child(i) => match &self.children[i] {
                    FigChild::Text(t) => t.lk().draw(renderer, &tc, &clip),
                    FigChild::Line(l) => l.lk().draw(renderer, &tc, &clip, None, false),
                    FigChild::Patch(p) => p.lk().draw(renderer, &tc, &clip),
                    FigChild::Legend(_) => {}
                    FigChild::Ext(e, _) => {
                        if e.visible() {
                            e.draw(renderer)
                        }
                    }
                },
            }
        }
    }

    fn execute_layout(&mut self, renderer: &mut Renderer) {
        if let LayoutEngine::Tight { pad, h_pad, w_pad, rect } = self.layout.clone() {
            super::tight_layout::tight_layout(self, renderer, pad, h_pad, w_pad, rect);
        }
    }

    /// Render the figure to a fresh renderer (`FigureCanvasAgg.draw`).
    pub fn render(&mut self) -> Renderer {
        let (w, h) = self.canvas_size();
        let mut r = Renderer::new(w as f64, h as f64, self.dpi);
        r.clear();
        self.draw(&mut r);
        r
    }

    /// Draw with drawing disabled (`print_figure`'s pre-draw that runs the
    /// layout engine and updates artist state).
    pub fn draw_disabled(&mut self) {
        let (w, h) = self.canvas_size();
        let mut r = Renderer::metrics_only(w as f64, h as f64, self.dpi);
        self.draw(&mut r);
    }

    /// Render with `savefig`'s semantics: optional tight/explicit bbox (in
    /// inches) via `_tight_bbox.adjust_bbox`.
    pub fn render_for_save(&mut self, bbox_inches: Option<Bbox>, tight: Option<f64>) -> Renderer {
        let had_layout = self.layout != LayoutEngine::None;
        let mut bbox = bbox_inches;
        if had_layout || tight.is_some() {
            let (w, h) = self.canvas_size();
            let mut r = Renderer::metrics_only(w as f64, h as f64, self.dpi);
            self.draw(&mut r);
            if let Some(pad) = tight {
                let b = self.get_tightbbox(&mut r);
                bbox = Some(Bbox::new(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad));
            }
        }
        let old_layout = std::mem::replace(&mut self.layout, LayoutEngine::None);
        let out = match bbox {
            None => self.render(),
            Some(b) => {
                let fig = self.ctx();
                let saved: Vec<(Bbox, bool)> = self.axes.iter().map(|a| (a.position, a.aspect_frozen)).collect();
                for a in self.axes.iter_mut() {
                    a.apply_aspect(None, &fig);
                    a.aspect_frozen = true;
                }
                let old_size = self.figsize;
                let old_patch = self.patch.clone();
                let tr = Transform::Affine(Affine2D::identity().scale(self.dpi, self.dpi).0);
                let (x0, y0) = tr.transform_point(b.x0, b.y0);
                let ob = self.fig_bbox();
                let (w1, h1) = (ob.width(), ob.height());
                self.boxout = Some(Bbox::from_bounds(-x0, -y0, w1, h1));
                self.figsize = (b.width(), b.height());
                let nb = self.fig_bbox();
                self.patch.set_bounds(x0 / w1, y0 / h1, nb.width() / w1, nb.height() / h1);
                let r = self.render();
                self.boxout = None;
                self.figsize = old_size;
                self.patch = old_patch;
                for (a, (p, f)) in self.axes.iter_mut().zip(saved) {
                    a.position = p;
                    a.aspect_frozen = f;
                }
                r
            }
        };
        self.layout = old_layout;
        out
    }

    /// `Figure.get_tightbbox(renderer)` in inches.
    pub fn get_tightbbox(&mut self, renderer: &mut Renderer) -> Bbox {
        let fig = self.ctx();
        let tc = self.trans_ctx();
        let _snap = self.publish_snapshot();
        let mut bb: Vec<Bbox> = Vec::new();
        for c in &self.children {
            match c {
                FigChild::Text(t) => {
                    let t = t.lk();
                    if t.base.visible && t.base.in_layout {
                        let b = t.get_window_extent(renderer, &tc);
                        if b.width() != 0.0 || b.height() != 0.0 {
                            bb.push(b);
                        }
                    }
                }
                FigChild::Ext(e, _) => {
                    if let Some(b) = e.tightbbox(renderer, false) {
                        if b.width() != 0.0 || b.height() != 0.0 {
                            bb.push(b);
                        }
                    }
                }
                _ => {}
            }
        }
        for ax in self.axes.iter_mut() {
            if ax.base.visible {
                if let Some(b) = ax.get_tightbbox(renderer, &fig, false) {
                    if b.width() != 0.0 || b.height() != 0.0 {
                        bb.push(b);
                    }
                }
            }
        }
        let fb = fig.fig_bbox;
        let isfinite: Vec<Bbox> = bb.into_iter().filter(|b| b.is_finite()).collect();
        let u = if isfinite.is_empty() { fb } else { Bbox::union(&isfinite) };
        let inv = fig.dpi_scale.inverted();
        u.transformed_bbox(&inv)
    }
}

/// Read-only figure state published while the figure is drawn, so that
/// external (Python) artists can query transforms and limits without
/// re-entering the locked figure.
#[derive(Clone, Debug)]
pub struct AxesSnapshot {
    pub uid: u64,
    pub tc: TransCtx,
    pub bbox: Bbox,
    pub view_lim: Bbox,
    pub data_lim: super::axes::DataLim,
    pub position: Bbox,
    pub original_position: Bbox,
}

#[derive(Clone, Debug)]
pub struct DrawSnapshot {
    /// `Figure::key` of the figure being drawn
    pub key: usize,
    pub fig: FigCtx,
    pub tc: TransCtx,
    pub canvas: (f64, f64),
    pub figsize: (f64, f64),
    pub axes: Vec<AxesSnapshot>,
}

impl DrawSnapshot {
    pub fn axes(&self, uid: u64) -> Option<&AxesSnapshot> {
        self.axes.iter().find(|a| a.uid == uid)
    }
}

thread_local! {
    static SNAPSHOTS: std::cell::RefCell<Vec<std::rc::Rc<DrawSnapshot>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The innermost published snapshot, if a figure is being drawn.
pub fn current_snapshot() -> Option<std::rc::Rc<DrawSnapshot>> {
    SNAPSHOTS.with(|s| s.borrow().last().cloned())
}

/// The snapshot of the figure with `key` if it is being drawn by this thread.
pub fn snapshot_for(key: usize) -> Option<std::rc::Rc<DrawSnapshot>> {
    if key == 0 {
        return None;
    }
    SNAPSHOTS.with(|s| s.borrow().iter().rev().find(|x| x.key == key).cloned())
}

pub struct SnapshotGuard;

impl Drop for SnapshotGuard {
    fn drop(&mut self) {
        SNAPSHOTS.with(|s| {
            s.borrow_mut().pop();
        });
    }
}

impl Figure {
    pub fn snapshot(&self) -> DrawSnapshot {
        let fig = self.ctx();
        let axes = self
            .axes
            .iter()
            .map(|a| AxesSnapshot {
                uid: a.uid,
                tc: a.trans_ctx(&fig),
                bbox: a.bbox(&fig),
                view_lim: a.view_lim,
                data_lim: a.data_lim,
                position: a.position,
                original_position: a.original_position,
            })
            .collect();
        let (w, h) = self.canvas_size();
        DrawSnapshot { key: self.key, tc: self.trans_ctx(), fig, canvas: (w as f64, h as f64), figsize: self.figsize, axes }
    }

    /// Publish the current state for external artists until the guard drops.
    pub fn publish_snapshot(&self) -> SnapshotGuard {
        let s = std::rc::Rc::new(self.snapshot());
        SNAPSHOTS.with(|v| v.borrow_mut().push(s));
        SnapshotGuard
    }
}
