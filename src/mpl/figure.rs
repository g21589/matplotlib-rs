//! Port of `matplotlib.figure.Figure` (drawing, subplots, savefig).

use super::artist::{shared, ClipCtx, Lock, Shared, TSpec, TransCtx};
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
}

#[derive(Debug)]
pub struct Figure {
    pub figsize: (f64, f64),
    pub dpi: f64,
    pub patch: Patch,
    pub axes: Vec<Axes>,
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
}

impl Figure {
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
            axstack: Vec::new(),
            children: Vec::new(),
            suptitle: None,
            supxlabel: None,
            supylabel: None,
            subplotpars: SubplotParams::from_rc(),
            gridspecs: Vec::new(),
            layout,
            visible: true,
        }
    }

    pub fn fig_bbox(&self) -> Bbox {
        let t = Transform::Affine(Affine2D::identity().scale(self.dpi, self.dpi).0);
        Bbox::from_bounds(0.0, 0.0, self.figsize.0, self.figsize.1).transformed_bbox(&t)
    }

    pub fn ctx(&self) -> FigCtx {
        let dpi_scale = Transform::Affine(Affine2D::identity().scale(self.dpi, self.dpi).0);
        let fig_bbox = self.fig_bbox();
        FigCtx { dpi: self.dpi, trans_figure: Transform::bbox_to(&fig_bbox), dpi_scale, fig_bbox }
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
        self.axes.iter_mut().find(|a| a.uid == uid)
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
            for a in self.axes.iter_mut() {
                if sx.contains(&a.uid) || a.uid == self.axes[i].uid {
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
        let clip = ClipCtx { axes_bbox: None, trans: &tc };
        self.patch.draw(renderer, &tc, &clip);
        // artists sorted by zorder: [*artists, *axes, *lines, *patches, *texts, *legends]
        enum Item {
            Axes(usize),
            Child(usize),
        }
        let mut items: Vec<(f64, usize, Item)> = Vec::new();
        let mut seq = 0;
        for (i, c) in self.children.iter().enumerate() {
            if let FigChild::Patch(_) = c {
                let _ = i;
            }
        }
        for (i, ax) in self.axes.iter().enumerate() {
            items.push((ax.base.zorder, seq, Item::Axes(i)));
            seq += 1;
        }
        let order = |c: &FigChild| match c {
            FigChild::Line(_) => 1,
            FigChild::Patch(_) => 2,
            FigChild::Text(_) => 3,
            FigChild::Legend(_) => 4,
        };
        let mut ch: Vec<(usize, usize)> = self.children.iter().enumerate().map(|(i, c)| (order(c), i)).collect();
        ch.sort_by_key(|x| x.0);
        for (_, i) in ch {
            let z = match &self.children[i] {
                FigChild::Text(t) => t.lk().base.zorder,
                FigChild::Line(l) => l.lk().base.zorder,
                FigChild::Patch(p) => p.lk().base.zorder,
                FigChild::Legend(l) => l.lk().zorder,
            };
            items.push((z, seq, Item::Child(i)));
            seq += 1;
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

    /// `Figure.get_tightbbox(renderer)` in inches.
    pub fn get_tightbbox(&mut self, renderer: &mut Renderer) -> Bbox {
        let fig = self.ctx();
        let tc = self.trans_ctx();
        let mut bb: Vec<Bbox> = Vec::new();
        for c in &self.children {
            if let FigChild::Text(t) = c {
                let t = t.lk();
                if t.base.visible && t.base.in_layout {
                    let b = t.get_window_extent(renderer, &tc);
                    if b.width() != 0.0 || b.height() != 0.0 {
                        bb.push(b);
                    }
                }
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
