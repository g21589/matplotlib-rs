//! Port of `matplotlib.axes._base._AxesBase` (and the Spines).

use super::artist::{shared, ArtistBase, ClipBoxSpec, ClipCtx, Lock, Shared, SpinePosition, TSpec, TransCtx};
use super::axis::{Axis, AxisCtx, GridKw, TickKw};
use super::collections::Collection;
use super::colors::ColorSpec;
use super::fonts::parse_weight;
use super::image::AxesImage;
use super::legend::Legend;
use super::lines::Line2D;
use super::patches::{Patch, PatchKind};
use super::path::{self as mpath, Path};
use super::rc::{rc, RcValue};
use super::renderer::Renderer;
use super::scale::Scale;
use super::text::{pymax, pymin, Text};
use super::ticker::AxisInfo;
use super::transforms::{Bbox, Transform};
use crate::agg::stroke::LineCap;

/// A child artist of an Axes.
#[derive(Clone, Debug)]
pub enum Child {
    Line(Shared<Line2D>),
    Patch(Shared<Patch>),
    Text(Shared<Text>),
    Collection(Shared<Collection>),
    Image(Shared<AxesImage>),
}

impl Child {
    pub fn zorder(&self) -> f64 {
        match self {
            Child::Line(a) => a.lk().base.zorder,
            Child::Patch(a) => a.lk().base.zorder,
            Child::Text(a) => a.lk().base.zorder,
            Child::Collection(a) => a.lk().base.zorder,
            Child::Image(a) => a.lk().base.zorder,
        }
    }

    pub fn same(&self, o: &Child) -> bool {
        match (self, o) {
            (Child::Line(a), Child::Line(b)) => std::sync::Arc::ptr_eq(a, b),
            (Child::Patch(a), Child::Patch(b)) => std::sync::Arc::ptr_eq(a, b),
            (Child::Text(a), Child::Text(b)) => std::sync::Arc::ptr_eq(a, b),
            (Child::Collection(a), Child::Collection(b)) => std::sync::Arc::ptr_eq(a, b),
            (Child::Image(a), Child::Image(b)) => std::sync::Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    pub fn sticky_edges(&self) -> (Vec<f64>, Vec<f64>) {
        match self {
            Child::Patch(p) => {
                let p = p.lk();
                (p.sticky_x.clone(), p.sticky_y.clone())
            }
            Child::Collection(c) => {
                let c = c.lk();
                (c.sticky_x.clone(), c.sticky_y.clone())
            }
            Child::Image(i) => {
                let i = i.lk();
                (i.sticky_x.clone(), i.sticky_y.clone())
            }
            _ => (Vec::new(), Vec::new()),
        }
    }

    pub fn in_autoscale(&self) -> bool {
        match self {
            Child::Text(_) => false,
            _ => true,
        }
    }
}

/// `Bbox` with `minpos` and the `ignore` semantics used for `dataLim`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DataLim {
    pub bbox: Bbox,
    pub minpos: [f64; 2],
    pub ignore: bool,
}

impl DataLim {
    pub fn null() -> DataLim {
        DataLim { bbox: Bbox::null(), minpos: [f64::INFINITY, f64::INFINITY], ignore: true }
    }

    /// `update_from_path(path, ignore, updatex, updatey)` on the vertices.
    pub fn update_from_xy(&mut self, xy: &[[f64; 2]], ignore: Option<bool>, updatex: bool, updatey: bool) {
        let ignore = ignore.unwrap_or(self.ignore);
        if xy.is_empty() || !(updatex || updatey) {
            return;
        }
        let (mut pts, mut minpos) = if ignore {
            (Bbox::null(), [f64::INFINITY, f64::INFINITY])
        } else {
            (self.bbox, self.minpos)
        };
        let valid: Vec<&[f64; 2]> = xy.iter().filter(|p| p[0].is_finite() && p[1].is_finite()).collect();
        if updatex {
            let minx = valid.iter().map(|p| p[0]).fold(f64::INFINITY, |a, b| if b < a { b } else { a });
            let maxx = valid.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
            pts.x0 = pymin(pts.x0, minx);
            pts.x1 = pymax(pts.x1, maxx);
            if minx > 0.0 {
                minpos[0] = pymin(minpos[0], minx);
            } else {
                let mp = valid.iter().map(|p| p[0]).filter(|&v| v > 0.0).fold(f64::INFINITY, |a, b| if b < a { b } else { a });
                minpos[0] = pymin(minpos[0], mp);
            }
        }
        if updatey {
            let miny = valid.iter().map(|p| p[1]).fold(f64::INFINITY, |a, b| if b < a { b } else { a });
            let maxy = valid.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
            pts.y0 = pymin(pts.y0, miny);
            pts.y1 = pymax(pts.y1, maxy);
            if miny > 0.0 {
                minpos[1] = pymin(minpos[1], miny);
            } else {
                let mp = valid.iter().map(|p| p[1]).filter(|&v| v > 0.0).fold(f64::INFINITY, |a, b| if b < a { b } else { a });
                minpos[1] = pymin(minpos[1], mp);
            }
        }
        if pts != self.bbox || minpos != self.minpos {
            if updatex {
                self.bbox.x0 = pts.x0;
                self.bbox.x1 = pts.x1;
                self.minpos[0] = minpos[0];
            }
            if updatey {
                self.bbox.y0 = pts.y0;
                self.bbox.y1 = pts.y1;
                self.minpos[1] = minpos[1];
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Spine {
    pub patch: Patch,
    pub idx: usize,
    pub position: Option<SpinePosition>,
    pub bounds: Option<(f64, f64)>,
}

impl Spine {
    /// `Spine.linear_spine(axes, side)`.
    pub fn linear(idx: usize) -> Spine {
        let r = rc();
        let verts = match idx {
            0 => vec![[0.0, 0.999], [0.0, 0.999]],
            1 => vec![[1.0, 0.999], [1.0, 0.999]],
            2 => vec![[0.999, 0.0], [0.999, 0.0]],
            _ => vec![[0.999, 1.0], [0.999, 1.0]],
        };
        let mut p = Patch::new(PatchKind::Path { path: Path::new(verts, None) });
        p.set_facecolor(Some(ColorSpec::None));
        p.set_edgecolor(Some(ColorSpec::Rgba(r.color("axes.edgecolor"))));
        p.set_linewidth(Some(r.num("axes.linewidth")));
        p.capstyle = LineCap::Square;
        p.base.zorder = 2.5;
        p.base.transform = TSpec::Spine(idx);
        let side = ["left", "right", "bottom", "top"][idx];
        p.base.visible = r.boolean(&format!("axes.spines.{side}"));
        Spine { patch: p, idx, position: None, bounds: None }
    }

    pub fn position(&self) -> SpinePosition {
        self.position.clone().unwrap_or(SpinePosition::Outward(0.0))
    }

    /// `_adjust_location` with the current view limits.
    pub fn adjust_location(&mut self, view: &Bbox) {
        let (low, high) = match self.bounds {
            Some(b) => b,
            None => {
                if self.idx < 2 {
                    view.intervaly()
                } else {
                    view.intervalx()
                }
            }
        };
        if let PatchKind::Path { path } = &mut self.patch.kind {
            if self.idx < 2 {
                path.vertices[0][1] = low;
                path.vertices[1][1] = high;
            } else {
                path.vertices[0][0] = low;
                path.vertices[1][0] = high;
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Aspect {
    Auto,
    Num(f64),
}

/// Figure-level information an Axes needs.
#[derive(Clone, Debug)]
pub struct FigCtx {
    pub dpi: f64,
    pub trans_figure: Transform,
    pub dpi_scale: Transform,
    pub fig_bbox: Bbox,
}

#[derive(Clone, Debug)]
pub struct SubplotSpec {
    pub nrows: usize,
    pub ncols: usize,
    pub num1: usize,
    pub num2: usize,
    /// id of the gridspec (for subplots_adjust / layout)
    pub gridspec: Shared<GridSpec>,
}

#[derive(Clone, Debug)]
pub struct GridSpec {
    pub nrows: usize,
    pub ncols: usize,
    pub left: Option<f64>,
    pub right: Option<f64>,
    pub bottom: Option<f64>,
    pub top: Option<f64>,
    pub wspace: Option<f64>,
    pub hspace: Option<f64>,
    pub width_ratios: Vec<f64>,
    pub height_ratios: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubplotParams {
    pub left: f64,
    pub right: f64,
    pub bottom: f64,
    pub top: f64,
    pub wspace: f64,
    pub hspace: f64,
}

impl SubplotParams {
    pub fn from_rc() -> SubplotParams {
        let r = rc();
        SubplotParams {
            left: r.num("figure.subplot.left"),
            right: r.num("figure.subplot.right"),
            bottom: r.num("figure.subplot.bottom"),
            top: r.num("figure.subplot.top"),
            wspace: r.num("figure.subplot.wspace"),
            hspace: r.num("figure.subplot.hspace"),
        }
    }
}

/// numpy cumsum (sequential).
fn cumsum(v: &[f64]) -> Vec<f64> {
    let mut acc = 0.0;
    v.iter()
        .map(|x| {
            acc += x;
            acc
        })
        .collect()
}

fn py_sum(v: &[f64]) -> f64 {
    // Python's sum() of floats starts from int 0
    let mut acc = 0.0;
    for x in v {
        acc += x;
    }
    acc
}

impl GridSpec {
    pub fn new(nrows: usize, ncols: usize) -> GridSpec {
        GridSpec {
            nrows,
            ncols,
            left: None,
            right: None,
            bottom: None,
            top: None,
            wspace: None,
            hspace: None,
            width_ratios: vec![1.0; ncols],
            height_ratios: vec![1.0; nrows],
        }
    }

    pub fn subplot_params(&self, fig: &SubplotParams) -> SubplotParams {
        let mut p = *fig;
        if let Some(v) = self.left {
            p.left = v;
        }
        if let Some(v) = self.right {
            p.right = v;
        }
        if let Some(v) = self.bottom {
            p.bottom = v;
        }
        if let Some(v) = self.top {
            p.top = v;
        }
        if let Some(v) = self.wspace {
            p.wspace = v;
        }
        if let Some(v) = self.hspace {
            p.hspace = v;
        }
        p
    }

    /// `get_grid_positions(fig)`: (bottoms, tops, lefts, rights).
    pub fn grid_positions(&self, sp: &SubplotParams) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let p = self.subplot_params(sp);
        let (nrows, ncols) = (self.nrows as f64, self.ncols as f64);
        let tot_width = p.right - p.left;
        let tot_height = p.top - p.bottom;
        let cell_h = tot_height / (nrows + p.hspace * (nrows - 1.0));
        let sep_h = p.hspace * cell_h;
        let norm = cell_h * nrows / py_sum(&self.height_ratios);
        let cell_heights: Vec<f64> = self.height_ratios.iter().map(|r| r * norm).collect();
        let mut flat = Vec::new();
        for i in 0..self.nrows {
            flat.push(if i == 0 { 0.0 } else { sep_h });
            flat.push(cell_heights[i]);
        }
        let cell_hs = cumsum(&flat);
        let cell_w = tot_width / (ncols + p.wspace * (ncols - 1.0));
        let sep_w = p.wspace * cell_w;
        let norm = cell_w * ncols / py_sum(&self.width_ratios);
        let cell_widths: Vec<f64> = self.width_ratios.iter().map(|r| r * norm).collect();
        let mut flat = Vec::new();
        for i in 0..self.ncols {
            flat.push(if i == 0 { 0.0 } else { sep_w });
            flat.push(cell_widths[i]);
        }
        let cell_ws = cumsum(&flat);
        let tops: Vec<f64> = (0..self.nrows).map(|i| p.top - cell_hs[2 * i]).collect();
        let bottoms: Vec<f64> = (0..self.nrows).map(|i| p.top - cell_hs[2 * i + 1]).collect();
        let lefts: Vec<f64> = (0..self.ncols).map(|i| p.left + cell_ws[2 * i]).collect();
        let rights: Vec<f64> = (0..self.ncols).map(|i| p.left + cell_ws[2 * i + 1]).collect();
        (bottoms, tops, lefts, rights)
    }
}

impl SubplotSpec {
    /// `SubplotSpec.get_position(figure)`.
    pub fn get_position(&self, sp: &SubplotParams) -> Bbox {
        let gs = self.gridspec.lk();
        let (b, t, l, r) = gs.grid_positions(sp);
        let ncols = gs.ncols;
        let rows = [self.num1 / ncols, self.num2 / ncols];
        let cols = [self.num1 % ncols, self.num2 % ncols];
        let fb = pymin(b[rows[0]], b[rows[1]]);
        let ft = pymax(t[rows[0]], t[rows[1]]);
        let fl = pymin(l[cols[0]], l[cols[1]]);
        let fr = pymax(r[cols[0]], r[cols[1]]);
        Bbox::from_extents(fl, fb, fr, ft)
    }

    pub fn rowspan(&self) -> (usize, usize) {
        let n = self.gridspec.lk().ncols;
        (self.num1 / n, self.num2 / n + 1)
    }

    pub fn colspan(&self) -> (usize, usize) {
        let n = self.gridspec.lk().ncols;
        let (a, b) = (self.num1 % n, self.num2 % n);
        (a.min(b), a.max(b) + 1)
    }
}

#[derive(Clone, Debug)]
pub struct Axes {
    pub uid: u64,
    pub base: ArtistBase,
    pub position: Bbox,
    pub original_position: Bbox,
    pub subplotspec: Option<SubplotSpec>,
    pub aspect: Aspect,
    pub adjustable: String,
    pub anchor: (f64, f64),
    pub box_aspect: Option<f64>,
    pub view_lim: Bbox,
    pub data_lim: DataLim,
    pub stale_viewlims: [bool; 2],
    pub ignore_existing_data_limits: bool,
    pub xmargin: f64,
    pub ymargin: f64,
    pub tight: Option<bool>,
    pub use_sticky_edges: bool,
    pub children: Vec<Child>,
    pub spines: [Spine; 4],
    pub xaxis: Axis,
    pub yaxis: Axis,
    pub patch: Patch,
    pub facecolor: ColorSpec,
    pub frameon: bool,
    pub axison: bool,
    pub axisbelow: String,
    pub title: Shared<Text>,
    pub left_title: Shared<Text>,
    pub right_title: Shared<Text>,
    pub autotitlepos: Option<bool>,
    pub title_offset_points: f64,
    pub legend: Option<Shared<Legend>>,
    pub label: String,
    pub grid_on: bool,
    /// shared axes siblings (uids), including self when shared
    pub shared_x: Vec<u64>,
    pub shared_y: Vec<u64>,
    pub twinned: Vec<u64>,
    pub prop_cycle_index: usize,
    pub patch_cycle_index: usize,
    pub name: String,
    /// positions frozen by `bbox_inches` adjustment (apply_aspect disabled)
    pub aspect_frozen: bool,
}

static NEXT_UID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub fn next_uid() -> u64 {
    NEXT_UID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn mk_title(x: f64, y: f64, ha: &str) -> Text {
    let r = rc();
    let mut t = Text::new(x, y, "");
    let size = r.font_size("axes.titlesize");
    t.fontproperties = t.fontproperties.clone().with_size(size);
    t.fontproperties.weight = parse_weight(&r.string("axes.titleweight"));
    t.va = "baseline".into();
    t.ha = ha.into();
    t
}

impl Axes {
    pub fn new(position: Bbox, subplotspec: Option<SubplotSpec>, sp: &SubplotParams) -> Axes {
        let r = rc();
        let pos = match &subplotspec {
            Some(s) => s.get_position(sp),
            None => position,
        };
        let mut ax = Axes {
            uid: next_uid(),
            base: ArtistBase { zorder: 0.0, ..ArtistBase::default() },
            position: pos,
            original_position: pos,
            subplotspec,
            aspect: Aspect::Auto,
            adjustable: "box".into(),
            anchor: (0.5, 0.5),
            box_aspect: None,
            view_lim: Bbox::unit(),
            data_lim: DataLim::null(),
            stale_viewlims: [false, false],
            ignore_existing_data_limits: true,
            xmargin: r.num("axes.xmargin"),
            ymargin: r.num("axes.ymargin"),
            tight: None,
            use_sticky_edges: true,
            children: Vec::new(),
            spines: [Spine::linear(0), Spine::linear(1), Spine::linear(2), Spine::linear(3)],
            xaxis: Axis::new(true),
            yaxis: Axis::new(false),
            patch: Patch::rectangle(0.0, 0.0, 1.0, 1.0),
            facecolor: ColorSpec::Rgba(r.color("axes.facecolor")),
            frameon: true,
            axison: true,
            axisbelow: "line".into(),
            title: shared(Text::new(0.5, 1.0, "")),
            left_title: shared(Text::new(0.0, 1.0, "")),
            right_title: shared(Text::new(1.0, 1.0, "")),
            autotitlepos: Some(true),
            title_offset_points: r.num("axes.titlepad"),
            legend: None,
            label: String::new(),
            grid_on: false,
            shared_x: Vec::new(),
            shared_y: Vec::new(),
            twinned: Vec::new(),
            prop_cycle_index: 0,
            patch_cycle_index: 0,
            name: "rectilinear".into(),
            aspect_frozen: false,
        };
        ax.set_axisbelow(&r.string("axes.axisbelow"));
        ax.clear();
        // tick_params from rc
        let b = |k: &str| r.boolean(k);
        for which in ["minor", "major"] {
            let xkw = TickKw {
                tick2_on: Some(b("xtick.top") && b(&format!("xtick.{which}.top"))),
                tick1_on: Some(b("xtick.bottom") && b(&format!("xtick.{which}.bottom"))),
                label2_on: Some(b("xtick.labeltop") && b(&format!("xtick.{which}.top"))),
                label1_on: Some(b("xtick.labelbottom") && b(&format!("xtick.{which}.bottom"))),
                ..TickKw::default()
            };
            ax.xaxis.set_tick_params(which, false, &xkw);
            let ykw = TickKw {
                tick1_on: Some(b("ytick.left") && b(&format!("ytick.{which}.left"))),
                tick2_on: Some(b("ytick.right") && b(&format!("ytick.{which}.right"))),
                label1_on: Some(b("ytick.labelleft") && b(&format!("ytick.{which}.left"))),
                label2_on: Some(b("ytick.labelright") && b(&format!("ytick.{which}.right"))),
                ..TickKw::default()
            };
            ax.yaxis.set_tick_params(which, false, &ykw);
        }
        ax
    }

    pub fn set_axisbelow(&mut self, b: &str) {
        let z = match b {
            "True" | "true" => 0.5,
            "False" | "false" => 2.5,
            _ => 1.5,
        };
        self.axisbelow = b.to_string();
        self.xaxis.zorder = z;
        self.yaxis.zorder = z;
    }

    /// `Axes.__clear()`.
    pub fn clear(&mut self) {
        let r = rc();
        let patch_visible = self.patch.base.visible;
        self.xaxis.clear();
        self.yaxis.clear();
        for s in self.spines.iter_mut() {
            s.position = None;
        }
        self.ignore_existing_data_limits = true;
        if r.boolean("xtick.minor.visible") {
            self.xaxis.minor_locator = super::ticker::Locator::AutoMinor { ndivs: None };
            self.xaxis.default_minloc = false;
        }
        if r.boolean("ytick.minor.visible") {
            self.yaxis.minor_locator = super::ticker::Locator::AutoMinor { ndivs: None };
            self.yaxis.default_minloc = false;
        }
        self.xmargin = r.num("axes.xmargin");
        self.ymargin = r.num("axes.ymargin");
        self.tight = None;
        self.use_sticky_edges = true;
        self.grid_on = r.boolean("axes.grid");
        self.children.clear();
        self.legend = None;
        self.prop_cycle_index = 0;
        self.patch_cycle_index = 0;
        self.grid(Some(false), "major", "both", &GridKw::default());
        let which = r.string("axes.grid.which");
        let axis = r.string("axes.grid.axis");
        let g = self.grid_on;
        self.grid(Some(g), &which, &axis, &GridKw::default());
        let y = match r.get("axes.titley") {
            Some(RcValue::Num(v)) => {
                self.autotitlepos = Some(false);
                *v
            }
            _ => {
                self.autotitlepos = Some(true);
                1.0
            }
        };
        self.title = shared(mk_title(0.5, y, "center"));
        self.left_title = shared(mk_title(0.0, y, "left"));
        self.right_title = shared(mk_title(1.0, y, "right"));
        self.title_offset_points = r.num("axes.titlepad");
        self.set_title_offset_trans(self.title_offset_points);
        let mut patch = Patch::rectangle(0.0, 0.0, 1.0, 1.0);
        patch.set_facecolor(Some(self.facecolor));
        patch.set_edgecolor(Some(ColorSpec::None));
        patch.set_linewidth(Some(0.0));
        patch.base.transform = TSpec::Axes;
        patch.base.visible = patch_visible;
        self.patch = patch;
        self.axison = true;
        self.xaxis.clip_box = ClipBoxSpec::AxesBbox;
        self.yaxis.clip_box = ClipBoxSpec::AxesBbox;
        self.set_lim(true, 0.0, 1.0, Some(true));
        self.set_lim(false, 0.0, 1.0, Some(true));
    }

    pub fn set_title_offset_trans(&mut self, pts: f64) {
        for t in [&self.title, &self.left_title, &self.right_title] {
            let mut t = t.lk();
            t.base.transform = TSpec::Title(pts);
            t.base.transform_set = true;
            t.base.clip_box = ClipBoxSpec::None;
        }
    }

    pub fn grid(&mut self, visible: Option<bool>, which: &str, axis: &str, kw: &GridKw) {
        if axis == "x" || axis == "both" {
            self.xaxis.grid(visible, which, kw);
        }
        if axis == "y" || axis == "both" {
            self.yaxis.grid(visible, which, kw);
        }
    }

    pub fn axis(&self, x: bool) -> &Axis {
        if x {
            &self.xaxis
        } else {
            &self.yaxis
        }
    }

    pub fn axis_mut(&mut self, x: bool) -> &mut Axis {
        if x {
            &mut self.xaxis
        } else {
            &mut self.yaxis
        }
    }

    // ----- transforms ---------------------------------------------------

    /// `self.bbox` (TransformedBbox(position, transSubfigure)).
    pub fn bbox(&self, fig: &FigCtx) -> Bbox {
        self.position.transformed_bbox(&fig.trans_figure)
    }

    pub fn trans_scale(&self) -> Transform {
        let sx = self.xaxis.scale.transform();
        let sy = self.yaxis.scale.transform();
        match (sx, sy) {
            (None, None) => Transform::identity(),
            (a, b) => Transform::Separable(a, b),
        }
    }

    pub fn trans_ctx(&self, fig: &FigCtx) -> TransCtx {
        let trans_axes = Transform::bbox_to(&self.bbox(fig));
        let ts = self.trans_scale();
        let lim_bbox = self.view_lim.transformed_bbox(&ts);
        let trans_limits = Transform::bbox_from(&lim_bbox);
        let trans_data = ts.then(&trans_limits.then(&trans_axes));
        let r = rc();
        TransCtx {
            dpi: fig.dpi,
            trans_figure: fig.trans_figure.clone(),
            dpi_scale: fig.dpi_scale.clone(),
            trans_axes: Some(trans_axes),
            trans_data: Some(trans_data),
            spine_pos: [
                self.spines[0].position(),
                self.spines[1].position(),
                self.spines[2].position(),
                self.spines[3].position(),
            ],
            xtick_alignment: r.string("xtick.alignment"),
            ytick_alignment: r.string("ytick.alignment"),
        }
    }

    /// Clip box of artists clipped to the axes patch.
    pub fn clip_bbox(&self, tc: &TransCtx) -> Bbox {
        let t = self.patch.get_transform(tc);
        Bbox::unit().transformed_bbox(&t)
    }

    pub fn axis_ctx<'a>(&self, x: bool, tc: &'a TransCtx, fig: &FigCtx) -> AxisCtx<'a> {
        AxisCtx {
            trans: tc,
            view: if x { self.view_lim.intervalx() } else { self.view_lim.intervaly() },
            data: if x { self.data_lim.bbox.intervalx() } else { self.data_lim.bbox.intervaly() },
            minpos: if x { self.data_lim.minpos[0] } else { self.data_lim.minpos[1] },
            axes_bbox: self.bbox(fig),
            dpi: fig.dpi,
        }
    }

    pub fn axis_info(&self, x: bool, fig: &FigCtx) -> AxisInfo {
        let tc = self.trans_ctx(fig);
        let ac = self.axis_ctx(x, &tc, fig);
        self.axis(x).axis_info(&ac)
    }

    // ----- limits -------------------------------------------------------

    pub fn get_lim(&self, x: bool) -> (f64, f64) {
        if x {
            self.view_lim.intervalx()
        } else {
            self.view_lim.intervaly()
        }
    }

    pub fn get_bound(&self, x: bool) -> (f64, f64) {
        let (a, b) = self.get_lim(x);
        if a < b {
            (a, b)
        } else {
            (b, a)
        }
    }

    fn set_view_interval(&mut self, x: bool, v0: f64, v1: f64) {
        if x {
            self.view_lim.x0 = v0;
            self.view_lim.x1 = v1;
        } else {
            self.view_lim.y0 = v0;
            self.view_lim.y1 = v1;
        }
    }

    /// `Axis._set_lim(v0, v1, auto)` without unit handling.
    pub fn set_lim(&mut self, x: bool, v0: f64, v1: f64, auto: Option<bool>) -> (f64, f64) {
        let (mut v0, mut v1) = (v0, v1);
        let (old0, old1) = self.get_lim(x);
        if v0.is_nan() {
            v0 = old0;
        }
        if v1.is_nan() {
            v1 = old1;
        }
        if self.axis(x).scale.name() == "log" {
            if v0 <= 0.0 {
                v0 = old0;
            }
            if v1 <= 0.0 {
                v1 = old1;
            }
        }
        let reverse = v0 > v1;
        let info = AxisInfo {
            view: self.get_lim(x),
            data: if x { self.data_lim.bbox.intervalx() } else { self.data_lim.bbox.intervaly() },
            minpos: if x { self.data_lim.minpos[0] } else { self.data_lim.minpos[1] },
            ..AxisInfo::default()
        };
        let (a, b) = self.axis(x).major_locator.nonsingular(v0, v1, &info);
        let minpos = info.minpos;
        let (a, b) = self.axis(x).scale.limit_range_for_scale(a, b, minpos);
        let (a, b) = sorted2(a, b, reverse);
        self.set_view_interval(x, a, b);
        self.stale_viewlims[if x { 0 } else { 1 }] = false;
        if let Some(au) = auto {
            self.axis_mut(x).autoscale_on = au;
        }
        (a, b)
    }

    pub fn set_bound(&mut self, x: bool, lower: Option<f64>, upper: Option<f64>) {
        let (ol, ou) = self.get_bound(x);
        let lower = lower.unwrap_or(ol);
        let upper = upper.unwrap_or(ou);
        let (l0, l1) = self.get_lim(x);
        let inverted = l1 < l0;
        let (a, b) = sorted2(lower, upper, inverted);
        self.set_lim(x, a, b, None);
    }

    pub fn request_autoscale_view(&mut self, x: bool, y: bool, tight: Option<bool>) {
        if x {
            self.stale_viewlims[0] = true;
        }
        if y {
            self.stale_viewlims[1] = true;
        }
        if tight.is_some() {
            self.tight = tight;
        }
    }

    /// `_unstale_viewLim` (no shared axes: those are handled by the figure).
    pub fn unstale_viewlim(&mut self, fig: &FigCtx) {
        let need = self.stale_viewlims;
        if need[0] || need[1] {
            self.stale_viewlims = [false, false];
            self.autoscale_view(None, need[0], need[1], fig, None);
        }
    }

    /// `autoscale_view(tight, scalex, scaley)`; `shared` supplies the data
    /// limits of shared siblings: (x datalims, x minpos, y datalims, y minpos).
    pub fn autoscale_view(&mut self, tight: Option<bool>, scalex: bool, scaley: bool, fig: &FigCtx, shared: Option<&SharedLims>) {
        if let Some(t) = tight {
            self.tight = Some(t);
        }
        let mut x_stickies: Vec<f64> = Vec::new();
        let mut y_stickies: Vec<f64> = Vec::new();
        if self.use_sticky_edges {
            if self.xmargin != 0.0 && scalex && self.xaxis.autoscale_on {
                for c in &self.children {
                    x_stickies.extend(c.sticky_edges().0);
                }
                if let Some(s) = shared {
                    x_stickies.extend(s.x_stickies.iter());
                }
                x_stickies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
            if self.ymargin != 0.0 && scaley && self.yaxis.autoscale_on {
                for c in &self.children {
                    y_stickies.extend(c.sticky_edges().1);
                }
                if let Some(s) = shared {
                    y_stickies.extend(s.y_stickies.iter());
                }
                y_stickies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        if self.xaxis.scale.name() == "log" {
            x_stickies.retain(|&v| v > 0.0);
        }
        if self.yaxis.scale.name() == "log" {
            y_stickies.retain(|&v| v > 0.0);
        }
        for (is_x, scale, stickies) in [(true, scalex, x_stickies), (false, scaley, y_stickies)] {
            if !(scale && self.axis(is_x).autoscale_on) {
                continue;
            }
            let own = if is_x { self.data_lim.bbox.intervalx() } else { self.data_lim.bbox.intervaly() };
            let mut values: Vec<f64> = Vec::new();
            let mut minposes: Vec<f64> = vec![if is_x { self.data_lim.minpos[0] } else { self.data_lim.minpos[1] }];
            match shared.map(|s| if is_x { &s.x } else { &s.y }) {
                Some(list) if !list.is_empty() => {
                    for (iv, mp) in list {
                        for v in [iv.0, iv.1] {
                            if v.is_finite() {
                                values.push(v);
                            }
                        }
                        minposes.push(*mp);
                    }
                    minposes.remove(0);
                }
                _ => {
                    for v in [own.0, own.1] {
                        if v.is_finite() {
                            values.push(v);
                        }
                    }
                }
            }
            let (mut x0, mut x1);
            if !values.is_empty() {
                x0 = values[1..].iter().copied().fold(values[0], pymin);
                x1 = values[1..].iter().copied().fold(values[0], pymax);
            } else {
                let mutated = if is_x {
                    self.view_lim.x0 != 0.0 || self.view_lim.x1 != 1.0
                } else {
                    self.view_lim.y0 != 0.0 || self.view_lim.y1 != 1.0
                };
                if mutated {
                    continue;
                }
                x0 = f64::NEG_INFINITY;
                x1 = f64::INFINITY;
            }
            let info = self.axis_info(is_x, fig);
            let (a, b) = self.axis(is_x).major_locator.nonsingular(x0, x1, &info);
            x0 = a;
            x1 = b;
            let minimum_minpos = minposes[1..].iter().copied().fold(minposes[0], pymin);
            let mut x0bound = None;
            let mut x1bound = None;
            if !stickies.is_empty() {
                let tol = 1e-5 * (x1 - x0).abs();
                let i0 = stickies.partition_point(|&s| s < x0 + tol) as i64 - 1;
                if i0 != -1 {
                    x0bound = Some(stickies[i0 as usize]);
                }
                let i1 = stickies.partition_point(|&s| s < x1 - tol);
                if i1 != stickies.len() {
                    x1bound = Some(stickies[i1]);
                }
            }
            let axis = self.axis(is_x);
            let (a, b) = axis.scale.limit_range_for_scale(x0, x1, minimum_minpos);
            x0 = a;
            x1 = b;
            let tr = axis.scale.transform();
            let (x0t, x1t) = match &tr {
                Some(t) => (t.apply(x0), t.apply(x1)),
                None => (x0, x1),
            };
            let margin = if is_x { self.xmargin } else { self.ymargin };
            let mut delta = (x1t - x0t) * margin;
            if !delta.is_finite() {
                delta = 0.0;
            }
            let (a, b) = match &tr {
                Some(t) => {
                    let inv = t.inverted();
                    (inv.apply(x0t - delta), inv.apply(x1t + delta))
                }
                None => (x0t - delta, x1t + delta),
            };
            x0 = a;
            x1 = b;
            if let Some(bd) = x0bound {
                x0 = pymax(x0, bd);
            }
            if let Some(bd) = x1bound {
                x1 = pymin(x1, bd);
            }
            if self.tight != Some(true) {
                let (a, b) = self.axis(is_x).major_locator.view_limits(x0, x1, &info);
                x0 = a;
                x1 = b;
            }
            self.set_bound(is_x, Some(x0), Some(x1));
        }
    }

    // ----- children -----------------------------------------------------

    /// `_set_artist_props` + `add_line`.
    pub fn add_line(&mut self, line: Shared<Line2D>) {
        {
            let mut l = line.lk();
            if !l.base.transform_set {
                l.base.transform = TSpec::Data;
            }
            if l.base.clip_path.is_none() && l.base.clip_box == ClipBoxSpec::None {
                l.base.clip_box = ClipBoxSpec::AxesBbox;
            }
        }
        self.update_line_limits(&line);
        {
            let mut l = line.lk();
            if l.base.label.is_empty() {
                l.base.label = format!("_child{}", self.children.len());
            }
        }
        self.children.push(Child::Line(line));
    }

    pub fn add_patch(&mut self, p: Shared<Patch>) {
        {
            let mut pp = p.lk();
            if !pp.base.transform_set {
                pp.base.transform = TSpec::Data;
            }
            if pp.base.clip_path.is_none() && pp.base.clip_box == ClipBoxSpec::None {
                pp.base.clip_box = ClipBoxSpec::AxesBbox;
            }
        }
        self.update_patch_limits(&p);
        self.children.push(Child::Patch(p));
    }

    pub fn add_text(&mut self, t: Shared<Text>) {
        {
            let mut tt = t.lk();
            if !tt.base.transform_set {
                tt.base.transform = TSpec::Data;
            }
        }
        self.children.push(Child::Text(t));
    }

    pub fn add_collection(&mut self, c: Shared<Collection>, autolim: bool, fig: &FigCtx) {
        {
            let mut cc = c.lk();
            if cc.base.label.is_empty() {
                cc.base.label = format!("_child{}", self.children.len());
            }
            if !cc.base.transform_set {
                cc.base.transform = TSpec::Data;
            }
            if cc.base.clip_path.is_none() && cc.base.clip_box == ClipBoxSpec::None {
                cc.base.clip_box = ClipBoxSpec::AxesBbox;
            }
        }
        self.children.push(Child::Collection(c.clone()));
        if autolim {
            self.unstale_viewlim(fig);
            self.update_collection_limits(&c, fig);
            self.request_autoscale_view(true, true, None);
        }
    }

    pub fn add_image(&mut self, im: Shared<AxesImage>) {
        {
            let mut i = im.lk();
            if !i.base.transform_set {
                i.base.transform = TSpec::Data;
            }
            if i.base.label.is_empty() {
                i.base.label = format!("_child{}", self.children.len());
            }
        }
        self.children.push(Child::Image(im));
    }

    fn update_xy(&mut self, xy: &[[f64; 2]], updatex: bool, updatey: bool) {
        if !xy.iter().any(|p| p[0].is_finite() || p[1].is_finite()) {
            return;
        }
        let ig = self.ignore_existing_data_limits;
        self.data_lim.update_from_xy(xy, Some(ig), updatex, updatey);
        self.ignore_existing_data_limits = false;
    }

    /// `update_datalim(xys, updatex, updatey)`.
    pub fn update_datalim(&mut self, xy: &[[f64; 2]], updatex: bool, updatey: bool) {
        self.update_xy(xy, updatex, updatey);
    }

    fn branch_flags(t: &TSpec) -> Option<(bool, bool)> {
        match t {
            TSpec::Data => Some((true, true)),
            TSpec::Blend(x, y) => {
                let fx = **x == TSpec::Data;
                let fy = **y == TSpec::Data;
                Some((fx, fy))
            }
            _ => None,
        }
    }

    /// `_update_line_limits`.
    pub fn update_line_limits(&mut self, line: &Shared<Line2D>) {
        let mut l = line.lk();
        let ok = l.base.transform == TSpec::Data;
        let path = l.get_path(ok).clone();
        if path.vertices.is_empty() {
            return;
        }
        let Some((ux, uy)) = Self::branch_flags(&l.base.transform) else { return };
        drop(l);
        if !(ux || uy) {
            return;
        }
        let ig = self.ignore_existing_data_limits;
        self.data_lim.update_from_xy(&path.vertices, Some(ig), ux, uy);
        self.ignore_existing_data_limits = false;
    }

    /// `_update_patch_limits`.
    pub fn update_patch_limits(&mut self, p: &Shared<Patch>) {
        let p = p.lk();
        if let PatchKind::Rectangle { width, height, .. } = p.kind {
            if width == 0.0 && height == 0.0 {
                return;
            }
        }
        let path = p.get_path();
        let mut vertices: Vec<[f64; 2]> = Vec::new();
        mpath::bezier_vertices_for_limits(&path, &mut vertices);
        let Some((ux, uy)) = Self::branch_flags(&p.base.transform) else { return };
        if !(ux || uy) {
            return;
        }
        let xys = match p.get_patch_transform(72.0) {
            Some(t) => t.transform_points(&vertices),
            None => vertices,
        };
        drop(p);
        self.update_datalim(&xys, ux, uy);
    }

    pub fn update_collection_limits(&mut self, c: &Shared<Collection>, fig: &FigCtx) {
        let tc = self.trans_ctx(fig);
        let (bb, minpos, ux, uy) = c.lk().get_datalim(&tc);
        let mut pts = vec![[bb.x0, bb.y0], [bb.x1, bb.y1]];
        if !(minpos[0].is_infinite() && minpos[1].is_infinite()) {
            pts.push(minpos);
        }
        self.update_datalim(&pts, ux, uy);
    }

    /// `relim(visible_only)`.
    pub fn relim(&mut self, visible_only: bool, fig: &FigCtx) {
        self.data_lim = DataLim::null();
        self.ignore_existing_data_limits = true;
        let children = self.children.clone();
        for c in &children {
            match c {
                Child::Line(l) => {
                    if !visible_only || l.lk().base.visible {
                        self.update_line_limits(l);
                    }
                }
                Child::Patch(p) => {
                    if !visible_only || p.lk().base.visible {
                        self.update_patch_limits(p);
                    }
                }
                Child::Collection(cc) => {
                    if !visible_only || cc.lk().base.visible {
                        self.update_collection_limits(cc, fig);
                    }
                }
                Child::Image(im) => {
                    let e = im.lk().get_extent();
                    self.update_datalim(&[[e.0, e.2], [e.1, e.3]], true, true);
                }
                Child::Text(_) => {}
            }
        }
    }

    // ----- drawing ------------------------------------------------------

    /// Window extent of spine `idx` (incl. tick padding of its axis).
    pub fn spine_window_extent(&mut self, idx: usize, fig: &FigCtx) -> Bbox {
        let view = self.view_lim;
        self.spines[idx].adjust_location(&view);
        let tc = self.trans_ctx(fig);
        let bb = self.spines[idx].patch.get_window_extent(&tc);
        let is_x = idx >= 2;
        let ac = self.axis_ctx(is_x, &tc, fig);
        let axis = if is_x { &mut self.xaxis } else { &mut self.yaxis };
        axis.spine_window_extent(idx, bb, &ac)
    }

    fn label_spine_idx(&self, x: bool) -> usize {
        if x {
            if self.xaxis.label_position == "bottom" {
                2
            } else {
                3
            }
        } else if self.yaxis.label_position == "left" {
            0
        } else {
            1
        }
    }

    /// `_update_title_position`.
    pub fn update_title_position(&mut self, renderer: &mut Renderer, fig: &FigCtx) {
        if self.autotitlepos == Some(false) {
            return;
        }
        let titles = [self.title.clone(), self.left_title.clone(), self.right_title.clone()];
        if !titles.iter().any(|t| !t.lk().text.is_empty()) {
            return;
        }
        let tc = self.trans_ctx(fig);
        let mut top = f64::NEG_INFINITY;
        {
            let mut bb = None;
            let tp = self.xaxis.ticks_position();
            if tp == "top" || tp == "unknown" || self.xaxis.label_position == "top" {
                let sidx = self.label_spine_idx(true);
                let me: *mut Axes = self;
                let ac = self.axis_ctx(true, &tc, fig);
                // SAFETY: the closure only touches the spines and y axis
                // of `self`, disjoint from the x axis borrowed mutably.
                bb = self.xaxis.get_tightbbox(
                    renderer,
                    &ac,
                    |_a| unsafe { (*me).spine_window_extent_noaxis(sidx, fig) },
                    false,
                );
            }
            let bb = bb.unwrap_or_else(|| self.bbox(fig));
            top = pymax(top, bb.ymax());
        }
        let axes_inv = tc.trans_axes.clone().unwrap().inverted();
        for title in &titles {
            let x = title.lk().x;
            title.lk().y = 1.0;
            if !title.lk().text.is_empty() {
                // offset text of the y axis overlapping the title
                let sidx = self.label_spine_idx(false);
                let me: *mut Axes = self;
                let ac = self.axis_ctx(false, &tc, fig);
                let _ = self.yaxis.get_tightbbox(renderer, &ac, |_a| unsafe { (*me).spine_window_extent_noaxis(sidx, fig) }, false);
                let ot = self.yaxis.offset_text.clone();
                let ot = ot.lk();
                if ot.base.visible && !ot.text.is_empty() {
                    let bb = ot.get_window_extent(renderer, &tc);
                    let tb = title.lk().get_window_extent(renderer, &tc);
                    if Bbox::intersection(&bb, &tb).is_some() {
                        top = bb.ymax();
                    }
                }
            }
            if top < 0.0 {
                return;
            }
            let ymin = title.lk().get_window_extent(renderer, &tc).ymin();
            if ymin < top {
                let (_, y) = axes_inv.transform_point(0.0, top);
                title.lk().y = y;
                let ymin2 = title.lk().get_window_extent(renderer, &tc).ymin();
                if ymin2 < top {
                    let (_, y) = axes_inv.transform_point(0.0, 2.0 * top - ymin2);
                    title.lk().y = y;
                }
            }
            title.lk().x = x;
        }
        let ymax = titles.iter().map(|t| t.lk().y).fold(f64::NEG_INFINITY, pymax);
        for t in &titles {
            t.lk().y = ymax;
        }
    }

    /// Spine extent without touching the axis' tick state (the caller holds
    /// the axis); used only from closures inside axis methods.
    fn spine_window_extent_noaxis(&mut self, idx: usize, fig: &FigCtx) -> Bbox {
        let view = self.view_lim;
        self.spines[idx].adjust_location(&view);
        let tc = self.trans_ctx(fig);
        let bb = self.spines[idx].patch.get_window_extent(&tc);
        let is_x = idx >= 2;
        let ac = self.axis_ctx(is_x, &tc, fig);
        // SAFETY: called from within a method of the same axis; the axis'
        // tick update is idempotent.
        let axis: *mut Axis = if is_x { &mut self.xaxis } else { &mut self.yaxis };
        unsafe { (*axis).spine_window_extent(idx, bb, &ac) }
    }

    /// `Axes.draw`.
    pub fn draw(&mut self, renderer: &mut Renderer, fig: &FigCtx) {
        if !self.base.visible {
            return;
        }
        self.unstale_viewlim(fig);
        self.apply_aspect(None, fig);
        // children + spines + axes + titles + legend, sorted by zorder
        #[derive(Clone)]
        enum Item {
            Child(Child),
            Spine(usize),
            Axis(bool),
            Title(usize),
            Legend,
        }
        let mut items: Vec<(f64, Item)> = Vec::new();
        for c in &self.children {
            items.push((c.zorder(), Item::Child(c.clone())));
        }
        if self.axison && self.frameon {
            for i in 0..4 {
                items.push((self.spines[i].patch.base.zorder, Item::Spine(i)));
            }
        }
        self.update_title_position(renderer, fig);
        if self.axison {
            items.push((self.xaxis.zorder, Item::Axis(true)));
            items.push((self.yaxis.zorder, Item::Axis(false)));
        }
        items.push((self.title.lk().base.zorder, Item::Title(0)));
        items.push((self.left_title.lk().base.zorder, Item::Title(1)));
        items.push((self.right_title.lk().base.zorder, Item::Title(2)));
        if let Some(l) = &self.legend {
            items.push((l.lk().zorder, Item::Legend));
        }
        items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let tc = self.trans_ctx(fig);
        let clip = ClipCtx { axes_bbox: Some(self.clip_bbox(&tc)), trans: &tc };
        if self.axison && self.frameon {
            self.patch.draw(renderer, &tc, &clip);
        }
        let xbound = Some(self.get_bound(true));
        let subslice_ok = self.name == "rectilinear" && self.xaxis.scale.name() == "linear";
        for (_, it) in items {
            match it {
                Item::Child(c) => match c {
                    Child::Line(l) => {
                        let mut l = l.lk();
                        let ok = subslice_ok && l.base.transform == TSpec::Data;
                        l.draw(renderer, &tc, &clip, xbound, ok);
                    }
                    Child::Patch(p) => p.lk().draw(renderer, &tc, &clip),
                    Child::Text(t) => t.lk().draw(renderer, &tc, &clip),
                    Child::Collection(cc) => cc.lk().draw(renderer, &tc, &clip),
                    Child::Image(im) => im.lk().draw(renderer, &tc, &clip),
                },
                Item::Spine(i) => {
                    let view = self.view_lim;
                    self.spines[i].adjust_location(&view);
                    self.spines[i].patch.draw(renderer, &tc, &clip);
                }
                Item::Axis(x) => {
                    let sidx = self.label_spine_idx(x);
                    let me: *mut Axes = self;
                    let ac = self.axis_ctx(x, &tc, fig);
                    let axis: *mut Axis = if x { &mut self.xaxis } else { &mut self.yaxis };
                    // SAFETY: the spine closure only reads/updates disjoint
                    // state (spines) plus idempotent tick updates.
                    unsafe {
                        (*axis).draw(renderer, &ac, &clip, |_a| (*me).spine_window_extent_noaxis(sidx, fig));
                    }
                }
                Item::Title(i) => {
                    let t = [&self.title, &self.left_title, &self.right_title][i].clone();
                    t.lk().draw(renderer, &tc, &clip);
                }
                Item::Legend => {
                    if let Some(l) = self.legend.clone() {
                        let bbox = self.bbox(fig);
                        l.lk().draw(renderer, &tc, &clip, bbox, self);
                    }
                }
            }
        }
    }

    /// `apply_aspect(position)`.
    pub fn apply_aspect(&mut self, position: Option<Bbox>, fig: &FigCtx) {
        if self.aspect_frozen {
            return;
        }
        let position = position.unwrap_or(self.original_position);
        let aspect = match self.aspect {
            Aspect::Auto if self.box_aspect.is_none() => {
                self.position = position;
                return;
            }
            Aspect::Auto => None,
            Aspect::Num(a) => Some(a),
        };
        let bb = Bbox::unit().transformed(&fig.trans_figure);
        let fig_aspect = bb.height() / bb.width();
        if self.adjustable == "box" {
            if let Some(a) = aspect {
                let box_aspect = a * self.get_data_ratio();
                let pb1 = shrunk_to_aspect(&position, box_aspect, &position, fig_aspect);
                self.position = anchored(&pb1, self.anchor, &position);
                return;
            }
        }
        let mut pos = position;
        if let Some(ba) = self.box_aspect {
            let pb1 = shrunk_to_aspect(&position, ba, &position, fig_aspect);
            self.position = anchored(&pb1, self.anchor, &position);
            pos = self.position;
            if aspect.is_none() {
                return;
            }
        } else {
            self.position = position;
        }
        let Some(aspect) = aspect else { return };
        let xt = self.xaxis.scale.transform();
        let yt = self.yaxis.scale.transform();
        let ap = |t: &Option<super::transforms::ScaleTransform>, v: f64| t.as_ref().map(|t| t.apply(v)).unwrap_or(v);
        let (xb0, xb1) = self.get_bound(true);
        let (yb0, yb1) = self.get_bound(false);
        let (xmin, xmax) = (ap(&xt, xb0), ap(&xt, xb1));
        let (ymin, ymax) = (ap(&yt, yb0), ap(&yt, yb1));
        let xsize = pymax((xmax - xmin).abs(), 1e-30);
        let ysize = pymax((ymax - ymin).abs(), 1e-30);
        let box_aspect = fig_aspect * (pos.height() / pos.width());
        let data_ratio = box_aspect / aspect;
        let y_expander = data_ratio * xsize / ysize - 1.0;
        if y_expander.abs() < 0.005 {
            return;
        }
        let dl = self.data_lim.bbox;
        let (x0, x1) = (ap(&xt, dl.x0), ap(&xt, dl.x1));
        let (y0, y1) = (ap(&yt, dl.y0), ap(&yt, dl.y1));
        let xr = 1.05 * (x1 - x0);
        let yr = 1.05 * (y1 - y0);
        let xmarg = xsize - xr;
        let ymarg = ysize - yr;
        let ysize_ = data_ratio * xsize;
        let xsize_ = ysize / data_ratio;
        let xmarg_ = xsize_ - xr;
        let ymarg_ = ysize_ - yr;
        let shared_x = !self.shared_x.is_empty();
        let shared_y = !self.shared_y.is_empty();
        let adjust_y = if shared_y {
            false
        } else {
            let adjy = if xmarg > 0.0 && ymarg > 0.0 {
                (ymarg_ > 0.0 && y_expander < 0.0) || (xmarg_ < 0.0 && y_expander > 0.0)
            } else {
                y_expander > 0.0
            };
            shared_x || adjy
        };
        if adjust_y {
            let yc = 0.5 * (ymin + ymax);
            let a = yc - ysize_ / 2.0;
            let b = yc + ysize_ / 2.0;
            let inv = yt.as_ref().map(|t| t.inverted());
            let (a, b) = (ap(&inv, a), ap(&inv, b));
            self.set_bound(false, Some(a), Some(b));
        } else {
            let xc = 0.5 * (xmin + xmax);
            let a = xc - xsize_ / 2.0;
            let b = xc + xsize_ / 2.0;
            let inv = xt.as_ref().map(|t| t.inverted());
            let (a, b) = (ap(&inv, a), ap(&inv, b));
            self.set_bound(true, Some(a), Some(b));
        }
    }

    pub fn get_data_ratio(&self) -> f64 {
        let xt = self.xaxis.scale.transform();
        let yt = self.yaxis.scale.transform();
        let ap = |t: &Option<super::transforms::ScaleTransform>, v: f64| t.as_ref().map(|t| t.apply(v)).unwrap_or(v);
        let (a, b) = self.get_bound(true);
        let (c, d) = self.get_bound(false);
        let xsize = pymax((ap(&xt, b) - ap(&xt, a)).abs(), 1e-30);
        let ysize = pymax((ap(&yt, d) - ap(&yt, c)).abs(), 1e-30);
        ysize / xsize
    }

    /// `get_tightbbox(renderer, for_layout_only)`.
    pub fn get_tightbbox(&mut self, renderer: &mut Renderer, fig: &FigCtx, for_layout_only: bool) -> Option<Bbox> {
        if !self.base.visible {
            return None;
        }
        self.apply_aspect(None, fig);
        let mut bb: Vec<Bbox> = Vec::new();
        let tc = self.trans_ctx(fig);
        for x in [true, false] {
            if self.axison && self.axis(x).visible {
                let sidx = self.label_spine_idx(x);
                let me: *mut Axes = self;
                let ac = self.axis_ctx(x, &tc, fig);
                let axis: *mut Axis = if x { &mut self.xaxis } else { &mut self.yaxis };
                // SAFETY: see `draw`.
                let ba = unsafe { (*axis).get_tightbbox(renderer, &ac, |_a| (*me).spine_window_extent_noaxis(sidx, fig), for_layout_only) };
                if let Some(b) = ba {
                    bb.push(b);
                }
            }
        }
        self.update_title_position(renderer, fig);
        bb.push(self.bbox(fig));
        for t in [&self.title, &self.left_title, &self.right_title] {
            let t = t.lk();
            if t.base.visible {
                let mut bt = t.get_window_extent(renderer, &tc);
                if for_layout_only && bt.width() > 0.0 {
                    bt.x0 = (bt.x0 + bt.x1) / 2.0 - 0.5;
                    bt.x1 = bt.x0 + 1.0;
                }
                bb.push(bt);
            }
        }
        // default bbox extra artists: children not fully clipped to the axes
        let axes_clip = self.clip_bbox(&tc);
        for c in &self.children {
            if let Some(b) = child_tightbbox(c, renderer, &tc, axes_clip) {
                if b.is_finite() {
                    bb.push(b);
                }
            }
        }
        if let Some(l) = &self.legend {
            let bbox = self.bbox(fig);
            if let Some(b) = l.lk().get_window_extent(renderer, &tc, bbox, self) {
                bb.push(b);
            }
        }
        let bb: Vec<Bbox> = bb.into_iter().filter(|b| b.width() != 0.0 || b.height() != 0.0).collect();
        if bb.is_empty() {
            None
        } else {
            Some(Bbox::union(&bb))
        }
    }
}

/// Python `sorted([a, b], reverse=reverse)`.
pub fn sorted2(a: f64, b: f64, reverse: bool) -> (f64, f64) {
    if !reverse {
        if b < a {
            (b, a)
        } else {
            (a, b)
        }
    } else if a < b {
        (b, a)
    } else {
        (a, b)
    }
}

/// Data limits from sibling axes for shared autoscaling.
#[derive(Clone, Debug, Default)]
pub struct SharedLims {
    pub x: Vec<((f64, f64), f64)>,
    pub y: Vec<((f64, f64), f64)>,
    pub x_stickies: Vec<f64>,
    pub y_stickies: Vec<f64>,
}

/// `Artist.get_tightbbox` for children that are not fully clipped to the axes.
fn child_tightbbox(c: &Child, renderer: &mut Renderer, tc: &TransCtx, _axes_clip: Bbox) -> Option<Bbox> {
    match c {
        Child::Text(t) => {
            let t = t.lk();
            if !t.base.visible || !t.base.in_layout {
                return None;
            }
            let bb = t.get_window_extent(renderer, tc);
            Some(clip_extent(bb, &t.base, tc, _axes_clip))
        }
        // lines, patches and collections clipped to the axes are fully
        // clipped (their extents never exceed the axes box)
        _ => None,
    }
}

fn clip_extent(bb: Bbox, base: &ArtistBase, _tc: &TransCtx, axes_clip: Bbox) -> Bbox {
    if base.clip_on {
        match &base.clip_box {
            ClipBoxSpec::AxesBbox => Bbox::intersection(&bb, &axes_clip).unwrap_or(bb),
            ClipBoxSpec::Display(b) => Bbox::intersection(&bb, b).unwrap_or(bb),
            ClipBoxSpec::None => bb,
        }
    } else {
        bb
    }
}

/// `BboxBase.shrunk_to_aspect`.
pub fn shrunk_to_aspect(b: &Bbox, box_aspect: f64, container: &Bbox, fig_aspect: f64) -> Bbox {
    let (w, h) = (container.width(), container.height());
    let mut hh = w * box_aspect / fig_aspect;
    let ww;
    if hh <= h {
        ww = w;
    } else {
        ww = h * fig_aspect / box_aspect;
        hh = h;
    }
    Bbox::new(b.x0, b.y0, b.x0 + ww, b.y0 + hh)
}

/// `BboxBase.anchored(c, container)`.
pub fn anchored(b: &Bbox, c: (f64, f64), container: &Bbox) -> Bbox {
    let (l, bo, w, h) = container.bounds();
    let (ll, bb, ww, hh) = b.bounds();
    let dx = (l + c.0 * (w - ww)) - ll;
    let dy = (bo + c.1 * (h - hh)) - bb;
    Bbox::new(b.x0 + dx, b.y0 + dy, b.x1 + dx, b.y1 + dy)
}

pub fn anchor_coefs(name: &str) -> Option<(f64, f64)> {
    Some(match name {
        "C" => (0.5, 0.5),
        "SW" => (0.0, 0.0),
        "S" => (0.5, 0.0),
        "SE" => (1.0, 0.0),
        "E" => (1.0, 0.5),
        "NE" => (1.0, 1.0),
        "N" => (0.5, 1.0),
        "NW" => (0.0, 1.0),
        "W" => (0.0, 0.5),
        _ => return None,
    })
}

pub fn scale_from_name(name: &str) -> Option<Scale> {
    Some(match name {
        "linear" => Scale::Linear,
        "log" => Scale::Log { base: 10.0, subs: None, clip: true },
        "symlog" => Scale::Symlog { base: 10.0, linthresh: 2.0, linscale: 1.0, subs: None },
        "logit" => Scale::Logit { clip: true },
        "asinh" => Scale::Asinh { linear_width: 1.0 },
        _ => return None,
    })
}
