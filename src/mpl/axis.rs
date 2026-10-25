//! Port of `matplotlib.axis` (Tick, XTick/YTick, Axis, XAxis/YAxis).

use super::artist::{shared, ClipBoxSpec, ClipCtx, Lock, Shared, TSpec, TransCtx, Which};
use super::colors::ColorSpec;
use super::fonts::NumOrName;
use super::lines::Line2D;
use super::markers::MarkerStyle;
use super::rc::{rc, RcValue};
use super::renderer::Renderer;
use super::scale::Scale;
use super::text::Text;
use super::ticker::{AxisInfo, Formatter, Locator, MaxNLocator, ScalarFormatter};
use super::transforms::{Bbox, Transform};

pub const GRIDLINE_INTERPOLATION_STEPS: usize = 180;

/// Grid line properties passed through `grid(**kwargs)` / `tick_params(grid_*)`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GridKw {
    pub color: Option<ColorSpec>,
    pub alpha: Option<Option<f64>>,
    pub linestyle: Option<String>,
    pub dashes: Option<(f64, Option<Vec<f64>>)>,
    pub linewidth: Option<f64>,
    pub zorder: Option<f64>,
    pub visible: Option<bool>,
    pub antialiased: Option<bool>,
}

/// `Axis._major_tick_kw` / `_minor_tick_kw`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TickKw {
    pub size: Option<f64>,
    pub width: Option<f64>,
    pub color: Option<ColorSpec>,
    pub tickdir: Option<String>,
    pub pad: Option<f64>,
    pub labelsize: Option<f64>,
    pub labelcolor: Option<ColorSpec>,
    pub labelfontfamily: Option<Vec<String>>,
    pub zorder: Option<f64>,
    pub grid_on: Option<bool>,
    pub tick1_on: Option<bool>,
    pub tick2_on: Option<bool>,
    pub label1_on: Option<bool>,
    pub label2_on: Option<bool>,
    pub labelrotation: Option<(String, f64)>,
    pub labelrotation_mode: Option<String>,
    pub grid: GridKw,
}

impl TickKw {
    pub fn update(&mut self, o: &TickKw) {
        macro_rules! up {
            ($($f:ident),*) => { $( if o.$f.is_some() { self.$f = o.$f.clone(); } )* };
        }
        up!(size, width, color, tickdir, pad, labelsize, labelcolor, labelfontfamily, zorder, grid_on, tick1_on, tick2_on, label1_on, label2_on, labelrotation, labelrotation_mode);
        macro_rules! upg {
            ($($f:ident),*) => { $( if o.grid.$f.is_some() { self.grid.$f = o.grid.$f.clone(); } )* };
        }
        upg!(color, alpha, linestyle, dashes, linewidth, zorder, visible, antialiased);
    }
}

fn rc_color_or(key: &str, fallback: &str) -> ColorSpec {
    let r = rc();
    match r.get(key) {
        Some(RcValue::Color(c)) => ColorSpec::Rgba(*c),
        Some(RcValue::Str(s)) if s == "inherit" => ColorSpec::Rgba(r.color(fallback)),
        Some(RcValue::Str(s)) if s == "none" => ColorSpec::None,
        _ => ColorSpec::Rgba(r.color(fallback)),
    }
}

#[derive(Clone, Debug)]
pub struct Tick {
    pub is_x: bool,
    pub major: bool,
    pub loc: f64,
    pub size: f64,
    pub width: f64,
    pub base_pad: f64,
    pub tickdir: String,
    pub labelrotation: (String, f64),
    pub zorder: f64,
    pub visible: bool,
    pub tick1line: Shared<Line2D>,
    pub tick2line: Shared<Line2D>,
    pub gridline: Shared<Line2D>,
    pub label1: Shared<Text>,
    pub label2: Shared<Text>,
}

fn tick_markers(is_x: bool, tickdir: &str) -> (&'static str, &'static str) {
    match (is_x, tickdir) {
        (true, "in") => ("tickup", "tickdown"),
        (true, "inout") => ("|", "|"),
        (true, _) => ("tickdown", "tickup"),
        (false, "in") => ("tickright", "tickleft"),
        (false, "inout") => ("_", "_"),
        (false, _) => ("tickleft", "tickright"),
    }
}

impl Tick {
    pub fn new(is_x: bool, loc: f64, major: bool, kw: &TickKw) -> Tick {
        let r = rc();
        let name = if is_x { "xtick" } else { "ytick" };
        let mm = if major { "major" } else { "minor" };
        let grid_on = kw.grid_on.unwrap_or_else(|| {
            let which = r.string("axes.grid.which");
            if major && (which == "both" || which == "major") {
                r.boolean("axes.grid")
            } else if !major && (which == "both" || which == "minor") {
                r.boolean("axes.grid")
            } else {
                false
            }
        });
        let size = kw.size.unwrap_or_else(|| r.num(&format!("{name}.{mm}.size")));
        let width = kw.width.unwrap_or_else(|| r.num(&format!("{name}.{mm}.width")));
        let base_pad = kw.pad.unwrap_or_else(|| r.num(&format!("{name}.{mm}.pad")));
        let color = kw.color.unwrap_or_else(|| ColorSpec::Rgba(r.color(&format!("{name}.color"))));
        let labelcolor = kw.labelcolor.unwrap_or_else(|| rc_color_or(&format!("{name}.labelcolor"), &format!("{name}.color")));
        let labelsize = kw.labelsize.unwrap_or_else(|| r.font_size(&format!("{name}.labelsize")));
        let labelrotation = kw.labelrotation.clone().unwrap_or(("default".into(), 0.0));
        let zorder = kw.zorder.unwrap_or(if major { 2.0 + 0.01 } else { 2.0 });
        // grid properties
        let grid_color = kw.grid.color.unwrap_or_else(|| match r.get(&format!("grid.{mm}.color")) {
            Some(RcValue::Color(c)) => ColorSpec::Rgba(*c),
            _ => ColorSpec::Rgba(r.color("grid.color")),
        });
        let grid_ls = kw.grid.linestyle.clone().unwrap_or_else(|| match r.get(&format!("grid.{mm}.linestyle")) {
            Some(RcValue::Str(s)) => s.clone(),
            _ => r.string("grid.linestyle"),
        });
        let grid_lw = kw.grid.linewidth.unwrap_or_else(|| match r.get(&format!("grid.{mm}.linewidth")) {
            Some(RcValue::Num(v)) => *v,
            _ => r.num("grid.linewidth"),
        });
        let mut grid_color_v = grid_color;
        let grid_alpha = match kw.grid.alpha {
            Some(a) => a,
            None => {
                // a colour with an explicit alpha channel provides the alpha
                match grid_color {
                    ColorSpec::Rgba(c) if c.a != 1.0 => {
                        grid_color_v = ColorSpec::rgba(c.r, c.g, c.b, 1.0);
                        Some(c.a)
                    }
                    _ => match r.get(&format!("grid.{mm}.alpha")) {
                        Some(RcValue::Num(v)) => Some(*v),
                        _ => Some(r.num("grid.alpha")),
                    },
                }
            }
        };

        let mk_tickline = |visible: bool| {
            let mut l = Line2D::new(Vec::new(), Vec::new());
            l.color = color;
            l.set_linestyle("none");
            l.base.zorder = zorder;
            l.base.visible = visible;
            l.markeredgecolor = color;
            l.markersize = size;
            l.markeredgewidth = width;
            l
        };
        let tick1 = mk_tickline(kw.tick1_on.unwrap_or(true));
        let tick2 = mk_tickline(kw.tick2_on.unwrap_or(true));
        let mut grid = Line2D::new(Vec::new(), Vec::new());
        grid.color = grid_color_v;
        grid.base.alpha = grid_alpha;
        grid.base.visible = grid_on;
        grid.set_linestyle(&grid_ls);
        grid.set_linewidth(grid_lw);
        grid.set_marker(MarkerStyle::named(""));
        if let Some((o, d)) = &kw.grid.dashes {
            grid.set_dashes(*o, d.clone());
        }
        if let Some(z) = kw.grid.zorder {
            grid.base.zorder = z;
        }
        if let Some(v) = kw.grid.visible {
            grid.base.visible = v;
        }
        if let Some(aa) = kw.grid.antialiased {
            grid.antialiased = aa;
        }
        grid.interpolation_steps = GRIDLINE_INTERPOLATION_STEPS;
        let mk_label = |visible: bool| {
            let mut t = Text::new(f64::NAN, f64::NAN, "");
            t.fontproperties = t.fontproperties.clone().with_size(labelsize);
            t.color = labelcolor.to_rgba(None);
            t.base.visible = visible;
            if let Some(f) = &kw.labelfontfamily {
                t.fontproperties.family = f.clone();
            }
            t.set_rotation_deg(labelrotation.1);
            if let Some(m) = &kw.labelrotation_mode {
                t.rotation_mode = m.clone();
            }
            t
        };
        let label1 = mk_label(kw.label1_on.unwrap_or(true));
        let label2 = mk_label(kw.label2_on.unwrap_or(false));
        let mut t = Tick {
            is_x,
            major,
            loc,
            size,
            width,
            base_pad,
            tickdir: String::new(),
            labelrotation,
            zorder,
            visible: true,
            tick1line: shared(tick1),
            tick2line: shared(tick2),
            gridline: shared(grid),
            label1: shared(label1),
            label2: shared(label2),
        };
        t.apply_tickdir(kw.tickdir.clone());
        // XTick / YTick specific initialisation
        let pad = t.pad();
        {
            let mut l1 = t.tick1line.lk();
            let mut l2 = t.tick2line.lk();
            let mut g = t.gridline.lk();
            let mut a = t.label1.lk();
            let mut b = t.label2.lk();
            if is_x {
                l1.set_data(vec![0.0], vec![0.0]);
                l1.base.transform = TSpec::XAxis(Which::Tick1);
                l2.set_data(vec![0.0], vec![1.0]);
                l2.base.transform = TSpec::XAxis(Which::Tick2);
                g.set_data(vec![0.0, 0.0], vec![0.0, 1.0]);
                g.base.transform = TSpec::XAxis(Which::Grid);
                a.x = 0.0;
                a.y = 0.0;
                a.va = "top".into();
                a.ha = r.string("xtick.alignment");
                a.base.transform = TSpec::XText1(pad);
                b.x = 0.0;
                b.y = 1.0;
                b.va = "bottom".into();
                b.ha = r.string("xtick.alignment");
                b.base.transform = TSpec::XText2(pad);
            } else {
                l1.set_data(vec![0.0], vec![0.0]);
                l1.base.transform = TSpec::YAxis(Which::Tick1);
                l2.set_data(vec![1.0], vec![0.0]);
                l2.base.transform = TSpec::YAxis(Which::Tick2);
                g.set_data(vec![0.0, 1.0], vec![0.0, 0.0]);
                g.base.transform = TSpec::YAxis(Which::Grid);
                a.x = 0.0;
                a.y = 0.0;
                a.va = r.string("ytick.alignment");
                a.ha = "right".into();
                a.base.transform = TSpec::YText1(pad);
                b.x = 1.0;
                b.y = 0.0;
                b.va = r.string("ytick.alignment");
                b.ha = "left".into();
                b.base.transform = TSpec::YText2(pad);
            }
            for l in [&mut *l1, &mut *l2, &mut *g] {
                l.base.transform_set = true;
            }
            a.base.transform_set = true;
            b.base.transform_set = true;
        }
        t.update_position(loc);
        t
    }

    pub fn tick_padding(&self) -> f64 {
        let p = match self.tickdir.as_str() {
            "in" => 0.0,
            "inout" => 0.5,
            _ => 1.0,
        };
        self.size * p
    }

    pub fn pad(&self) -> f64 {
        self.base_pad + self.tick_padding()
    }

    pub fn apply_tickdir(&mut self, tickdir: Option<String>) {
        let name = if self.is_x { "xtick" } else { "ytick" };
        let td = tickdir.unwrap_or_else(|| rc().string(&format!("{name}.direction")));
        self.tickdir = td;
        let (m1, m2) = tick_markers(self.is_x, &self.tickdir);
        let fs1 = self.tick1line.lk().marker.fillstyle;
        let fs2 = self.tick2line.lk().marker.fillstyle;
        self.tick1line.lk().set_marker(MarkerStyle::new(super::markers::MarkerSpec::Named(m1.into()), fs1));
        self.tick2line.lk().set_marker(MarkerStyle::new(super::markers::MarkerSpec::Named(m2.into()), fs2));
    }

    pub fn update_position(&mut self, loc: f64) {
        if self.is_x {
            self.tick1line.lk().set_xdata(vec![loc]);
            self.tick2line.lk().set_xdata(vec![loc]);
            self.gridline.lk().set_xdata(vec![loc]);
            self.label1.lk().x = loc;
            self.label2.lk().x = loc;
        } else {
            self.tick1line.lk().set_ydata(vec![loc]);
            self.tick2line.lk().set_ydata(vec![loc]);
            self.gridline.lk().set_ydata(vec![loc]);
            self.label1.lk().y = loc;
            self.label2.lk().y = loc;
        }
        self.loc = loc;
    }

    /// `Tick._apply_params(**kw)` for the parameters in `kw`.
    pub fn apply_params(&mut self, kw: &TickKw) {
        if let Some(v) = kw.grid_on {
            self.gridline.lk().base.visible = v;
        }
        if let Some(v) = kw.tick1_on {
            self.tick1line.lk().base.visible = v;
        }
        if let Some(v) = kw.tick2_on {
            self.tick2line.lk().base.visible = v;
        }
        if let Some(v) = kw.label1_on {
            self.label1.lk().base.visible = v;
        }
        if let Some(v) = kw.label2_on {
            self.label2.lk().base.visible = v;
        }
        if kw.size.is_some() || kw.width.is_some() || kw.pad.is_some() || kw.tickdir.is_some() {
            if let Some(s) = kw.size {
                self.size = s;
            }
            if let Some(w) = kw.width {
                self.width = w;
            }
            if let Some(p) = kw.pad {
                self.base_pad = p;
            }
            let td = kw.tickdir.clone().unwrap_or_else(|| self.tickdir.clone());
            self.apply_tickdir(Some(td));
            for l in [&self.tick1line, &self.tick2line] {
                let mut l = l.lk();
                l.markersize = self.size;
                l.markeredgewidth = self.width;
            }
            let pad = self.pad();
            self.label1.lk().base.transform = if self.is_x { TSpec::XText1(pad) } else { TSpec::YText1(pad) };
            self.label2.lk().base.transform = if self.is_x { TSpec::XText2(pad) } else { TSpec::YText2(pad) };
        }
        if let Some(c) = kw.color {
            for l in [&self.tick1line, &self.tick2line] {
                let mut l = l.lk();
                l.color = c;
                l.markeredgecolor = c;
            }
        }
        if let Some(z) = kw.zorder {
            self.tick1line.lk().base.zorder = z;
            self.tick2line.lk().base.zorder = z;
            self.zorder = z;
        }
        if let Some(r) = &kw.labelrotation {
            self.labelrotation = r.clone();
            self.label1.lk().set_rotation_deg(r.1);
            self.label2.lk().set_rotation_deg(r.1);
        }
        for lab in [&self.label1, &self.label2] {
            let mut t = lab.lk();
            if let Some(s) = kw.labelsize {
                t.fontproperties.size = super::fonts::resolve_size(Some(s), None);
            }
            if let Some(c) = kw.labelcolor {
                t.color = c.to_rgba(None);
            }
            if let Some(f) = &kw.labelfontfamily {
                t.fontproperties.family = f.clone();
            }
            if let Some(m) = &kw.labelrotation_mode {
                t.rotation_mode = m.clone();
            }
        }
        let g = &kw.grid;
        let mut gl = self.gridline.lk();
        if let Some(c) = g.color {
            match (c, g.alpha) {
                (ColorSpec::Rgba(cc), None) if cc.a != 1.0 => {
                    gl.color = ColorSpec::rgba(cc.r, cc.g, cc.b, 1.0);
                    gl.base.alpha = Some(cc.a);
                }
                _ => gl.color = c,
            }
        }
        if let Some(a) = g.alpha {
            gl.base.alpha = a;
        }
        if let Some(ls) = &g.linestyle {
            gl.set_linestyle(ls);
        }
        if let Some((o, d)) = &g.dashes {
            gl.set_dashes(*o, d.clone());
        }
        if let Some(lw) = g.linewidth {
            gl.set_linewidth(lw);
        }
        if let Some(z) = g.zorder {
            gl.base.zorder = z;
        }
        if let Some(v) = g.visible {
            gl.base.visible = v;
        }
        if let Some(aa) = g.antialiased {
            gl.antialiased = aa;
        }
    }

    /// `_copy_tick_props(src, self)`.
    pub fn copy_props_from(&mut self, src: &Tick) {
        text_update_from(&mut self.label1.lk(), &src.label1.lk());
        text_update_from(&mut self.label2.lk(), &src.label2.lk());
        self.tick1line.lk().update_from(&src.tick1line.lk());
        self.tick2line.lk().update_from(&src.tick2line.lk());
        self.gridline.lk().update_from(&src.gridline.lk());
        self.visible = src.visible;
        self.loc = src.loc;
        self.size = src.size;
        self.width = src.width;
        self.base_pad = src.base_pad;
        self.labelrotation = src.labelrotation.clone();
        self.zorder = src.zorder;
        self.tickdir = src.tickdir.clone();
    }

    pub fn draw(&self, renderer: &mut Renderer, ctx: &TransCtx, clip: &ClipCtx) {
        if !self.visible {
            return;
        }
        self.gridline.lk().draw(renderer, ctx, clip, None, false);
        self.tick1line.lk().draw(renderer, ctx, clip, None, false);
        self.tick2line.lk().draw(renderer, ctx, clip, None, false);
        self.label1.lk().draw(renderer, ctx, clip);
        self.label2.lk().draw(renderer, ctx, clip);
    }
}

/// `Text.update_from`.
pub fn text_update_from(dst: &mut Text, src: &Text) {
    dst.base.transform = src.base.transform.clone();
    dst.base.transform_set = src.base.transform_set;
    dst.base.visible = src.base.visible;
    dst.base.alpha = src.base.alpha;
    dst.base.clip_box = src.base.clip_box.clone();
    dst.base.clip_on = src.base.clip_on;
    dst.base.clip_path = src.base.clip_path.clone();
    dst.base.label = src.base.label.clone();
    dst.base.sketch = src.base.sketch;
    dst.color = src.color;
    dst.multialignment = src.multialignment.clone();
    dst.va = src.va.clone();
    dst.ha = src.ha.clone();
    dst.fontproperties = src.fontproperties.clone();
    dst.usetex = src.usetex;
    dst.rotation = src.rotation;
    dst.transform_rotates_text = src.transform_rotates_text;
    dst.linespacing = src.linespacing;
    dst.antialiased = src.antialiased;
}

/// Everything an Axis needs to know about its Axes while computing ticks.
pub struct AxisCtx<'a> {
    pub trans: &'a TransCtx,
    pub view: (f64, f64),
    pub data: (f64, f64),
    pub minpos: f64,
    pub axes_bbox: Bbox,
    pub dpi: f64,
}

#[derive(Clone, Debug)]
pub struct Axis {
    pub is_x: bool,
    pub visible: bool,
    pub zorder: f64,
    pub major_locator: Locator,
    pub minor_locator: Locator,
    pub major_formatter: Formatter,
    pub minor_formatter: Formatter,
    pub default_majloc: bool,
    pub default_minloc: bool,
    pub default_majfmt: bool,
    pub default_minfmt: bool,
    pub label: Shared<Text>,
    pub offset_text: Shared<Text>,
    pub labelpad: f64,
    pub major_ticks: Option<Vec<Tick>>,
    pub minor_ticks: Option<Vec<Tick>>,
    pub major_tick_kw: TickKw,
    pub minor_tick_kw: TickKw,
    pub autolabelpos: bool,
    pub label_position: String,
    pub tick_position: String,
    pub remove_overlapping_locs: bool,
    pub scale: Scale,
    pub autoscale_on: bool,
    pub is_default_label: bool,
    /// clip box of the tick gridlines (set via `set_clip_path(axes.patch)`)
    pub clip_box: ClipBoxSpec,
}

pub const OFFSETTEXTPAD: f64 = 3.0;

impl Axis {
    pub fn new(is_x: bool) -> Axis {
        let r = rc();
        let mut label = Text::new(f64::NAN, f64::NAN, "");
        label.fontproperties = label.fontproperties.clone().with_size(r.font_size("axes.labelsize"));
        label.fontproperties.weight = super::fonts::parse_weight(&r.string("axes.labelweight"));
        label.color = r.color("axes.labelcolor");
        let offset_text = Text::new(f64::NAN, f64::NAN, "");
        let mut a = Axis {
            is_x,
            visible: true,
            zorder: 0.0,
            major_locator: Locator::MaxN(MaxNLocator::auto()),
            minor_locator: Locator::Null,
            major_formatter: Formatter::Scalar(ScalarFormatter::new()),
            minor_formatter: Formatter::Null,
            default_majloc: true,
            default_minloc: true,
            default_majfmt: true,
            default_minfmt: true,
            label: shared(label),
            offset_text: shared(offset_text),
            labelpad: r.num("axes.labelpad"),
            major_ticks: None,
            minor_ticks: None,
            major_tick_kw: TickKw::default(),
            minor_tick_kw: TickKw::default(),
            autolabelpos: true,
            label_position: if is_x { "bottom".into() } else { "left".into() },
            tick_position: if is_x { "bottom".into() } else { "left".into() },
            remove_overlapping_locs: true,
            scale: Scale::Linear,
            autoscale_on: true,
            is_default_label: true,
            clip_box: ClipBoxSpec::AxesBbox,
        };
        a.init();
        a
    }

    pub fn name(&self) -> &'static str {
        if self.is_x {
            "x"
        } else {
            "y"
        }
    }

    /// `XAxis._init` / `YAxis._init`.
    pub fn init(&mut self) {
        let r = rc();
        let tick = if self.is_x { "xtick" } else { "ytick" };
        let tick_color = rc_color_or(&format!("{tick}.labelcolor"), &format!("{tick}.color"));
        {
            let mut l = self.label.lk();
            let mut o = self.offset_text.lk();
            if self.is_x {
                l.x = 0.5;
                l.y = 0.0;
                l.va = "top".into();
                l.ha = "center".into();
                l.base.transform = TSpec::blend(TSpec::Axes, TSpec::Identity);
                o.x = 1.0;
                o.y = 0.0;
                o.va = "top".into();
                o.ha = "right".into();
                o.base.transform = TSpec::blend(TSpec::Axes, TSpec::Identity);
            } else {
                l.x = 0.0;
                l.y = 0.5;
                l.va = "bottom".into();
                l.ha = "center".into();
                l.rotation = 90.0;
                l.rotation_mode = "anchor".into();
                l.base.transform = TSpec::blend(TSpec::Identity, TSpec::Axes);
                o.x = 0.0;
                o.y = 0.5;
                o.va = "baseline".into();
                o.ha = "left".into();
                o.base.transform = TSpec::blend(TSpec::Axes, TSpec::Identity);
            }
            l.base.transform_set = true;
            o.base.transform_set = true;
            o.fontproperties = o.fontproperties.clone().with_size(r.font_size(&format!("{tick}.labelsize")));
            o.color = tick_color.to_rgba(None);
        }
        self.label_position = if self.is_x { "bottom".into() } else { "left".into() };
    }

    /// `Axis.clear()`.
    pub fn clear(&mut self) {
        let r = rc();
        {
            let mut l = self.label.lk();
            l.reset_visual_defaults();
            l.color = r.color("axes.labelcolor");
            l.fontproperties = l.fontproperties.clone().with_size(r.font_size("axes.labelsize"));
            l.fontproperties.weight = super::fonts::parse_weight(&r.string("axes.labelweight"));
        }
        self.offset_text.lk().reset_visual_defaults();
        self.labelpad = r.num("axes.labelpad");
        self.init();
        self.set_scale(Scale::Linear);
        let which = r.string("axes.grid.which");
        self.major_tick_kw.grid_on = Some(r.boolean("axes.grid") && (which == "both" || which == "major"));
        self.minor_tick_kw.grid_on = Some(r.boolean("axes.grid") && (which == "both" || which == "minor"));
        self.reset_ticks();
    }

    pub fn reset_ticks(&mut self) {
        self.major_ticks = None;
        self.minor_ticks = None;
    }

    /// `_set_scale`.
    pub fn set_scale(&mut self, scale: Scale) {
        let (majloc, minloc, majfmt, minfmt) = scale.default_locators_and_formatters(self.is_x);
        self.scale = scale;
        self.major_locator = majloc;
        self.minor_locator = minloc;
        self.major_formatter = majfmt;
        self.minor_formatter = minfmt;
        self.default_majloc = true;
        self.default_minloc = true;
        self.default_majfmt = true;
        self.default_minfmt = true;
    }

    pub fn get_transform(&self) -> Option<super::transforms::ScaleTransform> {
        self.scale.transform()
    }

    fn scale_t(&self, v: f64) -> f64 {
        match self.scale.transform() {
            Some(t) => t.apply(v),
            None => v,
        }
    }

    fn get_tick(&self, major: bool) -> Tick {
        let kw = if major { &self.major_tick_kw } else { &self.minor_tick_kw };
        let t = Tick::new(self.is_x, 0.0, major, kw);
        t.gridline.lk().base.clip_box = self.clip_box.clone();
        t
    }

    pub fn major_ticks_mut(&mut self) -> &mut Vec<Tick> {
        if self.major_ticks.is_none() {
            let t = self.get_tick(true);
            self.major_ticks = Some(vec![t]);
        }
        self.major_ticks.as_mut().unwrap()
    }

    pub fn minor_ticks_mut(&mut self) -> &mut Vec<Tick> {
        if self.minor_ticks.is_none() {
            let t = self.get_tick(false);
            self.minor_ticks = Some(vec![t]);
        }
        self.minor_ticks.as_mut().unwrap()
    }

    /// `get_major_ticks(numticks)` growing the list.
    pub fn ensure_major_ticks(&mut self, n: usize) {
        self.major_ticks_mut();
        while self.major_ticks.as_ref().unwrap().len() < n {
            let mut t = self.get_tick(true);
            t.copy_props_from(&self.major_ticks.as_ref().unwrap()[0]);
            self.major_ticks.as_mut().unwrap().push(t);
        }
    }

    pub fn ensure_minor_ticks(&mut self, n: usize) {
        self.minor_ticks_mut();
        while self.minor_ticks.as_ref().unwrap().len() < n {
            let mut t = self.get_tick(false);
            t.copy_props_from(&self.minor_ticks.as_ref().unwrap()[0]);
            self.minor_ticks.as_mut().unwrap().push(t);
        }
    }

    /// `set_tick_params(which, reset, **kw)`.
    pub fn set_tick_params(&mut self, which: &str, reset: bool, kw: &TickKw) {
        let r = rc();
        let gridwhich = r.string("axes.grid.which");
        if reset {
            if which == "major" || which == "both" {
                self.major_tick_kw = TickKw::default();
                self.major_tick_kw.grid_on = Some(r.boolean("axes.grid") && (gridwhich == "both" || gridwhich == "major"));
                self.major_tick_kw.update(kw);
            }
            if which == "minor" || which == "both" {
                self.minor_tick_kw = TickKw::default();
                self.minor_tick_kw.grid_on = Some(r.boolean("axes.grid") && (gridwhich == "both" || gridwhich == "minor"));
                self.minor_tick_kw.update(kw);
            }
            self.reset_ticks();
        } else {
            if which == "major" || which == "both" {
                self.major_tick_kw.update(kw);
                for t in self.major_ticks_mut().iter_mut() {
                    t.apply_params(kw);
                }
            }
            if which == "minor" || which == "both" {
                self.minor_tick_kw.update(kw);
                for t in self.minor_ticks_mut().iter_mut() {
                    t.apply_params(kw);
                }
            }
            if kw.label1_on.is_some() || kw.label2_on.is_some() {
                let v = self.major_tick_kw.label1_on.unwrap_or(false) || self.major_tick_kw.label2_on.unwrap_or(false);
                self.offset_text.lk().base.visible = v;
            }
            if let Some(c) = kw.labelcolor {
                self.offset_text.lk().color = c.to_rgba(None);
            }
        }
    }

    /// `grid(visible, which, **kwargs)`.
    pub fn grid(&mut self, visible: Option<bool>, which: &str, gkw: &GridKw) {
        let has_kw = *gkw != GridKw::default();
        let visible = if has_kw && visible.is_none() { Some(true) } else if has_kw { Some(true) } else { visible };
        let mut kw = TickKw { grid: gkw.clone(), ..TickKw::default() };
        if which == "minor" || which == "both" {
            kw.grid_on = Some(match visible {
                None => !self.minor_tick_kw.grid_on.unwrap_or(false),
                Some(v) => v,
            });
            self.set_tick_params("minor", false, &kw);
        }
        if which == "major" || which == "both" {
            kw.grid_on = Some(match visible {
                None => !self.major_tick_kw.grid_on.unwrap_or(false),
                Some(v) => v,
            });
            self.set_tick_params("major", false, &kw);
        }
    }

    pub fn tick_label_size(&self) -> f64 {
        let r = rc();
        let size = self.major_tick_kw.labelsize.unwrap_or_else(|| r.font_size(&format!("{}tick.labelsize", self.name())));
        super::fonts::resolve_size(Some(size), None)
    }

    /// `get_tick_space()`.
    pub fn tick_space(&self, ctx: &TransCtx) -> i64 {
        let ta = ctx.trans_axes.clone().unwrap_or_else(|| ctx.trans_figure.clone());
        let t = ta.then(&ctx.dpi_scale.inverted());
        let ends = Bbox::unit().transformed(&t);
        let (length, size) = if self.is_x {
            (ends.width() * 72.0, self.tick_label_size() * 3.0)
        } else {
            (ends.height() * 72.0, self.tick_label_size() * 2.0)
        };
        if size > 0.0 {
            (length / size).floor() as i64
        } else {
            i64::MAX
        }
    }

    pub fn axis_info(&self, ctx: &AxisCtx) -> AxisInfo {
        AxisInfo {
            view: ctx.view,
            data: ctx.data,
            minpos: ctx.minpos,
            tick_space: self.tick_space(ctx.trans),
            axis_name: if self.is_x { 'x' } else { 'y' },
            scale: self.scale.name().to_string(),
            major_locs: Vec::new(),
        }
    }

    pub fn get_majorticklocs(&self, ctx: &AxisCtx) -> Vec<f64> {
        let info = self.axis_info(ctx);
        self.major_locator.call(&info)
    }

    pub fn get_minorticklocs(&self, ctx: &AxisCtx) -> Vec<f64> {
        let mut info = self.axis_info(ctx);
        let major = self.major_locator.call(&info);
        info.major_locs = major.clone();
        let minor = self.minor_locator.call(&info);
        if !self.remove_overlapping_locs {
            return minor;
        }
        let tmaj: Vec<f64> = major.iter().map(|&v| self.scale_t(v)).collect();
        let (a, b) = (self.scale_t(ctx.view.0), self.scale_t(ctx.view.1));
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let tol = (hi - lo) * 1e-5;
        minor
            .into_iter()
            .filter(|&m| {
                let tm = self.scale_t(m);
                !tmaj.iter().any(|&t| (tm - t).abs() <= tol)
            })
            .collect()
    }

    fn tick_group_visible(kw: &TickKw) -> bool {
        kw.tick1_on != Some(false)
            || kw.tick2_on != Some(false)
            || kw.label1_on != Some(false)
            || kw.label2_on != Some(false)
            || kw.grid_on != Some(false)
    }

    /// `_update_ticks()`: returns (major indices, minor indices) to draw.
    pub fn update_ticks(&mut self, ctx: &AxisCtx) -> (Vec<usize>, Vec<usize>) {
        let mut info = self.axis_info(ctx);
        let mut nmaj = 0;
        let mut major_locs = Vec::new();
        if Self::tick_group_visible(&self.major_tick_kw) && !self.major_locator.is_null() {
            major_locs = self.major_locator.call(&info);
            let labels = self.major_formatter.format_ticks(&major_locs, &info);
            self.ensure_major_ticks(major_locs.len());
            let ticks = self.major_ticks.as_mut().unwrap();
            for (i, (&loc, label)) in major_locs.iter().zip(labels.iter()).enumerate() {
                ticks[i].update_position(loc);
                ticks[i].label1.lk().text = label.clone();
                ticks[i].label2.lk().text = label.clone();
            }
            nmaj = major_locs.len();
        }
        let mut nmin = 0;
        if Self::tick_group_visible(&self.minor_tick_kw) && !self.minor_locator.is_null() {
            if major_locs.is_empty() && !self.major_locator.is_null() {
                major_locs = self.major_locator.call(&info);
            }
            info.major_locs = major_locs.clone();
            let mut minor_locs = self.minor_locator.call(&info);
            if self.remove_overlapping_locs {
                let tmaj: Vec<f64> = major_locs.iter().map(|&v| self.scale_t(v)).collect();
                let (a, b) = (self.scale_t(ctx.view.0), self.scale_t(ctx.view.1));
                let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                let tol = (hi - lo) * 1e-5;
                minor_locs.retain(|&m| {
                    let tm = self.scale_t(m);
                    !tmaj.iter().any(|&t| (tm - t).abs() <= tol)
                });
            }
            let labels = self.minor_formatter.format_ticks(&minor_locs, &info);
            self.ensure_minor_ticks(minor_locs.len());
            let ticks = self.minor_ticks.as_mut().unwrap();
            for (i, (&loc, label)) in minor_locs.iter().zip(labels.iter()).enumerate() {
                ticks[i].update_position(loc);
                ticks[i].label1.lk().text = label.clone();
                ticks[i].label2.lk().text = label.clone();
            }
            nmin = minor_locs.len();
        }
        if nmaj + nmin == 0 {
            return (Vec::new(), Vec::new());
        }
        let (mut lo, mut hi) = ctx.view;
        if lo > hi {
            std::mem::swap(&mut lo, &mut hi);
        }
        let (ia, ib) = (self.scale_t(lo), self.scale_t(hi));
        let contains = |v: f64| {
            let (a, b) = if ia > ib { (ib, ia) } else { (ia, ib) };
            let rtol = (b - a) * 1e-10;
            a - rtol <= v && v <= b + rtol
        };
        let mj: Vec<usize> = (0..nmaj)
            .filter(|&i| contains(self.scale_t(self.major_ticks.as_ref().unwrap()[i].loc)))
            .collect();
        let mn: Vec<usize> = (0..nmin)
            .filter(|&i| contains(self.scale_t(self.minor_ticks.as_ref().unwrap()[i].loc)))
            .collect();
        (mj, mn)
    }

    fn ticks_iter<'a>(&'a self, sel: &'a (Vec<usize>, Vec<usize>)) -> impl Iterator<Item = &'a Tick> + 'a {
        let maj = self.major_ticks.as_deref().unwrap_or(&[]);
        let min = self.minor_ticks.as_deref().unwrap_or(&[]);
        sel.0.iter().map(move |&i| &maj[i]).chain(sel.1.iter().map(move |&i| &min[i]))
    }

    /// `_get_ticklabel_bboxes`.
    pub fn ticklabel_bboxes(&self, sel: &(Vec<usize>, Vec<usize>), renderer: &mut Renderer, ctx: &TransCtx) -> (Vec<Bbox>, Vec<Bbox>) {
        let mut b1 = Vec::new();
        let mut b2 = Vec::new();
        for t in self.ticks_iter(sel) {
            let l1 = t.label1.lk();
            if l1.base.visible && l1.base.in_layout {
                b1.push(l1.get_window_extent(renderer, ctx));
            }
        }
        for t in self.ticks_iter(sel) {
            let l2 = t.label2.lk();
            if l2.base.visible && l2.base.in_layout {
                b2.push(l2.get_window_extent(renderer, ctx));
            }
        }
        (b1, b2)
    }

    /// `Spine.get_window_extent` padding for this axis' drawn ticks.
    pub fn spine_window_extent(&mut self, spine_idx: usize, bb: Bbox, ctx: &AxisCtx) -> Bbox {
        if !self.visible {
            return bb;
        }
        let sel = self.update_ticks(ctx);
        let mut bboxes = vec![bb];
        let maj = self.major_ticks.as_deref().unwrap_or(&[]);
        let min = self.minor_ticks.as_deref().unwrap_or(&[]);
        let picks: Vec<&Tick> = sel.0.first().map(|&i| &maj[i]).into_iter().chain(sel.1.first().map(|&i| &min[i])).collect();
        for tick in picks {
            let mut bb0 = bb;
            let tickl = tick.size;
            let (po, pi) = match tick.tickdir.as_str() {
                "out" => (1.0, 0.0),
                "in" => (0.0, 1.0),
                _ => (0.5, 0.5),
            };
            let padout = po * tickl / 72.0 * ctx.dpi;
            let padin = pi * tickl / 72.0 * ctx.dpi;
            if tick.tick1line.lk().base.visible {
                if spine_idx == 0 {
                    bb0.x0 -= padout;
                    bb0.x1 += padin;
                } else if spine_idx == 2 {
                    bb0.y0 -= padout;
                    bb0.y1 += padin;
                }
            }
            if tick.tick2line.lk().base.visible {
                if spine_idx == 1 {
                    bb0.x1 += padout;
                    bb0.x0 -= padin;
                } else if spine_idx == 3 {
                    bb0.y1 += padout;
                    bb0.y0 -= padout;
                }
            }
            bboxes.push(bb0);
        }
        Bbox::union(&bboxes)
    }

    /// `_update_label_position`; `spine_bb` is the window extent of the
    /// spine on the label side.
    pub fn update_label_position(&mut self, bboxes: &[Bbox], bboxes2: &[Bbox], spine_bb: Bbox, dpi: f64) {
        if !self.autolabelpos {
            return;
        }
        let mut l = self.label.lk();
        if self.is_x {
            if self.label_position == "bottom" {
                let mut all = bboxes.to_vec();
                all.push(spine_bb);
                let bbox = Bbox::union(&all);
                l.y = bbox.y0 - self.labelpad * dpi / 72.0;
            } else {
                let mut all = bboxes2.to_vec();
                all.push(spine_bb);
                let bbox = Bbox::union(&all);
                l.y = bbox.y1 + self.labelpad * dpi / 72.0;
            }
        } else if self.label_position == "left" {
            let mut all = bboxes.to_vec();
            all.push(spine_bb);
            let bbox = Bbox::union(&all);
            l.x = bbox.x0 - self.labelpad * dpi / 72.0;
        } else {
            let mut all = bboxes2.to_vec();
            all.push(spine_bb);
            let bbox = Bbox::union(&all);
            l.x = bbox.x1 + self.labelpad * dpi / 72.0;
        }
    }

    /// `_update_offset_text_position`.
    pub fn update_offset_text_position(&mut self, bboxes: &[Bbox], bboxes2: &[Bbox], axes_bbox: Bbox, dpi: f64) {
        let mut o = self.offset_text.lk();
        if self.is_x {
            if self.tick_position == "bottom" {
                let bottom = if bboxes.is_empty() { axes_bbox.ymin() } else { Bbox::union(bboxes).y0 };
                o.y = bottom - OFFSETTEXTPAD * dpi / 72.0;
            } else {
                let top = if bboxes2.is_empty() { axes_bbox.ymax() } else { Bbox::union(bboxes2).y1 };
                o.y = top + OFFSETTEXTPAD * dpi / 72.0;
            }
        } else {
            let top = axes_bbox.ymax();
            o.y = top + OFFSETTEXTPAD * dpi / 72.0;
        }
    }

    /// `get_ticks_position()` helper: 1 ("bottom"/"left"), 2, "default" or "unknown".
    pub fn ticks_position(&mut self) -> &'static str {
        let mut reps: Vec<(bool, bool, bool, bool)> = Vec::new();
        if !self.major_locator.is_null() {
            let t = &self.major_ticks_mut()[0];
            reps.push((t.tick1line.lk().base.visible, t.tick2line.lk().base.visible, t.label1.lk().base.visible, t.label2.lk().base.visible));
        }
        if !self.minor_locator.is_null() {
            let t = &self.minor_ticks_mut()[0];
            reps.push((t.tick1line.lk().base.visible, t.tick2line.lk().base.visible, t.label1.lk().base.visible, t.label2.lk().base.visible));
        }
        let (one, two) = if self.is_x { ("bottom", "top") } else { ("left", "right") };
        if reps.iter().all(|r| r.0 && !r.1 && r.2 && !r.3) {
            one
        } else if reps.iter().all(|r| r.1 && !r.0 && r.3 && !r.2) {
            two
        } else if reps.iter().all(|r| r.0 && r.1 && r.2 && !r.3) {
            "default"
        } else {
            "unknown"
        }
    }

    /// `Axis.draw` given the spine extent on the label side.
    pub fn draw(&mut self, renderer: &mut Renderer, ctx: &AxisCtx, clip: &ClipCtx, spine_bb: impl FnOnce(&mut Axis) -> Bbox) {
        if !self.visible {
            return;
        }
        let sel = self.update_ticks(ctx);
        let (tlb1, tlb2) = self.ticklabel_bboxes(&sel, renderer, ctx.trans);
        for t in self.ticks_iter(&sel) {
            t.draw(renderer, ctx.trans, clip);
        }
        // _update_label_position: tick boxes of the (unaligned) siblings,
        // i.e. this axis again
        let sel2 = self.update_ticks(ctx);
        let (b1, b2) = self.ticklabel_bboxes(&sel2, renderer, ctx.trans);
        let sbb = spine_bb(self);
        self.update_label_position(&b1, &b2, sbb, ctx.dpi);
        self.label.lk().draw(renderer, ctx.trans, clip);
        self.update_offset_text_position(&tlb1, &tlb2, ctx.axes_bbox, ctx.dpi);
        let off = self.major_formatter.get_offset();
        self.offset_text.lk().text = off;
        self.offset_text.lk().draw(renderer, ctx.trans, clip);
    }

    /// `get_tightbbox`.
    pub fn get_tightbbox(&mut self, renderer: &mut Renderer, ctx: &AxisCtx, spine_bb: impl FnOnce(&mut Axis) -> Bbox, for_layout_only: bool) -> Option<Bbox> {
        if !self.visible {
            return None;
        }
        let sel = self.update_ticks(ctx);
        let sel2 = self.update_ticks(ctx);
        let (b1, b2) = self.ticklabel_bboxes(&sel2, renderer, ctx.trans);
        let sbb = spine_bb(self);
        self.update_label_position(&b1, &b2, sbb, ctx.dpi);
        let (tlb1, tlb2) = self.ticklabel_bboxes(&sel, renderer, ctx.trans);
        self.update_offset_text_position(&tlb1, &tlb2, ctx.axes_bbox, ctx.dpi);
        let off = self.major_formatter.get_offset();
        self.offset_text.lk().text = off;
        let mut bboxes: Vec<Bbox> = Vec::new();
        {
            let o = self.offset_text.lk();
            if o.base.visible {
                bboxes.push(o.get_window_extent(renderer, ctx.trans));
            }
        }
        bboxes.extend(tlb1);
        bboxes.extend(tlb2);
        {
            let l = self.label.lk();
            if l.base.visible {
                let mut bb = l.get_window_extent(renderer, ctx.trans);
                if for_layout_only {
                    if self.is_x && bb.width() > 0.0 {
                        bb.x0 = (bb.x0 + bb.x1) / 2.0 - 0.5;
                        bb.x1 = bb.x0 + 1.0;
                    }
                    if !self.is_x && bb.height() > 0.0 {
                        bb.y0 = (bb.y0 + bb.y1) / 2.0 - 0.5;
                        bb.y1 = bb.y0 + 1.0;
                    }
                }
                bboxes.push(bb);
            }
        }
        let bboxes: Vec<Bbox> = bboxes.into_iter().filter(|b| b.is_finite()).collect();
        if bboxes.is_empty() {
            None
        } else {
            Some(Bbox::union(&bboxes))
        }
    }

    pub fn label_text(&self) -> String {
        self.label.lk().text.clone()
    }
}

/// Weight helper for tick label font defaults.
pub fn weight_of(n: &NumOrName) -> i32 {
    n.weight()
}

/// Locator nonsingular for autoscaling (`axis.get_major_locator().nonsingular`).
pub fn locator_nonsingular(axis: &Axis, v0: f64, v1: f64, ctx: &AxisCtx) -> (f64, f64) {
    let info = axis.axis_info(ctx);
    axis.major_locator.nonsingular(v0, v1, &info)
}

/// Transform used for an axis (`axis.get_transform()`), as a 2-D transform.
pub fn axis_transform_1d(axis: &Axis) -> Transform {
    match axis.scale.transform() {
        None => Transform::identity(),
        Some(t) => {
            if axis.is_x {
                Transform::Separable(Some(t), None)
            } else {
                Transform::Separable(None, Some(t))
            }
        }
    }
}
