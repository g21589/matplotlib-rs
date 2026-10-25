//! Port of `matplotlib.lines.Line2D`.

use super::artist::{ArtistBase, ClipCtx, GcState, TSpec, TransCtx};
use super::colors::ColorSpec;
use super::markers::{MarkerStyle, SnapThreshold};
use super::path::{new_path, Path, SimplifyParams};
use super::rc::rc;
use super::renderer::Renderer;
use super::transforms::{Bbox, Transform};
use crate::agg::stroke::{LineCap, LineJoin};

pub type DashPattern = (f64, Option<Vec<f64>>);

/// `_get_dash_pattern` for named styles.
pub fn get_dash_pattern(style: &str) -> DashPattern {
    let style = match style {
        "-" => "solid",
        "--" => "dashed",
        "-." => "dashdot",
        ":" => "dotted",
        s => s,
    };
    match style {
        "dashed" | "dashdot" | "dotted" => {
            let d = rc().num_list(&format!("lines.{style}_pattern"));
            normalize_dashes(0.0, Some(d))
        }
        _ => (0.0, None),
    }
}

/// Offset normalisation of `_get_dash_pattern` for explicit `(offset, dashes)`.
pub fn normalize_dashes(mut offset: f64, dashes: Option<Vec<f64>>) -> DashPattern {
    if let Some(d) = &dashes {
        let dsum: f64 = d.iter().fold(0.0, |a, b| a + b);
        if dsum != 0.0 {
            offset = super::text::py_mod(offset, dsum);
        }
    }
    (offset, dashes)
}

/// `_scale_dashes`.
pub fn scale_dashes(p: &DashPattern, lw: f64) -> DashPattern {
    if !rc().boolean("lines.scale_dashes") {
        return p.clone();
    }
    (p.0 * lw, p.1.as_ref().map(|d| d.iter().map(|x| x * lw).collect()))
}

/// `_get_inverse_dash_pattern`.
pub fn inverse_dash_pattern(p: &DashPattern) -> DashPattern {
    let d = p.1.clone().unwrap_or_default();
    if d.is_empty() {
        return (p.0, Some(d));
    }
    let mut gaps = vec![d[d.len() - 1]];
    gaps.extend_from_slice(&d[..d.len() - 1]);
    (p.0 + d[d.len() - 1], Some(gaps))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawStyle {
    Default,
    StepsPre,
    StepsMid,
    StepsPost,
}

impl DrawStyle {
    pub fn parse(s: &str) -> Option<DrawStyle> {
        Some(match s {
            "default" => DrawStyle::Default,
            "steps" | "steps-pre" => DrawStyle::StepsPre,
            "steps-mid" => DrawStyle::StepsMid,
            "steps-post" => DrawStyle::StepsPost,
            _ => return None,
        })
    }
}

/// `pts_to_prestep` / `pts_to_poststep` / `pts_to_midstep` on (x, y) pairs.
pub fn step_vertices(style: DrawStyle, xy: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let n = xy.len();
    match style {
        DrawStyle::Default => xy.to_vec(),
        DrawStyle::StepsPre | DrawStyle::StepsPost => {
            let m = if n == 0 { 0 } else { 2 * n - 1 };
            let mut out = vec![[0.0, 0.0]; m];
            for i in 0..n {
                out[2 * i] = xy[i];
            }
            for i in 0..n.saturating_sub(1) {
                out[2 * i + 1] = if style == DrawStyle::StepsPre {
                    [xy[i][0], xy[i + 1][1]]
                } else {
                    [xy[i + 1][0], xy[i][1]]
                };
            }
            out
        }
        DrawStyle::StepsMid => {
            let mut out = vec![[0.0, 0.0]; 2 * n];
            if n == 0 {
                return out;
            }
            for i in 0..n - 1 {
                let m = (xy[i][0] + xy[i + 1][0]) / 2.0;
                out[2 * i + 1][0] = m;
                out[2 * i + 2][0] = m;
            }
            out[0][0] = xy[0][0];
            out[2 * n - 1][0] = xy[n - 1][0];
            for i in 0..n {
                out[2 * i][1] = xy[i][1];
                out[2 * i + 1][1] = xy[i][1];
            }
            out
        }
    }
}

/// `markevery` settings.
#[derive(Clone, Debug, PartialEq)]
pub enum MarkEvery {
    /// (start, step) with integer step
    Every(usize, usize),
    /// explicit indices
    Indices(Vec<usize>),
    /// (start, step) fractions of the axes diagonal
    Fraction(f64, f64),
}

/// `_path.is_sorted_and_has_non_nan`.
pub fn is_sorted_and_has_non_nan(x: &[f64]) -> bool {
    let mut last = f64::NEG_INFINITY;
    let mut found = false;
    for &v in x {
        if v == v {
            found = true;
            if v < last {
                return false;
            }
            last = v;
        }
    }
    found
}

/// numpy `searchsorted` on a sorted array.
pub fn searchsorted(a: &[f64], v: f64, right: bool) -> usize {
    if right {
        a.partition_point(|&x| x <= v)
    } else {
        a.partition_point(|&x| x < v)
    }
}

#[derive(Clone, Debug)]
pub struct Line2D {
    pub base: ArtistBase,
    pub xorig: Vec<f64>,
    pub yorig: Vec<f64>,
    pub linewidth: f64,
    /// one of '-', '--', '-.', ':', 'None'
    pub linestyle: String,
    pub unscaled_dash_pattern: DashPattern,
    pub dash_pattern: DashPattern,
    pub drawstyle: DrawStyle,
    pub color: ColorSpec,
    pub gapcolor: Option<ColorSpec>,
    pub marker: MarkerStyle,
    pub markevery: Option<MarkEvery>,
    pub markersize: f64,
    pub antialiased: bool,
    pub markeredgecolor: ColorSpec,
    pub markeredgewidth: f64,
    pub markerfacecolor: ColorSpec,
    pub markerfacecoloralt: ColorSpec,
    pub dash_capstyle: LineCap,
    pub solid_capstyle: LineCap,
    pub dash_joinstyle: LineJoin,
    pub solid_joinstyle: LineJoin,
    pub interpolation_steps: usize,
    /// cached (xy, path) built by `recache`
    cache: Option<LineCache>,
}

#[derive(Clone, Debug)]
struct LineCache {
    xy: Vec<[f64; 2]>,
    path: Path,
    subslice: bool,
    x_filled: Option<Vec<f64>>,
}

impl Line2D {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Line2D {
        let r = rc();
        let lw = r.num("lines.linewidth");
        let mut l = Line2D {
            base: ArtistBase { zorder: 2.0, transform: TSpec::Identity, ..ArtistBase::default() },
            xorig: x,
            yorig: y,
            linewidth: lw,
            linestyle: "-".into(),
            unscaled_dash_pattern: (0.0, None),
            dash_pattern: (0.0, None),
            drawstyle: DrawStyle::Default,
            color: ColorSpec::Rgba(r.color("lines.color")),
            gapcolor: None,
            marker: MarkerStyle::named("none"),
            markevery: None,
            markersize: r.num("lines.markersize"),
            antialiased: r.boolean("lines.antialiased"),
            markeredgecolor: rc_marker_color(&r, "lines.markeredgecolor"),
            markeredgewidth: r.num("lines.markeredgewidth"),
            markerfacecolor: rc_marker_color(&r, "lines.markerfacecolor"),
            markerfacecoloralt: ColorSpec::None,
            dash_capstyle: super::artist::parse_capstyle(&r.string("lines.dash_capstyle")),
            solid_capstyle: super::artist::parse_capstyle(&r.string("lines.solid_capstyle")),
            dash_joinstyle: super::artist::parse_joinstyle(&r.string("lines.dash_joinstyle")),
            solid_joinstyle: super::artist::parse_joinstyle(&r.string("lines.solid_joinstyle")),
            interpolation_steps: 1,
            cache: None,
        };
        let ls = r.string("lines.linestyle");
        l.set_linestyle(&ls);
        let m = r.string("lines.marker");
        l.set_marker(MarkerStyle::named(&m));
        l
    }

    pub fn set_data(&mut self, x: Vec<f64>, y: Vec<f64>) {
        self.xorig = x;
        self.yorig = y;
        self.cache = None;
    }

    pub fn set_xdata(&mut self, x: Vec<f64>) {
        self.xorig = x;
        self.cache = None;
    }

    pub fn set_ydata(&mut self, y: Vec<f64>) {
        self.yorig = y;
        self.cache = None;
    }

    pub fn invalidate(&mut self) {
        self.cache = None;
    }

    pub fn set_linewidth(&mut self, w: f64) {
        self.linewidth = w;
        self.dash_pattern = scale_dashes(&self.unscaled_dash_pattern, w);
    }

    /// `set_linestyle` for named styles.
    pub fn set_linestyle(&mut self, ls: &str) {
        let ls = match ls {
            " " | "" | "none" => "None",
            "solid" => "-",
            "dashed" => "--",
            "dashdot" => "-.",
            "dotted" => ":",
            s => s,
        };
        self.linestyle = ls.to_string();
        self.unscaled_dash_pattern = get_dash_pattern(ls);
        self.dash_pattern = scale_dashes(&self.unscaled_dash_pattern, self.linewidth);
    }

    /// `set_linestyle((offset, dashes))`.
    pub fn set_dashes(&mut self, offset: f64, dashes: Option<Vec<f64>>) {
        self.linestyle = "--".into();
        self.unscaled_dash_pattern = normalize_dashes(offset, dashes);
        self.dash_pattern = scale_dashes(&self.unscaled_dash_pattern, self.linewidth);
    }

    /// `set_marker`: the caller builds the style with this line's fillstyle.
    pub fn set_marker(&mut self, m: MarkerStyle) {
        self.marker = m;
    }

    pub fn set_drawstyle(&mut self, d: DrawStyle) {
        if self.drawstyle != d {
            self.cache = None;
        }
        self.drawstyle = d;
    }

    pub fn is_dashed(&self) -> bool {
        matches!(self.linestyle.as_str(), "--" | "-." | ":")
    }

    /// `update_from` (as used by `Tick._copy_tick_props`).
    pub fn update_from(&mut self, other: &Line2D) {
        self.base.transform = other.base.transform.clone();
        self.base.transform_set = other.base.transform_set;
        self.base.visible = other.base.visible;
        self.base.alpha = other.base.alpha;
        self.base.clip_box = other.base.clip_box.clone();
        self.base.clip_on = other.base.clip_on;
        self.base.clip_path = other.base.clip_path.clone();
        self.base.label = other.base.label.clone();
        self.base.sketch = other.base.sketch;
        self.linestyle = other.linestyle.clone();
        self.linewidth = other.linewidth;
        self.color = other.color;
        self.gapcolor = other.gapcolor;
        self.markersize = other.markersize;
        self.markerfacecolor = other.markerfacecolor;
        self.markerfacecoloralt = other.markerfacecoloralt;
        self.markeredgecolor = other.markeredgecolor;
        self.markeredgewidth = other.markeredgewidth;
        self.unscaled_dash_pattern = other.unscaled_dash_pattern.clone();
        self.dash_pattern = other.dash_pattern.clone();
        self.dash_capstyle = other.dash_capstyle;
        self.dash_joinstyle = other.dash_joinstyle;
        self.solid_capstyle = other.solid_capstyle;
        self.solid_joinstyle = other.solid_joinstyle;
        self.marker = other.marker.clone();
        self.drawstyle = other.drawstyle;
        self.cache = None;
    }

    /// Broadcast x/y like `np.broadcast_arrays`.
    fn build_xy(&self) -> Vec<[f64; 2]> {
        let (nx, ny) = (self.xorig.len(), self.yorig.len());
        if nx == ny {
            self.xorig.iter().zip(self.yorig.iter()).map(|(&x, &y)| [x, y]).collect()
        } else if nx == 1 {
            self.yorig.iter().map(|&y| [self.xorig[0], y]).collect()
        } else if ny == 1 {
            self.xorig.iter().map(|&x| [x, self.yorig[0]]).collect()
        } else {
            let n = nx.min(ny);
            (0..n).map(|i| [self.xorig[i], self.yorig[i]]).collect()
        }
    }

    /// `recache`.  `subslice_ok` tells whether the artist sits in a
    /// rectilinear linear-x Axes with `transform == transData`.
    pub fn recache(&mut self, subslice_ok: bool) {
        let xy = self.build_xy();
        let x: Vec<f64> = xy.iter().map(|p| p[0]).collect();
        let mut subslice = false;
        let mut x_filled = None;
        if subslice_ok
            && x.len() > 1000
            && is_sorted_and_has_non_nan(&x)
            && self.markevery.is_none()
            && self.base.clip_on
        {
            subslice = true;
            if x.iter().any(|v| v.is_nan()) {
                // np.interp over the NaN positions
                let idx: Vec<f64> = (0..x.len()).filter(|&i| !x[i].is_nan()).map(|i| i as f64).collect();
                let vals: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
                let mut xf = x.clone();
                for (i, v) in xf.iter_mut().enumerate() {
                    if v.is_nan() {
                        *v = np_interp(i as f64, &idx, &vals);
                    }
                }
                x_filled = Some(xf);
            } else {
                x_filled = Some(x.clone());
            }
        }
        let vertices = step_vertices(self.drawstyle, &xy);
        let path = new_path(vertices, None, simplify_params());
        self.cache = Some(LineCache { xy, path, subslice, x_filled });
    }

    fn ensure_cache(&mut self, subslice_ok: bool) {
        if self.cache.is_none() {
            self.recache(subslice_ok);
        }
    }

    pub fn get_xydata(&mut self, subslice_ok: bool) -> &[[f64; 2]] {
        self.ensure_cache(subslice_ok);
        &self.cache.as_ref().unwrap().xy
    }

    pub fn get_path(&mut self, subslice_ok: bool) -> &Path {
        self.ensure_cache(subslice_ok);
        &self.cache.as_ref().unwrap().path
    }

    fn marker_edgecolor(&self) -> ColorSpec {
        if self.markeredgecolor.is_auto() {
            self.color
        } else {
            self.markeredgecolor
        }
    }

    fn marker_facecolor(&self, alt: bool) -> ColorSpec {
        if self.marker.fillstyle == super::markers::FillStyle::None {
            return ColorSpec::None;
        }
        let fc = if alt { self.markerfacecoloralt } else { self.markerfacecolor };
        if fc.is_auto() {
            self.color
        } else {
            fc
        }
    }

    /// `get_window_extent`.
    pub fn get_window_extent(&mut self, ctx: &TransCtx, subslice_ok: bool) -> Bbox {
        let t = ctx.resolve(&self.base.transform);
        let xy = self.get_xydata(subslice_ok).to_vec();
        let pts = t.transform_points(&xy);
        let mut bb = bbox_from_data_xy(&pts);
        if marker_is_truthy(&self.marker) {
            let ms = (self.markersize / 72.0 * ctx.dpi) * 0.5;
            bb = bb.padded(ms, ms);
        }
        bb
    }

    /// `Line2D.draw`.  `xbound` is `axes.get_xbound()` when the line lives
    /// in an Axes (for the subslice optimisation).
    pub fn draw(&mut self, renderer: &mut Renderer, ctx: &TransCtx, clip: &ClipCtx, xbound: Option<(f64, f64)>, subslice_ok: bool) {
        renderer.artist_start();
        if !self.base.visible {
            return;
        }
        self.ensure_cache(subslice_ok);
        let transform = ctx.resolve(&self.base.transform);
        let cache = self.cache.as_ref().unwrap();
        // subslice
        let mut sub_path_store: Option<Path> = None;
        let mut sub_xy: Option<Vec<[f64; 2]>> = None;
        if cache.subslice {
            if let Some((x0, x1)) = xbound {
                let xf = cache.x_filled.as_ref().unwrap();
                let i0 = searchsorted(xf, x0, false);
                let i1 = searchsorted(xf, x1, true);
                let start = i0.saturating_sub(1);
                let end = (i1 + 1).min(cache.xy.len());
                let xy = &cache.xy[start.min(end)..end];
                let verts = step_vertices(self.drawstyle, xy);
                sub_path_store = Some(new_path(verts, None, simplify_params()));
                sub_xy = Some(xy.to_vec());
            }
        }
        let path: &Path = sub_path_store.as_ref().unwrap_or(&cache.path);
        let affine = transform.get_affine();
        renderer.open_group("line2d", self.base.gid.as_deref());

        if self.linestyle != "None" {
            let tpath = if transform.is_affine() {
                std::borrow::Cow::Borrowed(path)
            } else {
                std::borrow::Cow::Owned(super::path::transform_path_non_affine(path, &transform))
            };
            if !tpath.vertices.is_empty() {
                let mut gc = GcState::default();
                gc.set_clip_from(&self.base, clip);
                gc.antialiased = self.antialiased;
                gc.linewidth = self.linewidth;
                let (cap, join) = if self.is_dashed() {
                    (self.dash_capstyle, self.dash_joinstyle)
                } else {
                    (self.solid_capstyle, self.solid_joinstyle)
                };
                gc.joinstyle = join;
                gc.capstyle = cap;
                gc.snap = self.base.snap;
                gc.sketch = self.base.sketch;
                if self.linewidth > 0.0 && self.is_dashed() {
                    if let Some(gcol) = self.gapcolor {
                        gc.set_foreground(gcol.to_rgba(self.base.alpha));
                        let (o, g) = inverse_dash_pattern(&self.dash_pattern);
                        gc.set_dashes(o, g);
                        renderer.draw_path(&gc, &tpath, &affine, None);
                    }
                }
                gc.set_foreground(self.color.to_rgba(self.base.alpha));
                if self.linewidth > 0.0 {
                    gc.set_dashes(self.dash_pattern.0, self.dash_pattern.1.clone());
                } else {
                    gc.set_dashes(0.0, None);
                }
                renderer.draw_path(&gc, &tpath, &affine, None);
            }
        }

        if marker_is_truthy(&self.marker) && self.markersize > 0.0 {
            let mut gc = GcState::default();
            gc.set_clip_from(&self.base, clip);
            gc.linewidth = self.markeredgewidth;
            gc.antialiased = self.antialiased;
            let mut ec = self.marker_edgecolor().to_rgba(self.base.alpha);
            let fc = self.marker_facecolor(false).to_rgba(self.base.alpha);
            let fcalt = self.marker_facecolor(true).to_rgba(self.base.alpha);
            if self.markeredgecolor.is_auto() && !self.marker_facecolor(false).is_none() {
                ec.a = fc.a;
            }
            gc.set_foreground(ec);
            if let Some((s, l, r)) = self.base.sketch {
                gc.sketch = Some((s / 2.0, l / 2.0, 2.0 * r));
            }
            // points (always the non-stepped data)
            let xy: &[[f64; 2]] = match &sub_xy {
                Some(v) => v,
                None => &self.cache.as_ref().unwrap().xy,
            };
            let pts_nonaffine = if transform.is_affine() { xy.to_vec() } else { transform.transform_points_non_affine(xy) };
            let src_path = &self.cache.as_ref().unwrap().path;
            let mut tpoints = Path {
                vertices: pts_nonaffine,
                codes: None,
                should_simplify: src_path.should_simplify,
                simplify_threshold: src_path.simplify_threshold,
            };
            if !tpoints.vertices.is_empty() {
                if let Some(me) = &self.markevery {
                    tpoints = mark_every_path(me, &tpoints, &affine, ctx);
                }
                let snap = match self.marker.snap_threshold {
                    SnapThreshold::None => None,
                    SnapThreshold::Inf => Some(false),
                    SnapThreshold::Value(v) => Some(renderer.points_to_pixels(self.markersize) >= v),
                };
                gc.snap = snap;
                gc.joinstyle = self.marker.joinstyle();
                gc.capstyle = self.marker.capstyle();
                let w = renderer.points_to_pixels(self.markersize);
                let mut mtrans = self.marker.get_transform();
                if self.marker.is_pixel {
                    gc.linewidth = 0.0;
                } else {
                    mtrans = mtrans.scale(w, w);
                }
                renderer.draw_markers(&gc, &self.marker.path, &mtrans.0, &tpoints, &affine, Some(fc));
                if let Some(alt) = &self.marker.alt_path {
                    let at = self.marker.get_alt_transform().unwrap().scale(w, w);
                    renderer.draw_markers(&gc, alt, &at.0, &tpoints, &affine, Some(fcalt));
                }
            }
        }
        renderer.close_group("line2d");
    }
}

fn rc_marker_color(r: &super::rc::Rc, key: &str) -> ColorSpec {
    match r.get(key) {
        Some(super::rc::RcValue::Color(c)) => ColorSpec::Rgba(*c),
        Some(super::rc::RcValue::Str(s)) if s == "none" => ColorSpec::None,
        _ => ColorSpec::Auto,
    }
}

/// `bool(MarkerStyle)`: whether the marker path has vertices.
pub fn marker_is_truthy(m: &MarkerStyle) -> bool {
    !m.path.vertices.is_empty()
}

pub fn simplify_params() -> SimplifyParams {
    let r = rc();
    SimplifyParams { simplify: r.boolean("path.simplify"), threshold: r.num("path.simplify_threshold") }
}

/// `np.interp(x, xp, fp)` for increasing `xp`.
pub fn np_interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if n == 0 {
        return f64::NAN;
    }
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    let j = xp.partition_point(|&v| v <= x) - 1;
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

/// `Bbox.update_from_data_xy(xy, ignore=True)` on a null bbox.
pub fn bbox_from_data_xy(xy: &[[f64; 2]]) -> Bbox {
    let mut x0 = f64::INFINITY;
    let mut y0 = f64::INFINITY;
    let mut x1 = f64::NEG_INFINITY;
    let mut y1 = f64::NEG_INFINITY;
    for p in xy {
        if p[0].is_finite() && p[1].is_finite() {
            x0 = x0.min(p[0]);
            x1 = x1.max(p[0]);
            y0 = y0.min(p[1]);
            y1 = y1.max(p[1]);
        }
    }
    Bbox::new(x0, y0, x1, y1)
}

fn mark_every_path(me: &MarkEvery, tpath: &Path, affine: &super::transforms::Mat, ctx: &TransCtx) -> Path {
    let verts = &tpath.vertices;
    let mk = |v: Vec<[f64; 2]>| Path {
        vertices: v,
        codes: None,
        should_simplify: tpath.should_simplify,
        simplify_threshold: tpath.simplify_threshold,
    };
    match me {
        MarkEvery::Every(start, step) => mk(verts.iter().skip(*start).step_by((*step).max(1)).copied().collect()),
        MarkEvery::Indices(idx) => mk(idx.iter().filter(|&&i| i < verts.len()).map(|&i| verts[i]).collect()),
        MarkEvery::Fraction(start, step) => {
            let fverts: Vec<[f64; 2]> = verts.iter().filter(|p| p[0].is_finite() && p[1].is_finite()).copied().collect();
            let disp = Transform::Affine(*affine).transform_points(&fverts);
            let mut delta = vec![0.0; disp.len()];
            let mut acc = 0.0;
            for i in 0..disp.len() {
                let d = if i == 0 { 0.0 } else { (disp[i][0] - disp[i - 1][0]).hypot(disp[i][1] - disp[i - 1][1]) };
                acc += d;
                delta[i] = acc;
            }
            let ta = ctx.resolve(&TSpec::Axes);
            let (x0, y0) = ta.transform_point(0.0, 0.0);
            let (x1, y1) = ta.transform_point(1.0, 1.0);
            let scale = (x1 - x0).hypot(y1 - y0);
            let marker_start = start * scale;
            let marker_step = step * scale;
            if marker_step <= 0.0 || delta.is_empty() {
                return mk(Vec::new());
            }
            let last = delta[delta.len() - 1];
            let mut md: Vec<f64> = Vec::new();
            for &d in &delta {
                let r = d - marker_start;
                let rem = r - marker_step * (r / marker_step).floor();
                let v = d - rem;
                md.push(v);
                md.push(v + marker_step);
            }
            md.sort_by(|a, b| a.partial_cmp(b).unwrap());
            md.dedup();
            md.retain(|&v| v >= marker_start && v < last);
            let mut inds: Vec<usize> = Vec::new();
            for &m in &md {
                let right = searchsorted(&delta, m, false);
                let left = right.saturating_sub(1);
                let right = right.min(delta.len() - 1);
                let i = if (delta[right] - m).abs() < (m - delta[left]).abs() { right } else { left };
                inds.push(searchsorted(&delta, delta[i], false));
            }
            inds.sort();
            inds.dedup();
            mk(inds.iter().map(|&i| fverts[i]).collect())
        }
    }
}
