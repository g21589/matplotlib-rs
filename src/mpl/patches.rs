//! Port of `matplotlib.patches` (Patch and the common subclasses).

use std::f64::consts::SQRT_2;

use super::artist::{ArtistBase, ClipCtx, GcState, TransCtx};
use super::colors::ColorSpec;
use super::lines::{get_dash_pattern, inverse_dash_pattern, normalize_dashes, scale_dashes, DashPattern};
use super::path::{self as mpath, closed_path, new_path, Path, SimplifyParams, CLOSEPOLY, CURVE3, CURVE4, LINETO, MOVETO};
use super::rc::rc;
use super::renderer::Renderer;
use super::transforms::{Affine2D, Bbox, PyRadians, Transform};
use crate::agg::pixfmt::Rgba;
use crate::agg::stroke::{LineCap, LineJoin};

#[derive(Clone, Debug, PartialEq)]
pub enum RotationPoint {
    Xy,
    Center,
    Point(f64, f64),
}

#[derive(Clone, Debug, PartialEq)]
pub enum BoxStyle {
    Square { pad: f64 },
    Circle { pad: f64 },
    Ellipse { pad: f64 },
    RArrow { pad: f64, head_width: f64, head_angle: f64 },
    LArrow { pad: f64, head_width: f64, head_angle: f64 },
    DArrow { pad: f64, head_width: f64, head_angle: f64 },
    Round { pad: f64, rounding_size: Option<f64> },
    Round4 { pad: f64, rounding_size: Option<f64> },
    Sawtooth { pad: f64, tooth_size: Option<f64> },
    Roundtooth { pad: f64, tooth_size: Option<f64> },
}

impl BoxStyle {
    /// Parse `"round,pad=0.2"`-style specifications.
    pub fn parse(spec: &str) -> Option<BoxStyle> {
        let mut parts = spec.split(',');
        let name = parts.next()?.trim().to_lowercase();
        let mut kw: Vec<(String, f64)> = Vec::new();
        for p in parts {
            let mut it = p.splitn(2, '=');
            let k = it.next()?.trim().to_string();
            let v: f64 = it.next()?.trim().parse().ok()?;
            kw.push((k, v));
        }
        let get = |k: &str, d: f64| kw.iter().find(|(n, _)| n == k).map(|(_, v)| *v).unwrap_or(d);
        let opt = |k: &str| kw.iter().find(|(n, _)| n == k).map(|(_, v)| *v);
        Some(match name.as_str() {
            "square" => BoxStyle::Square { pad: get("pad", 0.3) },
            "circle" => BoxStyle::Circle { pad: get("pad", 0.3) },
            "ellipse" => BoxStyle::Ellipse { pad: get("pad", 0.3) },
            "rarrow" => BoxStyle::RArrow { pad: get("pad", 0.3), head_width: get("head_width", 1.5), head_angle: get("head_angle", 90.0) },
            "larrow" => BoxStyle::LArrow { pad: get("pad", 0.3), head_width: get("head_width", 1.5), head_angle: get("head_angle", 90.0) },
            "darrow" => BoxStyle::DArrow { pad: get("pad", 0.3), head_width: get("head_width", 1.5), head_angle: get("head_angle", 90.0) },
            "round" => BoxStyle::Round { pad: get("pad", 0.3), rounding_size: opt("rounding_size") },
            "round4" => BoxStyle::Round4 { pad: get("pad", 0.3), rounding_size: opt("rounding_size") },
            "sawtooth" => BoxStyle::Sawtooth { pad: get("pad", 0.3), tooth_size: opt("tooth_size") },
            "roundtooth" => BoxStyle::Roundtooth { pad: get("pad", 0.3), tooth_size: opt("tooth_size") },
            _ => return None,
        })
    }

    pub fn pad(&self) -> f64 {
        match self {
            BoxStyle::Square { pad }
            | BoxStyle::Circle { pad }
            | BoxStyle::Ellipse { pad }
            | BoxStyle::RArrow { pad, .. }
            | BoxStyle::LArrow { pad, .. }
            | BoxStyle::DArrow { pad, .. }
            | BoxStyle::Round { pad, .. }
            | BoxStyle::Round4 { pad, .. }
            | BoxStyle::Sawtooth { pad, .. }
            | BoxStyle::Roundtooth { pad, .. } => *pad,
        }
    }

    pub fn set_pad(&mut self, p: f64) {
        match self {
            BoxStyle::Square { pad }
            | BoxStyle::Circle { pad }
            | BoxStyle::Ellipse { pad }
            | BoxStyle::RArrow { pad, .. }
            | BoxStyle::LArrow { pad, .. }
            | BoxStyle::DArrow { pad, .. }
            | BoxStyle::Round { pad, .. }
            | BoxStyle::Round4 { pad, .. }
            | BoxStyle::Sawtooth { pad, .. }
            | BoxStyle::Roundtooth { pad, .. } => *pad = p,
        }
    }

    /// `BoxStyle.__call__(x0, y0, width, height, mutation_size)`.
    pub fn path(&self, x0: f64, y0: f64, width: f64, height: f64, ms: f64) -> Path {
        let sp = SimplifyParams::default();
        let cl = |v: Vec<[f64; 2]>| {
            let mut v = v;
            v.push(v[0]);
            closed_path(v, sp)
        };
        match self {
            BoxStyle::Square { pad } => {
                let pad = ms * pad;
                let (width, height) = (width + 2.0 * pad, height + 2.0 * pad);
                let (x0, y0) = (x0 - pad, y0 - pad);
                let (x1, y1) = (x0 + width, y0 + height);
                cl(vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            }
            BoxStyle::Circle { pad } => {
                let pad = ms * pad;
                let (width, height) = (width + 2.0 * pad, height + 2.0 * pad);
                let (x0, y0) = (x0 - pad, y0 - pad);
                mpath::circle((x0 + width / 2.0, y0 + height / 2.0), super::text::pymax(width, height) / 2.0)
            }
            BoxStyle::Ellipse { pad } => {
                let pad = ms * pad;
                let (width, height) = (width + 2.0 * pad, height + 2.0 * pad);
                let (x0, y0) = (x0 - pad, y0 - pad);
                let a = width / SQRT_2;
                let b = height / SQRT_2;
                let t = Affine2D::identity().scale(a, b).translate(x0 + width / 2.0, y0 + height / 2.0);
                mpath::transform_path(&mpath::unit_circle(), &Transform::Affine(t.0))
            }
            BoxStyle::RArrow { pad, head_width, head_angle } | BoxStyle::LArrow { pad, head_width, head_angle } => {
                let p = rarrow(x0, y0, width, height, ms, *pad, *head_width, *head_angle);
                if let BoxStyle::LArrow { .. } = self {
                    let mut p = p;
                    for v in p.vertices.iter_mut() {
                        v[0] = 2.0 * x0 + width - v[0];
                    }
                    p
                } else {
                    p
                }
            }
            BoxStyle::DArrow { pad, head_width, head_angle } => {
                let pad = ms * pad;
                let (dx, dy) = (width + 2.0 * pad, height + 2.0 * pad);
                let (x0, y0) = (x0 - pad, y0 - pad);
                let (x1, y1) = (x0 + dx, y0 + dy);
                let head_dy = head_width * dy;
                let mid_y = (y0 + y1) / 2.0;
                let shaft_y0 = mid_y - head_dy / 2.0;
                let shaft_y1 = mid_y + head_dy / 2.0;
                let cot = 1.0 / (head_angle / 2.0).to_radians_py().tan();
                if cot > 0.0 {
                    let tip_x0 = x0 - cot * super::text::pymin(dy, head_dy) / 2.0;
                    let shaft_x0 = tip_x0 + cot * head_dy / 2.0;
                    let tip_x1 = x1 + cot * super::text::pymin(dy, head_dy) / 2.0;
                    let shaft_x1 = tip_x1 - cot * head_dy / 2.0;
                    cl(vec![
                        [shaft_x0, y1],
                        [shaft_x0, shaft_y1],
                        [tip_x0, mid_y],
                        [shaft_x0, shaft_y0],
                        [shaft_x0, y0],
                        [shaft_x1, y0],
                        [shaft_x1, shaft_y0],
                        [tip_x1, mid_y],
                        [shaft_x1, shaft_y1],
                        [shaft_x1, y1],
                    ])
                } else {
                    let dx = super::text::pymin(-cot * super::text::pymax(head_dy - dy, 0.0) / 2.0, dx / 2.0);
                    let mid_y0 = super::text::pymin(shaft_y0, y0) - dx / cot;
                    let mid_y1 = super::text::pymax(shaft_y1, y1) + dx / cot;
                    cl(vec![
                        [x0, shaft_y0],
                        [x0 + dx, mid_y0],
                        [x1 - dx, mid_y0],
                        [x1, shaft_y0],
                        [x1, shaft_y1],
                        [x1 - dx, mid_y1],
                        [x0 + dx, mid_y1],
                        [x0, shaft_y1],
                    ])
                }
            }
            BoxStyle::Round { pad, rounding_size } => {
                let pad = ms * pad;
                let dr = match rounding_size {
                    Some(r) if *r != 0.0 => ms * r,
                    _ => pad,
                };
                let (width, height) = (width + 2.0 * pad, height + 2.0 * pad);
                let (x0, y0) = (x0 - pad, y0 - pad);
                let (x1, y1) = (x0 + width, y0 + height);
                let cp = vec![
                    [x0 + dr, y0],
                    [x1 - dr, y0],
                    [x1, y0],
                    [x1, y0 + dr],
                    [x1, y1 - dr],
                    [x1, y1],
                    [x1 - dr, y1],
                    [x0 + dr, y1],
                    [x0, y1],
                    [x0, y1 - dr],
                    [x0, y0 + dr],
                    [x0, y0],
                    [x0 + dr, y0],
                    [x0 + dr, y0],
                ];
                let com = vec![
                    MOVETO, LINETO, CURVE3, CURVE3, LINETO, CURVE3, CURVE3, LINETO, CURVE3, CURVE3, LINETO, CURVE3, CURVE3,
                    CLOSEPOLY,
                ];
                new_path(cp, Some(com), sp)
            }
            BoxStyle::Round4 { pad, rounding_size } => {
                let pad = ms * pad;
                let dr = match rounding_size {
                    Some(r) if *r != 0.0 => ms * r,
                    _ => pad / 2.0,
                };
                let width = width + 2.0 * pad - 2.0 * dr;
                let height = height + 2.0 * pad - 2.0 * dr;
                let (x0, y0) = (x0 - pad + dr, y0 - pad + dr);
                let (x1, y1) = (x0 + width, y0 + height);
                let cp = vec![
                    [x0, y0],
                    [x0 + dr, y0 - dr],
                    [x1 - dr, y0 - dr],
                    [x1, y0],
                    [x1 + dr, y0 + dr],
                    [x1 + dr, y1 - dr],
                    [x1, y1],
                    [x1 - dr, y1 + dr],
                    [x0 + dr, y1 + dr],
                    [x0, y1],
                    [x0 - dr, y1 - dr],
                    [x0 - dr, y0 + dr],
                    [x0, y0],
                    [x0, y0],
                ];
                let mut com = vec![MOVETO];
                com.extend(std::iter::repeat_n(CURVE4, 12));
                com.push(CLOSEPOLY);
                new_path(cp, Some(com), sp)
            }
            BoxStyle::Sawtooth { pad, tooth_size } => {
                let v = sawtooth_vertices(x0, y0, width, height, ms, *pad, *tooth_size);
                closed_path(v, sp)
            }
            BoxStyle::Roundtooth { pad, tooth_size } => {
                let mut v = sawtooth_vertices(x0, y0, width, height, ms, *pad, *tooth_size);
                v.push(v[0]);
                let mut codes = vec![MOVETO];
                for _ in 0..(v.len() - 1) / 2 {
                    codes.push(CURVE3);
                    codes.push(CURVE3);
                }
                codes.push(CLOSEPOLY);
                new_path(v, Some(codes), sp)
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn rarrow(x0: f64, y0: f64, width: f64, height: f64, ms: f64, pad: f64, head_width: f64, head_angle: f64) -> Path {
    let pad = ms * pad;
    let (dx, dy) = (width + 2.0 * pad, height + 2.0 * pad);
    let (x0, y0) = (x0 - pad, y0 - pad);
    let (x1, y1) = (x0 + dx, y0 + dy);
    let head_dy = head_width * dy;
    let mid_y = (y0 + y1) / 2.0;
    let shaft_y0 = mid_y - head_dy / 2.0;
    let shaft_y1 = mid_y + head_dy / 2.0;
    let cot = 1.0 / (head_angle / 2.0).to_radians_py().tan();
    let mut v = if cot > 0.0 {
        let tip_x = x1 + cot * super::text::pymin(dy, head_dy) / 2.0;
        let shaft_x = tip_x - cot * head_dy / 2.0;
        vec![[x0, y0], [shaft_x, y0], [shaft_x, shaft_y0], [tip_x, mid_y], [shaft_x, shaft_y1], [shaft_x, y1], [x0, y1]]
    } else {
        let dx = super::text::pymin(-cot * super::text::pymax(head_dy - dy, 0.0) / 2.0, dx);
        let mid_y0 = super::text::pymin(shaft_y0, y0) - dx / cot;
        let mid_y1 = super::text::pymax(shaft_y1, y1) + dx / cot;
        vec![[x0, y0], [x1 - dx, mid_y0], [x1, shaft_y0], [x1, shaft_y1], [x1 - dx, mid_y1], [x0, y1]]
    };
    v.push(v[0]);
    closed_path(v, SimplifyParams::default())
}

fn sawtooth_vertices(x0: f64, y0: f64, width: f64, height: f64, ms: f64, pad: f64, tooth: Option<f64>) -> Vec<[f64; 2]> {
    let padv = ms * pad;
    let tooth_size = match tooth {
        None => pad * 0.5 * ms,
        Some(t) => t * ms,
    };
    let hsz = tooth_size / 2.0;
    let width = width + 2.0 * padv - tooth_size;
    let height = height + 2.0 * padv - tooth_size;
    let dsx_n = ((width - tooth_size) / (tooth_size * 2.0)).round_ties_even() as i64 * 2;
    let dsy_n = ((height - tooth_size) / (tooth_size * 2.0)).round_ties_even() as i64 * 2;
    let (x0, y0) = (x0 - padv + hsz, y0 - padv + hsz);
    let (x1, y1) = (x0 + width, y0 + height);
    let rep = |pat: [f64; 4], n: i64, take: i64| -> Vec<f64> {
        let mut v = Vec::new();
        for _ in 0..n.max(0) {
            v.extend_from_slice(&pat);
        }
        v.truncate(take.max(0) as usize);
        v
    };
    let mut xs = vec![x0];
    xs.extend(mpath::linspace(x0 + hsz, x1 - hsz, (2 * dsx_n + 1).max(0) as usize));
    xs.extend(rep([x1, x1 + hsz, x1, x1 - hsz], dsy_n, 2 * dsy_n + 2));
    xs.push(x1);
    xs.extend(mpath::linspace(x1 - hsz, x0 + hsz, (2 * dsx_n + 1).max(0) as usize));
    xs.extend(rep([x0, x0 - hsz, x0, x0 + hsz], dsy_n, 2 * dsy_n + 2));
    let mut ys = rep([y0, y0 - hsz, y0, y0 + hsz], dsx_n, 2 * dsx_n + 2);
    ys.push(y0);
    ys.extend(mpath::linspace(y0 + hsz, y1 - hsz, (2 * dsy_n + 1).max(0) as usize));
    ys.extend(rep([y1, y1 + hsz, y1, y1 - hsz], dsx_n, 2 * dsx_n + 2));
    ys.push(y1);
    ys.extend(mpath::linspace(y1 - hsz, y0 + hsz, (2 * dsy_n + 1).max(0) as usize));
    let n = xs.len().min(ys.len());
    let mut v: Vec<[f64; 2]> = (0..n).map(|i| [xs[i], ys[i]]).collect();
    v.push([xs[0], ys[0]]);
    v
}

#[derive(Clone, Debug)]
pub enum PatchKind {
    Rectangle { x0: f64, y0: f64, width: f64, height: f64, angle: f64, rotation_point: RotationPoint },
    /// Polygon / PathPatch / Wedge / StepPatch etc.: a fixed path in data coordinates.
    Path { path: Path },
    Ellipse { center: (f64, f64), width: f64, height: f64, angle: f64 },
    RegularPolygon { xy: (f64, f64), numvertices: usize, radius: f64, orientation: f64 },
    FancyBbox { x: f64, y: f64, width: f64, height: f64, boxstyle: BoxStyle },
    /// Shadow of another patch, offset by (ox, oy) points.
    Shadow { patch: Box<Patch>, ox: f64, oy: f64 },
    /// A path plus a precomputed patch transform (from the Python side).
    Generic { path: Path, patch_transform: Option<super::transforms::Mat> },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HatchColor {
    Edge,
    Color(Rgba),
}

#[derive(Clone, Debug)]
pub struct Patch {
    pub base: ArtistBase,
    pub kind: PatchKind,
    pub orig_edgecolor: Option<ColorSpec>,
    pub orig_facecolor: Option<ColorSpec>,
    pub orig_hatchcolor: Option<Option<ColorSpec>>,
    pub edgecolor: Rgba,
    pub facecolor: Rgba,
    pub hatch_color: HatchColor,
    pub fill: bool,
    pub linewidth: f64,
    /// 'solid', '-', 'dashed', ..., 'None' or 'custom' for explicit dashes
    pub linestyle: String,
    pub unscaled_dash_pattern: DashPattern,
    pub dash_pattern: DashPattern,
    pub antialiased: bool,
    pub hatch: Option<String>,
    pub hatch_linewidth: f64,
    pub capstyle: LineCap,
    pub joinstyle: LineJoin,
    pub gapcolor: Option<Rgba>,
    pub mutation_scale: f64,
    pub mutation_aspect: f64,
    pub edge_default: bool,
    /// sticky edges (x, y) used by autoscaling
    pub sticky_x: Vec<f64>,
    pub sticky_y: Vec<f64>,
}

impl Patch {
    pub fn new(kind: PatchKind) -> Patch {
        let r = rc();
        let edge_default = matches!(kind, PatchKind::FancyBbox { .. });
        let mut p = Patch {
            base: ArtistBase { zorder: 1.0, ..ArtistBase::default() },
            kind,
            orig_edgecolor: None,
            orig_facecolor: None,
            orig_hatchcolor: None,
            edgecolor: Rgba::new(0.0, 0.0, 0.0, 0.0),
            facecolor: Rgba::new(0.0, 0.0, 0.0, 0.0),
            hatch_color: HatchColor::Edge,
            fill: true,
            linewidth: 0.0,
            linestyle: "solid".into(),
            unscaled_dash_pattern: (0.0, None),
            dash_pattern: (0.0, None),
            antialiased: r.boolean("patch.antialiased"),
            hatch: None,
            hatch_linewidth: r.num("hatch.linewidth"),
            capstyle: LineCap::Butt,
            joinstyle: LineJoin::MiterRevert,
            gapcolor: None,
            mutation_scale: 1.0,
            mutation_aspect: 1.0,
            edge_default,
            sticky_x: Vec::new(),
            sticky_y: Vec::new(),
        };
        p.set_edgecolor(None);
        p.set_hatchcolor(None);
        p.set_facecolor(None);
        p.set_linestyle("solid");
        p.set_linewidth(None);
        p
    }

    pub fn rectangle(x0: f64, y0: f64, width: f64, height: f64) -> Patch {
        Patch::new(PatchKind::Rectangle { x0, y0, width, height, angle: 0.0, rotation_point: RotationPoint::Xy })
    }

    /// `Polygon(xy, closed=closed)`.
    pub fn polygon(xy: Vec<[f64; 2]>, closed: bool) -> Patch {
        Patch::new(PatchKind::Path { path: polygon_path(xy, closed) })
    }

    pub fn path_patch(path: Path) -> Patch {
        let mut p = Patch::new(PatchKind::Path { path });
        p.edge_default = true;
        p.set_edgecolor(p.orig_edgecolor);
        p
    }

    pub fn set_edgecolor(&mut self, c: Option<ColorSpec>) {
        self.orig_edgecolor = c;
        let r = rc();
        let c = match c {
            Some(c) => c,
            None => {
                if r.boolean("patch.force_edgecolor") || !self.fill || self.edge_default {
                    ColorSpec::Rgba(r.color("patch.edgecolor"))
                } else {
                    ColorSpec::None
                }
            }
        };
        self.edgecolor = c.to_rgba(self.base.alpha);
    }

    pub fn set_facecolor(&mut self, c: Option<ColorSpec>) {
        self.orig_facecolor = c;
        let c = c.unwrap_or_else(|| ColorSpec::Rgba(rc().color("patch.facecolor")));
        let alpha = if self.fill { self.base.alpha } else { Some(0.0) };
        self.facecolor = c.to_rgba(alpha);
    }

    /// `set_hatchcolor`; `Some(None)` means `'edge'`.
    pub fn set_hatchcolor(&mut self, c: Option<Option<ColorSpec>>) {
        self.orig_hatchcolor = c;
        let c = match c {
            Some(c) => c,
            None => match rc().get("hatch.color") {
                Some(super::rc::RcValue::Color(c)) => Some(ColorSpec::Rgba(*c)),
                _ => None,
            },
        };
        self.hatch_color = match c {
            None => HatchColor::Edge,
            Some(c) => HatchColor::Color(c.to_rgba(self.base.alpha)),
        };
    }

    pub fn set_color(&mut self, c: ColorSpec) {
        self.set_edgecolor(Some(c));
        self.set_hatchcolor(Some(Some(c)));
        self.set_facecolor(Some(c));
    }

    pub fn set_alpha(&mut self, a: Option<f64>) {
        self.base.alpha = a;
        self.set_facecolor(self.orig_facecolor);
        self.set_edgecolor(self.orig_edgecolor);
        self.set_hatchcolor(self.orig_hatchcolor);
    }

    pub fn set_fill(&mut self, b: bool) {
        self.fill = b;
        self.set_facecolor(self.orig_facecolor);
        self.set_edgecolor(self.orig_edgecolor);
        self.set_hatchcolor(self.orig_hatchcolor);
    }

    pub fn set_linewidth(&mut self, w: Option<f64>) {
        let w = w.unwrap_or_else(|| rc().num("patch.linewidth"));
        self.linewidth = w;
        self.dash_pattern = scale_dashes(&self.unscaled_dash_pattern, w);
    }

    pub fn set_linestyle(&mut self, ls: &str) {
        let ls = match ls {
            " " | "" | "none" => "None",
            s => s,
        };
        self.linestyle = ls.to_string();
        self.unscaled_dash_pattern = get_dash_pattern(ls);
        self.dash_pattern = scale_dashes(&self.unscaled_dash_pattern, self.linewidth);
    }

    pub fn set_dashes(&mut self, offset: f64, dashes: Option<Vec<f64>>) {
        self.linestyle = "custom".into();
        self.unscaled_dash_pattern = normalize_dashes(offset, dashes);
        self.dash_pattern = scale_dashes(&self.unscaled_dash_pattern, self.linewidth);
    }

    fn has_dashed_edge(&self) -> bool {
        !(self.linestyle == "solid" || self.linestyle == "-")
    }

    pub fn get_hatchcolor(&self) -> Rgba {
        match self.hatch_color {
            HatchColor::Edge => {
                if self.edgecolor.a == 0.0 {
                    rc().color("patch.edgecolor")
                } else {
                    self.edgecolor
                }
            }
            HatchColor::Color(c) => c,
        }
    }

    pub fn set_bounds(&mut self, l: f64, b: f64, w: f64, h: f64) {
        match &mut self.kind {
            PatchKind::Rectangle { x0, y0, width, height, .. } => {
                *x0 = l;
                *y0 = b;
                *width = w;
                *height = h;
            }
            PatchKind::FancyBbox { x, y, width, height, .. } => {
                *x = l;
                *y = b;
                *width = w;
                *height = h;
            }
            _ => {}
        }
    }

    /// `get_path()`.
    pub fn get_path(&self) -> Path {
        match &self.kind {
            PatchKind::Rectangle { .. } => mpath::unit_rectangle(),
            PatchKind::Path { path } => path.clone(),
            PatchKind::Ellipse { .. } => mpath::unit_circle(),
            PatchKind::RegularPolygon { numvertices, .. } => mpath::unit_regular_polygon(*numvertices),
            PatchKind::FancyBbox { x, y, width, height, boxstyle } => {
                let m = self.mutation_aspect;
                let p = boxstyle.path(*x, y / m, *width, height / m, self.mutation_scale);
                let verts = p.vertices.iter().map(|v| [v[0] * 1.0, v[1] * m]).collect();
                new_path(verts, p.codes, SimplifyParams::default())
            }
            PatchKind::Shadow { patch, .. } => patch.get_path(),
            PatchKind::Generic { path, .. } => path.clone(),
        }
    }

    /// `get_patch_transform()`; `None` is the identity transform.
    pub fn get_patch_transform(&self, dpi: f64) -> Option<Transform> {
        match &self.kind {
            PatchKind::Rectangle { x0, y0, width, height, angle, rotation_point } => {
                let bbox = Bbox::from_extents(*x0, *y0, x0 + width, y0 + height);
                let rp = match rotation_point {
                    RotationPoint::Center => {
                        let (w, h) = (bbox.x1 - bbox.x0, bbox.y1 - bbox.y0);
                        (bbox.x0 + w / 2.0, bbox.y0 + h / 2.0)
                    }
                    RotationPoint::Xy => (bbox.x0, bbox.y0),
                    RotationPoint::Point(x, y) => (*x, *y),
                };
                let arc = 1.0;
                let chain = Affine2D::identity()
                    .translate(-rp.0, -rp.1)
                    .scale(1.0, arc)
                    .rotate_deg(*angle)
                    .scale(1.0, 1.0 / arc)
                    .translate(rp.0, rp.1);
                Some(Transform::bbox_to(&bbox).then(&Transform::Affine(chain.0)))
            }
            PatchKind::Ellipse { center, width, height, angle } => {
                let arc = 1.0;
                let t = Affine2D::identity()
                    .scale(width * 0.5, height * 0.5 * arc)
                    .rotate_deg(*angle)
                    .scale(1.0, 1.0 / arc)
                    .translate(center.0, center.1);
                Some(Transform::Affine(t.0))
            }
            PatchKind::RegularPolygon { xy, radius, orientation, .. } => {
                let t = Affine2D::identity().scale(*radius, *radius).rotate(*orientation).translate(xy.0, xy.1);
                Some(Transform::Affine(t.0))
            }
            PatchKind::Shadow { patch, ox, oy } => {
                let st = Transform::Affine(Affine2D::identity().translate(ox * dpi / 72.0, oy * dpi / 72.0).0);
                Some(match patch.get_patch_transform(dpi) {
                    Some(pt) => pt.then(&st),
                    None => st,
                })
            }
            PatchKind::Generic { patch_transform, .. } => patch_transform.map(Transform::Affine),
            _ => None,
        }
    }

    /// `get_transform()` = patch transform + artist transform.
    pub fn get_transform(&self, ctx: &TransCtx) -> Transform {
        let at = match &self.kind {
            PatchKind::Shadow { patch, .. } => ctx.resolve(&patch.base.transform),
            _ => ctx.resolve(&self.base.transform),
        };
        match self.get_patch_transform(ctx.dpi) {
            Some(pt) => pt.then(&at),
            None => at,
        }
    }

    pub fn get_window_extent(&self, ctx: &TransCtx) -> Bbox {
        mpath::get_extents(&self.get_path(), Some(&self.get_transform(ctx)))
    }

    pub fn draw(&self, renderer: &mut Renderer, ctx: &TransCtx, clip: &ClipCtx) {
        if !self.base.visible {
            return;
        }
        let path = self.get_path();
        let transform = self.get_transform(ctx);
        let tpath = if transform.is_affine() { path } else { mpath::transform_path_non_affine(&path, &transform) };
        let affine = transform.get_affine();
        let face = if self.facecolor.a != 0.0 { Some(self.facecolor) } else { None };
        self.draw_paths_with_artist_properties(renderer, clip, &[(tpath, affine, face)]);
    }

    pub fn draw_paths_with_artist_properties(
        &self,
        renderer: &mut Renderer,
        clip: &ClipCtx,
        args: &[(Path, super::transforms::Mat, Option<Rgba>)],
    ) {
        let mut gc = GcState::default();
        let mut lw = self.linewidth;
        if self.edgecolor.a == 0.0 || self.linestyle == "None" {
            lw = 0.0;
        }
        gc.linewidth = lw;
        gc.capstyle = self.capstyle;
        gc.joinstyle = self.joinstyle;
        gc.antialiased = self.antialiased;
        gc.set_clip_from(&self.base, clip);
        gc.snap = self.base.snap;
        gc.set_alpha(self.base.alpha);
        if let Some(h) = &self.hatch {
            if let Some(hp) = super::hatch::get_path(h, 6) {
                gc.hatch = Some((hp, self.get_hatchcolor(), self.hatch_linewidth));
            }
        }
        gc.sketch = self.base.sketch;
        if lw > 0.0 && self.has_dashed_edge() {
            if let Some(g) = self.gapcolor {
                gc.set_foreground(g);
                let (o, gaps) = inverse_dash_pattern(&self.dash_pattern);
                gc.set_dashes(o, gaps);
                for (p, t, f) in args {
                    renderer.draw_path(&gc, p, t, *f);
                }
            }
        }
        gc.set_foreground(self.edgecolor);
        if lw > 0.0 {
            gc.set_dashes(self.dash_pattern.0, self.dash_pattern.1.clone());
        } else {
            gc.set_dashes(0.0, None);
        }
        for (p, t, f) in args {
            renderer.draw_path(&gc, p, t, *f);
        }
    }
}

/// `Polygon.set_xy` path construction.
pub fn polygon_path(xy: Vec<[f64; 2]>, closed: bool) -> Path {
    let mut xy = xy;
    let n = xy.len();
    if closed {
        if n == 1 || (n > 1 && xy[0] != xy[n - 1]) {
            xy.push(xy[0]);
        }
    } else if n > 2 && xy[0] == xy[n - 1] {
        xy.pop();
    }
    if closed {
        closed_path(xy, super::lines::simplify_params())
    } else {
        new_path(xy, None, super::lines::simplify_params())
    }
}

/// `Wedge._recompute_path`.
pub fn wedge_path(center: (f64, f64), r: f64, theta1: f64, theta2: f64, width: Option<f64>) -> Path {
    let (t1, t2, connector) = if ((theta2 - theta1) - 360.0).abs() <= 1e-12 { (0.0, 360.0, MOVETO) } else { (theta1, theta2, LINETO) };
    let arc = mpath::arc(t1, t2, None, false);
    let acodes = arc.codes.clone().unwrap();
    let (v, c) = match width {
        Some(w) => {
            let mut v = arc.vertices.clone();
            for p in arc.vertices.iter().rev() {
                v.push([p[0] * (r - w) / r, p[1] * (r - w) / r]);
            }
            v.push([0.0, 0.0]);
            let mut c = acodes.clone();
            c.push(connector);
            c.extend_from_slice(&acodes[1..]);
            c.push(CLOSEPOLY);
            (v, c)
        }
        None => {
            let mut v = arc.vertices.clone();
            v.push([0.0, 0.0]);
            v.push([0.0, 0.0]);
            let mut c = acodes.clone();
            c.push(connector);
            c.push(CLOSEPOLY);
            (v, c)
        }
    };
    let v = v.iter().map(|p| [p[0] * r + center.0, p[1] * r + center.1]).collect();
    new_path(v, Some(c), super::lines::simplify_params())
}
