//! `matplotlib.markers.MarkerStyle`.

use super::path::{self, new_path, Path, SimplifyParams, CLOSEPOLY, LINETO, MOVETO};
use super::transforms::Affine2D;
use crate::agg::stroke::{LineCap, LineJoin};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillStyle {
    Full,
    Left,
    Right,
    Bottom,
    Top,
    None,
}

impl FillStyle {
    pub fn parse(s: &str) -> Option<FillStyle> {
        Some(match s {
            "full" => FillStyle::Full,
            "left" => FillStyle::Left,
            "right" => FillStyle::Right,
            "bottom" => FillStyle::Bottom,
            "top" => FillStyle::Top,
            "none" => FillStyle::None,
            _ => return None,
        })
    }
    fn is_half(&self) -> bool {
        matches!(self, FillStyle::Left | FillStyle::Right | FillStyle::Bottom | FillStyle::Top)
    }
}

/// Marker specification as given by the user.
#[derive(Clone, Debug, PartialEq)]
pub enum MarkerSpec {
    /// A named marker (`'o'`, `'^'`, `'tickdown'` codes 0..11 as "0".."11").
    Named(String),
    /// `(numsides, style, angle)`.
    Tuple(usize, u8, f64),
    /// Custom vertices / path.
    Path(Path),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SnapThreshold {
    None,
    Inf,
    Value(f64),
}

#[derive(Clone, Debug)]
pub struct MarkerStyle {
    pub spec: MarkerSpec,
    pub fillstyle: FillStyle,
    pub user_transform: Option<Affine2D>,
    pub user_capstyle: Option<LineCap>,
    pub user_joinstyle: Option<LineJoin>,
    pub path: Path,
    pub transform: Affine2D,
    pub alt_path: Option<Path>,
    pub alt_transform: Option<Affine2D>,
    pub snap_threshold: SnapThreshold,
    pub joinstyle: LineJoin,
    pub capstyle: LineCap,
    pub filled: bool,
    pub is_nothing: bool,
    pub is_pixel: bool,
}

fn p(verts: &[[f64; 2]], codes: Option<&[u8]>) -> Path {
    new_path(verts.to_vec(), codes.map(|c| c.to_vec()), SimplifyParams::default())
}

fn triangle_path() -> Path {
    p(&[[0.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]], Some(&[MOVETO, LINETO, LINETO, CLOSEPOLY]))
}

fn triangle_paths_half() -> [Path; 4] {
    [
        p(&[[0.0, 1.0], [-0.6, -0.2], [0.6, -0.2], [0.0, 1.0]], Some(&[1, 2, 2, 79])),
        p(&[[0.0, 1.0], [0.0, -1.0], [-1.0, -1.0], [0.0, 1.0]], Some(&[1, 2, 2, 79])),
        p(&[[-0.6, -0.2], [0.6, -0.2], [1.0, -1.0], [-1.0, -1.0], [-0.6, -0.2]], Some(&[1, 2, 2, 2, 79])),
        p(&[[0.0, 1.0], [0.0, -1.0], [1.0, -1.0], [0.0, 1.0]], Some(&[1, 2, 2, 79])),
    ]
}

const ONE6: f64 = 0.16666666666666666;

impl MarkerStyle {
    pub fn new(spec: MarkerSpec, fillstyle: FillStyle) -> MarkerStyle {
        let mut m = MarkerStyle {
            spec,
            fillstyle,
            user_transform: None,
            user_capstyle: None,
            user_joinstyle: None,
            path: Path::new(Vec::new(), None),
            transform: Affine2D::identity(),
            alt_path: None,
            alt_transform: None,
            snap_threshold: SnapThreshold::None,
            joinstyle: LineJoin::Round,
            capstyle: LineCap::Butt,
            filled: true,
            is_nothing: false,
            is_pixel: false,
        };
        m.recache();
        m
    }

    /// A marker fully described by the Python side (`MarkerStyle`'s path,
    /// transforms and styles).
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        path: Path,
        transform: Affine2D,
        alt_path: Option<Path>,
        alt_transform: Option<Affine2D>,
        joinstyle: LineJoin,
        capstyle: LineCap,
        snap_threshold: SnapThreshold,
        filled: bool,
        fillstyle: FillStyle,
        is_pixel: bool,
    ) -> MarkerStyle {
        MarkerStyle {
            spec: MarkerSpec::Path(path.clone()),
            fillstyle,
            user_transform: None,
            user_capstyle: None,
            user_joinstyle: None,
            path,
            transform,
            alt_path,
            alt_transform,
            snap_threshold,
            joinstyle,
            capstyle,
            filled,
            is_nothing: false,
            is_pixel,
        }
    }

    pub fn named(s: &str) -> MarkerStyle {
        MarkerStyle::new(MarkerSpec::Named(s.to_string()), FillStyle::Full)
    }

    pub fn is_valid_name(s: &str) -> bool {
        matches!(
            s,
            "." | ","
                | "o"
                | "v"
                | "^"
                | "<"
                | ">"
                | "1"
                | "2"
                | "3"
                | "4"
                | "8"
                | "s"
                | "p"
                | "*"
                | "h"
                | "H"
                | "+"
                | "x"
                | "D"
                | "d"
                | "|"
                | "_"
                | "P"
                | "X"
                | "None"
                | "none"
                | " "
                | ""
                | "tickleft"
                | "tickright"
                | "tickup"
                | "tickdown"
                | "caretleft"
                | "caretright"
                | "caretup"
                | "caretdown"
                | "caretleftbase"
                | "caretrightbase"
                | "caretupbase"
                | "caretdownbase"
        )
    }

    pub fn recache(&mut self) {
        self.path = Path::new(Vec::new(), None);
        self.transform = Affine2D::identity();
        self.alt_path = None;
        self.alt_transform = None;
        self.snap_threshold = SnapThreshold::None;
        self.joinstyle = LineJoin::Round;
        self.capstyle = self.user_capstyle.unwrap_or(LineCap::Butt);
        self.filled = self.fillstyle != FillStyle::None;
        self.is_nothing = false;
        self.is_pixel = false;
        let spec = self.spec.clone();
        match spec {
            MarkerSpec::Named(name) => self.set_named(&name),
            MarkerSpec::Tuple(numsides, style, rotation) => {
                match style {
                    0 => {
                        self.path = path::unit_regular_polygon(numsides);
                        self.joinstyle = self.user_joinstyle.unwrap_or(LineJoin::MiterRevert);
                    }
                    1 => {
                        self.path = path::unit_regular_star(numsides, 0.5);
                        self.joinstyle = self.user_joinstyle.unwrap_or(LineJoin::Bevel);
                    }
                    _ => {
                        self.path = path::unit_regular_asterisk(numsides);
                        self.filled = false;
                        self.joinstyle = self.user_joinstyle.unwrap_or(LineJoin::Bevel);
                    }
                }
                self.transform = Affine2D::identity().scale(0.5, 0.5).rotate_deg(rotation);
            }
            MarkerSpec::Path(path) => {
                let rescale = path.vertices.iter().fold(f64::NEG_INFINITY, |m, v| m.max(v[0].abs()).max(v[1].abs()));
                self.transform = Affine2D::identity().scale(0.5 / rescale, 0.5 / rescale);
                self.path = path;
            }
        }
    }

    fn miter(&self) -> LineJoin {
        self.user_joinstyle.unwrap_or(LineJoin::MiterRevert)
    }

    fn set_circle(&mut self, size: f64) {
        self.transform = Affine2D::identity().scale(0.5 * size, 0.5 * size);
        self.snap_threshold = SnapThreshold::Inf;
        if !self.fillstyle.is_half() {
            self.path = path::unit_circle();
        } else {
            self.path = path::unit_circle_righthalf();
            self.alt_path = Some(self.path.clone());
            let r = match self.fillstyle {
                FillStyle::Right => 0.0,
                FillStyle::Top => 90.0,
                FillStyle::Left => 180.0,
                _ => 270.0,
            };
            self.transform = self.transform.rotate_deg(r);
            self.alt_transform = Some(self.transform.rotate_deg(180.0));
        }
    }

    fn set_triangle(&mut self, rot: f64, skip: usize) {
        self.transform = Affine2D::identity().scale(0.5, 0.5).rotate_deg(rot);
        self.snap_threshold = SnapThreshold::Value(5.0);
        if !self.fillstyle.is_half() {
            self.path = triangle_path();
        } else {
            let mp = triangle_paths_half();
            // order in matplotlib: [u, l, d, r]
            let (a, b) = match self.fillstyle {
                FillStyle::Top => (skip % 4, (2 + skip) % 4),
                FillStyle::Bottom => ((2 + skip) % 4, skip % 4),
                FillStyle::Left => ((1 + skip) % 4, (3 + skip) % 4),
                _ => ((3 + skip) % 4, (1 + skip) % 4),
            };
            self.path = mp[a].clone();
            self.alt_path = Some(mp[b].clone());
            self.alt_transform = Some(self.transform);
        }
        self.joinstyle = self.miter();
    }

    fn set_named(&mut self, name: &str) {
        match name {
            "." => self.set_circle(0.5),
            "," => {
                self.path = path::unit_rectangle();
                self.transform = Affine2D::identity().translate(-0.49999, -0.49999);
                self.snap_threshold = SnapThreshold::None;
                self.is_pixel = true;
            }
            "o" => self.set_circle(1.0),
            "v" => self.set_triangle(180.0, 2),
            "^" => self.set_triangle(0.0, 0),
            "<" => self.set_triangle(90.0, 3),
            ">" => self.set_triangle(270.0, 1),
            "1" | "2" | "3" | "4" => {
                self.transform = Affine2D::identity().scale(0.5, 0.5);
                self.snap_threshold = SnapThreshold::Value(5.0);
                self.filled = false;
                self.path = p(
                    &[[0.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.8, 0.5], [0.0, 0.0], [-0.8, 0.5]],
                    Some(&[1, 2, 1, 2, 1, 2]),
                );
                match name {
                    "2" => self.transform = self.transform.rotate_deg(180.0),
                    "3" => self.transform = self.transform.rotate_deg(270.0),
                    "4" => self.transform = self.transform.rotate_deg(90.0),
                    _ => {}
                }
            }
            "8" => {
                self.transform = Affine2D::identity().scale(0.5, 0.5);
                self.snap_threshold = SnapThreshold::Value(5.0);
                let polypath = path::unit_regular_polygon(8);
                if !self.fillstyle.is_half() {
                    self.transform = self.transform.rotate_deg(22.5);
                    self.path = polypath;
                } else {
                    let x = 2f64.sqrt() / 4.0;
                    let pth = p(&[[0.0, -1.0], [0.0, 1.0], [-x, 1.0], [-1.0, x], [-1.0, -x], [-x, -1.0], [0.0, -1.0]], None);
                    self.path = pth.clone();
                    self.alt_path = Some(pth);
                    let r = match self.fillstyle {
                        FillStyle::Left => 0.0,
                        FillStyle::Bottom => 90.0,
                        FillStyle::Right => 180.0,
                        _ => 270.0,
                    };
                    self.transform = self.transform.rotate_deg(r);
                    self.alt_transform = Some(self.transform.rotate_deg(180.0));
                }
                self.joinstyle = self.miter();
            }
            "s" => {
                self.transform = Affine2D::identity().translate(-0.5, -0.5);
                self.snap_threshold = SnapThreshold::Value(2.0);
                if !self.fillstyle.is_half() {
                    self.path = path::unit_rectangle();
                } else {
                    self.path = p(&[[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5], [0.0, 0.0]], None);
                    self.alt_path = Some(p(&[[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0], [0.0, 0.5]], None));
                    let r = match self.fillstyle {
                        FillStyle::Bottom => 0.0,
                        FillStyle::Right => 90.0,
                        FillStyle::Top => 180.0,
                        _ => 270.0,
                    };
                    self.transform = self.transform.rotate_deg(r);
                    self.alt_transform = Some(self.transform);
                }
                self.joinstyle = self.miter();
            }
            "p" => {
                self.transform = Affine2D::identity().scale(0.5, 0.5);
                self.snap_threshold = SnapThreshold::Value(5.0);
                let polypath = path::unit_regular_polygon(5);
                if !self.fillstyle.is_half() {
                    self.path = polypath;
                } else {
                    let v = &polypath.vertices;
                    let y = (1.0 + 5f64.sqrt()) / 4.0;
                    let top = p(&[v[0], v[1], v[4], v[0]], None);
                    let bottom = p(&[v[1], v[2], v[3], v[4], v[1]], None);
                    let left = p(&[v[0], v[1], v[2], [0.0, -y], v[0]], None);
                    let right = p(&[v[0], v[4], v[3], [0.0, -y], v[0]], None);
                    let (a, b) = match self.fillstyle {
                        FillStyle::Top => (top, bottom),
                        FillStyle::Bottom => (bottom, top),
                        FillStyle::Left => (left, right),
                        _ => (right, left),
                    };
                    self.path = a;
                    self.alt_path = Some(b);
                    self.alt_transform = Some(self.transform);
                }
                self.joinstyle = self.miter();
            }
            "*" => {
                self.transform = Affine2D::identity().scale(0.5, 0.5);
                self.snap_threshold = SnapThreshold::Value(5.0);
                let polypath = path::unit_regular_star(5, 0.381966);
                if !self.fillstyle.is_half() {
                    self.path = polypath;
                } else {
                    let v = &polypath.vertices;
                    let cat = |idx: &[usize]| -> Path { p(&idx.iter().map(|&i| v[i]).collect::<Vec<_>>(), None) };
                    let top = cat(&[0, 1, 2, 3, 7, 8, 9, 0]);
                    let bottom = cat(&[3, 4, 5, 6, 7, 3]);
                    let left = cat(&[0, 1, 2, 3, 4, 5, 0]);
                    let right = cat(&[0, 5, 6, 7, 8, 9, 0]);
                    let (a, b) = match self.fillstyle {
                        FillStyle::Top => (top, bottom),
                        FillStyle::Bottom => (bottom, top),
                        FillStyle::Left => (left, right),
                        _ => (right, left),
                    };
                    self.path = a;
                    self.alt_path = Some(b);
                    self.alt_transform = Some(self.transform);
                }
                self.joinstyle = self.user_joinstyle.unwrap_or(LineJoin::Bevel);
            }
            "h" | "H" => {
                if name == "h" {
                    self.transform = Affine2D::identity().scale(0.5, 0.5);
                } else {
                    self.transform = Affine2D::identity().scale(0.5, 0.5).rotate_deg(30.0);
                }
                self.snap_threshold = SnapThreshold::None;
                let polypath = path::unit_regular_polygon(6);
                if !self.fillstyle.is_half() {
                    self.path = polypath;
                } else {
                    let v = &polypath.vertices;
                    let (top, bottom, left, right);
                    if name == "h" {
                        let x = (5.0 * std::f64::consts::PI / 6.0).cos().abs();
                        top = p(&[[-x, 0.0], v[1], v[0], v[5], [x, 0.0]], None);
                        bottom = p(&[[-x, 0.0], v[2], v[3], v[4], [x, 0.0]], None);
                        left = p(&[v[0], v[1], v[2], v[3]], None);
                        right = p(&[v[0], v[5], v[4], v[3]], None);
                    } else {
                        let (x, y) = (3f64.sqrt() / 4.0, 3.0 / 4.0);
                        top = p(&[v[1], v[0], v[5], v[4], v[1]], None);
                        bottom = p(&[v[1], v[2], v[3], v[4]], None);
                        left = p(&[[x, y], v[0], v[1], v[2], [-x, -y], [x, y]], None);
                        right = p(&[[x, y], v[5], v[4], v[3], [-x, -y]], None);
                    }
                    let (a, b) = match self.fillstyle {
                        FillStyle::Top => (top, bottom),
                        FillStyle::Bottom => (bottom, top),
                        FillStyle::Left => (left, right),
                        _ => (right, left),
                    };
                    self.path = a;
                    self.alt_path = Some(b);
                    self.alt_transform = Some(self.transform);
                }
                self.joinstyle = self.miter();
            }
            "+" => {
                self.transform = Affine2D::identity().scale(0.5, 0.5);
                self.snap_threshold = SnapThreshold::Value(1.0);
                self.filled = false;
                self.path = p(&[[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], Some(&[1, 2, 1, 2]));
            }
            "x" => {
                self.transform = Affine2D::identity().scale(0.5, 0.5);
                self.snap_threshold = SnapThreshold::Value(3.0);
                self.filled = false;
                self.path = p(&[[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]], Some(&[1, 2, 1, 2]));
            }
            "D" | "d" => {
                self.transform = Affine2D::identity().translate(-0.5, -0.5).rotate_deg(45.0);
                self.snap_threshold = SnapThreshold::Value(5.0);
                if !self.fillstyle.is_half() {
                    self.path = path::unit_rectangle();
                } else {
                    self.path = p(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], None);
                    self.alt_path = Some(p(&[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]], None));
                    let r = match self.fillstyle {
                        FillStyle::Right => 0.0,
                        FillStyle::Top => 90.0,
                        FillStyle::Left => 180.0,
                        _ => 270.0,
                    };
                    self.transform = self.transform.rotate_deg(r);
                    self.alt_transform = Some(self.transform);
                }
                self.joinstyle = self.miter();
                if name == "d" {
                    self.transform = self.transform.scale(0.6, 1.0);
                    // matplotlib's alt transform is the same (mutated) object
                    if self.alt_transform.is_some() {
                        self.alt_transform = Some(self.transform);
                    }
                }
            }
            "|" | "_" => {
                self.transform = Affine2D::identity().scale(0.5, 0.5);
                self.snap_threshold = SnapThreshold::Value(1.0);
                self.filled = false;
                self.path = p(&[[0.0, -1.0], [0.0, 1.0]], None);
                if name == "_" {
                    self.transform = self.transform.rotate_deg(90.0);
                }
            }
            "P" | "X" => {
                self.transform = Affine2D::identity();
                self.snap_threshold = SnapThreshold::Value(5.0);
                self.joinstyle = self.miter();
                let full = if name == "P" {
                    p(
                        &[
                            [-ONE6, -0.5],
                            [ONE6, -0.5],
                            [ONE6, -ONE6],
                            [0.5, -ONE6],
                            [0.5, ONE6],
                            [ONE6, ONE6],
                            [ONE6, 0.5],
                            [-ONE6, 0.5],
                            [-ONE6, ONE6],
                            [-0.5, ONE6],
                            [-0.5, -ONE6],
                            [-ONE6, -ONE6],
                            [-ONE6, -0.5],
                        ],
                        Some(&[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 79]),
                    )
                } else {
                    p(
                        &[
                            [-0.25, -0.5],
                            [0.0, -0.25],
                            [0.25, -0.5],
                            [0.5, -0.25],
                            [0.25, 0.0],
                            [0.5, 0.25],
                            [0.25, 0.5],
                            [0.0, 0.25],
                            [-0.25, 0.5],
                            [-0.5, 0.25],
                            [-0.25, 0.0],
                            [-0.5, -0.25],
                            [-0.25, -0.5],
                        ],
                        Some(&[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 79]),
                    )
                };
                if !self.fillstyle.is_half() {
                    self.path = full;
                } else {
                    let half = if name == "P" {
                        p(
                            &[
                                [0.5, 0.0],
                                [0.5, ONE6],
                                [ONE6, ONE6],
                                [ONE6, 0.5],
                                [-ONE6, 0.5],
                                [-ONE6, ONE6],
                                [-0.5, ONE6],
                                [-0.5, 0.0],
                                [0.5, 0.0],
                            ],
                            Some(&[1, 2, 2, 2, 2, 2, 2, 2, 79]),
                        )
                    } else {
                        p(
                            &[
                                [0.25, 0.0],
                                [0.5, 0.25],
                                [0.25, 0.5],
                                [0.0, 0.25],
                                [-0.25, 0.5],
                                [-0.5, 0.25],
                                [-0.25, 0.0],
                                [0.25, 0.0],
                            ],
                            Some(&[1, 2, 2, 2, 2, 2, 2, 79]),
                        )
                    };
                    self.path = half.clone();
                    self.alt_path = Some(half);
                    let r = match self.fillstyle {
                        FillStyle::Top => 0.0,
                        FillStyle::Left => 90.0,
                        FillStyle::Bottom => 180.0,
                        _ => 270.0,
                    };
                    self.transform = self.transform.rotate_deg(r);
                    self.alt_transform = Some(self.transform.rotate_deg(180.0));
                }
            }
            "tickleft" => self.set_tick(-1.0, 1.0, true),
            "tickright" => self.set_tick(1.0, 1.0, true),
            "tickup" => self.set_tick(1.0, 1.0, false),
            "tickdown" => self.set_tick(1.0, -1.0, false),
            "caretdown" | "caretup" | "caretleft" | "caretright" | "caretdownbase" | "caretupbase"
            | "caretleftbase" | "caretrightbase" => {
                self.transform = Affine2D::identity().scale(0.5, 0.5);
                self.snap_threshold = SnapThreshold::Value(3.0);
                self.filled = false;
                self.joinstyle = self.miter();
                if name.ends_with("base") {
                    self.path = p(&[[-1.0, 0.0], [0.0, -1.5], [1.0, 0.0]], None);
                } else {
                    self.path = p(&[[-1.0, 1.5], [0.0, 0.0], [1.0, 1.5]], None);
                }
                let dir = name.trim_start_matches("caret").trim_end_matches("base");
                match dir {
                    "up" => self.transform = self.transform.rotate_deg(180.0),
                    "left" => self.transform = self.transform.rotate_deg(270.0),
                    "right" => self.transform = self.transform.rotate_deg(90.0),
                    _ => {}
                }
            }
            _ => {
                // 'None', 'none', ' ', ''
                self.filled = false;
                self.is_nothing = true;
            }
        }
    }

    fn set_tick(&mut self, sx: f64, sy: f64, horiz: bool) {
        self.transform = Affine2D::identity().scale(sx, sy);
        self.snap_threshold = SnapThreshold::Value(1.0);
        self.filled = false;
        self.path = if horiz { p(&[[0.0, 0.0], [1.0, 0.0]], None) } else { p(&[[-0.0, 0.0], [-0.0, 1.0]], None) };
    }

    /// `get_transform()` (with user transform appended).
    pub fn get_transform(&self) -> Affine2D {
        match &self.user_transform {
            None => self.transform,
            Some(u) => Affine2D(super::transforms::dot(&u.0, &self.transform.0)),
        }
    }

    pub fn get_alt_transform(&self) -> Option<Affine2D> {
        self.alt_transform.map(|t| match &self.user_transform {
            None => t,
            Some(u) => Affine2D(super::transforms::dot(&u.0, &t.0)),
        })
    }

    pub fn joinstyle(&self) -> LineJoin {
        self.joinstyle
    }

    pub fn capstyle(&self) -> LineCap {
        self.capstyle
    }

    pub fn is_filled(&self) -> bool {
        self.filled
    }
}
