//! Shared artist machinery: transform specifications resolved at draw time,
//! graphics-context construction mirroring `GraphicsContextBase`, clipping.

use super::transforms::{Affine2D, Bbox, Mat, Transform, IDENTITY};
use crate::agg::pixfmt::Rgba;
use crate::agg::stroke::{LineCap, LineJoin};
use crate::backend::path_converters::SnapMode;
use crate::backend::renderer_agg::{ClipPath, Dashes, Gc, MplPath};

/// Which of an axis' transforms (`get_xaxis_transform(which)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    Grid,
    Tick1,
    Tick2,
}

/// Symbolic transform, resolved against a [`TransCtx`] when drawing.
#[derive(Clone, Debug, PartialEq)]
pub enum TSpec {
    Identity,
    Data,
    Axes,
    Figure,
    /// `fig.dpi_scale_trans`
    DpiScale,
    XAxis(Which),
    YAxis(Which),
    /// `get_xaxis_text1_transform(pad)` / text2 (pad in points)
    XText1(f64),
    XText2(f64),
    YText1(f64),
    YText2(f64),
    /// Title transform: transAxes + ScaledTranslation(0, pad/72, dpi_scale_trans)
    Title(f64),
    Affine(Mat),
    /// `ScaledTranslation(xt, yt, dpi_scale_trans)`
    ScaledTranslation(f64, f64),
    /// `a + b`
    Then(Box<TSpec>, Box<TSpec>),
    /// `blended_transform_factory(x, y)`
    Blend(Box<TSpec>, Box<TSpec>),
    /// An explicit transform.
    Fixed(Transform),
    /// `spine.get_spine_transform()` for spine 0..4 (left, right, bottom, top).
    Spine(usize),
}

impl TSpec {
    pub fn then(self, b: TSpec) -> TSpec {
        TSpec::Then(Box::new(self), Box::new(b))
    }
    pub fn blend(x: TSpec, y: TSpec) -> TSpec {
        TSpec::Blend(Box::new(x), Box::new(y))
    }
}

/// Spine placement (`Spine._position`).
#[derive(Clone, Debug, PartialEq)]
pub enum SpinePosition {
    Outward(f64),
    Axes(f64),
    Data(f64),
}

/// Everything needed to resolve a [`TSpec`].
#[derive(Clone, Debug)]
pub struct TransCtx {
    pub dpi: f64,
    pub trans_figure: Transform,
    pub dpi_scale: Transform,
    pub trans_axes: Option<Transform>,
    pub trans_data: Option<Transform>,
    /// spine positions: left, right, bottom, top
    pub spine_pos: [SpinePosition; 4],
    pub xtick_alignment: String,
    pub ytick_alignment: String,
}

impl TransCtx {
    pub fn figure_only(dpi: f64, trans_figure: Transform) -> TransCtx {
        TransCtx {
            dpi,
            trans_figure,
            dpi_scale: Transform::Affine(Affine2D::identity().scale(dpi, dpi).0),
            trans_axes: None,
            trans_data: None,
            spine_pos: [
                SpinePosition::Outward(0.0),
                SpinePosition::Outward(0.0),
                SpinePosition::Outward(0.0),
                SpinePosition::Outward(0.0),
            ],
            xtick_alignment: "center".into(),
            ytick_alignment: "center_baseline".into(),
        }
    }

    fn axes(&self) -> Transform {
        self.trans_axes.clone().unwrap_or_else(|| self.trans_figure.clone())
    }
    fn data(&self) -> Transform {
        self.trans_data.clone().unwrap_or_else(|| self.trans_figure.clone())
    }

    /// `Spine.get_spine_transform()` for spine index 0..4 (l, r, b, t).
    pub fn spine_transform(&self, idx: usize) -> Transform {
        let base = if idx < 2 {
            self.resolve(&TSpec::YAxis(Which::Grid))
        } else {
            self.resolve(&TSpec::XAxis(Which::Grid))
        };
        match &self.spine_pos[idx] {
            SpinePosition::Outward(amount) => {
                if *amount == 0.0 {
                    base
                } else {
                    let v = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)][idx];
                    let dx = amount * v.0 / 72.0;
                    let dy = amount * v.1 / 72.0;
                    base.then(&Transform::scaled_translation(dx, dy, &self.dpi_scale))
                }
            }
            SpinePosition::Axes(amount) => {
                let a = if idx < 2 {
                    Affine2D::from_values(0.0, 0.0, 0.0, 1.0, *amount, 0.0)
                } else {
                    Affine2D::from_values(1.0, 0.0, 0.0, 0.0, 0.0, *amount)
                };
                Transform::Affine(a.0).then(&base)
            }
            SpinePosition::Data(amount) => {
                let mut amount = *amount;
                if idx == 1 || idx == 3 {
                    amount -= 1.0;
                }
                if idx < 2 {
                    let x = Transform::Affine(Affine2D::identity().translate(amount, 0.0).0).then(&self.data());
                    Transform::blend(&x, &self.data())
                } else {
                    let y = Transform::Affine(Affine2D::identity().translate(0.0, amount).0).then(&self.data());
                    Transform::blend(&self.data(), &y)
                }
            }
        }
    }

    pub fn resolve(&self, spec: &TSpec) -> Transform {
        match spec {
            TSpec::Identity => Transform::Affine(IDENTITY),
            TSpec::Data => self.data(),
            TSpec::Axes => self.axes(),
            TSpec::Figure => self.trans_figure.clone(),
            TSpec::DpiScale => self.dpi_scale.clone(),
            TSpec::XAxis(w) => match w {
                Which::Grid => Transform::blend(&self.data(), &self.axes()),
                Which::Tick1 => self.spine_transform(2),
                Which::Tick2 => self.spine_transform(3),
            },
            TSpec::YAxis(w) => match w {
                Which::Grid => Transform::blend(&self.axes(), &self.data()),
                Which::Tick1 => self.spine_transform(0),
                Which::Tick2 => self.spine_transform(1),
            },
            TSpec::XText1(pad) => self
                .resolve(&TSpec::XAxis(Which::Tick1))
                .then(&Transform::scaled_translation(0.0, -1.0 * pad / 72.0, &self.dpi_scale)),
            TSpec::XText2(pad) => self
                .resolve(&TSpec::XAxis(Which::Tick2))
                .then(&Transform::scaled_translation(0.0, pad / 72.0, &self.dpi_scale)),
            TSpec::YText1(pad) => self
                .resolve(&TSpec::YAxis(Which::Tick1))
                .then(&Transform::scaled_translation(-1.0 * pad / 72.0, 0.0, &self.dpi_scale)),
            TSpec::YText2(pad) => self
                .resolve(&TSpec::YAxis(Which::Tick2))
                .then(&Transform::scaled_translation(pad / 72.0, 0.0, &self.dpi_scale)),
            TSpec::Title(pad) => self.axes().then(&Transform::scaled_translation(0.0, pad / 72.0, &self.dpi_scale)),
            TSpec::Affine(m) => Transform::Affine(*m),
            TSpec::ScaledTranslation(x, y) => Transform::scaled_translation(*x, *y, &self.dpi_scale),
            TSpec::Then(a, b) => self.resolve(a).then(&self.resolve(b)),
            TSpec::Blend(x, y) => Transform::blend(&self.resolve(x), &self.resolve(y)),
            TSpec::Fixed(t) => t.clone(),
            TSpec::Spine(i) => self.spine_transform(*i),
        }
    }
}

/// Clip box specification of an artist.
#[derive(Clone, Debug, PartialEq)]
pub enum ClipBoxSpec {
    None,
    /// The axes bbox (`TransformedBbox(unit, axes.patch transform)`).
    AxesBbox,
    /// A bbox in display coordinates.
    Display(Bbox),
}

#[derive(Clone, Debug)]
pub struct ClipPathSpec {
    pub path: std::sync::Arc<MplPath>,
    pub trans: TSpec,
}

/// Common artist properties (`matplotlib.artist.Artist`).
#[derive(Clone, Debug)]
pub struct ArtistBase {
    pub visible: bool,
    pub alpha: Option<f64>,
    pub zorder: f64,
    pub clip_on: bool,
    pub clip_box: ClipBoxSpec,
    pub clip_path: Option<ClipPathSpec>,
    pub label: String,
    pub gid: Option<String>,
    pub snap: Option<bool>,
    pub sketch: Option<(f64, f64, f64)>,
    pub transform: TSpec,
    pub transform_set: bool,
    pub animated: bool,
    pub in_layout: bool,
    pub rasterized: bool,
}

impl Default for ArtistBase {
    fn default() -> Self {
        ArtistBase {
            visible: true,
            alpha: None,
            zorder: 0.0,
            clip_on: true,
            clip_box: ClipBoxSpec::None,
            clip_path: None,
            label: String::new(),
            gid: None,
            snap: None,
            sketch: None,
            transform: TSpec::Identity,
            transform_set: false,
            animated: false,
            in_layout: true,
            rasterized: false,
        }
    }
}

/// Context needed to turn an artist's clip settings into a GC.
pub struct ClipCtx<'a> {
    pub axes_bbox: Option<Bbox>,
    pub trans: &'a TransCtx,
}

/// A graphics context under construction (`GraphicsContextBase`).
#[derive(Clone, Debug)]
pub struct GcState {
    pub alpha: f64,
    pub forced_alpha: bool,
    pub antialiased: bool,
    pub capstyle: LineCap,
    pub joinstyle: LineJoin,
    pub cliprect: Option<Bbox>,
    pub clippath: Option<ClipPath>,
    pub dashes: (f64, Option<Vec<f64>>),
    pub linewidth: f64,
    pub rgb: Rgba,
    /// whether `_rgb` was given as a 3-tuple (affects face alpha in Agg)
    pub snap: Option<bool>,
    pub sketch: Option<(f64, f64, f64)>,
    pub hatch: Option<(MplPath, Rgba, f64)>,
}

impl Default for GcState {
    fn default() -> Self {
        GcState {
            alpha: 1.0,
            forced_alpha: false,
            antialiased: true,
            capstyle: LineCap::Butt,
            joinstyle: LineJoin::Round,
            cliprect: None,
            clippath: None,
            dashes: (0.0, None),
            linewidth: 1.0,
            rgb: Rgba::new(0.0, 0.0, 0.0, 1.0),
            snap: None,
            sketch: None,
            hatch: None,
        }
    }
}

impl GcState {
    /// `set_alpha`.
    pub fn set_alpha(&mut self, alpha: Option<f64>) {
        match alpha {
            Some(a) => {
                self.alpha = a;
                self.forced_alpha = true;
            }
            None => {
                self.alpha = 1.0;
                self.forced_alpha = false;
            }
        }
        let rgb = self.rgb;
        self.set_foreground(rgb);
    }

    /// `set_foreground(fg, isRGBA=True)`.
    pub fn set_foreground(&mut self, fg: Rgba) {
        if self.forced_alpha {
            self.rgb = Rgba::new(fg.r, fg.g, fg.b, self.alpha);
        } else {
            self.rgb = fg;
        }
    }

    pub fn set_dashes(&mut self, offset: f64, dashes: Option<Vec<f64>>) {
        self.dashes = (offset, dashes);
    }

    /// Apply `Artist._set_gc_clip`.
    pub fn set_clip_from(&mut self, base: &ArtistBase, ctx: &ClipCtx) {
        if base.clip_on {
            match &base.clip_box {
                ClipBoxSpec::None => {}
                ClipBoxSpec::AxesBbox => self.cliprect = ctx.axes_bbox,
                ClipBoxSpec::Display(b) => self.cliprect = Some(*b),
            }
            self.clippath = base.clip_path.as_ref().map(|cp| {
                let t = ctx.trans.resolve(&cp.trans);
                // TransformedPath: non-affine part applied to the path, the
                // affine part passed to the renderer.
                let p = super::path::transform_path_non_affine(&cp.path, &t);
                ClipPath { path: std::sync::Arc::new(p), trans: crate::agg::TransAffine::from_matrix(&t.get_affine()) }
            });
        } else {
            self.cliprect = None;
            self.clippath = None;
        }
    }

    /// Convert to the Agg backend's GC (`GCAgg` conversion).
    pub fn to_agg(&self) -> Gc {
        let dashes = match &self.dashes.1 {
            Some(seq) if !seq.is_empty() => Dashes::from_seq(self.dashes.0, seq),
            _ => Dashes::default(),
        };
        Gc {
            linewidth: self.linewidth,
            alpha: self.alpha,
            forced_alpha: self.forced_alpha,
            color: self.rgb,
            isaa: self.antialiased,
            cap: self.capstyle,
            join: self.joinstyle,
            cliprect: self.cliprect.map(|b| [b.x0, b.y0, b.x1, b.y1]),
            clippath: self.clippath.clone(),
            dashes,
            snap_mode: match self.snap {
                None => SnapMode::Auto,
                Some(true) => SnapMode::True,
                Some(false) => SnapMode::False,
            },
            hatch: self.hatch.as_ref().map(|(p, c, lw)| crate::backend::renderer_agg::Hatch {
                path: p.clone(),
                color: *c,
                linewidth: *lw,
            }),
            sketch: self.sketch,
        }
    }

    /// Face colour as `RendererAgg.draw_path` sees it (`face.a = gc.alpha`
    /// when alpha is forced or a 3-tuple was given).
    pub fn face(&self, face: Option<Rgba>, is_rgb_tuple: bool) -> Option<Rgba> {
        face.map(|mut f| {
            if self.forced_alpha || is_rgb_tuple {
                f.a = self.alpha;
            }
            f
        })
    }
}

/// `to_rgba(c, alpha)`: override alpha when given.
pub fn with_alpha(c: Rgba, alpha: Option<f64>) -> Rgba {
    match alpha {
        Some(a) => Rgba::new(c.r, c.g, c.b, a),
        None => c,
    }
}

pub fn parse_capstyle(s: &str) -> LineCap {
    match s {
        "round" => LineCap::Round,
        "projecting" => LineCap::Square,
        _ => LineCap::Butt,
    }
}

pub fn parse_joinstyle(s: &str) -> LineJoin {
    match s {
        "miter" => LineJoin::MiterRevert,
        "bevel" => LineJoin::Bevel,
        _ => LineJoin::Round,
    }
}

/// Shared, mutable artist storage (artists are referenced from Python
/// handles as well as from their container).
pub type Shared<T> = std::sync::Arc<std::sync::Mutex<T>>;

pub fn shared<T>(t: T) -> Shared<T> {
    std::sync::Arc::new(std::sync::Mutex::new(t))
}

pub trait Lock<T> {
    fn lk(&self) -> std::sync::MutexGuard<'_, T>;
}

impl<T> Lock<T> for Shared<T> {
    fn lk(&self) -> std::sync::MutexGuard<'_, T> {
        self.lock().unwrap_or_else(|e| e.into_inner())
    }
}
