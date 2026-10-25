//! Numerically faithful equivalents of the parts of `matplotlib.transforms`
//! used for drawing.  Affine transforms are represented by their 3x3
//! matrix; composition reproduces numpy's `np.dot` (OpenBLAS) results.

use std::sync::atomic::{AtomicBool, Ordering};

pub type Mat = [[f64; 3]; 3];

pub const IDENTITY: Mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Whether `np.dot` accumulates with fused multiply-adds on this machine
/// (true for OpenBLAS on FMA-capable CPUs).  Set from Python at import time.
static DOT_USES_FMA: AtomicBool = AtomicBool::new(true);

pub fn set_dot_uses_fma(v: bool) {
    DOT_USES_FMA.store(v, Ordering::Relaxed);
}

/// `np.dot(b, a)` for 3x3 matrices.
pub fn dot(b: &Mat, a: &Mat) -> Mat {
    let mut r = [[0.0; 3]; 3];
    if DOT_USES_FMA.load(Ordering::Relaxed) {
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0f64;
                for k in 0..3 {
                    s = b[i][k].mul_add(a[k][j], s);
                }
                r[i][j] = s;
            }
        }
    } else {
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = b[i][0] * a[0][j] + b[i][1] * a[1][j] + b[i][2] * a[2][j];
            }
        }
    }
    r
}

/// matplotlib's C++ `affine_transform` for a single point.
#[inline(always)]
pub fn affine_point(m: &Mat, x: f64, y: f64) -> (f64, f64) {
    let t0 = m[0][0] * x;
    let t1 = m[0][1] * y;
    let xo = t0 + t1 + m[0][2];
    let t0 = m[1][0] * x;
    let t1 = m[1][1] * y;
    let yo = t0 + t1 + m[1][2];
    (xo, yo)
}

/// `numpy.linalg.inv` of an affine 3x3 matrix (LAPACK getrf/getri on the
/// generic matrix).  For the axis-aligned matrices used in practice this
/// reduces to reciprocals; the general case uses Gauss-Jordan elimination
/// with partial pivoting like LAPACK.
pub fn inv(m: &Mat) -> Mat {
    // Fast exact path for diagonal + translation (most common).
    if m[0][1] == 0.0 && m[1][0] == 0.0 && m[2][0] == 0.0 && m[2][1] == 0.0 && m[2][2] == 1.0 {
        let a = 1.0 / m[0][0];
        let d = 1.0 / m[1][1];
        return [[a, 0.0, -(m[0][2] * a)], [0.0, d, -(m[1][2] * d)], [0.0, 0.0, 1.0]];
    }
    let mut a = *m;
    let mut r = IDENTITY;
    for c in 0..3 {
        let mut p = c;
        for i in c + 1..3 {
            if a[i][c].abs() > a[p][c].abs() {
                p = i;
            }
        }
        a.swap(c, p);
        r.swap(c, p);
        let piv = a[c][c];
        for j in 0..3 {
            a[c][j] /= piv;
            r[c][j] /= piv;
        }
        for i in 0..3 {
            if i != c {
                let f = a[i][c];
                for j in 0..3 {
                    a[i][j] -= f * a[c][j];
                    r[i][j] -= f * r[c][j];
                }
            }
        }
    }
    r
}

/// `Affine2D` building operations (in-place semantics of matplotlib).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine2D(pub Mat);

impl Default for Affine2D {
    fn default() -> Self {
        Affine2D(IDENTITY)
    }
}

impl Affine2D {
    pub fn identity() -> Self {
        Affine2D(IDENTITY)
    }
    pub fn from_values(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Self {
        Affine2D([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])
    }
    pub fn translate(mut self, tx: f64, ty: f64) -> Self {
        self.0[0][2] += tx;
        self.0[1][2] += ty;
        self
    }
    pub fn scale(mut self, sx: f64, sy: f64) -> Self {
        self.0[0][0] *= sx;
        self.0[0][1] *= sx;
        self.0[0][2] *= sx;
        self.0[1][0] *= sy;
        self.0[1][1] *= sy;
        self.0[1][2] *= sy;
        self
    }
    pub fn rotate(mut self, theta: f64) -> Self {
        let a = theta.cos();
        let b = theta.sin();
        let [[xx, xy, x0], [yx, yy, y0], _] = self.0;
        self.0[0][0] = a * xx - b * yx;
        self.0[0][1] = a * xy - b * yy;
        self.0[0][2] = a * x0 - b * y0;
        self.0[1][0] = b * xx + a * yx;
        self.0[1][1] = b * xy + a * yy;
        self.0[1][2] = b * x0 + a * y0;
        self
    }
    pub fn rotate_deg(self, degrees: f64) -> Self {
        self.rotate(degrees.to_radians_py())
    }
    pub fn skew(mut self, x_shear: f64, y_shear: f64) -> Self {
        let rx = x_shear.tan();
        let ry = y_shear.tan();
        let [[xx, xy, x0], [yx, yy, y0], _] = self.0;
        self.0[0][0] += rx * yx;
        self.0[0][1] += rx * yy;
        self.0[0][2] += rx * y0;
        self.0[1][0] += ry * xx;
        self.0[1][1] += ry * xy;
        self.0[1][2] += ry * x0;
        self
    }
    pub fn matrix(&self) -> Mat {
        self.0
    }
}

/// Python's `math.radians` (x * (pi / 180)).
pub trait PyRadians {
    fn to_radians_py(self) -> f64;
}

impl PyRadians for f64 {
    #[inline]
    fn to_radians_py(self) -> f64 {
        // CPython: `return PyFloat_FromDouble(x * degToRad)` with
        // degToRad = Py_MATH_PI / 180.0
        self * (std::f64::consts::PI / 180.0)
    }
}

// ---------------------------------------------------------------------------
// Bbox

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bbox {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl Bbox {
    pub const fn new(x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        Bbox { x0, y0, x1, y1 }
    }
    pub fn unit() -> Self {
        Bbox::new(0.0, 0.0, 1.0, 1.0)
    }
    pub fn null() -> Self {
        Bbox::new(f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY)
    }
    pub fn from_bounds(x0: f64, y0: f64, w: f64, h: f64) -> Self {
        Bbox::new(x0, y0, x0 + w, y0 + h)
    }
    pub fn from_extents(x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        Bbox::new(x0, y0, x1, y1)
    }
    #[inline]
    pub fn width(&self) -> f64 {
        self.x1 - self.x0
    }
    #[inline]
    pub fn height(&self) -> f64 {
        self.y1 - self.y0
    }
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        (self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0)
    }
    pub fn xmin(&self) -> f64 {
        self.x0.min(self.x1)
    }
    pub fn xmax(&self) -> f64 {
        self.x0.max(self.x1)
    }
    pub fn ymin(&self) -> f64 {
        self.y0.min(self.y1)
    }
    pub fn ymax(&self) -> f64 {
        self.y0.max(self.y1)
    }
    pub fn intervalx(&self) -> (f64, f64) {
        (self.x0, self.x1)
    }
    pub fn intervaly(&self) -> (f64, f64) {
        (self.y0, self.y1)
    }
    pub fn translated(&self, tx: f64, ty: f64) -> Self {
        Bbox::new(self.x0 + tx, self.y0 + ty, self.x1 + tx, self.y1 + ty)
    }
    pub fn padded(&self, w: f64, h: f64) -> Self {
        Bbox::new(self.x0 + -w, self.y0 + -h, self.x1 + w, self.y1 + h)
    }
    pub fn is_finite(&self) -> bool {
        self.x0.is_finite() && self.y0.is_finite() && self.x1.is_finite() && self.y1.is_finite()
    }
    /// `Bbox.union`: elementwise numpy min/max over x/ymin/max.
    pub fn union(bboxes: &[Bbox]) -> Bbox {
        let mut x0 = f64::NAN;
        let mut x1 = f64::NAN;
        let mut y0 = f64::NAN;
        let mut y1 = f64::NAN;
        for (i, b) in bboxes.iter().enumerate() {
            if i == 0 {
                x0 = b.xmin();
                x1 = b.xmax();
                y0 = b.ymin();
                y1 = b.ymax();
            } else {
                x0 = np_min(x0, b.xmin());
                x1 = np_max(x1, b.xmax());
                y0 = np_min(y0, b.ymin());
                y1 = np_max(y1, b.ymax());
            }
        }
        Bbox::new(x0, y0, x1, y1)
    }
    /// `BboxBase.transformed`.
    pub fn transformed(&self, t: &Transform) -> Bbox {
        let ll = t.transform_point(self.x0, self.y0);
        let ul = t.transform_point(self.x0, self.y1);
        let lr = t.transform_point(self.x1, self.y0);
        Bbox::new(ll.0, ll.1, lr.0, ul.1)
    }
    /// `TransformedBbox(self, t).get_points()`.
    pub fn transformed_bbox(&self, t: &Transform) -> Bbox {
        let pts = [
            t.transform_point(self.x0, self.y0),
            t.transform_point(self.x1, self.y0),
            t.transform_point(self.x0, self.y1),
            t.transform_point(self.x1, self.y1),
        ];
        let fill = |v: f64| if v.is_nan() { v } else { v };
        let mut xs = (py_min(pts.iter().map(|p| fill(p.0))), py_max(pts.iter().map(|p| fill(p.0))));
        if self.x0 > self.x1 {
            xs = (xs.1, xs.0);
        }
        let mut ys = (py_min(pts.iter().map(|p| fill(p.1))), py_max(pts.iter().map(|p| fill(p.1))));
        if self.y0 > self.y1 {
            ys = (ys.1, ys.0);
        }
        Bbox::new(xs.0, ys.0, xs.1, ys.1)
    }
    pub fn overlaps(&self, other: &Bbox) -> bool {
        let (ax1, ay1, ax2, ay2) = (self.xmin(), self.ymin(), self.xmax(), self.ymax());
        let (bx1, by1, bx2, by2) = (other.xmin(), other.ymin(), other.xmax(), other.ymax());
        ax1 <= bx2 && bx1 <= ax2 && ay1 <= by2 && by1 <= ay2
    }
    /// `Bbox.intersection`.
    pub fn intersection(a: &Bbox, b: &Bbox) -> Option<Bbox> {
        let x0 = a.xmin().max(b.xmin());
        let x1 = a.xmax().min(b.xmax());
        let y0 = a.ymin().max(b.ymin());
        let y1 = a.ymax().min(b.ymax());
        if x0 <= x1 && y0 <= y1 {
            Some(Bbox::new(x0, y0, x1, y1))
        } else {
            None
        }
    }
    pub fn contains(&self, x: f64, y: f64) -> bool {
        let (x0, x1) = (self.x0.min(self.x1), self.x0.max(self.x1));
        let (y0, y1) = (self.y0.min(self.y1), self.y0.max(self.y1));
        x0 <= x && x <= x1 && y0 <= y && y <= y1
    }
}

/// numpy `min` of two values (NaN-propagating).
#[inline]
pub fn np_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if b < a {
        b
    } else {
        a
    }
}

#[inline]
pub fn np_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if b > a {
        b
    } else {
        a
    }
}

/// Python builtin `min` over floats (first element wins ties, NaN quirks).
pub fn py_min<I: Iterator<Item = f64>>(mut it: I) -> f64 {
    let mut m = it.next().unwrap_or(f64::NAN);
    for v in it {
        if v < m {
            m = v;
        }
    }
    m
}

pub fn py_max<I: Iterator<Item = f64>>(mut it: I) -> f64 {
    let mut m = it.next().unwrap_or(f64::NAN);
    for v in it {
        if v > m {
            m = v;
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Scale transforms (non-affine, 1-D)

#[derive(Clone, Debug, PartialEq)]
pub enum ScaleTransform {
    Log { base: f64, clip: bool },
    InvertedLog { base: f64 },
    Symlog { base: f64, linthresh: f64, linscale: f64 },
    InvertedSymlog { base: f64, linthresh: f64, linscale: f64 },
    Logit { clip: bool },
    Logistic,
    AsinH { a: f64 },
    InvertedAsinH { a: f64 },
}

impl ScaleTransform {
    pub fn apply(&self, v: f64) -> f64 {
        match *self {
            ScaleTransform::Log { base, clip } => {
                let out = if base == std::f64::consts::E {
                    v.ln()
                } else if base == 2.0 {
                    v.log2()
                } else if base == 10.0 {
                    v.log10()
                } else {
                    v.ln() / base.ln()
                };
                if clip && v <= 0.0 {
                    -1000.0
                } else {
                    out
                }
            }
            ScaleTransform::InvertedLog { base } => base.powf(v),
            ScaleTransform::Symlog { base, linthresh, linscale } => {
                let linscale_adj = linscale / (1.0 - 1.0 / base);
                let log_base = base.ln();
                let abs_a = v.abs();
                if abs_a <= linthresh {
                    v * linscale_adj
                } else {
                    let sign = if v < 0.0 { -1.0 } else if v > 0.0 { 1.0 } else { 0.0 };
                    sign * linthresh * (linscale_adj + (abs_a / linthresh).ln() / log_base)
                }
            }
            ScaleTransform::InvertedSymlog { base, linthresh, linscale } => {
                let linscale_adj = linscale / (1.0 - 1.0 / base);
                let invlinthresh = linthresh * linscale_adj;
                let abs_a = v.abs();
                if abs_a <= invlinthresh {
                    v / linscale_adj
                } else {
                    let sign = if v < 0.0 { -1.0 } else if v > 0.0 { 1.0 } else { 0.0 };
                    sign * linthresh * base.powf(abs_a / linthresh - linscale_adj)
                }
            }
            ScaleTransform::Logit { clip } => {
                let out = (v / (1.0 - v)).log10();
                if clip {
                    if v <= 0.0 {
                        return -1000.0;
                    }
                    if 1.0 <= v {
                        return 1000.0;
                    }
                }
                out
            }
            ScaleTransform::Logistic => 1.0 / (1.0 + 10f64.powf(-v)),
            ScaleTransform::AsinH { a } => a * (v / a).asinh(),
            ScaleTransform::InvertedAsinH { a } => a * (v / a).sinh(),
        }
    }

    pub fn inverted(&self) -> ScaleTransform {
        match *self {
            ScaleTransform::Log { base, .. } => ScaleTransform::InvertedLog { base },
            ScaleTransform::InvertedLog { base } => ScaleTransform::Log { base, clip: true },
            ScaleTransform::Symlog { base, linthresh, linscale } => {
                ScaleTransform::InvertedSymlog { base, linthresh, linscale }
            }
            ScaleTransform::InvertedSymlog { base, linthresh, linscale } => {
                ScaleTransform::Symlog { base, linthresh, linscale }
            }
            ScaleTransform::Logit { .. } => ScaleTransform::Logistic,
            ScaleTransform::Logistic => ScaleTransform::Logit { clip: true },
            ScaleTransform::AsinH { a } => ScaleTransform::InvertedAsinH { a },
            ScaleTransform::InvertedAsinH { a } => ScaleTransform::AsinH { a },
        }
    }
}

// ---------------------------------------------------------------------------
// Generic transform tree

#[derive(Clone, Debug, PartialEq)]
pub enum Transform {
    /// Any affine transform, with its matrix already evaluated.
    Affine(Mat),
    /// `transScale`-like blend of per-axis 1-D transforms (None = identity).
    Separable(Option<ScaleTransform>, Option<ScaleTransform>),
    /// `CompositeGenericTransform(a, b)`: a, then b.
    Composite(Box<Transform>, Box<Transform>),
    /// `BlendedGenericTransform(x, y)`.
    Blended(Box<Transform>, Box<Transform>),
}

impl Transform {
    pub fn identity() -> Transform {
        Transform::Affine(IDENTITY)
    }

    pub fn affine(m: Mat) -> Transform {
        Transform::Affine(m)
    }

    /// `BboxTransformTo(bbox)`.
    pub fn bbox_to(b: &Bbox) -> Transform {
        let (outl, outb, outw, outh) = b.bounds();
        Transform::Affine([[outw, 0.0, outl], [0.0, outh, outb], [0.0, 0.0, 1.0]])
    }

    /// `BboxTransformFrom(bbox)`.
    pub fn bbox_from(b: &Bbox) -> Transform {
        let (inl, inb, inw, inh) = b.bounds();
        let x_scale = 1.0 / inw;
        let y_scale = 1.0 / inh;
        Transform::Affine([[x_scale, 0.0, -inl * x_scale], [0.0, y_scale, -inb * y_scale], [0.0, 0.0, 1.0]])
    }

    /// `ScaledTranslation(xt, yt, scale_trans)`.
    pub fn scaled_translation(xt: f64, yt: f64, scale_trans: &Transform) -> Transform {
        let (tx, ty) = scale_trans.transform_point(xt, yt);
        let mut m = IDENTITY;
        m[0][2] = tx;
        m[1][2] = ty;
        Transform::Affine(m)
    }

    pub fn is_affine(&self) -> bool {
        matches!(self, Transform::Affine(_))
            || matches!(self, Transform::Separable(None, None))
            || match self {
                Transform::Composite(a, b) => a.is_affine() && b.is_affine(),
                Transform::Blended(x, y) => x.is_affine() && y.is_affine(),
                _ => false,
            }
    }

    /// `a + b` (composite_transform_factory semantics for our representation).
    pub fn then(&self, b: &Transform) -> Transform {
        match (self, b) {
            (Transform::Affine(ma), Transform::Affine(mb)) => Transform::Affine(dot(mb, ma)),
            _ => Transform::Composite(Box::new(self.clone()), Box::new(b.clone())),
        }
    }

    /// `blended_transform_factory(x, y)`.
    pub fn blend(x: &Transform, y: &Transform) -> Transform {
        match (x, y) {
            (Transform::Affine(mx), Transform::Affine(my)) => {
                if mx == my {
                    Transform::Affine(*mx)
                } else {
                    Transform::Affine([mx[0], my[1], [0.0, 0.0, 1.0]])
                }
            }
            _ => Transform::Blended(Box::new(x.clone()), Box::new(y.clone())),
        }
    }

    pub fn get_affine(&self) -> Mat {
        match self {
            Transform::Affine(m) => *m,
            Transform::Separable(..) => IDENTITY,
            Transform::Composite(a, b) => {
                if !b.is_affine() {
                    b.get_affine()
                } else {
                    dot(&b.get_affine(), &a.get_affine())
                }
            }
            Transform::Blended(x, y) => {
                if x == y {
                    x.get_affine()
                } else {
                    let mx = x.get_affine();
                    let my = y.get_affine();
                    [mx[0], my[1], [0.0, 0.0, 1.0]]
                }
            }
        }
    }

    /// `transform_non_affine` for one point.
    pub fn transform_non_affine(&self, x: f64, y: f64) -> (f64, f64) {
        match self {
            Transform::Affine(_) => (x, y),
            Transform::Separable(sx, sy) => {
                let xo = sx.as_ref().map(|s| s.apply(x)).unwrap_or(x);
                let yo = sy.as_ref().map(|s| s.apply(y)).unwrap_or(y);
                (xo, yo)
            }
            Transform::Composite(a, b) => {
                if a.is_affine() && b.is_affine() {
                    (x, y)
                } else if !a.is_affine() && b.is_affine() {
                    a.transform_non_affine(x, y)
                } else {
                    let (x1, y1) = a.transform_point(x, y);
                    b.transform_non_affine(x1, y1)
                }
            }
            Transform::Blended(tx, ty) => {
                if tx.is_affine() && ty.is_affine() {
                    return (x, y);
                }
                if tx == ty {
                    return tx.transform_non_affine(x, y);
                }
                let xo = tx.transform_non_affine(x, y).0;
                let yo = ty.transform_non_affine(x, y).1;
                (xo, yo)
            }
        }
    }

    /// `transform` for one point: affine(non_affine(p)).
    pub fn transform_point(&self, x: f64, y: f64) -> (f64, f64) {
        match self {
            Transform::Affine(m) => affine_point(m, x, y),
            _ => {
                let (xn, yn) = self.transform_non_affine(x, y);
                affine_point(&self.get_affine(), xn, yn)
            }
        }
    }

    pub fn transform_points(&self, pts: &[[f64; 2]]) -> Vec<[f64; 2]> {
        match self {
            Transform::Affine(m) => pts
                .iter()
                .map(|p| {
                    let (x, y) = affine_point(m, p[0], p[1]);
                    [x, y]
                })
                .collect(),
            _ => {
                let a = self.get_affine();
                pts.iter()
                    .map(|p| {
                        let (xn, yn) = self.transform_non_affine(p[0], p[1]);
                        let (x, y) = affine_point(&a, xn, yn);
                        [x, y]
                    })
                    .collect()
            }
        }
    }

    pub fn transform_points_non_affine(&self, pts: &[[f64; 2]]) -> Vec<[f64; 2]> {
        if self.is_affine() {
            return pts.to_vec();
        }
        pts.iter()
            .map(|p| {
                let (x, y) = self.transform_non_affine(p[0], p[1]);
                [x, y]
            })
            .collect()
    }

    /// `inverted()` for affine transforms and scale-separable ones.
    pub fn inverted(&self) -> Transform {
        match self {
            Transform::Affine(m) => Transform::Affine(inv(m)),
            Transform::Separable(sx, sy) => {
                Transform::Separable(sx.as_ref().map(|s| s.inverted()), sy.as_ref().map(|s| s.inverted()))
            }
            Transform::Composite(a, b) => b.inverted().then(&a.inverted()),
            Transform::Blended(x, y) => Transform::Blended(Box::new(x.inverted()), Box::new(y.inverted())),
        }
    }
}
