//! Port of `agg::trans_affine` and `agg::conv_transform`.

use super::basics::{is_vertex, VertexSource};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransAffine {
    pub sx: f64,
    pub shy: f64,
    pub shx: f64,
    pub sy: f64,
    pub tx: f64,
    pub ty: f64,
}

impl Default for TransAffine {
    fn default() -> Self {
        Self::identity()
    }
}

impl TransAffine {
    pub const fn identity() -> Self {
        TransAffine { sx: 1.0, shy: 0.0, shx: 0.0, sy: 1.0, tx: 0.0, ty: 0.0 }
    }

    /// Construct from Agg's (sx, shy, shx, sy, tx, ty) ordering.
    pub const fn new(sx: f64, shy: f64, shx: f64, sy: f64, tx: f64, ty: f64) -> Self {
        TransAffine { sx, shy, shx, sy, tx, ty }
    }

    /// Construct from a matplotlib 3x3 row-major matrix
    /// `[[a, c, e], [b, d, f], [0, 0, 1]]`.
    pub fn from_matrix(m: &[[f64; 3]; 3]) -> Self {
        TransAffine { sx: m[0][0], shx: m[0][1], tx: m[0][2], shy: m[1][0], sy: m[1][1], ty: m[1][2] }
    }

    pub fn scaling(x: f64, y: f64) -> Self {
        TransAffine::new(x, 0.0, 0.0, y, 0.0, 0.0)
    }

    pub fn translation(x: f64, y: f64) -> Self {
        TransAffine::new(1.0, 0.0, 0.0, 1.0, x, y)
    }

    pub fn rotation(a: f64) -> Self {
        let (s, c) = (a.sin(), a.cos());
        TransAffine::new(c, s, -s, c, 0.0, 0.0)
    }

    /// `this *= m` (apply `m` after `this`).
    #[inline]
    pub fn multiply(&mut self, m: &TransAffine) -> &mut Self {
        let t0 = self.sx * m.sx + self.shy * m.shx;
        let t2 = self.shx * m.sx + self.sy * m.shx;
        let t4 = self.tx * m.sx + self.ty * m.shx + m.tx;
        self.shy = self.sx * m.shy + self.shy * m.sy;
        self.sy = self.shx * m.shy + self.sy * m.sy;
        self.ty = self.tx * m.shy + self.ty * m.sy + m.ty;
        self.sx = t0;
        self.shx = t2;
        self.tx = t4;
        self
    }

    pub fn then(mut self, m: &TransAffine) -> Self {
        self.multiply(m);
        self
    }

    pub fn determinant_reciprocal(&self) -> f64 {
        1.0 / (self.sx * self.sy - self.shy * self.shx)
    }

    pub fn invert(&mut self) -> &mut Self {
        let d = self.determinant_reciprocal();
        let t0 = self.sy * d;
        self.sy = self.sx * d;
        self.shy = -self.shy * d;
        self.shx = -self.shx * d;
        let t4 = -self.tx * t0 - self.ty * self.shx;
        self.ty = -self.tx * self.shy - self.ty * self.sy;
        self.sx = t0;
        self.tx = t4;
        self
    }

    #[inline(always)]
    pub fn transform(&self, x: &mut f64, y: &mut f64) {
        let tmp = *x;
        *x = tmp * self.sx + *y * self.shx + self.tx;
        *y = tmp * self.shy + *y * self.sy + self.ty;
    }

    #[inline(always)]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (x * self.sx + y * self.shx + self.tx, x * self.shy + y * self.sy + self.ty)
    }

    pub fn is_identity(&self) -> bool {
        *self == Self::identity()
    }
}

/// `agg::conv_transform`.
pub struct ConvTransform<S: VertexSource> {
    pub source: S,
    pub trans: TransAffine,
}

impl<S: VertexSource> ConvTransform<S> {
    pub fn new(source: S, trans: TransAffine) -> Self {
        ConvTransform { source, trans }
    }
}

impl<S: VertexSource> VertexSource for ConvTransform<S> {
    #[inline]
    fn rewind(&mut self, path_id: u32) {
        self.source.rewind(path_id);
    }
    #[inline]
    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let cmd = self.source.vertex(x, y);
        if is_vertex(cmd) {
            self.trans.transform(x, y);
        }
        cmd
    }
}
