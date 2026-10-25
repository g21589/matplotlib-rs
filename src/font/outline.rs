//! FreeType-style outlines (26.6 fixed point) and FreeType's fixed-point
//! helpers (`FT_MulFix`, `FT_DivFix`, `FT_MulDiv`).

#[derive(Clone, Debug, Default)]
pub struct Outline {
    /// Points in 26.6 fixed point.
    pub points: Vec<(i64, i64)>,
    /// FreeType tags: bit 0 = on curve, bit 1 = cubic (for off-curve).
    pub tags: Vec<u8>,
    /// Index of the last point of each contour.
    pub contours: Vec<u16>,
    /// `FT_OUTLINE_OVERLAP`.
    pub overlap: bool,
    /// `FT_OUTLINE_EVEN_ODD_FILL`.
    pub even_odd: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BBox {
    pub x_min: i64,
    pub y_min: i64,
    pub x_max: i64,
    pub y_max: i64,
}

impl Outline {
    /// `FT_Outline_Translate`.
    pub fn translate(&mut self, dx: i64, dy: i64) {
        for p in &mut self.points {
            p.0 = p.0.wrapping_add(dx);
            p.1 = p.1.wrapping_add(dy);
        }
    }

    /// `FT_Outline_Transform` with a 16.16 matrix (xx, xy, yx, yy).
    pub fn transform(&mut self, m: &FtMatrix) {
        for p in &mut self.points {
            *p = m.apply(p.0, p.1);
        }
    }

    /// `FT_Outline_Get_CBox`.
    pub fn cbox(&self) -> BBox {
        if self.points.is_empty() {
            return BBox::default();
        }
        let mut b = BBox { x_min: self.points[0].0, x_max: self.points[0].0, y_min: self.points[0].1, y_max: self.points[0].1 };
        for &(x, y) in &self.points[1..] {
            if x < b.x_min {
                b.x_min = x;
            }
            if x > b.x_max {
                b.x_max = x;
            }
            if y < b.y_min {
                b.y_min = y;
            }
            if y > b.y_max {
                b.y_max = y;
            }
        }
        b
    }

    /// Append another outline (used for composite glyphs).
    pub fn append(&mut self, other: &Outline) {
        let base = self.points.len() as u16;
        self.points.extend_from_slice(&other.points);
        self.tags.extend_from_slice(&other.tags);
        self.contours.extend(other.contours.iter().map(|c| c + base));
    }
}

/// `FT_Matrix` (16.16 fixed point).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FtMatrix {
    pub xx: i64,
    pub xy: i64,
    pub yx: i64,
    pub yy: i64,
}

impl FtMatrix {
    pub const IDENTITY: FtMatrix = FtMatrix { xx: 0x10000, xy: 0, yx: 0, yy: 0x10000 };

    /// `FT_Vector_Transform`.
    #[inline]
    pub fn apply(&self, x: i64, y: i64) -> (i64, i64) {
        let xz = mul_fix(x, self.xx) + mul_fix(y, self.xy);
        let yz = mul_fix(x, self.yx) + mul_fix(y, self.yy);
        (xz, yz)
    }

    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }
}

/// `FT_MulFix` (64-bit version): `(a * b + 0x8000 + (ab >> 63)) >> 16`.
#[inline(always)]
pub fn mul_fix(a: i64, b: i64) -> i64 {
    let ab = (a as i128 * b as i128) as i64;
    let ab = ab.wrapping_add(0x8000 + (ab >> 63));
    ab >> 16
}

#[inline(always)]
fn move_sign(v: i64, s: &mut i32) -> u64 {
    if v < 0 {
        *s = -*s;
        (v as i128).unsigned_abs() as u64
    } else {
        v as u64
    }
}

/// `FT_DivFix`.
pub fn div_fix(a: i64, b: i64) -> i64 {
    let mut s = 1;
    let a = move_sign(a, &mut s);
    let b = move_sign(b, &mut s);
    let q: u64 = if b > 0 { ((a << 16) + (b >> 1)) / b } else { 0x7FFFFFFF };
    let q = q as i64;
    if s < 0 {
        q.wrapping_neg()
    } else {
        q
    }
}

/// `FT_MulDiv`.
pub fn mul_div(a: i64, b: i64, c: i64) -> i64 {
    let mut s = 1;
    let a = move_sign(a, &mut s);
    let b = move_sign(b, &mut s);
    let c = move_sign(c, &mut s);
    let d: u64 = if c > 0 { (a.wrapping_mul(b) + (c >> 1)) / c } else { 0x7FFFFFFF };
    let d = d as i64;
    if s < 0 {
        d.wrapping_neg()
    } else {
        d
    }
}

/// `FT_Hypot` (uses `FT_Vector_Length`).
pub fn hypot(x: i64, y: i64) -> i64 {
    // FreeType computes this with a CORDIC-based vector length; components
    // of composite glyph scales are in 16.16 and the result is only used for
    // rarely-set SCALED_COMPONENT_OFFSET flags.  A float computation rounds
    // to the same value for all realistic inputs.
    ((x as f64).hypot(y as f64)).round() as i64
}
