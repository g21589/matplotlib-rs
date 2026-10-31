//! Pixel formats: RGBA32 with matplotlib's `fixed_blender_rgba_plain`,
//! an 8-bit alpha mask (`amask_no_clip_gray8`), and `pixfmt_amask_adaptor`.

use super::basics::uround;

/// Floating point colour (`agg::rgba`).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Rgba {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl Rgba {
    pub const fn new(r: f64, g: f64, b: f64, a: f64) -> Self {
        Rgba { r, g, b, a }
    }
    pub fn to_rgba8(&self) -> Rgba8 {
        Rgba8 {
            r: uround(self.r * 255.0) as u8,
            g: uround(self.g * 255.0) as u8,
            b: uround(self.b * 255.0) as u8,
            a: uround(self.a * 255.0) as u8,
        }
    }
}

/// 8-bit RGBA colour (`agg::rgba8`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Rgba8 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba8 {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Rgba8 { r, g, b, a }
    }
    #[inline(always)]
    pub fn is_transparent(&self) -> bool {
        self.a == 0
    }
    #[inline(always)]
    pub fn is_opaque(&self) -> bool {
        self.a == 255
    }
}

/// `rgba8::multiply` / `mult_cover`.
#[inline(always)]
pub fn mult_cover(a: u8, b: u8) -> u8 {
    let t = a as u32 * b as u32 + 128;
    (((t >> 8) + t) >> 8) as u8
}

/// `gray8::lerp` / `rgba8::lerp`.
#[inline(always)]
pub fn lerp(p: u8, q: u8, a: u8) -> u8 {
    let t: i32 = (q as i32 - p as i32) * a as i32 + 128 - (p > q) as i32;
    (p as i32 + (((t >> 8) + t) >> 8)) as u8
}

/// matplotlib's `fixed_blender_rgba_plain::blend_pix` (without cover), with
/// Agg's integer divisions (kept for reference and tests).
#[inline(always)]
#[cfg_attr(not(test), allow(dead_code))]
fn blend_pix_plain_div(p: &mut [u8], cr: u8, cg: u8, cb: u8, alpha: u8) {
    if alpha == 0 {
        return;
    }
    // calc_type is unsigned 32-bit in Agg; intermediate results always fit
    // into the non-negative i64 range so plain i64 arithmetic is identical.
    let a0 = p[3] as i64;
    let r = p[0] as i64 * a0;
    let g = p[1] as i64 * a0;
    let b = p[2] as i64 * a0;
    let alpha = alpha as i64;
    let a = ((alpha + a0) << 8) - alpha * a0;
    p[3] = (a >> 8) as u8;
    p[0] = ((((cr as i64) << 8) - r) * alpha + (r << 8)).wrapping_div(a) as u8;
    p[1] = ((((cg as i64) << 8) - g) * alpha + (g << 8)).wrapping_div(a) as u8;
    p[2] = ((((cb as i64) << 8) - b) * alpha + (b << 8)).wrapping_div(a) as u8;
}

/// The division of `blend_pix_plain` for the last (alpha, destination
/// alpha) pair: within a span the pair rarely changes (an opaque destination
/// keeps it constant), so most pixels need no division.
///
/// The numerators are below 2^26 and the divisors d at most 2^16, so with
/// m = ceil(2^42 / d), floor(n / d) == (n * m) >> 42 exactly (Granlund &
/// Montgomery: m * d - 2^42 < d <= 2^16).
#[derive(Clone, Copy)]
pub struct Recip {
    key: u32,
    m: u64,
}

impl Recip {
    #[inline(always)]
    pub const fn new() -> Self {
        Recip { key: u32::MAX, m: 0 }
    }
}

#[inline(always)]
fn magic(d: u32) -> u64 {
    ((1u64 << 42) + d as u64 - 1) / d as u64
}

#[inline(always)]
fn quotient(n: u32, m: u64) -> u8 {
    ((n as u64 * m) >> 42) as u8
}

/// matplotlib's `fixed_blender_rgba_plain::blend_pix` (without cover):
/// identical results to `blend_pix_plain_div`.
#[inline(always)]
fn blend_pix_plain(p: &mut [u8], cr: u8, cg: u8, cb: u8, alpha: u8, rc: &mut Recip) {
    if alpha == 0 {
        return;
    }
    let a0 = p[3] as u32;
    let alpha = alpha as u32;
    let a = ((alpha + a0) << 8) - alpha * a0;
    let key = (alpha << 8) | a0;
    if key != rc.key {
        rc.key = key;
        rc.m = magic(a);
    }
    let r = rc.m;
    // ((c << 8) - x) * alpha + (x << 8) with x = channel * a0, rearranged
    // into its two non-negative terms
    let inv = 256 - alpha;
    let ca = alpha << 8;
    p[3] = (a >> 8) as u8;
    p[0] = quotient(cr as u32 * ca + p[0] as u32 * a0 * inv, r);
    p[1] = quotient(cg as u32 * ca + p[1] as u32 * a0 * inv, r);
    p[2] = quotient(cb as u32 * ca + p[2] as u32 * a0 * inv, r);
}

/// `blend_pix_plain` of one colour and alpha over a row: pixels with an
/// opaque destination (the usual case) use constants computed once.
#[inline(always)]
fn blend_row_plain(row: &mut [u8], c: &Rgba8, alpha: u8) {
    if alpha == 0 {
        return;
    }
    let al = alpha as u32;
    // For a0 == 255 the divisor d = 65280 + alpha is constant; the
    // numerators are below 2^26, so (Granlund & Montgomery) floor(n / d) ==
    // (n * ceil(2^42 / d)) >> 42 exactly.
    let a = ((al + 255) << 8) - al * 255;
    let m = magic(a);
    let ca = al << 8;
    let k = 255 * (256 - al);
    let (nr, ng, nb) = (c.r as u32 * ca, c.g as u32 * ca, c.b as u32 * ca);
    let a_out = (a >> 8) as u8;
    let q = |n: u32| quotient(n, m);
    let mut rc = Recip::new();
    for p in row.chunks_exact_mut(4) {
        if p[3] == 255 {
            p[0] = q(nr + p[0] as u32 * k);
            p[1] = q(ng + p[1] as u32 * k);
            p[2] = q(nb + p[2] as u32 * k);
            p[3] = a_out;
        } else {
            blend_pix_plain(p, c.r, c.g, c.b, alpha, &mut rc);
        }
    }
}

#[inline(always)]
fn blend_pix_cover(p: &mut [u8], c: &Rgba8, cover: u8, rc: &mut Recip) {
    blend_pix_plain(p, c.r, c.g, c.b, mult_cover(c.a, cover), rc);
}

#[inline(always)]
fn blend_pix(p: &mut [u8], c: &Rgba8, rc: &mut Recip) {
    blend_pix_plain(p, c.r, c.g, c.b, c.a, rc);
}

#[inline(always)]
fn set_pix(p: &mut [u8], c: &Rgba8) {
    p[0] = c.r;
    p[1] = c.g;
    p[2] = c.b;
    p[3] = c.a;
}

#[inline(always)]
fn copy_or_blend_pix_cover(p: &mut [u8], c: &Rgba8, cover: u8, rc: &mut Recip) {
    if !c.is_transparent() {
        if c.is_opaque() && cover == 255 {
            set_pix(p, c);
        } else {
            blend_pix_cover(p, c, cover, rc);
        }
    }
}

#[inline(always)]
fn copy_or_blend_pix(p: &mut [u8], c: &Rgba8, rc: &mut Recip) {
    if !c.is_transparent() {
        if c.is_opaque() {
            set_pix(p, c);
        } else {
            blend_pix(p, c, rc);
        }
    }
}

/// Abstract pixel format used by [`super::renderer::RendererBase`].
pub trait PixFmt {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn copy_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8);
    fn blend_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8, cover: u8);
    fn blend_solid_hspan(&mut self, x: i32, y: i32, len: u32, c: &Rgba8, covers: &[u8]);
    fn blend_color_hspan(&mut self, x: i32, y: i32, len: u32, colors: &[Rgba8], covers: Option<&[u8]>, cover: u8);
}

/// RGBA32 image buffer (top row first) with matplotlib's plain blender.
#[derive(Clone)]
pub struct PixfmtRgba32 {
    pub width: u32,
    pub height: u32,
    pub buf: Vec<u8>,
}

impl PixfmtRgba32 {
    pub fn new(width: u32, height: u32) -> Self {
        PixfmtRgba32 { width, height, buf: vec![0; width as usize * height as usize * 4] }
    }

    #[inline(always)]
    fn offset(&self, x: i32, y: i32) -> usize {
        (y as usize * self.width as usize + x as usize) * 4
    }

    pub fn pixel(&self, x: i32, y: i32) -> Rgba8 {
        let o = self.offset(x, y);
        Rgba8::new(self.buf[o], self.buf[o + 1], self.buf[o + 2], self.buf[o + 3])
    }

    pub fn blend_pixel(&mut self, x: i32, y: i32, c: &Rgba8, cover: u8) {
        let o = self.offset(x, y);
        copy_or_blend_pix_cover(&mut self.buf[o..o + 4], c, cover, &mut Recip::new());
    }
}

impl PixFmt for PixfmtRgba32 {
    #[inline(always)]
    fn width(&self) -> u32 {
        self.width
    }
    #[inline(always)]
    fn height(&self) -> u32 {
        self.height
    }

    fn copy_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8) {
        let o = self.offset(x, y);
        for p in self.buf[o..o + len as usize * 4].chunks_exact_mut(4) {
            set_pix(p, c);
        }
    }

    fn blend_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8, cover: u8) {
        if c.is_transparent() {
            return;
        }
        let o = self.offset(x, y);
        let row = &mut self.buf[o..o + len as usize * 4];
        if c.is_opaque() && cover == 255 {
            for p in row.chunks_exact_mut(4) {
                set_pix(p, c);
            }
        } else if cover == 255 {
            blend_row_plain(row, c, c.a);
        } else {
            blend_row_plain(row, c, mult_cover(c.a, cover));
        }
    }

    fn blend_solid_hspan(&mut self, x: i32, y: i32, len: u32, c: &Rgba8, covers: &[u8]) {
        if c.is_transparent() {
            return;
        }
        let o = self.offset(x, y);
        let row = &mut self.buf[o..o + len as usize * 4];
        let opaque = c.is_opaque();
        let mut rc = Recip::new();
        for (p, &cv) in row.chunks_exact_mut(4).zip(covers.iter()) {
            if opaque && cv == 255 {
                set_pix(p, c);
            } else {
                blend_pix_cover(p, c, cv, &mut rc);
            }
        }
    }

    fn blend_color_hspan(&mut self, x: i32, y: i32, len: u32, colors: &[Rgba8], covers: Option<&[u8]>, cover: u8) {
        let o = self.offset(x, y);
        let row = &mut self.buf[o..o + len as usize * 4];
        let mut rc = Recip::new();
        match covers {
            Some(cvs) => {
                for ((p, c), &cv) in row.chunks_exact_mut(4).zip(colors.iter()).zip(cvs.iter()) {
                    copy_or_blend_pix_cover(p, c, cv, &mut rc);
                }
            }
            None => {
                if cover == 255 {
                    for (p, c) in row.chunks_exact_mut(4).zip(colors.iter()) {
                        copy_or_blend_pix(p, c, &mut rc);
                    }
                } else {
                    for (p, c) in row.chunks_exact_mut(4).zip(colors.iter()) {
                        copy_or_blend_pix_cover(p, c, cover, &mut rc);
                    }
                }
            }
        }
    }
}

/// 8-bit alpha mask buffer (`amask_no_clip_gray8` over a gray8 buffer).
#[derive(Clone)]
pub struct AlphaMask {
    pub width: u32,
    pub height: u32,
    pub buf: Vec<u8>,
}

impl AlphaMask {
    pub fn new(width: u32, height: u32) -> Self {
        AlphaMask { width, height, buf: vec![0; width as usize * height as usize] }
    }

    pub fn clear(&mut self) {
        self.buf.iter_mut().for_each(|v| *v = 0);
    }

    #[inline(always)]
    pub fn pixel(&self, x: i32, y: i32) -> u8 {
        self.buf[y as usize * self.width as usize + x as usize]
    }

    #[inline(always)]
    pub fn combine_pixel(&self, x: i32, y: i32, val: u8) -> u8 {
        ((255u32 + val as u32 * self.pixel(x, y) as u32) >> 8) as u8
    }

    #[inline]
    pub fn fill_hspan(&self, x: i32, y: i32, dst: &mut [u8]) {
        let o = y as usize * self.width as usize + x as usize;
        let n = dst.len();
        dst.copy_from_slice(&self.buf[o..o + n]);
    }

    #[inline]
    pub fn combine_hspan(&self, x: i32, y: i32, dst: &mut [u8]) {
        let o = y as usize * self.width as usize + x as usize;
        let n = dst.len();
        for (d, &m) in dst.iter_mut().zip(self.buf[o..o + n].iter()) {
            *d = ((255u32 + *d as u32 * m as u32) >> 8) as u8;
        }
    }
}

/// Gray8 pixel format used to render clip paths into the alpha mask.
pub struct PixfmtGray8<'a> {
    pub mask: &'a mut AlphaMask,
}

impl<'a> PixfmtGray8<'a> {
    #[inline(always)]
    fn blend_gray(p: &mut u8, v: u8, alpha: u8, cover: u8) {
        *p = lerp(*p, v, mult_cover(alpha, cover));
    }
}

impl<'a> PixFmt for PixfmtGray8<'a> {
    fn width(&self) -> u32 {
        self.mask.width
    }
    fn height(&self) -> u32 {
        self.mask.height
    }
    // For gray8 the colour's `r` channel is used as the gray value `v`.
    fn copy_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8) {
        let o = y as usize * self.mask.width as usize + x as usize;
        for p in &mut self.mask.buf[o..o + len as usize] {
            *p = c.r;
        }
    }
    fn blend_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8, cover: u8) {
        if c.is_transparent() {
            return;
        }
        let o = y as usize * self.mask.width as usize + x as usize;
        let row = &mut self.mask.buf[o..o + len as usize];
        if c.is_opaque() && cover == 255 {
            for p in row {
                *p = c.r;
            }
        } else {
            for p in row {
                Self::blend_gray(p, c.r, c.a, cover);
            }
        }
    }
    fn blend_solid_hspan(&mut self, x: i32, y: i32, len: u32, c: &Rgba8, covers: &[u8]) {
        if c.is_transparent() {
            return;
        }
        let o = y as usize * self.mask.width as usize + x as usize;
        let row = &mut self.mask.buf[o..o + len as usize];
        for (p, &cv) in row.iter_mut().zip(covers.iter()) {
            if c.is_opaque() && cv == 255 {
                *p = c.r;
            } else {
                Self::blend_gray(p, c.r, c.a, cv);
            }
        }
    }
    fn blend_color_hspan(&mut self, x: i32, y: i32, len: u32, colors: &[Rgba8], covers: Option<&[u8]>, cover: u8) {
        let o = y as usize * self.mask.width as usize + x as usize;
        let row = &mut self.mask.buf[o..o + len as usize];
        for (i, p) in row.iter_mut().enumerate() {
            let c = colors[i];
            let cv = covers.map(|c| c[i]).unwrap_or(cover);
            if !c.is_transparent() {
                if c.is_opaque() && cv == 255 {
                    *p = c.r;
                } else {
                    Self::blend_gray(p, c.r, c.a, cv);
                }
            }
        }
    }
}

/// `agg::pixfmt_amask_adaptor<pixfmt, amask_no_clip_gray8>`.
pub struct PixfmtAmaskAdaptor<'a, P: PixFmt> {
    pub pixf: &'a mut P,
    pub mask: &'a AlphaMask,
    span: Vec<u8>,
}

impl<'a, P: PixFmt> PixfmtAmaskAdaptor<'a, P> {
    pub fn new(pixf: &'a mut P, mask: &'a AlphaMask) -> Self {
        PixfmtAmaskAdaptor { pixf, mask, span: Vec::new() }
    }
}

impl<'a, P: PixFmt> PixFmt for PixfmtAmaskAdaptor<'a, P> {
    fn width(&self) -> u32 {
        self.pixf.width()
    }
    fn height(&self) -> u32 {
        self.pixf.height()
    }
    fn copy_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8) {
        self.span.resize(len as usize, 0);
        self.mask.fill_hspan(x, y, &mut self.span);
        let span = std::mem::take(&mut self.span);
        self.pixf.blend_solid_hspan(x, y, len, c, &span);
        self.span = span;
    }
    fn blend_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8, _cover: u8) {
        // Note: Agg ignores `cover` here and starts from a full cover.
        self.span.clear();
        self.span.resize(len as usize, 255);
        self.mask.combine_hspan(x, y, &mut self.span);
        let span = std::mem::take(&mut self.span);
        self.pixf.blend_solid_hspan(x, y, len, c, &span);
        self.span = span;
    }
    fn blend_solid_hspan(&mut self, x: i32, y: i32, len: u32, c: &Rgba8, covers: &[u8]) {
        self.span.clear();
        self.span.extend_from_slice(&covers[..len as usize]);
        self.mask.combine_hspan(x, y, &mut self.span);
        let span = std::mem::take(&mut self.span);
        self.pixf.blend_solid_hspan(x, y, len, c, &span);
        self.span = span;
    }
    fn blend_color_hspan(&mut self, x: i32, y: i32, len: u32, colors: &[Rgba8], covers: Option<&[u8]>, cover: u8) {
        self.span.clear();
        match covers {
            Some(cvs) => {
                self.span.extend_from_slice(&cvs[..len as usize]);
                self.mask.combine_hspan(x, y, &mut self.span);
            }
            None => {
                self.span.resize(len as usize, 0);
                self.mask.fill_hspan(x, y, &mut self.span);
            }
        }
        let span = std::mem::take(&mut self.span);
        self.pixf.blend_color_hspan(x, y, len, colors, Some(&span), cover);
        self.span = span;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The division-free blend equals Agg's for every colour channel,
    /// destination channel, alpha and destination alpha (2^32 cases; run
    /// with `cargo test --release -- --ignored`).
    #[test]
    #[ignore]
    fn plain_blend_exhaustive() {
        for alpha in 1..=255u8 {
            let mut rc = Recip::new();
            for a0 in 0..=255u8 {
                for c in 0..=255u8 {
                    for x in 0..=255u8 {
                        let mut p = [x, x, x, a0];
                        let mut q = p;
                        blend_pix_plain_div(&mut p, c, c, c, alpha);
                        blend_pix_plain(&mut q, c, c, c, alpha, &mut rc);
                        assert_eq!(p, q, "alpha {alpha} a0 {a0} c {c} x {x}");
                    }
                }
            }
        }
    }

    #[test]
    fn row_blend_opaque_exhaustive() {
        for alpha in 0..=255u8 {
            for c in 0..=255u8 {
                let mut row: Vec<u8> = (0..=255u8).flat_map(|x| [x, x, 255 - x, 255]).collect();
                let mut expect = row.clone();
                for p in expect.chunks_exact_mut(4) {
                    blend_pix_plain_div(p, c, 255 - c, c, alpha);
                }
                blend_row_plain(&mut row, &Rgba8::new(c, 255 - c, c, 255), alpha);
                assert_eq!(row, expect, "alpha {alpha} c {c}");
            }
        }
    }

    #[test]
    fn row_blend_sampled() {
        let mut s = 987654321u32;
        let mut next = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            s
        };
        for _ in 0..20000 {
            let n = (next() % 64) as usize + 1;
            let mut row: Vec<u8> = (0..4 * n).map(|_| next() as u8).collect();
            for p in row.chunks_exact_mut(4) {
                if next() % 3 != 0 {
                    p[3] = 255;
                }
            }
            let [r, g, b, a] = next().to_le_bytes();
            let c = Rgba8::new(r, g, b, a);
            let alpha = next() as u8;
            let mut expect = row.clone();
            for p in expect.chunks_exact_mut(4) {
                blend_pix_plain_div(p, c.r, c.g, c.b, alpha);
            }
            blend_row_plain(&mut row, &c, alpha);
            assert_eq!(row, expect);
        }
    }

    #[test]
    fn plain_blend_sampled() {
        let mut rc = Recip::new();
        let mut s = 12345u32;
        for _ in 0..2_000_000 {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let [a, b, c, d] = s.to_le_bytes();
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let [e, f, g, h] = s.to_le_bytes();
            let mut p = [a, b, c, d];
            let mut q = p;
            blend_pix_plain_div(&mut p, e, f, g, h);
            blend_pix_plain(&mut q, e, f, g, h, &mut rc);
            assert_eq!(p, q);
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    #[test]
    #[ignore]
    fn row_blend_speed() {
        let mut row = vec![255u8; 4 * 1000];
        let c = Rgba8::new(10, 200, 30, 255);
        let t = std::time::Instant::now();
        for i in 0..20000 {
            blend_row_plain(&mut row, &c, (i % 250 + 3) as u8);
        }
        let dt = t.elapsed().as_secs_f64();
        eprintln!("blend_row_plain: {:.3} ns/pixel", dt / 20e6 * 1e9);
        let mut rc = Recip::new();
        let t = std::time::Instant::now();
        for i in 0..20000 {
            for p in row.chunks_exact_mut(4) {
                blend_pix_plain_div(p, 10, 200, 30, (i % 250 + 3) as u8);
            }
        }
        let dt = t.elapsed().as_secs_f64();
        eprintln!("division blend: {:.3} ns/pixel {}", dt / 20e6 * 1e9, row[5]);
        let t = std::time::Instant::now();
        for i in 0..20000 {
            for p in row.chunks_exact_mut(4) {
                blend_pix_plain(p, 10, 200, 30, (i % 250 + 3) as u8, &mut rc);
            }
        }
        let dt = t.elapsed().as_secs_f64();
        eprintln!("cached blend: {:.3} ns/pixel {}", dt / 20e6 * 1e9, row[5]);
    }
}
