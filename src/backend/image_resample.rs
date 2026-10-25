//! Port of matplotlib's `_image_resample.h` (Agg image resampling).
//!
//! Resamples a gray (one channel) or RGBA image of `u8`, `u16`, `f32` or
//! `f64` values into an output image through an affine transform or a
//! lookup mesh, blending into the output exactly like matplotlib's
//! `resample<color_type>` does: same span generators (nearest neighbour,
//! general LUT filter, affine resampling), same interpolators, image
//! accessors, filter LUTs and pixel blenders, and the same fixed-point and
//! float/double arithmetic.

use crate::agg::basics::{PATH_CMD_LINE_TO, PATH_CMD_MOVE_TO};
use crate::agg::basics::CLOSEPOLY;
use crate::agg::rasterizer::RasterizerScanlineAa;
use crate::agg::scanline::{Scanline, ScanlineSpans, ScanlineU8};
use crate::agg::trans_affine::TransAffine;

const SUBPIXEL_SHIFT: i32 = 8;
const SUBPIXEL_SCALE: i32 = 1 << SUBPIXEL_SHIFT;
const SUBPIXEL_MASK: i32 = SUBPIXEL_SCALE - 1;
const FILTER_SHIFT: i32 = 14;
const FILTER_SCALE: i32 = 1 << FILTER_SHIFT;

/// C++ `int(double)` as compiled for x86-64 (`cvttsd2si`): NaN and
/// out-of-range values give `INT_MIN`.
#[inline(always)]
fn trunc_i32(v: f64) -> i32 {
    if v.is_nan() || v >= 2147483648.0 || v <= -2147483649.0 {
        i32::MIN
    } else {
        v as i32
    }
}

/// `agg::iround`.
#[inline(always)]
fn iround(v: f64) -> i32 {
    trunc_i32(if v < 0.0 { v - 0.5 } else { v + 0.5 })
}

/// `agg::uround` (returned as `int`, as Agg does).
#[inline(always)]
fn uround(v: f64) -> i32 {
    let t = v + 0.5;
    if t.is_nan() || t >= 4294967296.0 || t <= -1.0 {
        0
    } else {
        (t as u32) as i32
    }
}

// ---------------------------------------------------------------------------
// Channel arithmetic (gray8/rgba8, gray16/rgba16, gray32/rgba32, gray64/rgba64)

pub trait Channel: Copy + Default + PartialOrd + 'static {
    /// `long_type`, the filter accumulator.
    type Acc: Copy + PartialOrd;
    const FULL: Self;
    const ACC_ZERO: Self::Acc;
    const ACC_FULL: Self::Acc;
    /// `fg += v * weight`.
    fn acc_mul_add(acc: Self::Acc, v: Self, w: i32) -> Self::Acc;
    /// `color_type::downshift(fg, image_filter_shift)`.
    fn acc_downshift(acc: Self::Acc) -> Self::Acc;
    /// `fg /= total_weight`.
    fn acc_div(acc: Self::Acc, total: i32) -> Self::Acc;
    /// `(value_type)fg`.
    fn from_acc(acc: Self::Acc) -> Self;
    fn mult_cover(a: Self, cover: u8) -> Self;
    fn lerp(p: Self, q: Self, a: Self) -> Self;
    fn prelerp(p: Self, q: Self, a: Self) -> Self;
    fn is_transparent(a: Self) -> bool;
    fn is_opaque(a: Self) -> bool;
    /// `span_conv_alpha`: `value_type(calc_type(a) * alpha)`.
    fn scale_alpha(a: Self, alpha: f64) -> Self;
}

#[inline(always)]
fn multiply8(a: u8, b: u8) -> u8 {
    let t = a as u32 * b as u32 + 128;
    (((t >> 8) + t) >> 8) as u8
}

#[inline(always)]
fn multiply16(a: u16, b: u16) -> u16 {
    let t = a as u32 * b as u32 + 32768;
    (((t >> 16) + t) >> 16) as u16
}

impl Channel for u8 {
    type Acc = i32;
    const FULL: u8 = 255;
    const ACC_ZERO: i32 = 0;
    const ACC_FULL: i32 = 255;
    #[inline(always)]
    fn acc_mul_add(acc: i32, v: u8, w: i32) -> i32 {
        acc.wrapping_add((v as i32).wrapping_mul(w))
    }
    #[inline(always)]
    fn acc_downshift(acc: i32) -> i32 {
        acc >> FILTER_SHIFT
    }
    #[inline(always)]
    fn acc_div(acc: i32, total: i32) -> i32 {
        acc.checked_div(total).unwrap_or(0)
    }
    #[inline(always)]
    fn from_acc(acc: i32) -> u8 {
        acc as u8
    }
    #[inline(always)]
    fn mult_cover(a: u8, cover: u8) -> u8 {
        multiply8(a, cover)
    }
    #[inline(always)]
    fn lerp(p: u8, q: u8, a: u8) -> u8 {
        let t: i32 = (q as i32 - p as i32) * a as i32 + 128 - (p > q) as i32;
        (p as i32 + (((t >> 8) + t) >> 8)) as u8
    }
    #[inline(always)]
    fn prelerp(p: u8, q: u8, a: u8) -> u8 {
        (p as i32 + q as i32 - multiply8(p, a) as i32) as u8
    }
    #[inline(always)]
    fn is_transparent(a: u8) -> bool {
        a == 0
    }
    #[inline(always)]
    fn is_opaque(a: u8) -> bool {
        a == 255
    }
    #[inline(always)]
    fn scale_alpha(a: u8, alpha: f64) -> u8 {
        (a as f64 * alpha) as u8
    }
}

impl Channel for u16 {
    type Acc = i64;
    const FULL: u16 = 65535;
    const ACC_ZERO: i64 = 0;
    const ACC_FULL: i64 = 65535;
    #[inline(always)]
    fn acc_mul_add(acc: i64, v: u16, w: i32) -> i64 {
        acc.wrapping_add((v as i32).wrapping_mul(w) as i64)
    }
    #[inline(always)]
    fn acc_downshift(acc: i64) -> i64 {
        acc >> FILTER_SHIFT
    }
    #[inline(always)]
    fn acc_div(acc: i64, total: i32) -> i64 {
        acc.checked_div(total as i64).unwrap_or(0)
    }
    #[inline(always)]
    fn from_acc(acc: i64) -> u16 {
        acc as u16
    }
    #[inline(always)]
    fn mult_cover(a: u16, cover: u8) -> u16 {
        multiply16(a, ((cover as u16) << 8) | cover as u16)
    }
    #[inline(always)]
    fn lerp(p: u16, q: u16, a: u16) -> u16 {
        let t: i32 = (q as i32 - p as i32)
            .wrapping_mul(a as i32)
            .wrapping_add(32768 - (p > q) as i32);
        (p as i32).wrapping_add(((t >> 16).wrapping_add(t)) >> 16) as u16
    }
    #[inline(always)]
    fn prelerp(p: u16, q: u16, a: u16) -> u16 {
        (p as i32 + q as i32 - multiply16(p, a) as i32) as u16
    }
    #[inline(always)]
    fn is_transparent(a: u16) -> bool {
        a == 0
    }
    #[inline(always)]
    fn is_opaque(a: u16) -> bool {
        a == 65535
    }
    #[inline(always)]
    fn scale_alpha(a: u16, alpha: f64) -> u16 {
        (a as f64 * alpha) as u16
    }
}

impl Channel for f32 {
    type Acc = f64;
    const FULL: f32 = 1.0;
    const ACC_ZERO: f64 = 0.0;
    const ACC_FULL: f64 = 1.0;
    #[inline(always)]
    fn acc_mul_add(acc: f64, v: f32, w: i32) -> f64 {
        // `weight * value` is evaluated in float.
        acc + (v * w as f32) as f64
    }
    #[inline(always)]
    fn acc_downshift(acc: f64) -> f64 {
        acc / FILTER_SCALE as f64
    }
    #[inline(always)]
    fn acc_div(acc: f64, total: i32) -> f64 {
        acc / total as f64
    }
    #[inline(always)]
    fn from_acc(acc: f64) -> f32 {
        acc as f32
    }
    #[inline(always)]
    fn mult_cover(a: f32, cover: u8) -> f32 {
        a * cover as f32 / 255.0
    }
    #[inline(always)]
    fn lerp(p: f32, q: f32, a: f32) -> f32 {
        (1.0 - a) * p + a * q
    }
    #[inline(always)]
    fn prelerp(p: f32, q: f32, a: f32) -> f32 {
        (1.0 - a) * p + q
    }
    #[inline(always)]
    fn is_transparent(a: f32) -> bool {
        a <= 0.0
    }
    #[inline(always)]
    fn is_opaque(a: f32) -> bool {
        a >= 1.0
    }
    #[inline(always)]
    fn scale_alpha(a: f32, alpha: f64) -> f32 {
        (a as f64 * alpha) as f32
    }
}

impl Channel for f64 {
    type Acc = f64;
    const FULL: f64 = 1.0;
    const ACC_ZERO: f64 = 0.0;
    const ACC_FULL: f64 = 1.0;
    #[inline(always)]
    fn acc_mul_add(acc: f64, v: f64, w: i32) -> f64 {
        acc + v * w as f64
    }
    #[inline(always)]
    fn acc_downshift(acc: f64) -> f64 {
        acc / FILTER_SCALE as f64
    }
    #[inline(always)]
    fn acc_div(acc: f64, total: i32) -> f64 {
        acc / total as f64
    }
    #[inline(always)]
    fn from_acc(acc: f64) -> f64 {
        acc
    }
    #[inline(always)]
    fn mult_cover(a: f64, cover: u8) -> f64 {
        a * cover as f64 / 255.0
    }
    #[inline(always)]
    fn lerp(p: f64, q: f64, a: f64) -> f64 {
        (1.0 - a) * p + a * q
    }
    #[inline(always)]
    fn prelerp(p: f64, q: f64, a: f64) -> f64 {
        (1.0 - a) * p + q
    }
    #[inline(always)]
    fn is_transparent(a: f64) -> bool {
        a <= 0.0
    }
    #[inline(always)]
    fn is_opaque(a: f64) -> bool {
        a >= 1.0
    }
    #[inline(always)]
    fn scale_alpha(a: f64, alpha: f64) -> f64 {
        a * alpha
    }
}

// ---------------------------------------------------------------------------
// Color types: gray (`blender_gray`), rgba8 (`fixed_blender_rgba_plain`) and
// the other rgba types (`blender_rgba_pre`).  A generated color is stored as
// `[c0, c1, c2, a]`; gray colors use `c0` for the value.

pub trait ColorType {
    type V: Channel;
    /// Channels per pixel in the image buffers.
    const N: usize;
    /// `copy_or_blend_pix(p, c, cover)`.
    fn copy_or_blend(p: &mut [Self::V], c: &[Self::V; 4], cover: u8);
}

pub struct Gray<V>(std::marker::PhantomData<V>);
pub struct RgbaPre<V>(std::marker::PhantomData<V>);
pub struct Rgba8Plain;

impl<V: Channel> ColorType for Gray<V> {
    type V = V;
    const N: usize = 1;
    #[inline(always)]
    fn copy_or_blend(p: &mut [V], c: &[V; 4], cover: u8) {
        let a = c[3];
        if !V::is_transparent(a) {
            if V::is_opaque(a) && cover == 255 {
                p[0] = c[0];
            } else {
                p[0] = V::lerp(p[0], c[0], V::mult_cover(a, cover));
            }
        }
    }
}

impl<V: Channel> ColorType for RgbaPre<V> {
    type V = V;
    const N: usize = 4;
    #[inline(always)]
    fn copy_or_blend(p: &mut [V], c: &[V; 4], cover: u8) {
        if !V::is_transparent(c[3]) {
            if V::is_opaque(c[3]) && cover == 255 {
                p[..4].copy_from_slice(c);
            } else {
                let r = V::mult_cover(c[0], cover);
                let g = V::mult_cover(c[1], cover);
                let b = V::mult_cover(c[2], cover);
                let a = V::mult_cover(c[3], cover);
                p[0] = V::prelerp(p[0], r, a);
                p[1] = V::prelerp(p[1], g, a);
                p[2] = V::prelerp(p[2], b, a);
                p[3] = V::prelerp(p[3], a, a);
            }
        }
    }
}

impl ColorType for Rgba8Plain {
    type V = u8;
    const N: usize = 4;
    #[inline(always)]
    fn copy_or_blend(p: &mut [u8], c: &[u8; 4], cover: u8) {
        if c[3] != 0 {
            if c[3] == 255 && cover == 255 {
                p[..4].copy_from_slice(c);
            } else {
                let alpha = multiply8(c[3], cover);
                if alpha == 0 {
                    return;
                }
                // calc_type is unsigned 32-bit; the results always fit.
                let a0 = p[3] as i64;
                let r = p[0] as i64 * a0;
                let g = p[1] as i64 * a0;
                let b = p[2] as i64 * a0;
                let alpha = alpha as i64;
                let a = ((alpha + a0) << 8) - alpha * a0;
                p[3] = (a >> 8) as u8;
                p[0] = ((((c[0] as i64) << 8) - r) * alpha + (r << 8)).wrapping_div(a) as u8;
                p[1] = ((((c[1] as i64) << 8) - g) * alpha + (g << 8)).wrapping_div(a) as u8;
                p[2] = ((((c[2] as i64) << 8) - b) * alpha + (b << 8)).wrapping_div(a) as u8;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Filters (`agg_image_filters.h`, with MPL_FIX_AGG_IMAGE_FILTER_LUT_BUGS)

pub const NEAREST: u32 = 0;
pub const BILINEAR: u32 = 1;
pub const BICUBIC: u32 = 2;
pub const SPLINE16: u32 = 3;
pub const SPLINE36: u32 = 4;
pub const HANNING: u32 = 5;
pub const HAMMING: u32 = 6;
pub const HERMITE: u32 = 7;
pub const KAISER: u32 = 8;
pub const QUADRIC: u32 = 9;
pub const CATROM: u32 = 10;
pub const GAUSSIAN: u32 = 11;
pub const BESSEL: u32 = 12;
pub const MITCHELL: u32 = 13;
pub const SINC: u32 = 14;
pub const LANCZOS: u32 = 15;
pub const BLACKMAN: u32 = 16;

const PI: f64 = std::f64::consts::PI;

/// `agg::besj`.
fn besj(x: f64, n: i32) -> f64 {
    if n < 0 {
        return 0.0;
    }
    let d = 1e-6;
    let mut b = 0.0;
    if x.abs() <= d {
        if n != 0 {
            return 0.0;
        }
        return 1.0;
    }
    let mut b1 = 0.0;
    let mut m1 = x.abs() as i32 + 6;
    if x.abs() > 5.0 {
        m1 = (1.4 * x + 60.0 / x).abs() as i32;
    }
    let mut m2 = (n as f64 + 2.0 + x.abs() / 4.0) as i32;
    if m1 > m2 {
        m2 = m1;
    }
    loop {
        let mut c3 = 0.0;
        let mut c2 = 1e-30;
        let mut c4 = 0.0;
        let mut m8 = 1;
        if m2 / 2 * 2 == m2 {
            m8 = -1;
        }
        let imax = m2 - 2;
        for i in 1..=imax {
            let c6 = (2 * (m2 - i)) as f64 * c2 / x - c3;
            c3 = c2;
            c2 = c6;
            if m2 - i - 1 == n {
                b = c6;
            }
            m8 = -m8;
            if m8 > 0 {
                c4 += 2.0 * c6;
            }
        }
        let c6 = 2.0 * c2 / x - c3;
        if n == 0 {
            b = c6;
        }
        c4 += c6;
        b /= c4;
        if (b - b1).abs() < d {
            return b;
        }
        b1 = b;
        m2 += 3;
    }
}

fn bessel_i0(x: f64) -> f64 {
    let epsilon = 1e-12;
    let mut sum = 1.0;
    let y = x * x / 4.0;
    let mut t = y;
    let mut i: i32 = 2;
    while t > epsilon {
        sum += t;
        t *= y / (i * i) as f64;
        i += 1;
    }
    sum
}

fn pow3(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        x * x * x
    }
}

/// The filter function and radius for an interpolation kind.
fn filter_fn(interpolation: u32, radius: f64) -> (f64, Box<dyn Fn(f64) -> f64>) {
    match interpolation {
        HANNING => (1.0, Box::new(|x: f64| 0.5 + 0.5 * (PI * x).cos())),
        HAMMING => (1.0, Box::new(|x: f64| 0.54 + 0.46 * (PI * x).cos())),
        HERMITE => (1.0, Box::new(|x: f64| (2.0 * x - 3.0) * x * x + 1.0)),
        BICUBIC => (
            2.0,
            Box::new(|x: f64| {
                (1.0 / 6.0) * (pow3(x + 2.0) - 4.0 * pow3(x + 1.0) + 6.0 * pow3(x) - 4.0 * pow3(x - 1.0))
            }),
        ),
        SPLINE16 => (
            2.0,
            Box::new(|x: f64| {
                if x < 1.0 {
                    ((x - 9.0 / 5.0) * x - 1.0 / 5.0) * x + 1.0
                } else {
                    ((-1.0 / 3.0 * (x - 1.0) + 4.0 / 5.0) * (x - 1.0) - 7.0 / 15.0) * (x - 1.0)
                }
            }),
        ),
        SPLINE36 => (
            3.0,
            Box::new(|x: f64| {
                if x < 1.0 {
                    ((13.0 / 11.0 * x - 453.0 / 209.0) * x - 3.0 / 209.0) * x + 1.0
                } else if x < 2.0 {
                    ((-6.0 / 11.0 * (x - 1.0) + 270.0 / 209.0) * (x - 1.0) - 156.0 / 209.0) * (x - 1.0)
                } else {
                    ((1.0 / 11.0 * (x - 2.0) - 45.0 / 209.0) * (x - 2.0) + 26.0 / 209.0) * (x - 2.0)
                }
            }),
        ),
        KAISER => {
            let a = 6.33;
            let i0a = 1.0 / bessel_i0(a);
            (1.0, Box::new(move |x: f64| bessel_i0(a * (1. - x * x).sqrt()) * i0a))
        }
        QUADRIC => (
            1.5,
            Box::new(|x: f64| {
                if x < 0.5 {
                    0.75 - x * x
                } else if x < 1.5 {
                    let t = x - 1.5;
                    0.5 * t * t
                } else {
                    0.0
                }
            }),
        ),
        CATROM => (
            2.0,
            Box::new(|x: f64| {
                if x < 1.0 {
                    0.5 * (2.0 + x * x * (-5.0 + x * 3.0))
                } else if x < 2.0 {
                    0.5 * (4.0 + x * (-8.0 + x * (5.0 - x)))
                } else {
                    0.
                }
            }),
        ),
        GAUSSIAN => (2.0, Box::new(|x: f64| (-2.0 * x * x).exp() * (2.0 / PI).sqrt())),
        BESSEL => (
            3.2383,
            Box::new(|x: f64| if x == 0.0 { PI / 4.0 } else { besj(PI * x, 1) / (2.0 * x) }),
        ),
        MITCHELL => {
            let (b, c) = (1.0 / 3.0, 1.0 / 3.0);
            let p0 = (6.0 - 2.0 * b) / 6.0;
            let p2 = (-18.0 + 12.0 * b + 6.0 * c) / 6.0;
            let p3 = (12.0 - 9.0 * b - 6.0 * c) / 6.0;
            let q0 = (8.0 * b + 24.0 * c) / 6.0;
            let q1 = (-12.0 * b - 48.0 * c) / 6.0;
            let q2 = (6.0 * b + 30.0 * c) / 6.0;
            let q3 = (-b - 6.0 * c) / 6.0;
            (
                2.0,
                Box::new(move |x: f64| {
                    if x < 1.0 {
                        p0 + x * x * (p2 + x * p3)
                    } else if x < 2.0 {
                        q0 + x * (q1 + x * (q2 + x * q3))
                    } else {
                        0.0
                    }
                }),
            )
        }
        SINC => {
            let r = if radius < 2.0 { 2.0 } else { radius };
            (
                r,
                Box::new(|x: f64| {
                    if x == 0.0 {
                        return 1.0;
                    }
                    let x = x * PI;
                    x.sin() / x
                }),
            )
        }
        LANCZOS => {
            let r = if radius < 2.0 { 2.0 } else { radius };
            (
                r,
                Box::new(move |x: f64| {
                    if x == 0.0 {
                        return 1.0;
                    }
                    if x > r {
                        return 0.0;
                    }
                    let x = x * PI;
                    let xr = x / r;
                    (x.sin() / x) * (xr.sin() / xr)
                }),
            )
        }
        BLACKMAN => {
            let r = if radius < 2.0 { 2.0 } else { radius };
            (
                r,
                Box::new(move |x: f64| {
                    if x == 0.0 {
                        return 1.0;
                    }
                    if x > r {
                        return 0.0;
                    }
                    let x = x * PI;
                    let xr = x / r;
                    (x.sin() / x) * (0.42 + 0.5 * xr.cos() + 0.08 * (2.0 * xr).cos())
                }),
            )
        }
        // BILINEAR (NEAREST never gets here)
        _ => (1.0, Box::new(|x: f64| 1.0 - x)),
    }
}

/// `agg::image_filter_lut`.
pub struct FilterLut {
    diameter: u32,
    start: i32,
    weights: Vec<i16>,
}

impl FilterLut {
    pub fn new(interpolation: u32, radius: f64, normalization: bool) -> Self {
        let (r, f) = filter_fn(interpolation, radius);
        let diameter = (r.ceil() as u32) * 2;
        let start = -((diameter / 2) as i32 - 1);
        let size = (diameter << SUBPIXEL_SHIFT) as usize;
        let mut weights = vec![0i16; size];
        let pivot = ((diameter << (SUBPIXEL_SHIFT - 1)) - 1) as usize;
        for i in 0..=pivot {
            let x = i as f64 / SUBPIXEL_SCALE as f64;
            let y = f(x);
            let w = iround(y * FILTER_SCALE as f64) as i16;
            weights[pivot + i] = w;
            weights[pivot - i] = w;
        }
        let end = size - 1;
        weights[end] = iround(f((diameter / 2) as f64) * FILTER_SCALE as f64) as i16;
        let mut lut = FilterLut { diameter, start, weights };
        if normalization {
            lut.normalize();
        }
        lut
    }

    fn normalize(&mut self) {
        let d = self.diameter as usize;
        let s = SUBPIXEL_SCALE as usize;
        let w = &mut self.weights;
        let mut flip = 1;
        for i in 0..s {
            loop {
                let mut sum: i32 = 0;
                for j in 0..d {
                    sum += w[j * s + i] as i32;
                }
                if sum == FILTER_SCALE {
                    break;
                }
                let k = FILTER_SCALE as f64 / sum as f64;
                sum = 0;
                for j in 0..d {
                    let v = iround(w[j * s + i] as f64 * k) as i16;
                    w[j * s + i] = v;
                    sum += v as i32;
                }
                sum -= FILTER_SCALE;
                let inc: i32 = if sum > 0 { -1 } else { 1 };
                let mut j = 0;
                while j < d && sum != 0 {
                    flip ^= 1;
                    let idx = if flip != 0 { d / 2 + j / 2 } else { d / 2 - j / 2 };
                    let v = w[idx * s + i] as i32;
                    if v < FILTER_SCALE {
                        w[idx * s + i] = (v + inc) as i16;
                        sum += inc;
                    }
                    j += 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Interpolators

trait Interpolator {
    fn begin(&mut self, x: f64, y: f64, len: u32);
    fn coordinates(&self) -> (i32, i32);
    fn next(&mut self);
}

/// `agg::dda2_line_interpolator` (forward-adjusted).
#[derive(Clone, Copy, Default)]
struct Dda2 {
    cnt: i32,
    lft: i32,
    rem: i32,
    modv: i32,
    y: i32,
}

impl Dda2 {
    fn new(y1: i32, y2: i32, count: i32) -> Self {
        let cnt = if count <= 0 { 1 } else { count };
        let d = y2.wrapping_sub(y1);
        let mut lft = d.wrapping_div(cnt);
        let mut rem = d.wrapping_rem(cnt);
        let mut modv = rem;
        if modv <= 0 {
            modv = modv.wrapping_add(count);
            rem = rem.wrapping_add(count);
            lft = lft.wrapping_sub(1);
        }
        modv = modv.wrapping_sub(count);
        Dda2 { cnt, lft, rem, modv, y: y1 }
    }

    #[inline(always)]
    fn inc(&mut self) {
        self.modv = self.modv.wrapping_add(self.rem);
        self.y = self.y.wrapping_add(self.lft);
        if self.modv > 0 {
            self.modv = self.modv.wrapping_sub(self.cnt);
            self.y = self.y.wrapping_add(1);
        }
    }
}

/// `agg::span_interpolator_linear<>` (MPL_FIX_AGG_INTERPOLATION_ENDPOINT_BUG).
struct LinearInterp {
    trans: TransAffine,
    li_x: Dda2,
    li_y: Dda2,
}

impl LinearInterp {
    fn new(trans: TransAffine) -> Self {
        LinearInterp { trans, li_x: Dda2::default(), li_y: Dda2::default() }
    }
}

impl Interpolator for LinearInterp {
    fn begin(&mut self, x: f64, y: f64, len: u32) {
        let len = len.wrapping_sub(1);
        let (tx, ty) = self.trans.apply(x, y);
        let x1 = iround(tx * SUBPIXEL_SCALE as f64);
        let y1 = iround(ty * SUBPIXEL_SCALE as f64);
        let (tx, ty) = self.trans.apply(x + len as f64, y);
        let x2 = iround(tx * SUBPIXEL_SCALE as f64);
        let y2 = iround(ty * SUBPIXEL_SCALE as f64);
        self.li_x = Dda2::new(x1, x2, len as i32);
        self.li_y = Dda2::new(y1, y2, len as i32);
    }
    #[inline(always)]
    fn coordinates(&self) -> (i32, i32) {
        (self.li_x.y, self.li_y.y)
    }
    #[inline(always)]
    fn next(&mut self) {
        self.li_x.inc();
        self.li_y.inc();
    }
}

/// matplotlib's `accurate_interpolator_affine_nn`.
struct AccurateNnInterp {
    trans: TransAffine,
    len: u32,
    cur: u32,
    stx1: f64,
    sty1: f64,
    stx2: f64,
    sty2: f64,
}

impl AccurateNnInterp {
    fn new(trans: TransAffine) -> Self {
        AccurateNnInterp { trans, len: 0, cur: 0, stx1: 0.0, sty1: 0.0, stx2: 0.0, sty2: 0.0 }
    }
}

impl Interpolator for AccurateNnInterp {
    fn begin(&mut self, x: f64, y: f64, len: u32) {
        self.len = len.wrapping_sub(1);
        self.cur = 0;
        let (x1, y1) = self.trans.apply(x, y);
        self.stx1 = x1 * SUBPIXEL_SCALE as f64;
        self.sty1 = y1 * SUBPIXEL_SCALE as f64;
        let (x2, y2) = self.trans.apply(x + self.len as f64, y);
        self.stx2 = x2 * SUBPIXEL_SCALE as f64;
        self.sty2 = y2 * SUBPIXEL_SCALE as f64;
    }
    #[inline(always)]
    fn coordinates(&self) -> (i32, i32) {
        if self.cur == self.len {
            (trunc_i32(self.stx2), trunc_i32(self.sty2))
        } else {
            let cur = self.cur as f64;
            let len = self.len as f64;
            (
                trunc_i32(self.stx1 + (self.stx2 - self.stx1) * cur / len + 1e-8),
                trunc_i32(self.sty1 + (self.sty2 - self.sty1) * cur / len + 1e-8),
            )
        }
    }
    #[inline(always)]
    fn next(&mut self) {
        self.cur = self.cur.wrapping_add(1);
    }
}

/// `span_interpolator_adaptor<span_interpolator_linear<>, lookup_distortion>`.
struct DistortedInterp<'a> {
    base: LinearInterp,
    mesh: Option<&'a [f64]>,
    out_width: i32,
    out_height: i32,
    offset: f64,
}

impl<'a> Interpolator for DistortedInterp<'a> {
    fn begin(&mut self, x: f64, y: f64, len: u32) {
        self.base.begin(x, y, len);
    }
    #[inline(always)]
    fn coordinates(&self) -> (i32, i32) {
        let (mut x, mut y) = self.base.coordinates();
        if let Some(mesh) = self.mesh {
            let dx = x as f64 / SUBPIXEL_SCALE as f64;
            let dy = y as f64 / SUBPIXEL_SCALE as f64;
            if dx >= 0.0 && dx < self.out_width as f64 && dy >= 0.0 && dy < self.out_height as f64 {
                let idx = ((dy as i32 * self.out_width + dx as i32) * 2) as usize;
                x = trunc_i32(mesh[idx] * SUBPIXEL_SCALE as f64 + self.offset + 1e-8);
                y = trunc_i32(mesh[idx + 1] * SUBPIXEL_SCALE as f64 + self.offset + 1e-8);
            }
        }
        (x, y)
    }
    #[inline(always)]
    fn next(&mut self) {
        self.base.next();
    }
}

// ---------------------------------------------------------------------------
// Image accessors

struct Image<'a, V> {
    data: &'a [V],
    width: i32,
    height: i32,
}

/// `agg::wrap_mode_reflect`.
struct WrapReflect {
    size: u32,
    size2: u32,
    add: u32,
    value: u32,
}

impl WrapReflect {
    fn new(size: u32) -> Self {
        let size2 = size.wrapping_mul(2);
        let add = if size2 == 0 { 0 } else { size2.wrapping_mul(0x3FFFFFFF / size2) };
        WrapReflect { size, size2, add, value: 0 }
    }
    #[inline(always)]
    fn call(&mut self, v: i32) -> u32 {
        self.value = (v as u32).wrapping_add(self.add) % self.size2;
        if self.value >= self.size {
            return self.size2 - self.value - 1;
        }
        self.value
    }
    #[inline(always)]
    fn inc(&mut self) -> u32 {
        self.value += 1;
        if self.value >= self.size2 {
            self.value = 0;
        }
        if self.value >= self.size {
            return self.size2 - self.value - 1;
        }
        self.value
    }
}

/// `agg::image_accessor_wrap<pixfmt, wrap_mode_reflect, wrap_mode_reflect>`;
/// returns element offsets of pixels.
struct AccessorWrap {
    width: usize,
    n: usize,
    row: usize,
    x: i32,
    wrap_x: WrapReflect,
    wrap_y: WrapReflect,
}

impl AccessorWrap {
    fn new(width: i32, height: i32, n: usize) -> Self {
        AccessorWrap {
            width: width as usize,
            n,
            row: 0,
            x: 0,
            wrap_x: WrapReflect::new(width as u32),
            wrap_y: WrapReflect::new(height as u32),
        }
    }
    #[inline(always)]
    fn span(&mut self, x: i32, y: i32) -> usize {
        self.x = x;
        self.row = self.wrap_y.call(y) as usize * self.width;
        (self.row + self.wrap_x.call(x) as usize) * self.n
    }
    #[inline(always)]
    fn next_x(&mut self) -> usize {
        (self.row + self.wrap_x.inc() as usize) * self.n
    }
    #[inline(always)]
    fn next_y(&mut self) -> usize {
        self.row = self.wrap_y.inc() as usize * self.width;
        (self.row + self.wrap_x.call(self.x) as usize) * self.n
    }
}

// ---------------------------------------------------------------------------
// Span generators

trait SpanGen<V: Channel> {
    fn prepare(&mut self) {}
    fn generate(&mut self, span: &mut [[V; 4]], x: i32, y: i32);
}

/// `span_image_filter_{gray,rgba}_nn` over `image_accessor_clip` with a
/// transparent (zero) background.
struct NnGen<'a, V, I> {
    src: Image<'a, V>,
    n: usize,
    interp: I,
}

impl<'a, V: Channel, I: Interpolator> SpanGen<V> for NnGen<'a, V, I> {
    fn generate(&mut self, span: &mut [[V; 4]], x: i32, y: i32) {
        self.interp.begin(x as f64 + 0.5, y as f64 + 0.5, span.len() as u32);
        for s in span.iter_mut() {
            let (ix, iy) = self.interp.coordinates();
            let px = ix >> SUBPIXEL_SHIFT;
            let py = iy >> SUBPIXEL_SHIFT;
            let inside = py >= 0 && py < self.src.height && px >= 0 && px < self.src.width;
            if self.n == 1 {
                s[0] = if inside {
                    self.src.data[(py as usize) * self.src.width as usize + px as usize]
                } else {
                    V::default()
                };
                s[3] = V::FULL;
            } else if inside {
                let o = ((py as usize) * self.src.width as usize + px as usize) * 4;
                s.copy_from_slice(&self.src.data[o..o + 4]);
            } else {
                *s = [V::default(); 4];
            }
            self.interp.next();
        }
    }
}

#[inline(always)]
fn clamp_rgba<V: Channel>(fg: &mut [V::Acc; 4]) {
    for v in fg.iter_mut() {
        if *v < V::ACC_ZERO {
            *v = V::ACC_ZERO;
        }
    }
    if fg[3] > V::ACC_FULL {
        fg[3] = V::ACC_FULL;
    }
    for c in 0..3 {
        if fg[c] > fg[3] {
            fg[c] = fg[3];
        }
    }
}

/// `span_image_filter_{gray,rgba}` (general LUT filter) over the reflecting
/// accessor.
struct FilterGen<'a, V, I> {
    src: Image<'a, V>,
    n: usize,
    acc: AccessorWrap,
    interp: I,
    lut: &'a FilterLut,
}

impl<'a, V: Channel, I: Interpolator> SpanGen<V> for FilterGen<'a, V, I> {
    fn generate(&mut self, span: &mut [[V; 4]], x: i32, y: i32) {
        self.interp.begin(x as f64 + 0.5, y as f64 + 0.5, span.len() as u32);
        let diameter = self.lut.diameter;
        let start = self.lut.start;
        let w = &self.lut.weights;
        let data = self.src.data;
        let n = self.n;
        for s in span.iter_mut() {
            let (mut x, mut y) = self.interp.coordinates();
            x = x.wrapping_sub(SUBPIXEL_SCALE / 2);
            y = y.wrapping_sub(SUBPIXEL_SCALE / 2);
            let x_lr = x >> SUBPIXEL_SHIFT;
            let y_lr = y >> SUBPIXEL_SHIFT;
            let mut fg = [V::ACC_ZERO; 4];
            let x_fract = x & SUBPIXEL_MASK;
            let mut y_count = diameter;
            let mut y_hr = SUBPIXEL_MASK - (y & SUBPIXEL_MASK);
            let mut p = self.acc.span(x_lr.wrapping_add(start), y_lr.wrapping_add(start));
            loop {
                let mut x_count = diameter;
                let weight_y = w[y_hr as usize] as i32;
                let mut x_hr = SUBPIXEL_MASK - x_fract;
                loop {
                    let weight = (weight_y * w[x_hr as usize] as i32 + FILTER_SCALE / 2) >> FILTER_SHIFT;
                    for c in 0..n {
                        fg[c] = V::acc_mul_add(fg[c], data[p + c], weight);
                    }
                    x_count -= 1;
                    if x_count == 0 {
                        break;
                    }
                    x_hr += SUBPIXEL_SCALE;
                    p = self.acc.next_x();
                }
                y_count -= 1;
                if y_count == 0 {
                    break;
                }
                y_hr += SUBPIXEL_SCALE;
                p = self.acc.next_y();
            }
            if n == 1 {
                // MPL_DISABLE_AGG_GRAY_CLIPPING
                s[0] = V::from_acc(V::acc_downshift(fg[0]));
                s[3] = V::FULL;
            } else {
                for v in fg.iter_mut() {
                    *v = V::acc_downshift(*v);
                }
                clamp_rgba::<V>(&mut fg);
                for c in 0..4 {
                    s[c] = V::from_acc(fg[c]);
                }
            }
            self.interp.next();
        }
    }
}

/// `span_image_resample_{gray,rgba}_affine`.
struct ResampleAffineGen<'a, V> {
    src: Image<'a, V>,
    n: usize,
    acc: AccessorWrap,
    interp: LinearInterp,
    lut: &'a FilterLut,
    rx: i32,
    ry: i32,
    rx_inv: i32,
    ry_inv: i32,
}

impl<'a, V: Channel> SpanGen<V> for ResampleAffineGen<'a, V> {
    fn prepare(&mut self) {
        let scale_limit = 200.0;
        let t = &self.interp.trans;
        let mut scale_x = (t.sx * t.sx + t.shx * t.shx).sqrt();
        let mut scale_y = (t.shy * t.shy + t.sy * t.sy).sqrt();
        if scale_x * scale_y > scale_limit {
            scale_x = scale_x * scale_limit / (scale_x * scale_y);
            scale_y = scale_y * scale_limit / (scale_x * scale_y);
        }
        if scale_x < 1.0 {
            scale_x = 1.0;
        }
        if scale_y < 1.0 {
            scale_y = 1.0;
        }
        if scale_x > scale_limit {
            scale_x = scale_limit;
        }
        if scale_y > scale_limit {
            scale_y = scale_limit;
        }
        // blur_x = blur_y = 1
        scale_x *= 1.0;
        scale_y *= 1.0;
        if scale_x < 1.0 {
            scale_x = 1.0;
        }
        if scale_y < 1.0 {
            scale_y = 1.0;
        }
        self.rx = uround(scale_x * SUBPIXEL_SCALE as f64);
        self.rx_inv = uround(1.0 / scale_x * SUBPIXEL_SCALE as f64);
        self.ry = uround(scale_y * SUBPIXEL_SCALE as f64);
        self.ry_inv = uround(1.0 / scale_y * SUBPIXEL_SCALE as f64);
    }

    fn generate(&mut self, span: &mut [[V; 4]], x: i32, y: i32) {
        self.interp.begin(x as f64 + 0.5, y as f64 + 0.5, span.len() as u32);
        let diameter = self.lut.diameter as i32;
        let filter_scale = diameter << SUBPIXEL_SHIFT;
        let radius_x = diameter.wrapping_mul(self.rx) >> 1;
        let radius_y = diameter.wrapping_mul(self.ry) >> 1;
        let w = &self.lut.weights;
        let data = self.src.data;
        let n = self.n;
        let (rx_inv, ry_inv) = (self.rx_inv, self.ry_inv);
        for s in span.iter_mut() {
            let (mut x, mut y) = self.interp.coordinates();
            x = x.wrapping_add(SUBPIXEL_SCALE / 2 - radius_x);
            y = y.wrapping_add(SUBPIXEL_SCALE / 2 - radius_y);
            let mut fg = [V::ACC_ZERO; 4];
            let y_lr = y >> SUBPIXEL_SHIFT;
            let mut y_hr = ((SUBPIXEL_MASK - (y & SUBPIXEL_MASK)) * ry_inv) >> SUBPIXEL_SHIFT;
            let mut total_weight: i32 = 0;
            let x_lr = x >> SUBPIXEL_SHIFT;
            let x_hr2 = ((SUBPIXEL_MASK - (x & SUBPIXEL_MASK)) * rx_inv) >> SUBPIXEL_SHIFT;
            let mut p = self.acc.span(x_lr, y_lr);
            loop {
                let weight_y = w[y_hr as usize] as i32;
                let mut x_hr = x_hr2;
                loop {
                    let weight = (weight_y * w[x_hr as usize] as i32 + FILTER_SCALE / 2) >> FILTER_SHIFT;
                    for c in 0..n {
                        fg[c] = V::acc_mul_add(fg[c], data[p + c], weight);
                    }
                    total_weight += weight;
                    x_hr += rx_inv;
                    if x_hr >= filter_scale {
                        break;
                    }
                    p = self.acc.next_x();
                }
                y_hr += ry_inv;
                if y_hr >= filter_scale {
                    break;
                }
                p = self.acc.next_y();
            }
            if n == 1 {
                s[0] = V::from_acc(V::acc_div(fg[0], total_weight));
                s[3] = V::FULL;
            } else {
                for v in fg.iter_mut() {
                    *v = V::acc_div(*v, total_weight);
                }
                clamp_rgba::<V>(&mut fg);
                for c in 0..4 {
                    s[c] = V::from_acc(fg[c]);
                }
            }
            self.interp.next();
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering

/// `render_scanlines(rasterizer, scanline32_u8, renderer_scanline_aa<
/// renderer_base, span_allocator, span_converter<gen, span_conv_alpha>>)`.
fn render<C: ColorType, G: SpanGen<C::V>>(
    ras: &mut RasterizerScanlineAa,
    out: &mut [C::V],
    out_width: i32,
    out_height: i32,
    gen: &mut G,
    alpha: f64,
) {
    let mut sl = ScanlineU8::new(None);
    if !ras.rewind_scanlines() {
        return;
    }
    sl.reset(ras.min_x(), ras.max_x());
    gen.prepare();
    let mut colors: Vec<[C::V; 4]> = Vec::new();
    let n = C::N;
    while ras.sweep_scanline(&mut sl) {
        let y = sl.y();
        let covers = sl.covers();
        for span in sl.spans() {
            let mut x = span.x;
            let mut len = span.len.abs();
            colors.clear();
            colors.resize(len as usize, [C::V::default(); 4]);
            gen.generate(&mut colors, x, y);
            if alpha != 1.0 {
                for c in colors.iter_mut() {
                    c[3] = C::V::scale_alpha(c[3], alpha);
                }
            }
            // renderer_base::blend_color_hspan clipping
            if y > out_height - 1 || y < 0 {
                continue;
            }
            let mut cstart = 0usize;
            let mut cov = span.cover_idx as usize;
            if x < 0 {
                let d = -x;
                len -= d;
                if len <= 0 {
                    continue;
                }
                cov += d as usize;
                cstart = d as usize;
                x = 0;
            }
            if x + len > out_width - 1 {
                len = out_width - 1 - x + 1;
                if len <= 0 {
                    continue;
                }
            }
            let row = y as usize * out_width as usize;
            for i in 0..len as usize {
                let o = (row + x as usize + i) * n;
                let cover = if span.len < 0 { covers[span.cover_idx as usize] } else { covers[cov + i] };
                C::copy_or_blend(&mut out[o..o + n], &colors[cstart + i], cover);
            }
        }
    }
}

pub struct ResampleParams<'a> {
    pub interpolation: u32,
    /// The affine transform, or `None` for a non-affine transform described
    /// by `mesh`.
    pub affine: Option<TransAffine>,
    /// For each output pixel (row-major), the input coordinates of its
    /// center.
    pub mesh: Option<&'a [f64]>,
    pub resample: bool,
    pub norm: bool,
    pub radius: f64,
    pub alpha: f64,
}

/// matplotlib's `resample<color_type>(input, in_w, in_h, output, out_w,
/// out_h, params)`.
pub fn resample<C: ColorType>(
    input: &[C::V],
    in_width: i32,
    in_height: i32,
    output: &mut [C::V],
    out_width: i32,
    out_height: i32,
    params: &ResampleParams,
) {
    let is_affine = params.affine.is_some();
    let affine = params.affine.unwrap_or_default();
    let mut interpolation = params.interpolation;
    if interpolation != NEAREST
        && is_affine
        && affine.sx.abs() == 1.0
        && affine.sy.abs() == 1.0
        && affine.shx == 0.0
        && affine.shy == 0.0
    {
        interpolation = NEAREST;
    }
    let n = C::N;
    let mut ras = RasterizerScanlineAa::new();
    let mut inverted = affine;
    inverted.invert();
    ras.clip_box(0.0, 0.0, out_width as f64, out_height as f64);

    let rect = |ras: &mut RasterizerScanlineAa, pts: [(f64, f64); 4]| {
        ras.add_vertex(pts[0].0, pts[0].1, PATH_CMD_MOVE_TO);
        for p in &pts[1..] {
            ras.add_vertex(p.0, p.1, PATH_CMD_LINE_TO);
        }
        ras.add_vertex(0.0, 0.0, CLOSEPOLY);
    };
    if is_affine && interpolation != NEAREST {
        if affine.shx != 0.0 || affine.shy != 0.0 {
            let (w, h) = (in_width as f64, in_height as f64);
            rect(
                &mut ras,
                [affine.apply(0.0, 0.0), affine.apply(w, 0.0), affine.apply(w, h), affine.apply(0.0, h)],
            );
        } else {
            // Bump out the rendering edges that are within a half pixel of a
            // full pixel so that axes are visually filled.
            let (mut left, mut bottom) = affine.apply(0.0, 0.0);
            let (mut right, mut top) = affine.apply(in_width as f64, in_height as f64);
            if left > right {
                std::mem::swap(&mut left, &mut right);
            }
            if bottom > top {
                std::mem::swap(&mut top, &mut bottom);
            }
            let rleft = iround(left - 1e-8);
            let rright = iround(right + 1e-8);
            let rbottom = iround(bottom - 1e-8);
            let rtop = iround(top + 1e-8);
            if (rleft as f64) < left {
                left = rleft as f64;
            }
            if (rright as f64) > right {
                right = rright as f64;
            }
            if (rbottom as f64) < bottom {
                bottom = rbottom as f64;
            }
            if (rtop as f64) > top {
                top = rtop as f64;
            }
            rect(&mut ras, [(left, bottom), (right, bottom), (right, top), (left, top)]);
        }
    } else {
        let (w, h) = (out_width as f64, out_height as f64);
        rect(&mut ras, [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]);
    }

    let src = || Image { data: input, width: in_width, height: in_height };
    let distorted = |offset: f64| DistortedInterp {
        base: LinearInterp::new(inverted),
        mesh: if is_affine { None } else { params.mesh },
        out_width,
        out_height,
        offset,
    };
    if interpolation == NEAREST {
        if is_affine {
            let mut gen = NnGen { src: src(), n, interp: AccurateNnInterp::new(inverted) };
            render::<C, _>(&mut ras, output, out_width, out_height, &mut gen, params.alpha);
        } else {
            let mut gen = NnGen { src: src(), n, interp: distorted(0.0) };
            render::<C, _>(&mut ras, output, out_width, out_height, &mut gen, params.alpha);
        }
    } else {
        let lut = FilterLut::new(interpolation, params.radius, params.norm);
        let acc = AccessorWrap::new(in_width, in_height, n);
        if is_affine && params.resample {
            let mut gen = ResampleAffineGen {
                src: src(),
                n,
                acc,
                interp: LinearInterp::new(inverted),
                lut: &lut,
                rx: 0,
                ry: 0,
                rx_inv: 0,
                ry_inv: 0,
            };
            render::<C, _>(&mut ras, output, out_width, out_height, &mut gen, params.alpha);
        } else {
            let mut gen = FilterGen { src: src(), n, acc, interp: distorted(0.5), lut: &lut };
            render::<C, _>(&mut ras, output, out_width, out_height, &mut gen, params.alpha);
        }
    }
}
