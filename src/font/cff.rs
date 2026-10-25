//! OpenType/CFF outlines: the `CFF ` table (`cff/cffload.c`, `cffparse.c`,
//! `cffobjs.c`) and FreeType's Adobe CFF engine (`psaux/psintrp.c`,
//! `pshints.c`, `psft.c`, `psobjs.c`) for unhinted glyph loading, scaled the
//! way `cff_slot_load` (`cff/cffgload.c`) does it.
//!
//! Unhinted, the engine renders at "unity" scale (1/64 as 16.16) without
//! stem darkening, so the hint machinery reduces to its bookkeeping: which
//! stems count towards the mask bytes, and when a hint mask becomes valid.

use super::outline::{div_fix, mul_div, mul_fix, FtMatrix, Outline};
use super::sfnt::FontError;

fn err(m: &str) -> FontError {
    FontError(m.to_string())
}

#[inline]
fn byte(d: &[u8], o: usize) -> Result<u8, FontError> {
    d.get(o).copied().ok_or_else(|| err("CFF table truncated"))
}

#[inline]
fn u16_at(d: &[u8], o: usize) -> Result<u16, FontError> {
    match d.get(o..o + 2) {
        Some(b) => Ok(u16::from_be_bytes([b[0], b[1]])),
        None => Err(err("CFF table truncated")),
    }
}

/// A CFF INDEX: element `i` spans `offs[i]..offs[i + 1]` (absolute
/// positions, sanitized like `cff_index_get_pointers`).
#[derive(Clone, Debug, Default)]
struct Index {
    offs: Vec<usize>,
}

impl Index {
    /// `cff_index_init` at `pos`; returns the index and its end.
    fn parse(d: &[u8], pos: usize) -> Result<(Index, usize), FontError> {
        let count = u16_at(d, pos)? as usize;
        if count == 0 {
            return Ok((Index::default(), pos + 2));
        }
        let off_size = byte(d, pos + 2)? as usize;
        if !(1..=4).contains(&off_size) {
            return Err(err("invalid CFF index"));
        }
        let ostart = pos + 3;
        let read = |i: usize| -> Result<usize, FontError> {
            let p = ostart + i * off_size;
            let b = d.get(p..p + off_size).ok_or_else(|| err("CFF table truncated"))?;
            Ok(b.iter().fold(0usize, |v, &x| (v << 8) | x as usize))
        };
        let last = read(count)?;
        if last == 0 {
            return Err(err("invalid CFF index"));
        }
        let data_size = last - 1;
        let data = ostart + (count + 1) * off_size;
        if data + data_size > d.len() {
            return Err(err("CFF table truncated"));
        }
        let mut offs = Vec::with_capacity(count + 1);
        let mut cur = 0usize;
        offs.push(data);
        for i in 1..=count {
            let mut next = read(i)?.wrapping_sub(1);
            if next < cur {
                next = cur;
            } else if next > data_size {
                next = data_size;
            }
            offs.push(data + next);
            cur = next;
        }
        Ok((Index { offs }, data + data_size))
    }

    fn count(&self) -> usize {
        self.offs.len().saturating_sub(1)
    }

    fn get(&self, i: usize) -> Option<(usize, usize)> {
        if i + 1 < self.offs.len() {
            Some((self.offs[i], self.offs[i + 1]))
        } else {
            None
        }
    }
}

// -- DICT parsing (cffparse.c) ------------------------------------------------

const POWER_TENS: [i64; 10] = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000];

/// `cff_parse_integer`.
fn parse_integer(d: &[u8], p: usize, limit: usize) -> i64 {
    let v = d[p] as i64;
    let q = p + 1;
    let avail = |n: usize| q + n <= limit;
    match v {
        28 => {
            if !avail(2) {
                return 0;
            }
            i16::from_be_bytes([d[q], d[q + 1]]) as i64
        }
        29 => {
            if !avail(4) {
                return 0;
            }
            i32::from_be_bytes([d[q], d[q + 1], d[q + 2], d[q + 3]]) as i64
        }
        _ if v < 247 => v - 139,
        _ if v < 251 => {
            if !avail(1) {
                return 0;
            }
            (v - 247) * 256 + d[q] as i64 + 108
        }
        _ => {
            if !avail(1) {
                return 0;
            }
            -(v - 251) * 256 - d[q] as i64 - 108
        }
    }
}

/// `cff_parse_real` (16.16 result, optionally with a dynamic `scaling`).
fn parse_real(d: &[u8], start: usize, limit: usize, power_ten: i64, scaling: Option<&mut i64>) -> i64 {
    let mut p = start;
    let mut phase = 4u32;
    let mut sign = false;
    let mut exponent_sign = false;
    let mut have_overflow = false;
    let (mut number, mut exponent, mut exponent_add) = (0i64, 0i64, 0i64);
    let (mut integer_length, mut fraction_length) = (0i64, 0i64);
    let mut want_scaling = scaling.is_some();
    let mut scale_out = 0i64;
    let mut nib;
    macro_rules! next_nibble {
        () => {{
            if phase != 0 {
                p += 1;
                if p + 1 > limit {
                    return 0;
                }
            }
            let n = (d[p] >> phase) & 0xF;
            phase = 4 - phase;
            n
        }};
    }
    // integer part
    loop {
        nib = next_nibble!();
        if nib == 0xE {
            sign = true;
        } else if nib > 9 {
            break;
        } else if number >= 0xCCCCCCC {
            exponent_add += 1;
        } else if nib != 0 || number != 0 {
            integer_length += 1;
            number = number * 10 + nib as i64;
        }
    }
    // fraction part
    if nib == 0xA {
        loop {
            nib = next_nibble!();
            if nib >= 10 {
                break;
            }
            if nib == 0 && number == 0 {
                exponent_add -= 1;
            } else if number < 0xCCCCCCC && fraction_length < 9 {
                fraction_length += 1;
                number = number * 10 + nib as i64;
            }
        }
    }
    // exponent
    if nib == 12 {
        exponent_sign = true;
        nib = 11;
    }
    if nib == 11 {
        loop {
            nib = next_nibble!();
            if nib >= 10 {
                break;
            }
            if exponent > 1000 {
                have_overflow = true;
            } else {
                exponent = exponent * 10 + nib as i64;
            }
        }
        if exponent_sign {
            exponent = -exponent;
        }
    }
    let mut result: i64;
    'done: {
        if number == 0 {
            result = 0;
            break 'done;
        }
        if have_overflow {
            result = if exponent_sign { 0 } else { 0x7FFFFFFF };
            want_scaling = false;
            break 'done;
        }
        exponent += power_ten + exponent_add;
        if want_scaling {
            fraction_length += integer_length;
            exponent += integer_length;
            if fraction_length <= 5 {
                if number > 0x7FFF {
                    result = div_fix(number, 10);
                    scale_out = exponent - fraction_length + 1;
                } else {
                    if exponent > 0 {
                        let new_fraction_length = exponent.min(5);
                        let shift = new_fraction_length - fraction_length;
                        if shift > 0 {
                            exponent -= new_fraction_length;
                            number *= POWER_TENS[shift as usize];
                            if number > 0x7FFF {
                                number /= 10;
                                exponent += 1;
                            }
                        } else {
                            exponent -= fraction_length;
                        }
                    } else {
                        exponent -= fraction_length;
                    }
                    result = ((number as u64) << 16) as i64;
                    scale_out = exponent;
                }
            } else if number / POWER_TENS[(fraction_length - 5) as usize] > 0x7FFF {
                result = div_fix(number, POWER_TENS[(fraction_length - 4) as usize]);
                scale_out = exponent - 4;
            } else {
                result = div_fix(number, POWER_TENS[(fraction_length - 5) as usize]);
                scale_out = exponent - 5;
            }
        } else {
            integer_length += exponent;
            fraction_length -= exponent;
            if integer_length > 5 {
                result = 0x7FFFFFFF;
                break 'done;
            }
            if integer_length < -5 {
                result = 0;
                break 'done;
            }
            if integer_length < 0 {
                number /= POWER_TENS[(-integer_length) as usize];
                fraction_length += integer_length;
            }
            if fraction_length == 10 {
                number /= 10;
                fraction_length -= 1;
            }
            if fraction_length > 0 {
                if number / POWER_TENS[fraction_length as usize] > 0x7FFF {
                    result = 0;
                    break 'done;
                }
                result = div_fix(number, POWER_TENS[fraction_length as usize]);
            } else {
                number *= POWER_TENS[(-fraction_length) as usize];
                if number > 0x7FFF {
                    result = 0x7FFFFFFF;
                    break 'done;
                }
                result = ((number as u64) << 16) as i64;
            }
        }
    }
    if let Some(s) = scaling {
        *s = if want_scaling { scale_out } else { 0 };
    }
    if sign {
        result = -result;
    }
    result
}

/// `cff_parse_num`.
fn parse_num(d: &[u8], p: usize, limit: usize) -> i64 {
    match d[p] {
        30 => parse_real(d, p, limit, 0, None) >> 16,
        255 => {
            let v = ((d[p + 1] as u32) << 16 | (d[p + 2] as u32) << 8 | d[p + 3] as u32).wrapping_add(0x80) >> 8;
            v as i16 as i64
        }
        _ => parse_integer(d, p, limit),
    }
}

/// `cff_parse_fixed_dynamic`.
fn parse_fixed_dynamic(d: &[u8], p: usize, limit: usize, scaling: &mut i64) -> i64 {
    if d[p] == 30 {
        return parse_real(d, p, limit, 0, Some(scaling));
    }
    let number = parse_integer(d, p, limit);
    if number > 0x7FFF {
        let mut integer_length = 5usize;
        while integer_length < 10 {
            if number < POWER_TENS[integer_length] {
                break;
            }
            integer_length += 1;
        }
        if number / POWER_TENS[integer_length - 5] > 0x7FFF {
            *scaling = integer_length as i64 - 4;
            div_fix(number, POWER_TENS[integer_length - 4])
        } else {
            *scaling = integer_length as i64 - 5;
            div_fix(number, POWER_TENS[integer_length - 5])
        }
    } else {
        *scaling = 0;
        ((number as u64) << 16) as i64
    }
}

/// `cff_parser_run`: calls `f(op, operand positions)` for each operator.
fn parse_dict(d: &[u8], start: usize, end: usize, mut f: impl FnMut(u16, &[usize])) -> Result<(), FontError> {
    let end = end.min(d.len());
    let mut p = start;
    let mut operands: Vec<usize> = Vec::new();
    while p < end {
        let v = d[p];
        if v >= 27 && v != 31 {
            operands.push(p);
            p += match v {
                28 => 3,
                29 => 5,
                30 => {
                    // nibbles until 0xF
                    let mut q = p + 1;
                    loop {
                        if q >= end {
                            // unterminated: invalid but harmless
                            return Ok(());
                        }
                        let b = d[q];
                        q += 1;
                        if b & 0xF0 == 0xF0 || b & 0x0F == 0x0F {
                            break;
                        }
                    }
                    q - p
                }
                255 => 5,
                247..=254 => 2,
                _ => 1,
            };
            if p > end {
                return Err(err("invalid CFF DICT"));
            }
            if operands.len() > 48 {
                return Err(err("invalid CFF DICT"));
            }
        } else {
            let mut op = v as u16;
            p += 1;
            if v == 12 {
                if p >= end {
                    return Err(err("invalid CFF DICT"));
                }
                op = 0x100 | d[p] as u16;
                p += 1;
            }
            f(op, &operands);
            operands.clear();
        }
    }
    Ok(())
}

/// `FT_Matrix_Check`.
fn matrix_check(m: &FtMatrix) -> bool {
    let (mut xx, mut xy, mut yx, mut yy) = (m.xx, m.xy, m.yx, m.yy);
    let val = xx.abs() | xy.abs() | yx.abs() | yy.abs();
    if val == 0 || val > 0x7FFFFFFF {
        return false;
    }
    let shift = (63 - val.leading_zeros() as i64) - 12;
    if shift > 0 {
        xx >>= shift;
        xy >>= shift;
        yx >>= shift;
        yy >>= shift;
    }
    let temp1 = 32u64.wrapping_mul((xx * yy - xy * yx).unsigned_abs());
    let temp2 = ((xx * xx) as u64).wrapping_add((xy * xy) as u64).wrapping_add((yx * yx) as u64).wrapping_add((yy * yy) as u64);
    temp1 > temp2
}

/// The font dict fields of a top DICT or FD dict (`CFF_FontRecDict`).
#[derive(Clone, Debug)]
struct FontDict {
    charset_offset: i64,
    charstrings_offset: i64,
    private_size: i64,
    private_offset: i64,
    charstring_type: i64,
    has_font_matrix: bool,
    matrix: FtMatrix,
    offset: (i64, i64),
    upm: i64,
    is_cid: bool,
    fd_array_offset: i64,
    fd_select_offset: i64,
}

impl Default for FontDict {
    fn default() -> Self {
        FontDict {
            charset_offset: 0,
            charstrings_offset: 0,
            private_size: 0,
            private_offset: 0,
            charstring_type: 2,
            has_font_matrix: false,
            matrix: FtMatrix::IDENTITY,
            offset: (0, 0),
            upm: 0,
            is_cid: false,
            fd_array_offset: 0,
            fd_select_offset: 0,
        }
    }
}

fn parse_font_dict(d: &[u8], start: usize, end: usize) -> Result<FontDict, FontError> {
    let mut fd = FontDict::default();
    let limit = end.min(d.len());
    parse_dict(d, start, end, |op, a| {
        let num = |i: usize| a.get(i).map(|&p| parse_num(d, p, limit)).unwrap_or(0);
        match op {
            15 if !a.is_empty() => fd.charset_offset = num(0),
            17 if !a.is_empty() => fd.charstrings_offset = num(0),
            18 if a.len() >= 2 => {
                fd.private_size = num(0);
                fd.private_offset = num(1);
            }
            0x106 if !a.is_empty() => fd.charstring_type = num(0),
            0x107 if a.len() >= 6 => {
                // cff_parse_font_matrix
                fd.has_font_matrix = true;
                let mut values = [0i64; 6];
                let mut scalings = [0i64; 6];
                let (mut max_s, mut min_s) = (i64::MIN, i64::MAX);
                for i in 0..6 {
                    values[i] = parse_fixed_dynamic(d, a[i], limit, &mut scalings[i]);
                    if values[i] != 0 {
                        max_s = max_s.max(scalings[i]);
                        min_s = min_s.min(scalings[i]);
                    }
                }
                let unlikely = max_s < -9 || max_s > 0 || max_s.wrapping_sub(min_s) < 0 || max_s.wrapping_sub(min_s) > 9;
                if !unlikely {
                    for i in 0..6 {
                        let value = values[i];
                        if value == 0 {
                            continue;
                        }
                        let divisor = POWER_TENS[(max_s - scalings[i]) as usize];
                        let half = divisor >> 1;
                        values[i] = if value < 0 {
                            if i64::MIN + half < value {
                                (value - half) / divisor
                            } else {
                                i64::MIN / divisor
                            }
                        } else if i64::MAX - half > value {
                            (value + half) / divisor
                        } else {
                            i64::MAX / divisor
                        };
                    }
                    fd.matrix = FtMatrix { xx: values[0], yx: values[1], xy: values[2], yy: values[3] };
                    fd.offset = (values[4], values[5]);
                    fd.upm = POWER_TENS[(-max_s) as usize];
                }
                if unlikely || !matrix_check(&fd.matrix) {
                    fd.matrix = FtMatrix::IDENTITY;
                    fd.offset = (0, 0);
                    fd.upm = 1;
                }
            }
            0x11E => fd.is_cid = true,
            0x124 if !a.is_empty() => fd.fd_array_offset = num(0),
            0x125 if !a.is_empty() => fd.fd_select_offset = num(0),
            _ => {}
        }
    })?;
    Ok(fd)
}

/// Normalize a font matrix so that `yy` (or `yx`) is 1 (`cff_face_init`).
fn normalize_matrix(fd: &mut FontDict) {
    let temp = if fd.matrix.yy != 0 { fd.matrix.yy.abs() } else { fd.matrix.yx.abs() };
    if temp != 0x10000 {
        fd.upm = div_fix(fd.upm, temp);
        fd.matrix.xx = div_fix(fd.matrix.xx, temp);
        fd.matrix.yx = div_fix(fd.matrix.yx, temp);
        fd.matrix.xy = div_fix(fd.matrix.xy, temp);
        fd.matrix.yy = div_fix(fd.matrix.yy, temp);
        fd.offset.0 = div_fix(fd.offset.0, temp);
        fd.offset.1 = div_fix(fd.offset.1, temp);
    }
    fd.offset.0 >>= 16;
    fd.offset.1 >>= 16;
}

/// `cff_compute_bias`.
fn compute_bias(charstring_type: i64, n: usize) -> i64 {
    if charstring_type == 1 {
        0
    } else if n < 1240 {
        107
    } else if n < 33900 {
        1131
    } else {
        32768
    }
}

/// A font dict with its private data (`CFF_SubFontRec`).
#[derive(Clone, Debug, Default)]
struct SubFont {
    dict: FontDict,
    subrs: Index,
    subrs_bias: i64,
    initial_random_seed: i64,
}

fn load_subfont(d: &[u8], dict: FontDict) -> Result<SubFont, FontError> {
    let mut sub = SubFont { dict, ..Default::default() };
    if sub.dict.is_cid {
        return Ok(sub);
    }
    let (psize, poff) = (sub.dict.private_size, sub.dict.private_offset);
    let mut subrs_offset = 0i64;
    if poff > 0 && psize > 0 {
        let (ps, pe) = (poff as usize, (poff + psize) as usize);
        if pe > d.len() {
            return Err(err("CFF private dict out of range"));
        }
        parse_dict(d, ps, pe, |op, a| {
            let num = |i: usize| a.get(i).map(|&p| parse_num(d, p, pe)).unwrap_or(0);
            match op {
                19 if !a.is_empty() => subrs_offset = num(0),
                0x113 if !a.is_empty() => sub.initial_random_seed = num(0),
                _ => {}
            }
        })?;
        if sub.initial_random_seed < 0 {
            sub.initial_random_seed = -sub.initial_random_seed;
        } else if sub.initial_random_seed == 0 {
            sub.initial_random_seed = 987654321;
        }
        if subrs_offset != 0 {
            let at = (poff + subrs_offset) as usize;
            sub.subrs = Index::parse(d, at)?.0;
        }
    }
    sub.subrs_bias = compute_bias(sub.dict.charstring_type, sub.subrs.count());
    Ok(sub)
}

#[derive(Clone, Debug)]
enum FdSelect {
    Format0(Vec<u8>),
    /// (first gid, fd) ranges and the sentinel.
    Format3(Vec<(u16, u8)>, u16),
}

impl FdSelect {
    /// `cff_fd_select_get`.
    fn get(&self, gid: u32) -> u8 {
        match self {
            FdSelect::Format0(v) => v.get(gid as usize).copied().unwrap_or(0),
            FdSelect::Format3(ranges, sentinel) => {
                for (i, &(first, fd)) in ranges.iter().enumerate() {
                    if gid < first as u32 {
                        break;
                    }
                    let limit = ranges.get(i + 1).map(|r| r.0).unwrap_or(*sentinel);
                    if gid < limit as u32 {
                        return fd;
                    }
                }
                0
            }
        }
    }
}

/// The parsed `CFF ` table.
#[derive(Clone, Debug)]
pub struct CffFont {
    /// The table bytes.
    data: Vec<u8>,
    charstrings: Index,
    gsubrs: Index,
    gsubrs_bias: i64,
    top: SubFont,
    subfonts: Vec<SubFont>,
    fd_select: Option<FdSelect>,
    /// Glyph SIDs (CIDs for CID-keyed fonts), when the charset has them.
    sids: Option<Vec<u16>>,
    pub num_glyphs: u32,
}

impl CffFont {
    /// `cff_font_load` + the matrix setup of `cff_face_init` for an
    /// SFNT-wrapped CFF with `head` units per em `face_upm`.
    pub fn parse(table: &[u8], face_upm: u16) -> Result<CffFont, FontError> {
        let d = table.to_vec();
        if d.len() < 4 || d[0] != 1 {
            return Err(err("unsupported CFF version"));
        }
        let hdr_size = d[2] as usize;
        let (names, p) = Index::parse(&d, hdr_size)?;
        let (top_dicts, p) = Index::parse(&d, p)?;
        let (_strings, p) = Index::parse(&d, p)?;
        let (gsubrs, _) = Index::parse(&d, p)?;
        if names.count() == 0 || top_dicts.count() == 0 {
            return Err(err("empty CFF font"));
        }
        let (ts, te) = top_dicts.get(0).unwrap();
        let top_dict = parse_font_dict(&d, ts, te)?;
        if top_dict.charstrings_offset <= 0 {
            return Err(err("missing CFF CharStrings"));
        }
        let charstrings = Index::parse(&d, top_dict.charstrings_offset as usize)?.0;
        let num_glyphs = charstrings.count();
        let charstring_type = top_dict.charstring_type;
        let is_cid = top_dict.is_cid;
        let (fd_array_offset, fd_select_offset, charset_offset) =
            (top_dict.fd_array_offset, top_dict.fd_select_offset, top_dict.charset_offset);
        let mut top = load_subfont(&d, top_dict)?;

        let mut subfonts = Vec::new();
        let mut fd_select = None;
        if is_cid {
            let fds = Index::parse(&d, fd_array_offset as usize)?.0;
            if fds.count() > 256 {
                return Err(err("FD array too large"));
            }
            for i in 0..fds.count() {
                let (s, e) = fds.get(i).unwrap();
                let mut fdict = parse_font_dict(&d, s, e)?;
                fdict.charstring_type = charstring_type;
                subfonts.push(load_subfont(&d, fdict)?);
            }
            let fp = fd_select_offset as usize;
            fd_select = Some(match byte(&d, fp)? {
                0 => FdSelect::Format0(d.get(fp + 1..fp + 1 + num_glyphs).ok_or_else(|| err("CFF table truncated"))?.to_vec()),
                3 => {
                    let n = u16_at(&d, fp + 1)? as usize;
                    if n == 0 {
                        return Err(err("invalid FDSelect"));
                    }
                    let mut ranges = Vec::with_capacity(n);
                    for i in 0..n {
                        let q = fp + 3 + 3 * i;
                        ranges.push((u16_at(&d, q)?, byte(&d, q + 2)?));
                    }
                    FdSelect::Format3(ranges, u16_at(&d, fp + 3 + 3 * n)?)
                }
                _ => return Err(err("invalid FDSelect")),
            });
        }

        // cff_charset_load (SIDs only)
        let sids = if num_glyphs == 0 {
            None
        } else if charset_offset > 2 {
            let cp = charset_offset as usize;
            let mut sids = vec![0u16; num_glyphs];
            match byte(&d, cp)? {
                0 => {
                    for (j, s) in sids.iter_mut().enumerate().skip(1) {
                        *s = u16_at(&d, cp + 1 + 2 * (j - 1))?;
                    }
                }
                fmt @ (1 | 2) => {
                    let mut j = 1;
                    let mut q = cp + 1;
                    while j < num_glyphs {
                        let mut sid = u16_at(&d, q)? as u32;
                        let mut nleft = if fmt == 2 {
                            q += 4;
                            u16_at(&d, q - 2)? as u32
                        } else {
                            q += 3;
                            byte(&d, q - 1)? as u32
                        };
                        if sid > 0xFFFF - nleft {
                            nleft = 0xFFFF - sid;
                        }
                        let mut i = 0;
                        while j < num_glyphs && i <= nleft {
                            sids[j] = sid as u16;
                            i += 1;
                            j += 1;
                            sid += 1;
                        }
                    }
                }
                _ => return Err(err("invalid CFF charset")),
            }
            Some(sids)
        } else {
            match charset_offset {
                0 if num_glyphs > 229 => return Err(err("implicit CFF charset too large")),
                1 if num_glyphs > 166 => return Err(err("implicit CFF charset too large")),
                2 if num_glyphs > 87 => return Err(err("implicit CFF charset too large")),
                0 => Some((0..num_glyphs as u16).collect()),
                _ => None,
            }
        };

        // cff_face_init: normalize the font matrices
        if !top.dict.has_font_matrix {
            top.dict.upm = face_upm as i64;
        }
        normalize_matrix(&mut top.dict);
        for sub in subfonts.iter_mut().rev() {
            let t = &top.dict;
            let s = &mut sub.dict;
            if s.has_font_matrix {
                if t.has_font_matrix {
                    let scaling = if t.upm > 1 && s.upm > 1 { t.upm.min(s.upm) } else { 1 };
                    let val = 0x10000 * scaling;
                    let (a, b) = (t.matrix, s.matrix);
                    s.matrix = FtMatrix {
                        xx: mul_div(a.xx, b.xx, val) + mul_div(a.xy, b.yx, val),
                        xy: mul_div(a.xx, b.xy, val) + mul_div(a.xy, b.yy, val),
                        yx: mul_div(a.yx, b.xx, val) + mul_div(a.yy, b.yx, val),
                        yy: mul_div(a.yx, b.xy, val) + mul_div(a.yy, b.yy, val),
                    };
                    let (vx, vy) = s.offset;
                    s.offset = (
                        mul_div(vx, a.xx, val) + mul_div(vy, a.xy, val),
                        mul_div(vx, a.yx, val) + mul_div(vy, a.yy, val),
                    );
                    s.upm = mul_div(s.upm, t.upm, scaling);
                }
            } else {
                s.matrix = t.matrix;
                s.offset = t.offset;
                s.upm = t.upm;
            }
            normalize_matrix(s);
        }

        let gsubrs_bias = compute_bias(charstring_type, gsubrs.count());
        Ok(CffFont {
            data: d,
            charstrings,
            gsubrs,
            gsubrs_bias,
            top,
            subfonts,
            fd_select,
            sids,
            num_glyphs: num_glyphs as u32,
        })
    }

    /// The font dict used for `gid` (`None`: invalid FD index).
    fn subfont(&self, gid: u32) -> (Option<&SubFont>, &SubFont) {
        match &self.fd_select {
            Some(sel) if !self.subfonts.is_empty() => {
                let fd = sel.get(gid) as usize;
                let clamped = &self.subfonts[fd.min(self.subfonts.len() - 1)];
                (self.subfonts.get(fd), clamped)
            }
            _ => (Some(&self.top), &self.top),
        }
    }

    /// The scale factors and font matrix `cff_slot_load` uses for `gid`.
    fn glyph_transform(&self, gid: u32, x_scale: i64, y_scale: i64) -> (i64, i64, FtMatrix, (i64, i64)) {
        let (_, sub) = self.subfont(gid);
        if self.subfonts.is_empty() {
            return (x_scale, y_scale, self.top.dict.matrix, self.top.dict.offset);
        }
        let (top_upm, sub_upm) = (self.top.dict.upm, sub.dict.upm);
        let (xs, ys) = if top_upm != sub_upm {
            (mul_div(x_scale, top_upm, sub_upm), mul_div(y_scale, top_upm, sub_upm))
        } else {
            (x_scale, y_scale)
        };
        (xs, ys, sub.dict.matrix, sub.dict.offset)
    }

    /// `metrics->horiAdvance` of `cff_slot_load` for an hmtx advance `aw`.
    pub fn hori_advance(&self, gid: u32, aw: i64, x_scale: i64) -> i64 {
        let (xs, _, m, off) = self.glyph_transform(gid, x_scale, x_scale);
        let mut adv = aw;
        if !m.is_identity() {
            adv = mul_fix(adv, m.xx);
        }
        adv += off.0;
        mul_fix(adv, xs)
    }

    /// `cff_slot_load` (unhinted, scaled): the outline in 26.6.
    pub fn load_glyph(&self, gid: u32, x_scale: i64, y_scale: i64) -> Result<Outline, FontError> {
        if gid >= self.num_glyphs {
            return Err(err("invalid glyph index"));
        }
        let (sub, _) = self.subfont(gid);
        let sub = sub.ok_or_else(|| err("invalid CID subfont index"))?;
        let (cs, ce) = self.charstrings.get(gid as usize).unwrap();
        let mut b = Builder::default();
        Interp { cff: self, sub, random: sub.initial_random_seed as u32 }.run(&mut b, (cs, ce), false, 0, 0)?;
        b.close_contour();
        let mut out = Outline { points: b.points, tags: b.tags, contours: b.contours.iter().map(|&c| c as u16).collect(), ..Default::default() };
        let (xs, ys, m, off) = self.glyph_transform(gid, x_scale, y_scale);
        if !m.is_identity() {
            out.transform(&m);
        }
        if off.0 != 0 || off.1 != 0 {
            out.translate(off.0, off.1);
        }
        for p in &mut out.points {
            p.0 = mul_fix(p.0, xs);
            p.1 = mul_fix(p.1, ys);
        }
        Ok(out)
    }

    /// `cff_lookup_glyph_by_stdcharcode`.
    fn glyph_by_stdcharcode(&self, code: i32) -> Option<u32> {
        let sids = self.sids.as_ref()?;
        if !(0..=255).contains(&code) {
            return None;
        }
        let sid = STANDARD_ENCODING[code as usize];
        sids.iter().position(|&s| s == sid).map(|g| g as u32)
    }
}

// -- outline builder (psobjs.c ps_builder_*) ---------------------------------

#[derive(Default)]
struct Builder {
    points: Vec<(i64, i64)>,
    tags: Vec<u8>,
    contours: Vec<i64>,
    path_begun: bool,
}

const TAG_ON: u8 = 1;
const TAG_CUBIC: u8 = 2;

impl Builder {
    fn add_point(&mut self, p: (i32, i32), on: bool) {
        self.points.push(((p.0 >> 10) as i64, (p.1 >> 10) as i64));
        self.tags.push(if on { TAG_ON } else { TAG_CUBIC });
    }

    fn add_contour(&mut self) {
        let n = self.points.len() as i64;
        if let Some(last) = self.contours.last_mut() {
            *last = n - 1;
        }
        self.contours.push(0);
    }

    fn start_point(&mut self, p: (i32, i32)) {
        if !self.path_begun {
            self.path_begun = true;
            self.add_contour();
            self.add_point(p, true);
        }
    }

    fn close_contour(&mut self) {
        let nc = self.contours.len();
        let first = if nc <= 1 { 0 } else { (self.contours[nc - 2] + 1) as usize };
        if nc > 0 && first == self.points.len() {
            self.contours.pop();
            return;
        }
        let np = self.points.len();
        if np > 1 {
            let last = np - 1;
            if self.points[first] == self.points[last] && self.tags[last] == TAG_ON {
                self.points.pop();
                self.tags.pop();
            }
        }
        if nc > 0 {
            let np = self.points.len();
            if first + 1 == np {
                self.contours.pop();
                self.points.pop();
                self.tags.pop();
            } else {
                self.contours[nc - 1] = np as i64 - 1;
            }
        }
    }

    // cf2_builder_* callbacks (psft.c)
    fn move_to(&mut self) {
        self.close_contour();
        self.path_begun = false;
    }

    fn line_to(&mut self, p0: (i32, i32), p1: (i32, i32)) {
        if !self.path_begun {
            self.start_point(p0);
        }
        self.add_point(p1, true);
    }

    fn cube_to(&mut self, p0: (i32, i32), p1: (i32, i32), p2: (i32, i32), p3: (i32, i32)) {
        if !self.path_begun {
            self.start_point(p0);
        }
        self.add_point(p1, false);
        self.add_point(p2, false);
        self.add_point(p3, true);
    }
}

// -- charstring interpreter (psintrp.c) and glyph path (pshints.c) -----------

const STACK_SIZE: usize = 48;
const MAX_SUBR: usize = 16;
const MAX_HINTS: usize = 96;
const STORAGE_SIZE: usize = 32;
/// The engine's unity scale (1/64 in 16.16).
const UNITY: i64 = 0x400;

#[derive(Clone, Copy)]
enum Num {
    Int(i32),
    Fixed(i32),
}

impl Num {
    #[inline]
    fn real(self) -> i32 {
        match self {
            Num::Int(i) => (i as u32).wrapping_shl(16) as i32,
            Num::Fixed(r) => r,
        }
    }
}

/// Glyph-wide error state: the first error aborts the glyph.
#[derive(Default)]
struct Stack {
    v: Vec<Num>,
    err: bool,
}

impl Stack {
    fn push(&mut self, n: Num) {
        if self.v.len() == STACK_SIZE {
            self.err = true;
            return;
        }
        self.v.push(n);
    }
    fn push_int(&mut self, i: i32) {
        self.push(Num::Int(i));
    }
    fn push_fixed(&mut self, r: i32) {
        self.push(Num::Fixed(r));
    }
    fn pop_int(&mut self) -> i32 {
        match self.v.last() {
            None => {
                self.err = true;
                0
            }
            Some(Num::Int(i)) => {
                let i = *i;
                self.v.pop();
                i
            }
            Some(_) => {
                self.err = true;
                0
            }
        }
    }
    fn pop_fixed(&mut self) -> i32 {
        match self.v.pop() {
            None => {
                self.err = true;
                0
            }
            Some(n) => n.real(),
        }
    }
    fn get(&mut self, idx: usize) -> i32 {
        match self.v.get(idx) {
            Some(n) => n.real(),
            None => {
                self.err = true;
                0
            }
        }
    }
    fn count(&self) -> usize {
        self.v.len()
    }
    fn clear(&mut self) {
        self.v.clear();
    }
    /// `cf2_stack_roll`.
    fn roll(&mut self, count: i32, shift: i32) {
        if count < 2 {
            return;
        }
        if count as usize > self.v.len() {
            self.err = true;
            return;
        }
        let shift = if shift < 0 { -((-shift) % count) } else { shift % count };
        if shift == 0 {
            return;
        }
        let n = count as usize;
        let k = shift.rem_euclid(count) as usize;
        self.v[..n].rotate_right(k);
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ElemOp {
    Line,
    Cube,
}

type Pt = (i32, i32);

/// Hint bookkeeping that affects parsing (`CF2_HintMask`, hint map validity).
#[derive(Default)]
struct Hints {
    h_stems: usize,
    v_stems: usize,
    mask_valid: bool,
    mask_new: bool,
    map_valid: bool,
}

/// `CF2_GlyphPath` without darkening and hinting.
#[derive(Default)]
struct GlyphPath {
    move_pending: bool,
    path_open: bool,
    path_closing: bool,
    elem_queued: bool,
    current_cs: Pt,
    start: Pt,
    current_ds: Pt,
    offset_start0: Pt,
    prev_op: Option<ElemOp>,
    prev: [Pt; 4],
}

#[inline]
fn hint_point(p: Pt) -> Pt {
    (mul_fix(p.0 as i64, UNITY) as i32, mul_fix(p.1 as i64, UNITY) as i32)
}

struct Interp<'a> {
    cff: &'a CffFont,
    sub: &'a SubFont,
    random: u32,
}

struct Path<'b> {
    gp: GlyphPath,
    hints: Hints,
    builder: &'b mut Builder,
    err: bool,
}

impl<'b> Path<'b> {
    /// `cf2_hintmap_build` (only its effect on validity and errors).
    fn build_hint_map(&mut self) {
        if self.hints.h_stems + self.hints.v_stems > MAX_HINTS {
            self.err = true;
            return;
        }
        self.hints.mask_valid = true;
        self.hints.map_valid = true;
        self.hints.mask_new = false;
    }

    fn move_to(&mut self, x: i32, y: i32) {
        self.close_open_path();
        self.gp.current_cs = (x, y);
        self.gp.start = (x, y);
        self.gp.move_pending = true;
        if !self.hints.map_valid || self.hints.mask_new {
            self.build_hint_map();
        }
    }

    fn push_move(&mut self, start: Pt) {
        if !self.hints.map_valid {
            let s = self.gp.start;
            self.move_to(s.0, s.1);
        }
        let pt1 = hint_point(start);
        self.builder.move_to();
        self.gp.current_ds = pt1;
        self.gp.offset_start0 = start;
    }

    fn push_prev_elem(&mut self, next_p0: Pt) {
        match self.gp.prev_op {
            Some(ElemOp::Line) => {
                let pt0 = self.gp.current_ds;
                let pt1 = hint_point(self.gp.prev[1]);
                if pt0 != pt1 {
                    self.builder.line_to(pt0, pt1);
                    self.gp.current_ds = pt1;
                }
            }
            Some(ElemOp::Cube) => {
                let pt0 = self.gp.current_ds;
                let (p1, p2, p3) = (hint_point(self.gp.prev[1]), hint_point(self.gp.prev[2]), hint_point(self.gp.prev[3]));
                self.builder.cube_to(pt0, p1, p2, p3);
                self.gp.current_ds = p3;
            }
            None => {}
        }
        // connecting line to the start of the next element
        let pt1 = hint_point(next_p0);
        if pt1 != self.gp.current_ds {
            let pt0 = self.gp.current_ds;
            self.builder.line_to(pt0, pt1);
            self.gp.current_ds = pt1;
        }
    }

    fn line_to(&mut self, x: i32, y: i32) {
        let new_map = self.hints.mask_new && !self.gp.path_closing;
        if self.gp.current_cs == (x, y) && !new_map {
            return;
        }
        let p0 = self.gp.current_cs;
        let p1 = (x, y);
        if self.gp.move_pending {
            self.push_move(p0);
            self.gp.move_pending = false;
            self.gp.path_open = true;
        }
        if self.gp.elem_queued {
            self.push_prev_elem(p0);
        }
        self.gp.elem_queued = true;
        self.gp.prev_op = Some(ElemOp::Line);
        self.gp.prev[0] = p0;
        self.gp.prev[1] = p1;
        if new_map {
            self.build_hint_map();
        }
        self.gp.current_cs = (x, y);
    }

    fn curve_to(&mut self, x1: i32, y1: i32, x2: i32, y2: i32, x3: i32, y3: i32) {
        let p0 = self.gp.current_cs;
        if self.gp.move_pending {
            self.push_move(p0);
            self.gp.move_pending = false;
            self.gp.path_open = true;
        }
        if self.gp.elem_queued {
            self.push_prev_elem(p0);
        }
        self.gp.elem_queued = true;
        self.gp.prev_op = Some(ElemOp::Cube);
        self.gp.prev = [p0, (x1, y1), (x2, y2), (x3, y3)];
        if self.hints.mask_new {
            self.build_hint_map();
        }
        self.gp.current_cs = (x3, y3);
    }

    fn close_open_path(&mut self) {
        if self.gp.path_open {
            self.gp.path_closing = true;
            let s = self.gp.start;
            self.line_to(s.0, s.1);
            if self.gp.elem_queued {
                let s0 = self.gp.offset_start0;
                self.push_prev_elem(s0);
            }
            self.gp.move_pending = true;
            self.gp.path_open = false;
            self.gp.path_closing = false;
            self.gp.elem_queued = false;
        }
    }
}

#[inline]
fn add(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}

/// `FT_SqrtFixed` (64-bit variant).
fn sqrt_fixed(v: u32) -> u32 {
    if v == 0 {
        return 0;
    }
    let r = ((v as u64) << 16) - 1;
    let mut q: u32 = 1 << ((17 + (31 - v.leading_zeros())) >> 1);
    loop {
        let t = q;
        q = ((t as u64 + r / t as u64 + 1) >> 1) as u32;
        if q == t {
            return q;
        }
    }
}

impl<'a> Interp<'a> {
    fn subr(&self, global: bool, num: i32) -> Option<(usize, usize)> {
        let (idx, bias) = if global { (&self.cff.gsubrs, self.cff.gsubrs_bias) } else { (&self.sub.subrs, self.sub.subrs_bias) };
        let i = (num as i64 + bias) as u32 as usize;
        if i >= idx.count() {
            return None;
        }
        idx.get(i)
    }

    /// `cf2_interpT2CharString` for a CFF (Type 2) charstring.
    fn run(&mut self, builder: &mut Builder, buf: (usize, usize), doing_seac: bool, cur_x: i32, cur_y: i32) -> Result<(), FontError> {
        let fail = || Err(err("invalid CFF charstring"));
        let cff: &'a CffFont = self.cff;
        let d: &[u8] = &cff.data;
        let mut st = Stack::default();
        let mut storage = [0i32; STORAGE_SIZE];
        let mut instruction_limit: u32 = 20_000_000;
        let mut bufs: Vec<(usize, usize)> = vec![buf];
        let (mut cx, mut cy) = (cur_x, cur_y);
        let mut path = Path { gp: GlyphPath { move_pending: true, ..Default::default() }, hints: Hints::default(), builder, err: false };

        // reads the next byte of the current charstring (0 past its end)
        macro_rules! read_byte {
            () => {{
                let b = bufs.last_mut().unwrap();
                if b.0 < b.1 {
                    b.0 += 1;
                    d[b.0 - 1] as i32
                } else {
                    0
                }
            }};
        }

        loop {
            let at_end = {
                let b = bufs.last().unwrap();
                b.0 >= b.1
            };
            let op1 = if at_end {
                if bufs.len() > 1 {
                    11
                } else {
                    14
                }
            } else {
                read_byte!()
            };
            if st.err || path.err {
                return fail();
            }
            instruction_limit -= 1;
            if instruction_limit == 0 {
                return fail();
            }
            match op1 {
                1 | 18 | 3 | 23 => {
                    // hstem(hm) / vstem(hm)
                    if !path.hints.mask_valid {
                        let n = st.count() / 2;
                        if op1 == 1 || op1 == 18 {
                            path.hints.h_stems += n;
                        } else {
                            path.hints.v_stems += n;
                        }
                    }
                }
                4 => {
                    cy = add(cy, st.pop_fixed());
                    path.move_to(cx, cy);
                }
                5 => {
                    let count = st.count();
                    let mut idx = 0;
                    while idx < count {
                        cx = add(cx, st.get(idx));
                        cy = add(cy, st.get(idx + 1));
                        path.line_to(cx, cy);
                        idx += 2;
                    }
                    st.clear();
                    continue;
                }
                6 | 7 => {
                    let count = st.count();
                    let mut is_x = op1 == 6;
                    for idx in 0..count {
                        let v = st.get(idx);
                        if is_x {
                            cx = add(cx, v);
                        } else {
                            cy = add(cy, v);
                        }
                        is_x = !is_x;
                        path.line_to(cx, cy);
                    }
                    st.clear();
                    continue;
                }
                8 | 24 => {
                    // rrcurveto / rcurveline
                    let count = st.count();
                    let mut idx = 0;
                    while idx + 6 <= count {
                        let x1 = add(st.get(idx), cx);
                        let y1 = add(st.get(idx + 1), cy);
                        let x2 = add(st.get(idx + 2), x1);
                        let y2 = add(st.get(idx + 3), y1);
                        let x3 = add(st.get(idx + 4), x2);
                        let y3 = add(st.get(idx + 5), y2);
                        path.curve_to(x1, y1, x2, y2, x3, y3);
                        cx = x3;
                        cy = y3;
                        idx += 6;
                    }
                    if op1 == 24 {
                        cx = add(cx, st.get(idx));
                        cy = add(cy, st.get(idx + 1));
                        path.line_to(cx, cy);
                    }
                    st.clear();
                    continue;
                }
                10 | 29 => {
                    // callsubr / callgsubr
                    if bufs.len() - 1 > MAX_SUBR {
                        return fail();
                    }
                    let num = st.pop_int();
                    match self.subr(op1 == 29, num) {
                        Some(b) => bufs.push(b),
                        None => return fail(),
                    }
                    continue;
                }
                11 => {
                    if bufs.len() < 2 {
                        return fail();
                    }
                    bufs.pop();
                    continue;
                }
                12 => {
                    let op2 = read_byte!();
                    match op2 {
                        34..=37 => {
                            let rfs: [bool; 12] = match op2 {
                                34 => [true, false, true, true, true, false, true, false, true, false, true, false],
                                35 => [true; 12],
                                36 => [true, true, true, true, true, false, true, false, true, true, true, false],
                                _ => [true, true, true, true, true, true, true, true, true, true, false, false],
                            };
                            // cf2_doFlex
                            let mut vals = [0i32; 14];
                            vals[0] = cx;
                            vals[1] = cy;
                            let mut idx = 0;
                            let is_hflex = !rfs[9];
                            let top = if is_hflex { 9 } else { 10 };
                            for i in 0..top {
                                vals[i + 2] = vals[i];
                                if rfs[i] {
                                    vals[i + 2] = add(vals[i + 2], st.get(idx));
                                    idx += 1;
                                }
                            }
                            if is_hflex {
                                vals[11] = cy;
                            }
                            if op2 == 37 {
                                let last_is_x = vals[10].wrapping_sub(cx).wrapping_abs() > vals[11].wrapping_sub(cy).wrapping_abs();
                                let last = st.get(idx);
                                if last_is_x {
                                    vals[12] = add(vals[10], last);
                                    vals[13] = cy;
                                } else {
                                    vals[12] = cx;
                                    vals[13] = add(vals[11], last);
                                }
                            } else {
                                vals[12] = if rfs[10] {
                                    let v = add(vals[10], st.get(idx));
                                    idx += 1;
                                    v
                                } else {
                                    cx
                                };
                                vals[13] = if rfs[11] { add(vals[11], st.get(idx)) } else { cy };
                            }
                            for j in 0..2 {
                                let k = j * 6;
                                path.curve_to(vals[k + 2], vals[k + 3], vals[k + 4], vals[k + 5], vals[k + 6], vals[k + 7]);
                            }
                            st.clear();
                            cx = vals[12];
                            cy = vals[13];
                            continue;
                        }
                        3 | 4 => {
                            let b = st.pop_fixed();
                            let a = st.pop_fixed();
                            st.push_int(if op2 == 3 { (a != 0 && b != 0) as i32 } else { (a != 0 || b != 0) as i32 });
                            continue;
                        }
                        5 => {
                            let a = st.pop_fixed();
                            st.push_int((a == 0) as i32);
                            continue;
                        }
                        9 | 14 => {
                            let a = st.pop_fixed();
                            st.push_fixed(if a < -0x7FFFFFFF {
                                0x7FFFFFFF
                            } else if op2 == 9 {
                                a.abs()
                            } else {
                                -a
                            });
                            continue;
                        }
                        10 | 11 | 24 => {
                            let b = st.pop_fixed();
                            let a = st.pop_fixed();
                            st.push_fixed(match op2 {
                                10 => a.wrapping_add(b),
                                11 => a.wrapping_sub(b),
                                _ => mul_fix(a as i64, b as i64) as i32,
                            });
                            continue;
                        }
                        12 => {
                            let b = st.pop_fixed();
                            let a = st.pop_fixed();
                            st.push_fixed(div_fix(a as i64, b as i64) as i32);
                            continue;
                        }
                        15 => {
                            let b = st.pop_fixed();
                            let a = st.pop_fixed();
                            st.push_int((a == b) as i32);
                            continue;
                        }
                        16 | 17 => continue,
                        18 => {
                            st.pop_fixed();
                            continue;
                        }
                        20 => {
                            let idx = st.pop_int() as u32 as usize;
                            let val = st.pop_fixed();
                            if idx < STORAGE_SIZE {
                                storage[idx] = val;
                            }
                            continue;
                        }
                        21 => {
                            let idx = st.pop_int() as u32 as usize;
                            if idx < STORAGE_SIZE {
                                st.push_fixed(storage[idx]);
                            }
                            continue;
                        }
                        22 => {
                            let c2 = st.pop_fixed();
                            let c1 = st.pop_fixed();
                            let a2 = st.pop_fixed();
                            let a1 = st.pop_fixed();
                            st.push_fixed(if c1 <= c2 { a1 } else { a2 });
                            continue;
                        }
                        23 => {
                            let r = ((self.random & 0xFFFF) + 1) as i32;
                            let mut x = self.random;
                            x ^= x << 13;
                            x ^= x >> 17;
                            x ^= x << 5;
                            self.random = x;
                            st.push_fixed(r);
                            continue;
                        }
                        26 => {
                            let a = st.pop_fixed();
                            st.push_fixed(if a > 0 { sqrt_fixed(a as u32) as i32 } else { 0 });
                            continue;
                        }
                        27 => {
                            let a = st.pop_fixed();
                            st.push_fixed(a);
                            st.push_fixed(a);
                            continue;
                        }
                        28 => {
                            let b = st.pop_fixed();
                            let a = st.pop_fixed();
                            st.push_fixed(b);
                            st.push_fixed(a);
                            continue;
                        }
                        29 => {
                            let idx = st.pop_int();
                            let size = st.count();
                            if size > 0 {
                                let gi = if idx < 0 {
                                    size - 1
                                } else if idx as usize >= size {
                                    0
                                } else {
                                    size - 1 - idx as usize
                                };
                                let v = st.get(gi);
                                st.push_fixed(v);
                            }
                            continue;
                        }
                        30 => {
                            let idx = st.pop_int();
                            let count = st.pop_int();
                            st.roll(count, idx);
                            continue;
                        }
                        _ => {}
                    }
                }
                14 => {
                    // endchar
                    path.close_open_path();
                    if path.err {
                        return fail();
                    }
                    if st.count() > 1 {
                        // implied seac
                        if doing_seac {
                            return fail();
                        }
                        let achar = st.pop_int();
                        let bchar = st.pop_int();
                        let ay = st.pop_fixed();
                        let ax = st.pop_fixed();
                        if st.err {
                            return fail();
                        }
                        for (code, x, y) in [(achar, ax, ay), (bchar, 0, 0)] {
                            let gid = self.cff.glyph_by_stdcharcode(code).ok_or_else(|| err("invalid seac component"))?;
                            let comp = self.cff.charstrings.get(gid as usize).ok_or_else(|| err("invalid seac component"))?;
                            self.run(&mut *path.builder, comp, true, x, y)?;
                        }
                    }
                    if st.err {
                        return fail();
                    }
                    return Ok(());
                }
                19 | 20 => {
                    // hintmask / cntrmask
                    if st.count() > 1 && path.hints.mask_valid {
                        st.clear();
                        continue;
                    }
                    path.hints.v_stems += st.count() / 2;
                    let bit_count = path.hints.h_stems + path.hints.v_stems;
                    if bit_count > MAX_HINTS {
                        return fail();
                    }
                    if op1 == 19 {
                        path.hints.mask_valid = true;
                        path.hints.mask_new = true;
                    }
                    for _ in 0..bit_count.div_ceil(8) {
                        read_byte!();
                    }
                }
                21 => {
                    cy = add(cy, st.pop_fixed());
                    cx = add(cx, st.pop_fixed());
                    path.move_to(cx, cy);
                }
                22 => {
                    cx = add(cx, st.pop_fixed());
                    path.move_to(cx, cy);
                }
                25 => {
                    // rlinecurve
                    let count = st.count();
                    let mut idx = 0;
                    while idx + 6 < count {
                        cx = add(cx, st.get(idx));
                        cy = add(cy, st.get(idx + 1));
                        path.line_to(cx, cy);
                        idx += 2;
                    }
                    while idx < count {
                        let x1 = add(st.get(idx), cx);
                        let y1 = add(st.get(idx + 1), cy);
                        let x2 = add(st.get(idx + 2), x1);
                        let y2 = add(st.get(idx + 3), y1);
                        let x3 = add(st.get(idx + 4), x2);
                        let y3 = add(st.get(idx + 5), y2);
                        path.curve_to(x1, y1, x2, y2, x3, y3);
                        cx = x3;
                        cy = y3;
                        idx += 6;
                    }
                    st.clear();
                    continue;
                }
                26 | 27 => {
                    // vvcurveto / hhcurveto
                    let count1 = st.count();
                    let count = count1 & !2;
                    let mut idx = count1 - count;
                    while idx < count {
                        let (x1, y1, x2, y2, x3, y3);
                        if op1 == 26 {
                            if (count - idx) & 1 != 0 {
                                x1 = add(st.get(idx), cx);
                                idx += 1;
                            } else {
                                x1 = cx;
                            }
                            y1 = add(st.get(idx), cy);
                            x2 = add(st.get(idx + 1), x1);
                            y2 = add(st.get(idx + 2), y1);
                            x3 = x2;
                            y3 = add(st.get(idx + 3), y2);
                        } else {
                            if (count - idx) & 1 != 0 {
                                y1 = add(st.get(idx), cy);
                                idx += 1;
                            } else {
                                y1 = cy;
                            }
                            x1 = add(st.get(idx), cx);
                            x2 = add(st.get(idx + 1), x1);
                            y2 = add(st.get(idx + 2), y1);
                            x3 = add(st.get(idx + 3), x2);
                            y3 = y2;
                        }
                        path.curve_to(x1, y1, x2, y2, x3, y3);
                        cx = x3;
                        cy = y3;
                        idx += 4;
                    }
                    st.clear();
                    continue;
                }
                30 | 31 => {
                    // vhcurveto / hvcurveto
                    let count1 = st.count();
                    let count = count1 & !2;
                    let mut idx = count1 - count;
                    let mut alternate = op1 == 31;
                    while idx < count {
                        let (x1, y1, x2, y2, x3, y3);
                        if alternate {
                            x1 = add(st.get(idx), cx);
                            y1 = cy;
                            x2 = add(st.get(idx + 1), x1);
                            y2 = add(st.get(idx + 2), y1);
                            y3 = add(st.get(idx + 3), y2);
                            if count - idx == 5 {
                                x3 = add(st.get(idx + 4), x2);
                                idx += 1;
                            } else {
                                x3 = x2;
                            }
                        } else {
                            x1 = cx;
                            y1 = add(st.get(idx), cy);
                            x2 = add(st.get(idx + 1), x1);
                            y2 = add(st.get(idx + 2), y1);
                            x3 = add(st.get(idx + 3), x2);
                            if count - idx == 5 {
                                y3 = add(st.get(idx + 4), y2);
                                idx += 1;
                            } else {
                                y3 = y2;
                            }
                        }
                        alternate = !alternate;
                        path.curve_to(x1, y1, x2, y2, x3, y3);
                        cx = x3;
                        cy = y3;
                        idx += 4;
                    }
                    st.clear();
                    continue;
                }
                28 => {
                    let b1 = read_byte!();
                    let b2 = read_byte!();
                    st.push_int(((b1 << 8) | b2) as u16 as i16 as i32);
                    continue;
                }
                32..=246 => {
                    st.push_int(op1 - 139);
                    continue;
                }
                247..=250 => {
                    let b = read_byte!();
                    st.push_int((op1 - 247) * 256 + b + 108);
                    continue;
                }
                251..=254 => {
                    let b = read_byte!();
                    st.push_int(-(op1 - 251) * 256 - b - 108);
                    continue;
                }
                255 => {
                    let mut v: u32 = 0;
                    for _ in 0..4 {
                        v = (v << 8) | read_byte!() as u32;
                    }
                    st.push_fixed(v as i32);
                    continue;
                }
                // reserved, closepath (Type 1 only), vsindex/blend (CFF2 only)
                _ => {}
            }
            st.clear();
        }
    }
}

/// `cff_standard_encoding`: standard character code -> SID.
const STANDARD_ENCODING: [u16; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    0, 111, 112, 113, 114, 0, 115, 116, 117, 118, 119, 120, 121, 122, 0, 123,
    0, 124, 125, 126, 127, 128, 129, 130, 131, 0, 132, 133, 0, 134, 135, 136,
    137, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 138, 0, 139, 0, 0, 0, 0, 140, 141, 142, 143, 0, 0, 0, 0,
    0, 144, 0, 0, 0, 145, 0, 0, 146, 147, 148, 149, 0, 0, 0, 0,
];
