//! Character maps as FreeType loads and selects them (`sfnt/ttcmap.c`,
//! `find_unicode_charmap`), for `FT_Get_Char_Index`.

#[inline]
fn u16_at(d: &[u8], o: usize) -> Option<u16> {
    d.get(o..o + 2).map(|b| u16::from_be_bytes([b[0], b[1]]))
}

#[inline]
fn u32_at(d: &[u8], o: usize) -> Option<u32> {
    d.get(o..o + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

/// One `cmap` subtable (a FreeType charmap).
#[derive(Clone, Copy, Debug)]
pub struct Subtable {
    pub platform: u16,
    pub encoding: u16,
    pub format: u16,
    /// offset of the subtable within the `cmap` table
    pub offset: usize,
}

impl Subtable {
    /// `sfnt_find_encoding` == `FT_ENCODING_UNICODE`.
    pub fn is_unicode(&self) -> bool {
        matches!(self.platform, 0 | 2) || (self.platform == 3 && matches!(self.encoding, 1 | 10))
    }
}

/// The charmaps of a `cmap` table, in table order.
pub fn subtables(cmap: &[u8]) -> Vec<Subtable> {
    let mut out = Vec::new();
    let Some(n) = u16_at(cmap, 2) else { return out };
    for i in 0..n as usize {
        let rec = 4 + 8 * i;
        let (Some(platform), Some(encoding), Some(off)) = (u16_at(cmap, rec), u16_at(cmap, rec + 2), u32_at(cmap, rec + 4)) else {
            break;
        };
        let offset = off as usize;
        if let Some(format) = u16_at(cmap, offset) {
            out.push(Subtable { platform, encoding, format, offset });
        }
    }
    out
}

/// `find_unicode_charmap`: the index of the charmap FreeType selects.
pub fn default_charmap(subs: &[Subtable]) -> Option<usize> {
    let ucs4 = |s: &Subtable| {
        (s.platform == 3 && s.encoding == 10) || (s.platform == 0 && s.encoding == 4) || (s.platform == 0 && s.encoding == 6 && s.format == 13)
    };
    (0..subs.len())
        .rev()
        .find(|&i| subs[i].is_unicode() && subs[i].format != 14 && ucs4(&subs[i]))
        .or_else(|| (0..subs.len()).rev().find(|&i| subs[i].is_unicode() && subs[i].format != 14))
}

/// The glyph of character `c` in subtable `s` (0 if unmapped).
pub fn char_index(cmap: &[u8], s: &Subtable, c: u32) -> u32 {
    let d = cmap;
    let o = s.offset;
    let r = || -> Option<u32> {
        match s.format {
            0 => {
                if c < 256 {
                    Some(*d.get(o + 6 + c as usize)? as u32)
                } else {
                    Some(0)
                }
            }
            4 => {
                if c > 0xFFFF {
                    return Some(0);
                }
                let segx2 = u16_at(d, o + 6)? as usize;
                let seg = segx2 / 2;
                let ends = o + 14;
                let starts = ends + segx2 + 2;
                let deltas = starts + segx2;
                let ranges = deltas + segx2;
                // binary search for the first segment with end >= c
                let (mut lo, mut hi) = (0usize, seg);
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if (u16_at(d, ends + 2 * mid)? as u32) < c {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                if lo >= seg {
                    return Some(0);
                }
                let start = u16_at(d, starts + 2 * lo)? as u32;
                if c < start {
                    return Some(0);
                }
                let delta = u16_at(d, deltas + 2 * lo)? as u32;
                let ro = u16_at(d, ranges + 2 * lo)? as usize;
                if ro == 0 {
                    Some((c + delta) & 0xFFFF)
                } else if ro == 0xFFFF {
                    Some(0)
                } else {
                    let q = ranges + 2 * lo + ro + 2 * (c - start) as usize;
                    let g = u16_at(d, q)? as u32;
                    Some(if g == 0 { 0 } else { (g + delta) & 0xFFFF })
                }
            }
            6 => {
                let first = u16_at(d, o + 6)? as u32;
                let count = u16_at(d, o + 8)? as u32;
                if c >= first && c < first + count {
                    Some(u16_at(d, o + 10 + 2 * (c - first) as usize)? as u32)
                } else {
                    Some(0)
                }
            }
            12 | 13 => {
                let n = u32_at(d, o + 12)? as usize;
                let (mut lo, mut hi) = (0usize, n);
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    let g = o + 16 + 12 * mid;
                    let (start, end) = (u32_at(d, g)?, u32_at(d, g + 4)?);
                    if c < start {
                        hi = mid;
                    } else if c > end {
                        lo = mid + 1;
                    } else {
                        let gid = u32_at(d, g + 8)?;
                        return Some(if s.format == 12 { gid + (c - start) } else { gid });
                    }
                }
                Some(0)
            }
            _ => Some(0),
        }
    };
    r().unwrap_or(0)
}
