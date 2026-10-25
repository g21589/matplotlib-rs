//! Minimal SFNT/TrueType table access and FreeType-compatible unhinted
//! glyph loading (`ttgload.c` semantics for `FT_LOAD_NO_HINTING`).

use super::cff::CffFont;
use super::outline::{hypot, mul_fix, FtMatrix, Outline};

#[derive(Debug)]
pub struct FontError(pub String);

impl std::fmt::Display for FontError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for FontError {}

#[inline(always)]
fn u16_at(d: &[u8], o: usize) -> u16 {
    u16::from_be_bytes([d[o], d[o + 1]])
}
#[inline(always)]
fn i16_at(d: &[u8], o: usize) -> i16 {
    i16::from_be_bytes([d[o], d[o + 1]])
}
#[inline(always)]
fn u32_at(d: &[u8], o: usize) -> u32 {
    u32::from_be_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TableRange {
    pub offset: usize,
    pub len: usize,
}

/// Parsed font data (owned bytes plus table locations).
pub struct SfntFace {
    pub data: Vec<u8>,
    pub tables: Vec<([u8; 4], TableRange)>,
    pub units_per_em: u16,
    pub head_flags: u16,
    pub index_to_loc_format: i16,
    pub bbox: (i16, i16, i16, i16),
    pub num_glyphs: u16,
    pub num_hmetrics: u16,
    pub ascender: i16,
    pub descender: i16,
    pub line_gap: i16,
    pub max_advance_width: u16,
    hmtx: TableRange,
    loca: TableRange,
    glyf: TableRange,
    /// CFF outlines (OpenType `CFF ` table) instead of `glyf`.
    pub cff: Option<CffFont>,
    pub family_name: String,
    pub style_name: String,
    pub postscript_name: String,
    pub os2_weight: u16,
    pub mac_style: u16,
    pub italic_angle: f64,
    pub underline_position: i16,
    pub underline_thickness: i16,
    pub is_fixed_pitch: bool,
    /// OS/2 table (None when missing, like `FT_Get_Sfnt_Table`).
    pub os2: Option<Os2>,
    /// Raw name records: (platform, encoding, language, name id, bytes).
    pub names: Vec<(u16, u16, u16, u16, Vec<u8>)>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Os2 {
    pub version: u16,
    pub us_weight_class: u16,
    pub fs_selection: u16,
    pub s_typo_ascender: i16,
    pub s_typo_descender: i16,
    pub s_typo_line_gap: i16,
}

pub const ARGS_ARE_WORDS: u16 = 0x0001;
pub const ARGS_ARE_XY_VALUES: u16 = 0x0002;
pub const WE_HAVE_A_SCALE: u16 = 0x0008;
pub const MORE_COMPONENTS: u16 = 0x0020;
pub const WE_HAVE_AN_XY_SCALE: u16 = 0x0040;
pub const WE_HAVE_A_2X2: u16 = 0x0080;
pub const USE_MY_METRICS: u16 = 0x0200;
pub const OVERLAP_COMPOUND: u16 = 0x0400;
pub const SCALED_COMPONENT_OFFSET: u16 = 0x0800;

const ON_CURVE_POINT: u8 = 0x01;
const X_SHORT_VECTOR: u8 = 0x02;
const Y_SHORT_VECTOR: u8 = 0x04;
const REPEAT_FLAG: u8 = 0x08;
const X_POSITIVE: u8 = 0x10;
const SAME_X: u8 = 0x10;
const Y_POSITIVE: u8 = 0x20;
const SAME_Y: u8 = 0x20;
const OVERLAP_SIMPLE: u8 = 0x40;

/// Phantom points (pp1, pp2) horizontal values in the current units.
#[derive(Clone, Copy, Debug, Default)]
struct Phantom {
    pp1x: i64,
    pp2x: i64,
}

impl SfntFace {
    pub fn parse(data: Vec<u8>, face_index: u32) -> Result<SfntFace, FontError> {
        let err = |m: &str| FontError(m.to_string());
        if data.len() < 12 {
            return Err(err("font file too short"));
        }
        let mut base = 0usize;
        let tag = &data[0..4];
        if tag == b"ttcf" {
            let n = u32_at(&data, 8);
            if face_index >= n {
                return Err(err("face index out of range"));
            }
            base = u32_at(&data, 12 + 4 * face_index as usize) as usize;
        }
        let num_tables = u16_at(&data, base + 4) as usize;
        let mut tables = Vec::with_capacity(num_tables);
        for i in 0..num_tables {
            let rec = base + 12 + 16 * i;
            if rec + 16 > data.len() {
                return Err(err("bad table directory"));
            }
            let mut t = [0u8; 4];
            t.copy_from_slice(&data[rec..rec + 4]);
            let offset = u32_at(&data, rec + 8) as usize;
            let len = u32_at(&data, rec + 12) as usize;
            tables.push((t, TableRange { offset, len }));
        }
        let find = |name: &[u8; 4]| tables.iter().find(|(t, _)| t == name).map(|(_, r)| *r);
        let head = find(b"head").ok_or_else(|| err("missing head table"))?;
        let hhea = find(b"hhea").ok_or_else(|| err("missing hhea table"))?;
        let maxp = find(b"maxp").ok_or_else(|| err("missing maxp table"))?;
        let hmtx = find(b"hmtx").ok_or_else(|| err("missing hmtx table"))?;
        let loca = find(b"loca").unwrap_or_default();
        let glyf = find(b"glyf").unwrap_or_default();
        let cff_table = find(b"CFF ").unwrap_or_default();
        if glyf.len == 0 && cff_table.len == 0 {
            return Err(err("only TrueType (glyf) and CFF outlines are supported"));
        }

        let h = head.offset;
        let units_per_em = u16_at(&data, h + 18);
        let head_flags = u16_at(&data, h + 16);
        let bbox = (i16_at(&data, h + 36), i16_at(&data, h + 38), i16_at(&data, h + 40), i16_at(&data, h + 42));
        let mac_style = u16_at(&data, h + 44);
        let index_to_loc_format = i16_at(&data, h + 50);
        let num_glyphs = u16_at(&data, maxp.offset + 4);
        let hh = hhea.offset;
        let ascender = i16_at(&data, hh + 4);
        let descender = i16_at(&data, hh + 6);
        let line_gap = i16_at(&data, hh + 8);
        let max_advance_width = u16_at(&data, hh + 10);
        let num_hmetrics = u16_at(&data, hh + 34);
        let cff = if glyf.len == 0 {
            let t = data.get(cff_table.offset..cff_table.offset + cff_table.len).ok_or_else(|| err("bad CFF table"))?;
            Some(CffFont::parse(t, units_per_em)?)
        } else {
            None
        };
        let num_glyphs = cff.as_ref().map(|c| c.num_glyphs as u16).unwrap_or(num_glyphs);

        let mut face = SfntFace {
            data,
            tables,
            units_per_em,
            head_flags,
            index_to_loc_format,
            bbox,
            num_glyphs,
            num_hmetrics,
            ascender,
            descender,
            line_gap,
            max_advance_width,
            hmtx,
            loca,
            glyf,
            cff,
            family_name: String::new(),
            style_name: String::new(),
            postscript_name: String::new(),
            os2_weight: 400,
            mac_style,
            italic_angle: 0.0,
            underline_position: 0,
            underline_thickness: 0,
            is_fixed_pitch: false,
            os2: None,
            names: Vec::new(),
        };
        face.read_names();
        if let Some(os2) = face.table(b"OS/2") {
            if os2.len >= 6 {
                face.os2_weight = u16_at(&face.data, os2.offset + 4);
            }
            if os2.len >= 78 {
                let o = os2.offset;
                let d = &face.data;
                face.os2 = Some(Os2 {
                    version: u16_at(d, o),
                    us_weight_class: u16_at(d, o + 4),
                    fs_selection: u16_at(d, o + 62),
                    s_typo_ascender: i16_at(d, o + 68),
                    s_typo_descender: i16_at(d, o + 70),
                    s_typo_line_gap: i16_at(d, o + 72),
                });
            }
        }
        if let Some(post) = face.table(b"post") {
            if post.len >= 16 {
                let o = post.offset;
                let ia = u32_at(&face.data, o + 4) as i32;
                face.italic_angle = ia as f64 / 65536.0;
                face.underline_position = i16_at(&face.data, o + 8);
                face.underline_thickness = i16_at(&face.data, o + 10);
                face.is_fixed_pitch = u32_at(&face.data, o + 12) != 0;
            }
        }
        Ok(face)
    }

    /// Raw bytes of a name record (`FT2Font.get_sfnt()[key]`).
    pub fn name_entry(&self, platform: u16, encoding: u16, language: u16, name_id: u16) -> Option<&[u8]> {
        self.names
            .iter()
            .rev()
            .find(|(p, e, l, n, _)| *p == platform && *e == encoding && *l == language && *n == name_id)
            .map(|r| r.4.as_slice())
    }

    pub fn table(&self, name: &[u8; 4]) -> Option<TableRange> {
        self.tables.iter().find(|(t, _)| t == name).map(|(_, r)| *r)
    }

    pub fn table_data(&self, name: &[u8; 4]) -> Option<&[u8]> {
        self.table(name).and_then(|r| self.data.get(r.offset..r.offset + r.len))
    }

    fn read_names(&mut self) {
        let Some(name) = self.table(b"name") else { return };
        let d = &self.data;
        let o = name.offset;
        if o + 6 > d.len() {
            return;
        }
        let count = u16_at(d, o + 2) as usize;
        let storage = o + u16_at(d, o + 4) as usize;
        let mut best: [(i32, String); 18] = Default::default();
        let mut names = Vec::new();
        for i in 0..count {
            let r = o + 6 + 12 * i;
            if r + 12 > d.len() {
                break;
            }
            let platform = u16_at(d, r);
            let encoding = u16_at(d, r + 2);
            let language = u16_at(d, r + 4);
            let name_id = u16_at(d, r + 6) as usize;
            let len = u16_at(d, r + 8) as usize;
            let off = u16_at(d, r + 10) as usize;
            let s = storage + off;
            if s + len > d.len() {
                continue;
            }
            if name_id >= best.len() {
                names.push((platform, encoding, language, name_id as u16, d[s..s + len].to_vec()));
                continue;
            }
            let bytes = &d[s..s + len];
            names.push((platform, encoding, language, name_id as u16, bytes.to_vec()));
            // FreeType prefers Microsoft Unicode English names.
            let (score, text) = if platform == 3 && (encoding == 1 || encoding == 10) {
                let u: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
                (if language == 0x409 { 3 } else { 2 }, String::from_utf16_lossy(&u))
            } else if platform == 1 && encoding == 0 {
                (1, bytes.iter().map(|&b| b as char).collect())
            } else if platform == 0 {
                let u: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
                (2, String::from_utf16_lossy(&u))
            } else {
                continue;
            };
            if score > best[name_id].0 {
                best[name_id] = (score, text);
            }
        }
        self.names = names;
        // FreeType uses the typographic family/subfamily names when present.
        self.family_name = if !best[16].1.is_empty() { std::mem::take(&mut best[16].1) } else { std::mem::take(&mut best[1].1) };
        self.style_name = if !best[17].1.is_empty() { std::mem::take(&mut best[17].1) } else { std::mem::take(&mut best[2].1) };
        self.postscript_name = std::mem::take(&mut best[6].1);
    }

    /// `tt_face_get_kerning`: the 'kern' table value of a glyph pair (font
    /// units), summed over the horizontal format-0 subtables FreeType uses.
    pub fn kerning(&self, left: u32, right: u32) -> i32 {
        let Some(t) = self.table_data(b"kern") else { return 0 };
        if t.len() < 4 || u16_at(t, 0) != 0 {
            return 0;
        }
        let ntables = u16_at(t, 2) as usize;
        let key = (left << 16) | (right & 0xFFFF);
        let mut p = 4usize;
        let mut result: i32 = 0;
        for _ in 0..ntables {
            if p + 6 > t.len() {
                break;
            }
            let length = u16_at(t, p + 2) as usize;
            let coverage = u16_at(t, p + 4);
            let next = p + length.max(6);
            // only horizontal format-0 kerning, optionally overriding
            if (coverage & !8u16) == 0x0001 && p + 14 <= t.len() {
                let npairs = u16_at(t, p + 6) as usize;
                let base = p + 14;
                let avail = t.len().saturating_sub(base) / 6;
                let n = npairs.min(avail);
                // binary search on the (left, right) key
                let (mut lo, mut hi) = (0usize, n);
                let mut found = None;
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    let q = base + 6 * mid;
                    let k = ((u16_at(t, q) as u32) << 16) | u16_at(t, q + 2) as u32;
                    if k == key {
                        found = Some(i16_at(t, q + 4) as i32);
                        break;
                    } else if k < key {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                if let Some(v) = found {
                    if coverage & 8 != 0 {
                        result = v;
                    } else {
                        result += v;
                    }
                }
            }
            p = next;
        }
        result
    }

    /// `TT_Get_HMetrics`: (left side bearing, advance width) in font units.
    pub fn hmetrics(&self, gid: u32) -> (i16, u16) {
        let d = &self.data;
        let base = self.hmtx.offset;
        let end = base + self.hmtx.len;
        let k = self.num_hmetrics as u32;
        if k == 0 {
            return (0, 0);
        }
        if gid < k {
            let p = base + 4 * gid as usize;
            if p + 4 > end {
                return (0, 0);
            }
            (i16_at(d, p + 2), u16_at(d, p))
        } else {
            let p = base + 4 * (k as usize - 1);
            if p + 2 > end {
                return (0, 0);
            }
            let aw = u16_at(d, p);
            let q = p + 4 + 2 * (gid - k) as usize;
            let lsb = if q + 2 > end { 0 } else { i16_at(d, q) };
            (lsb, aw)
        }
    }

    /// `FT_Get_Glyph_Name`: the glyph's PostScript name (`post` table, or
    /// the CFF charset).
    pub fn glyph_name(&self, gid: u32) -> Option<String> {
        if let Some(cff) = &self.cff {
            return cff.glyph_name(gid);
        }
        let post = self.table(b"post")?;
        let d = &self.data[post.offset..post.offset + post.len];
        if d.len() < 32 {
            return None;
        }
        let format = u32_at(d, 0);
        let mac = |i: usize| super::names::MAC_GLYPH_NAMES.get(i).map(|s| s.to_string());
        match format {
            0x00010000 => mac(gid as usize).or_else(|| mac(0)),
            0x00020000 => {
                let n = u16_at(d, 32) as usize;
                if gid as usize >= n || d.len() < 34 + 2 * n {
                    return mac(0);
                }
                let idx = u16_at(d, 34 + 2 * gid as usize) as usize;
                if idx < 258 {
                    return mac(idx);
                }
                // Pascal strings after the index array
                let mut p = 34 + 2 * n;
                let mut k = 258;
                while p < d.len() {
                    let len = d[p] as usize;
                    if k == idx {
                        let end = (p + 1 + len).min(d.len());
                        return Some(String::from_utf8_lossy(&d[p + 1..end]).into_owned());
                    }
                    p += 1 + len;
                    k += 1;
                }
                Some(String::new())
            }
            _ => mac(0),
        }
    }

    /// Location (offset, length) of a glyph within the glyf table.
    fn glyph_location(&self, gid: u32) -> (usize, usize) {
        let d = &self.data;
        let lo = self.loca.offset;
        let (start, end) = if self.index_to_loc_format == 0 {
            let p = lo + 2 * gid as usize;
            if p + 4 > lo + self.loca.len {
                return (0, 0);
            }
            (u16_at(d, p) as usize * 2, u16_at(d, p + 2) as usize * 2)
        } else {
            let p = lo + 4 * gid as usize;
            if p + 8 > lo + self.loca.len {
                return (0, 0);
            }
            (u32_at(d, p) as usize, u32_at(d, p + 4) as usize)
        };
        if end <= start || start >= self.glyf.len {
            return (start, 0);
        }
        let end = end.min(self.glyf.len);
        (start, end - start)
    }

    /// Load a glyph outline scaled by (x_scale, y_scale) (16.16), unhinted,
    /// exactly like `FT_Load_Glyph(..., FT_LOAD_NO_HINTING)` for TrueType.
    pub fn load_glyph(&self, gid: u32, x_scale: i64, y_scale: i64) -> Result<Outline, FontError> {
        Ok(self.load_glyph_advance(gid, x_scale, y_scale)?.0)
    }

    /// `load_glyph` and the glyph's `metrics.horiAdvance` (for TrueType the
    /// distance of the scaled phantom points).
    pub fn load_glyph_advance(&self, gid: u32, x_scale: i64, y_scale: i64) -> Result<(Outline, i64), FontError> {
        if gid >= self.num_glyphs as u32 {
            return Err(FontError("invalid glyph index".into()));
        }
        if let Some(cff) = &self.cff {
            let (_, aw) = self.hmetrics(gid);
            return Ok((cff.load_glyph(gid, x_scale, y_scale)?, cff.hori_advance(gid, aw as i64, x_scale)));
        }
        let mut out = Outline::default();
        let mut pp = Phantom::default();
        self.load_rec(gid, x_scale, y_scale, 0, &mut out, &mut pp)?;
        // Translate array so that (0,0) is the glyph's origin.
        if pp.pp1x != 0 {
            out.translate(-pp.pp1x, 0);
        }
        Ok((out, pp.pp2x - pp.pp1x))
    }

    fn load_rec(
        &self,
        gid: u32,
        x_scale: i64,
        y_scale: i64,
        depth: u32,
        out: &mut Outline,
        pp: &mut Phantom,
    ) -> Result<(), FontError> {
        if depth > 100 {
            return Err(FontError("recursion depth exceeded".into()));
        }
        let d = &self.data;
        let (goff, glen) = self.glyph_location(gid);
        let g = self.glyf.offset + goff;
        let mut n_contours: i32 = 0;
        let mut bbox_xmin: i64 = 0;
        if glen > 0 {
            if glen < 10 {
                return Err(FontError("invalid outline".into()));
            }
            n_contours = i16_at(d, g) as i32;
            bbox_xmin = i16_at(d, g + 2) as i64;
        }
        if glen == 0 || n_contours == 0 {
            bbox_xmin = 0;
        }
        let (lsb, aw) = self.hmetrics(gid);
        // tt_loader_set_pp (font units)
        let pp1x = bbox_xmin - lsb as i64;
        let pp2x = pp1x + aw as i64;

        if glen == 0 || n_contours == 0 {
            pp.pp1x = mul_fix(pp1x, x_scale);
            pp.pp2x = mul_fix(pp2x, x_scale);
            return Ok(());
        }

        let limit = g + glen;
        if n_contours > 0 {
            // simple glyph
            let mut p = g + 10;
            let nc = n_contours as usize;
            if p + 2 * nc + 2 > limit {
                return Err(FontError("invalid outline".into()));
            }
            let mut contours = Vec::with_capacity(nc);
            let mut last: i32 = -1;
            for _ in 0..nc {
                let c = u16_at(d, p) as i32;
                p += 2;
                if c <= last {
                    return Err(FontError("invalid outline".into()));
                }
                last = c;
                contours.push(c as u16);
            }
            let n_points = (last + 1) as usize;
            let n_ins = u16_at(d, p) as usize;
            p += 2;
            if p + n_ins > limit {
                return Err(FontError("too many hints".into()));
            }
            p += n_ins;

            let mut flags = vec![0u8; n_points];
            let mut i = 0;
            while i < n_points {
                if p + 1 > limit {
                    return Err(FontError("invalid outline".into()));
                }
                let c = d[p];
                p += 1;
                flags[i] = c;
                i += 1;
                if c & REPEAT_FLAG != 0 {
                    if p + 1 > limit {
                        return Err(FontError("invalid outline".into()));
                    }
                    let count = d[p] as usize;
                    p += 1;
                    if i + count > n_points {
                        return Err(FontError("invalid outline".into()));
                    }
                    for _ in 0..count {
                        flags[i] = c;
                        i += 1;
                    }
                }
            }
            let overlap = n_points > 0 && flags[0] & OVERLAP_SIMPLE != 0;

            let mut xs = vec![0i64; n_points];
            let mut x: i64 = 0;
            for k in 0..n_points {
                let f = flags[k];
                let mut delta: i64 = 0;
                if f & X_SHORT_VECTOR != 0 {
                    if p + 1 > limit {
                        return Err(FontError("invalid outline".into()));
                    }
                    delta = d[p] as i64;
                    p += 1;
                    if f & X_POSITIVE == 0 {
                        delta = -delta;
                    }
                } else if f & SAME_X == 0 {
                    if p + 2 > limit {
                        return Err(FontError("invalid outline".into()));
                    }
                    delta = i16_at(d, p) as i64;
                    p += 2;
                }
                x += delta;
                xs[k] = x;
            }
            let mut ys = vec![0i64; n_points];
            let mut y: i64 = 0;
            for k in 0..n_points {
                let f = flags[k];
                let mut delta: i64 = 0;
                if f & Y_SHORT_VECTOR != 0 {
                    if p + 1 > limit {
                        return Err(FontError("invalid outline".into()));
                    }
                    delta = d[p] as i64;
                    p += 1;
                    if f & Y_POSITIVE == 0 {
                        delta = -delta;
                    }
                } else if f & SAME_Y == 0 {
                    if p + 2 > limit {
                        return Err(FontError("invalid outline".into()));
                    }
                    delta = i16_at(d, p) as i64;
                    p += 2;
                }
                y += delta;
                ys[k] = y;
            }

            // TT_Process_Simple_Glyph: scale points (and phantom points)
            let base = out.points.len() as u16;
            out.points.reserve(n_points);
            for k in 0..n_points {
                out.points.push((mul_fix(xs[k], x_scale), mul_fix(ys[k], y_scale)));
                out.tags.push(flags[k] & ON_CURVE_POINT);
            }
            for c in contours {
                out.contours.push(c + base);
            }
            if overlap {
                out.overlap = true;
            }
            pp.pp1x = mul_fix(pp1x, x_scale);
            pp.pp2x = mul_fix(pp2x, x_scale);
            return Ok(());
        }

        // composite glyph: scale phantom points
        pp.pp1x = mul_fix(pp1x, x_scale);
        pp.pp2x = mul_fix(pp2x, x_scale);

        struct Sub {
            flags: u16,
            index: u32,
            arg1: i64,
            arg2: i64,
            m: FtMatrix,
        }
        let mut subs: Vec<Sub> = Vec::new();
        let mut p = g + 10;
        loop {
            if p + 4 > limit {
                return Err(FontError("invalid composite".into()));
            }
            let flags = u16_at(d, p);
            let index = u16_at(d, p + 2) as u32;
            p += 4;
            if index >= self.num_glyphs as u32 {
                return Err(FontError("invalid composite".into()));
            }
            let mut count = 2;
            if flags & ARGS_ARE_WORDS != 0 {
                count += 2;
            }
            if flags & WE_HAVE_A_SCALE != 0 {
                count += 2;
            } else if flags & WE_HAVE_AN_XY_SCALE != 0 {
                count += 4;
            } else if flags & WE_HAVE_A_2X2 != 0 {
                count += 8;
            }
            if p + count > limit {
                return Err(FontError("invalid composite".into()));
            }
            let (arg1, arg2);
            if flags & ARGS_ARE_XY_VALUES != 0 {
                if flags & ARGS_ARE_WORDS != 0 {
                    arg1 = i16_at(d, p) as i64;
                    arg2 = i16_at(d, p + 2) as i64;
                    p += 4;
                } else {
                    arg1 = d[p] as i8 as i64;
                    arg2 = d[p + 1] as i8 as i64;
                    p += 2;
                }
            } else if flags & ARGS_ARE_WORDS != 0 {
                arg1 = u16_at(d, p) as i64;
                arg2 = u16_at(d, p + 2) as i64;
                p += 4;
            } else {
                arg1 = d[p] as i64;
                arg2 = d[p + 1] as i64;
                p += 2;
            }
            let mut m = FtMatrix::IDENTITY;
            if flags & WE_HAVE_A_SCALE != 0 {
                m.xx = i16_at(d, p) as i64 * 4;
                m.yy = m.xx;
                p += 2;
            } else if flags & WE_HAVE_AN_XY_SCALE != 0 {
                m.xx = i16_at(d, p) as i64 * 4;
                m.yy = i16_at(d, p + 2) as i64 * 4;
                p += 4;
            } else if flags & WE_HAVE_A_2X2 != 0 {
                m.xx = i16_at(d, p) as i64 * 4;
                m.yx = i16_at(d, p + 2) as i64 * 4;
                m.xy = i16_at(d, p + 4) as i64 * 4;
                m.yy = i16_at(d, p + 6) as i64 * 4;
                p += 8;
            }
            subs.push(Sub { flags, index, arg1, arg2, m });
            if flags & MORE_COMPONENTS == 0 {
                break;
            }
        }

        let start_point = out.points.len();
        for sub in &subs {
            let saved = *pp;
            let num_base_points = out.points.len();
            self.load_rec(sub.index, x_scale, y_scale, depth + 1, out, pp)?;
            if sub.flags & USE_MY_METRICS == 0 {
                *pp = saved;
            }
            let num_points = out.points.len();
            if num_points == num_base_points {
                continue;
            }
            // TT_Process_Composite_Component
            let have_scale = sub.flags & (WE_HAVE_A_SCALE | WE_HAVE_AN_XY_SCALE | WE_HAVE_A_2X2) != 0;
            if have_scale {
                for pt in &mut out.points[num_base_points..] {
                    *pt = sub.m.apply(pt.0, pt.1);
                }
            }
            let (mut x, mut y);
            if sub.flags & ARGS_ARE_XY_VALUES == 0 {
                let k = sub.arg1 as usize + start_point;
                let l = sub.arg2 as usize + num_base_points;
                if k >= num_base_points || l >= num_points {
                    return Err(FontError("invalid composite".into()));
                }
                x = out.points[k].0 - out.points[l].0;
                y = out.points[k].1 - out.points[l].1;
            } else {
                x = sub.arg1;
                y = sub.arg2;
                if x == 0 && y == 0 {
                    continue;
                }
                if have_scale && (sub.flags & SCALED_COMPONENT_OFFSET) != 0 {
                    let mac_xscale = hypot(sub.m.xx, sub.m.xy);
                    let mac_yscale = hypot(sub.m.yy, sub.m.yx);
                    x = mul_fix(x, mac_xscale);
                    y = mul_fix(y, mac_yscale);
                }
                x = mul_fix(x, x_scale);
                y = mul_fix(y, y_scale);
            }
            if x != 0 || y != 0 {
                for pt in &mut out.points[num_base_points..] {
                    pt.0 += x;
                    pt.1 += y;
                }
            }
        }
        if let Some(first) = subs.first() {
            if first.flags & OVERLAP_COMPOUND != 0 {
                out.overlap = true;
            }
        }
        Ok(())
    }
}
