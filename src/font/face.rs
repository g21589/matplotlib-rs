//! `FT2Font` equivalent: a TrueType face with FreeType-compatible sizing,
//! transforms and rendering, and libraqm/HarfBuzz(hb-ft)-compatible layout.

use std::sync::Arc;

use super::grays;
use super::outline::{div_fix, mul_fix, BBox, FtMatrix, Outline};
use super::sfnt::{FontError, SfntFace};

/// A rendered glyph bitmap (`FT_Bitmap` + `bitmap_left/top`).
#[derive(Clone, Debug, Default)]
pub struct GlyphBitmap {
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub rows: u32,
    /// Row-major coverage, row 0 is the top row.
    pub buffer: Vec<u8>,
}

/// A shaped glyph as returned by libraqm (`raqm_glyph_t`), 26.6 units.
#[derive(Clone, Copy, Debug)]
pub struct ShapedGlyph {
    pub index: u32,
    pub cluster: u32,
    pub x_advance: i32,
    pub y_advance: i32,
    pub x_offset: i32,
    pub y_offset: i32,
}

/// Glyph metrics in 26.6 (`linear_hori_advance` in 16.16).
#[derive(Clone, Copy, Debug, Default)]
pub struct GlyphMetrics {
    pub width: i64,
    pub height: i64,
    pub hori_bearing_x: i64,
    pub hori_bearing_y: i64,
    pub hori_advance: i64,
    pub linear_hori_advance: i64,
    pub bbox: BBox,
}

/// Size state (`FT_Size_Metrics`, unhinted root metrics).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SizeMetrics {
    pub x_ppem: u16,
    pub y_ppem: u16,
    pub x_scale: i64,
    pub y_scale: i64,
}

pub struct FontData {
    pub sfnt: SfntFace,
    pub hb_font: harfrust::Font,
}

/// An `FT2Font`-like object; cheap to create from shared [`FontData`].
pub struct FtFont {
    pub data: Arc<FontData>,
    pub size: SizeMetrics,
    pub matrix: FtMatrix,
    pub delta: (i64, i64),
    transform_flags: u32,
}

impl FontData {
    pub fn from_bytes(bytes: Vec<u8>, index: u32) -> Result<FontData, FontError> {
        let hb_font = harfrust::Font::new(bytes.clone(), index)
            .ok_or_else(|| FontError("could not parse font for shaping".into()))?;
        let sfnt = SfntFace::parse(bytes, index)?;
        Ok(FontData { sfnt, hb_font })
    }
}

struct HbFtFuncs<'a> {
    font: &'a FtFont,
}

impl<'a> harfrust::FontFuncs for HbFtFuncs<'a> {
    fn glyph_h_advance(&self, _font: &harfrust::ShaperFont, glyph: harfrust::GlyphId) -> i32 {
        self.font.hb_ft_advance(glyph.to_u32())
    }
}

impl FtFont {
    pub fn new(data: Arc<FontData>) -> Self {
        FtFont { data, size: SizeMetrics::default(), matrix: FtMatrix::IDENTITY, delta: (0, 0), transform_flags: 0 }
    }

    #[inline]
    pub fn sfnt(&self) -> &SfntFace {
        &self.data.sfnt
    }

    /// `FT2Font::set_size`: `FT_Set_Char_Size(ptsize * 64, 0, dpi, dpi)` and
    /// reset the transform to identity.
    pub fn set_size(&mut self, ptsize: f64, dpi: f64) {
        let mut char_width = (ptsize * 64.0) as i64;
        if char_width < 64 {
            char_width = 64;
        }
        let mut res = dpi as u32 as i64;
        if res == 0 {
            res = 72;
        }
        let upem = self.sfnt().units_per_em as i64;
        // FT_REQUEST_WIDTH / FT_REQUEST_HEIGHT
        let scaled = (char_width * res + 36) / 72;
        let y_scale = div_fix(scaled, upem);
        let x_scale = div_fix(scaled, upem);
        let ppem = ((scaled + 32) >> 6) as u16;
        self.size = SizeMetrics { x_ppem: ppem, y_ppem: ppem, x_scale, y_scale };
        self.set_transform(FtMatrix::IDENTITY, (0, 0));
    }

    /// `FT_Set_Transform`.
    pub fn set_transform(&mut self, matrix: FtMatrix, delta: (i64, i64)) {
        self.transform_flags = 0;
        self.matrix = matrix;
        if (matrix.xy | matrix.yx) != 0 || matrix.xx != 0x10000 || matrix.yy != 0x10000 {
            self.transform_flags |= 1;
        }
        self.delta = delta;
        if (delta.0 | delta.1) != 0 {
            self.transform_flags |= 2;
        }
    }

    /// `FT_Load_Glyph(NO_HINTING)`: scaled outline with the face transform applied.
    pub fn load_glyph(&self, gid: u32) -> Result<Outline, FontError> {
        let mut o = self.sfnt().load_glyph(gid, self.size.x_scale, self.size.y_scale)?;
        if self.transform_flags != 0 {
            // ft_smooth_transform applies both unconditionally
            o.transform(&self.matrix);
            o.translate(self.delta.0, self.delta.1);
        }
        Ok(o)
    }

    /// `FT_Load_Glyph` + `FT_Render_Glyph(FT_RENDER_MODE_NORMAL)`.
    pub fn render_glyph(&self, gid: u32) -> Result<GlyphBitmap, FontError> {
        let mut outline = self.load_glyph(gid)?;
        render_outline(&mut outline, Some(self.size))
    }

    /// Metrics of `FT_Load_Glyph(gid, FT_LOAD_NO_HINTING)` as matplotlib's
    /// `FT2Font.load_glyph` reports them (26.6, `linearHoriAdvance` 16.16).
    pub fn glyph_metrics(&self, gid: u32) -> Result<GlyphMetrics, FontError> {
        let o = self.sfnt().load_glyph(gid, self.size.x_scale, self.size.y_scale)?;
        let cb = if o.points.is_empty() { BBox::default() } else { o.cbox() };
        let (lsb, aw) = self.sfnt().hmetrics(gid);
        let _ = lsb;
        let linear = super::outline::mul_div(aw as i64, self.size.x_scale, 64);
        // the glyph image after the face transform
        let mut t = o.clone();
        if self.transform_flags != 0 {
            t.transform(&self.matrix);
            t.translate(self.delta.0, self.delta.1);
        }
        let bbox = if t.points.is_empty() { BBox::default() } else { t.cbox() };
        Ok(GlyphMetrics {
            width: cb.x_max - cb.x_min,
            height: cb.y_max - cb.y_min,
            hori_bearing_x: cb.x_min,
            hori_bearing_y: cb.y_max,
            hori_advance: mul_fix(aw as i64, self.size.x_scale),
            linear_hori_advance: linear,
            bbox,
        })
    }

    /// `FT_Get_Kerning` (mode 0 default, 1 unfitted, 2 unscaled), 26.6.
    pub fn kerning(&self, left: u32, right: u32, mode: u32) -> i64 {
        let k = self.sfnt().kerning(left, right) as i64;
        if mode == 2 {
            return k;
        }
        let mut x = mul_fix(k, self.size.x_scale);
        if mode == 0 {
            if self.size.x_ppem < 25 {
                x = super::outline::mul_div(x, self.size.x_ppem as i64, 25);
            }
            x = (x + 32) & !63;
        }
        x
    }

    /// `FT_Get_Advance(face, gid, FT_LOAD_NO_HINTING)`, 16.16.
    pub fn advance_16_16(&self, gid: u32) -> i64 {
        let (_, aw) = self.sfnt().hmetrics(gid);
        mul_fix(1024 * aw as i64, self.size.x_scale)
    }

    /// `hb_ft_get_glyph_h_advances` (26.6).
    pub fn hb_ft_advance(&self, gid: u32) -> i32 {
        let v = self.advance_16_16(gid).abs();
        let x_mult: f32 = 1.0;
        ((v as f32 * x_mult + (1 << 9) as f32) as i32) >> 10
    }

    /// hb-ft font scale (26.6 per em).
    pub fn hb_scale(&self) -> (i32, i32) {
        let upem = self.sfnt().units_per_em as u64;
        let xs = ((self.size.x_scale as u64 * upem + (1 << 15)) >> 16) as i32;
        let ys = ((self.size.y_scale as u64 * upem + (1 << 15)) >> 16) as i32;
        (xs, ys)
    }

    pub fn glyph_index(&self, c: u32) -> u32 {
        let shaper = harfrust::ShaperFont::new(&self.data.hb_font);
        shaper.default_nominal_glyph(c).map(|g| g.to_u32()).unwrap_or(0)
    }

    /// libraqm layout of `text` (LTR, single font): glyph ids and 26.6 positions.
    pub fn shape(&self, text: &[u32]) -> Vec<ShapedGlyph> {
        if text.is_empty() {
            return Vec::new();
        }
        let funcs = HbFtFuncs { font: self };
        let (xs, ys) = self.hb_scale();
        let shaper = harfrust::ShaperFont::new(&self.data.hb_font)
            .with_scale_separate(xs, ys)
            .with_font_funcs(Some(&funcs));

        let mut out = Vec::with_capacity(text.len());
        for (start, end, script) in resolve_script_runs(text) {
            let mut buffer = harfrust::Buffer::new();
            // raqm adds the full text, shaping only the run (context is kept).
            buffer.set_pre_context_codepoints(&text[..start]);
            for (i, &c) in text.iter().enumerate().take(end).skip(start) {
                buffer.push(c, i as u32);
            }
            buffer.set_post_context_codepoints(&text[end..]);
            buffer.set_direction(harfrust::Direction::LeftToRight);
            buffer.set_script(Some(script));
            buffer.set_flags(harfrust::BufferFlags::BEGINNING_OF_TEXT | harfrust::BufferFlags::END_OF_TEXT);
            if harfrust::shape(&shaper, &mut buffer, harfrust::ShapeOptions::new()).is_err() {
                continue;
            }
            for (info, pos) in buffer.glyph_infos().iter().zip(buffer.glyph_positions().iter()) {
                out.push(ShapedGlyph {
                    index: info.glyph_id,
                    cluster: info.cluster,
                    x_advance: pos.x_advance,
                    y_advance: pos.y_advance,
                    x_offset: pos.x_offset,
                    y_offset: pos.y_offset,
                });
            }
        }
        out
    }

    /// `FT2Font::set_text` with angle 0: returns (advance, bbox) in 26.6.
    pub fn set_text(&self, text: &[u32]) -> (i64, BBox) {
        let glyphs = self.shape(text);
        let mut bbox = BBox { x_min: 32000, y_min: 32000, x_max: -32000, y_max: -32000 };
        let mut pen = (0i64, 0i64);
        for g in &glyphs {
            if g.index == 0 {
                continue;
            }
            let Ok(mut o) = self.sfnt().load_glyph(g.index, self.size.x_scale, self.size.y_scale) else {
                continue;
            };
            pen.0 += g.x_offset as i64;
            pen.1 += g.y_offset as i64;
            o.translate(pen.0, pen.1);
            let gb = o.cbox();
            bbox.x_min = bbox.x_min.min(gb.x_min);
            bbox.x_max = bbox.x_max.max(gb.x_max);
            bbox.y_min = bbox.y_min.min(gb.y_min);
            bbox.y_max = bbox.y_max.max(gb.y_max);
            pen.0 += (g.x_advance - g.x_offset) as i64;
            pen.1 += (g.y_advance - g.y_offset) as i64;
        }
        if bbox.x_min > bbox.x_max {
            bbox = BBox::default();
        }
        (pen.0, bbox)
    }

    /// matplotlib's `RendererAgg.get_text_width_height_descent` for plain text.
    pub fn text_width_height_descent(&self, text: &[u32]) -> (f64, f64, f64) {
        let (adv, bbox) = self.set_text(text);
        (adv as f64 / 64.0, (bbox.y_max - bbox.y_min) as f64 / 64.0, (-bbox.y_min) as f64 / 64.0)
    }

    /// `FT2Font._layout`: (glyph index, x, y) in pixels for each glyph.
    pub fn layout(&self, text: &[u32]) -> Vec<(u32, f64, f64)> {
        let glyphs = self.shape(text);
        let mut items = Vec::with_capacity(glyphs.len());
        let (mut x, mut y) = (0i64, 0i64);
        for g in &glyphs {
            items.push((g.index, (x + g.x_offset as i64) as f64 / 64.0, (y + g.y_offset as i64) as f64 / 64.0));
            x += g.x_advance as i64;
            y += g.y_advance as i64;
        }
        items
    }
}

/// `ft_glyphslot_preset_bitmap` + `ft_smooth_render` for NORMAL mode.
pub fn render_outline(outline: &mut Outline, size: Option<SizeMetrics>) -> Result<GlyphBitmap, FontError> {
    let cbox = outline.cbox();
    // rough estimate of pixel box
    let mut pbox = BBox { x_min: cbox.x_min >> 6, y_min: cbox.y_min >> 6, x_max: cbox.x_max >> 6, y_max: cbox.y_max >> 6 };
    // tiny remainder box
    let rem = BBox { x_min: cbox.x_min & 63, y_min: cbox.y_min & 63, x_max: cbox.x_max & 63, y_max: cbox.y_max & 63 };
    pbox.x_min += rem.x_min >> 6;
    pbox.y_min += rem.y_min >> 6;
    pbox.x_max += (rem.x_max + 63) >> 6;
    pbox.y_max += (rem.y_max + 63) >> 6;

    let x_left = pbox.x_min;
    let y_top = pbox.y_max;
    let width = pbox.x_max - pbox.x_min;
    let height = pbox.y_max - pbox.y_min;

    let too_big = width >= 0x10000
        || height >= 0x10000
        || pbox.x_min < -0x1000000
        || pbox.x_max >= 0x1000000
        || pbox.y_min < -0x1000000
        || pbox.y_max >= 0x1000000
        || size.map(|s| width > 10 * s.x_ppem as i64 || height > 10 * s.y_ppem as i64).unwrap_or(false);
    if too_big {
        return Err(FontError("Raster_Overflow".into()));
    }

    let mut bm = GlyphBitmap { left: x_left as i32, top: y_top as i32, width: width as u32, rows: height as u32, buffer: Vec::new() };
    if bm.rows == 0 || bm.width == 0 {
        return Ok(bm);
    }
    bm.buffer = vec![0u8; (bm.rows * bm.width) as usize];

    let x_shift = 64 * -(bm.left as i64);
    let y_shift = 64 * -(bm.top as i64) + 64 * bm.rows as i64;
    outline.translate(x_shift, y_shift);

    if outline.overlap {
        // ft_smooth_raster_overlap: 4x oversampling with averaged spans
        const SCALE: i64 = 4;
        if (bm.width as i64) * SCALE > 0x7FFF {
            return Err(FontError("Raster_Overflow".into()));
        }
        let mut inflated = outline.clone();
        for p in &mut inflated.points {
            p.0 *= SCALE;
            p.1 *= SCALE;
        }
        let pitch = bm.width as usize;
        let rows = bm.rows as usize;
        let buf = &mut bm.buffer;
        grays::render_direct(&inflated, (bm.width as i64 * SCALE) as i32, (bm.rows as i64 * SCALE) as i32, |y, spans| {
            let row = rows - 1 - (y as usize / SCALE as usize);
            let dst = &mut buf[row * pitch..(row + 1) * pitch];
            for sp in spans {
                let cover = (sp.coverage as u32 + (SCALE * SCALE / 2) as u32) / (SCALE * SCALE) as u32;
                for x in 0..sp.len as usize {
                    let idx = (sp.x as usize + x) / SCALE as usize;
                    let sum = dst[idx] as u32 + cover;
                    dst[idx] = (sum - (sum >> 8)) as u8;
                }
            }
        });
    } else {
        grays::render_to_bitmap(outline, bm.width, bm.rows, &mut bm.buffer);
    }
    outline.translate(-x_shift, -y_shift);
    Ok(bm)
}

const PAIRED_CHARS: [u32; 34] = [
    0x0028, 0x0029, 0x003c, 0x003e, 0x005b, 0x005d, 0x007b, 0x007d, 0x00ab, 0x00bb, 0x2018, 0x2019, 0x201c, 0x201d,
    0x2039, 0x203a, 0x3008, 0x3009, 0x300a, 0x300b, 0x300c, 0x300d, 0x300e, 0x300f, 0x3010, 0x3011, 0x3014, 0x3015,
    0x3016, 0x3017, 0x3018, 0x3019, 0x301a, 0x301b,
];

fn pair_index(c: u32) -> i32 {
    match PAIRED_CHARS.binary_search(&c) {
        Ok(i) => i as i32,
        Err(_) => -1,
    }
}

/// libraqm's `_raqm_resolve_scripts` followed by splitting into script runs.
pub fn resolve_script_runs(text: &[u32]) -> Vec<(usize, usize, harfrust::Script)> {
    use harfrust::Script;
    let n = text.len();
    let mut scripts: Vec<Script> = text.iter().map(|&c| harfrust::unicode::script(c)).collect();
    let mut last_script_index: i32 = -1;
    let mut last_set_index: i32 = -1;
    let mut last_script = Script::UNKNOWN;
    // 1-based stack of (script, pair index)
    let mut stack: Vec<(Script, i32)> = Vec::new();
    for i in 0..n {
        if scripts[i] == Script::COMMON && last_script_index != -1 {
            let pi = pair_index(text[i]);
            if pi >= 0 {
                if pi & 1 == 0 {
                    scripts[i] = last_script;
                    last_set_index = i as i32;
                    stack.push((scripts[i], pi));
                } else {
                    while let Some(&(_, top_pi)) = stack.last() {
                        if top_pi == (pi & !1) {
                            break;
                        }
                        stack.pop();
                    }
                    if let Some(&(s, _)) = stack.last() {
                        scripts[i] = s;
                        last_script = s;
                        last_set_index = i as i32;
                    } else {
                        scripts[i] = last_script;
                        last_set_index = i as i32;
                    }
                }
            } else {
                scripts[i] = last_script;
                last_set_index = i as i32;
            }
        } else if scripts[i] == Script::INHERITED && last_script_index != -1 {
            scripts[i] = last_script;
            last_set_index = i as i32;
        } else {
            for j in (last_set_index + 1) as usize..i {
                scripts[j] = scripts[i];
            }
            last_script = scripts[i];
            last_script_index = i as i32;
            last_set_index = i as i32;
        }
    }
    if n >= 2 {
        for i in (0..n - 1).rev() {
            if scripts[i] == Script::INHERITED || scripts[i] == Script::COMMON {
                scripts[i] = scripts[i + 1];
            }
        }
    }
    let mut runs = Vec::new();
    let mut start = 0;
    for i in 1..=n {
        if i == n || scripts[i] != scripts[start] {
            runs.push((start, i, scripts[start]));
            start = i;
        }
    }
    runs
}
