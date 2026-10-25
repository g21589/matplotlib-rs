//! Port of `agg::renderer_base`, `renderer_scanline_aa_solid`,
//! `renderer_scanline_bin_solid`, `render_scanlines` and span rendering.

use super::basics::RectI;
use super::pixfmt::{PixFmt, Rgba8};
use super::rasterizer::RasterizerScanlineAa;
use super::scanline::{Scanline, ScanlineSpans, ScanlineStorage};

pub struct RendererBase<P: PixFmt> {
    pub ren: P,
    pub clip_box: RectI,
}

impl<P: PixFmt> RendererBase<P> {
    pub fn new(ren: P) -> Self {
        let w = ren.width() as i32;
        let h = ren.height() as i32;
        RendererBase { ren, clip_box: RectI::new(0, 0, w - 1, h - 1) }
    }

    #[inline(always)]
    pub fn width(&self) -> u32 {
        self.ren.width()
    }
    #[inline(always)]
    pub fn height(&self) -> u32 {
        self.ren.height()
    }
    #[inline(always)]
    pub fn xmin(&self) -> i32 {
        self.clip_box.x1
    }
    #[inline(always)]
    pub fn ymin(&self) -> i32 {
        self.clip_box.y1
    }
    #[inline(always)]
    pub fn xmax(&self) -> i32 {
        self.clip_box.x2
    }
    #[inline(always)]
    pub fn ymax(&self) -> i32 {
        self.clip_box.y2
    }

    pub fn clip_box(&mut self, x1: i32, y1: i32, x2: i32, y2: i32) -> bool {
        let mut cb = RectI::new(x1, y1, x2, y2);
        cb.normalize();
        if cb.clip(&RectI::new(0, 0, self.width() as i32 - 1, self.height() as i32 - 1)) {
            self.clip_box = cb;
            return true;
        }
        self.clip_box = RectI::new(1, 1, 0, 0);
        false
    }

    pub fn reset_clipping(&mut self, visibility: bool) {
        if visibility {
            self.clip_box = RectI::new(0, 0, self.width() as i32 - 1, self.height() as i32 - 1);
        } else {
            self.clip_box = RectI::new(1, 1, 0, 0);
        }
    }

    pub fn clear(&mut self, c: &Rgba8) {
        let w = self.width();
        if w != 0 {
            for y in 0..self.height() {
                self.ren.copy_hline(0, y as i32, w, c);
            }
        }
    }

    #[inline]
    pub fn blend_hline(&mut self, x1: i32, y: i32, x2: i32, c: &Rgba8, cover: u8) {
        let (mut x1, mut x2) = (x1, x2);
        if x1 > x2 {
            std::mem::swap(&mut x1, &mut x2);
        }
        if y > self.ymax() {
            return;
        }
        if y < self.ymin() {
            return;
        }
        if x1 > self.xmax() {
            return;
        }
        if x2 < self.xmin() {
            return;
        }
        if x1 < self.xmin() {
            x1 = self.xmin();
        }
        if x2 > self.xmax() {
            x2 = self.xmax();
        }
        self.ren.blend_hline(x1, y, (x2 - x1 + 1) as u32, c, cover);
    }

    #[inline]
    pub fn blend_solid_hspan(&mut self, x: i32, y: i32, len: i32, c: &Rgba8, covers: &[u8]) {
        let (mut x, mut len) = (x, len);
        let mut covers = covers;
        if y > self.ymax() {
            return;
        }
        if y < self.ymin() {
            return;
        }
        if x < self.xmin() {
            len -= self.xmin() - x;
            if len <= 0 {
                return;
            }
            covers = &covers[(self.xmin() - x) as usize..];
            x = self.xmin();
        }
        if x + len > self.xmax() {
            len = self.xmax() - x + 1;
            if len <= 0 {
                return;
            }
        }
        self.ren.blend_solid_hspan(x, y, len as u32, c, covers);
    }

    #[inline]
    pub fn blend_color_hspan(&mut self, x: i32, y: i32, len: i32, colors: &[Rgba8], covers: Option<&[u8]>, cover: u8) {
        let (mut x, mut len) = (x, len);
        let mut colors = colors;
        let mut covers = covers;
        if y > self.ymax() {
            return;
        }
        if y < self.ymin() {
            return;
        }
        if x < self.xmin() {
            let d = self.xmin() - x;
            len -= d;
            if len <= 0 {
                return;
            }
            covers = covers.map(|c| &c[d as usize..]);
            colors = &colors[d as usize..];
            x = self.xmin();
        }
        if x + len > self.xmax() {
            len = self.xmax() - x + 1;
            if len <= 0 {
                return;
            }
        }
        self.ren.blend_color_hspan(x, y, len as u32, colors, covers, cover);
    }
}

/// `render_scanline_aa_solid`.
#[inline]
pub fn render_scanline_aa_solid<S: ScanlineSpans, P: PixFmt>(sl: &S, ren: &mut RendererBase<P>, color: &Rgba8) {
    let y = sl.y();
    let covers = sl.covers();
    for span in sl.spans() {
        let x = span.x;
        if span.len > 0 {
            ren.blend_solid_hspan(x, y, span.len, color, &covers[span.cover_idx as usize..]);
        } else {
            ren.blend_hline(x, y, x - span.len - 1, color, covers[span.cover_idx as usize]);
        }
    }
}

/// `render_scanline_bin_solid`.
#[inline]
pub fn render_scanline_bin_solid<S: ScanlineSpans, P: PixFmt>(sl: &S, ren: &mut RendererBase<P>, color: &Rgba8) {
    let y = sl.y();
    for span in sl.spans() {
        let len = if span.len < 0 { -span.len } else { span.len };
        ren.blend_hline(span.x, y, span.x - 1 + len, color, 255);
    }
}

/// `render_scanlines(ras, sl, renderer_scanline_aa_solid)`.
pub fn render_scanlines_aa_solid<S: Scanline + ScanlineSpans, P: PixFmt>(
    ras: &mut RasterizerScanlineAa,
    sl: &mut S,
    ren: &mut RendererBase<P>,
    color: &Rgba8,
) {
    if ras.rewind_scanlines() {
        sl.reset(ras.min_x(), ras.max_x());
        while ras.sweep_scanline(sl) {
            render_scanline_aa_solid(sl, ren, color);
        }
    }
}

/// `render_scanlines(ras, sl, renderer_scanline_bin_solid)`.
pub fn render_scanlines_bin_solid<S: Scanline + ScanlineSpans, P: PixFmt>(
    ras: &mut RasterizerScanlineAa,
    sl: &mut S,
    ren: &mut RendererBase<P>,
    color: &Rgba8,
) {
    if ras.rewind_scanlines() {
        sl.reset(ras.min_x(), ras.max_x());
        while ras.sweep_scanline(sl) {
            render_scanline_bin_solid(sl, ren, color);
        }
    }
}

/// `render_scanlines(ras, sl, scanline_storage_aa8)`.
pub fn render_scanlines_to_storage<S: Scanline + ScanlineSpans>(
    ras: &mut RasterizerScanlineAa,
    sl: &mut S,
    storage: &mut ScanlineStorage,
) {
    if ras.rewind_scanlines() {
        sl.reset(ras.min_x(), ras.max_x());
        storage.prepare();
        while ras.sweep_scanline(sl) {
            storage.render(sl);
        }
    }
}

/// Replay stored scanlines shifted by (dx, dy) through an aa-solid renderer
/// (`serialized_scanlines_adaptor_aa8` + `embedded_scanline`).
pub fn render_storage_aa_solid<P: PixFmt>(
    storage: &ScanlineStorage,
    dx: i32,
    dy: i32,
    ren: &mut RendererBase<P>,
    color: &Rgba8,
) {
    for sl in &storage.scanlines {
        if sl.num_spans == 0 {
            continue;
        }
        let y = sl.y + dy;
        let spans = &storage.spans[sl.span_start as usize..(sl.span_start + sl.num_spans) as usize];
        for span in spans {
            let x = span.x + dx;
            if span.len > 0 {
                ren.blend_solid_hspan(x, y, span.len, color, &storage.covers[span.cover_idx as usize..]);
            } else {
                ren.blend_hline(x, y, x - span.len - 1, color, storage.covers[span.cover_idx as usize]);
            }
        }
    }
}

/// A span generator for `render_scanlines_aa` (`renderer_scanline_aa`).
pub trait SpanGenerator {
    fn prepare(&mut self) {}
    fn generate(&mut self, span: &mut [Rgba8], x: i32, y: i32);
}

/// `render_scanlines_aa(ras, sl, ren, alloc, span_gen)`.
pub fn render_scanlines_aa<S: Scanline + ScanlineSpans, P: PixFmt, G: SpanGenerator>(
    ras: &mut RasterizerScanlineAa,
    sl: &mut S,
    ren: &mut RendererBase<P>,
    gen: &mut G,
) {
    if ras.rewind_scanlines() {
        sl.reset(ras.min_x(), ras.max_x());
        gen.prepare();
        let mut colors: Vec<Rgba8> = Vec::new();
        while ras.sweep_scanline(sl) {
            let y = sl.y();
            let covers = sl.covers();
            for span in sl.spans() {
                let x = span.x;
                let mut len = span.len;
                if len < 0 {
                    len = -len;
                }
                colors.clear();
                colors.resize(len as usize, Rgba8::default());
                gen.generate(&mut colors, x, y);
                let cv = &covers[span.cover_idx as usize..];
                ren.blend_color_hspan(x, y, len, &colors, if span.len < 0 { None } else { Some(cv) }, cv[0]);
            }
        }
    }
}
