//! Port of Agg scanline containers: `scanline32_p8`, `scanline32_u8(_am)`,
//! `scanline32_bin` and `scanline_storage_aa8` (+ its serialized replay).

use super::pixfmt::AlphaMask;

/// A span within a scanline.  `len < 0` denotes a solid span whose single
/// cover value is `covers[0]` (packed scanlines only).
#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub x: i32,
    pub len: i32,
    pub cover_idx: u32,
}

pub trait Scanline {
    fn reset(&mut self, min_x: i32, max_x: i32);
    fn reset_spans(&mut self);
    fn add_cell(&mut self, x: i32, cover: u32);
    fn add_span(&mut self, x: i32, len: u32, cover: u32);
    fn finalize(&mut self, y: i32);
    fn num_spans(&self) -> usize;
}

/// Read access to the spans of a finished scanline, used by renderers.
pub trait ScanlineSpans {
    fn y(&self) -> i32;
    fn spans(&self) -> &[Span];
    fn covers(&self) -> &[u8];
}

// ---------------------------------------------------------------------------
// scanline32_p8

#[derive(Default)]
pub struct ScanlineP8 {
    last_x: i32,
    y: i32,
    covers: Vec<u8>,
    spans: Vec<Span>,
}

impl ScanlineP8 {
    pub fn new() -> Self {
        ScanlineP8 { last_x: 0x7FFFFFF0, y: 0, covers: Vec::new(), spans: Vec::new() }
    }
}

impl Scanline for ScanlineP8 {
    fn reset(&mut self, _min_x: i32, _max_x: i32) {
        self.last_x = 0x7FFFFFF0;
        self.covers.clear();
        self.spans.clear();
    }
    #[inline]
    fn reset_spans(&mut self) {
        self.last_x = 0x7FFFFFF0;
        self.covers.clear();
        self.spans.clear();
    }
    #[inline]
    fn add_cell(&mut self, x: i32, cover: u32) {
        let idx = self.covers.len() as u32;
        self.covers.push(cover as u8);
        if x == self.last_x.wrapping_add(1) {
            if let Some(last) = self.spans.last_mut() {
                if last.len > 0 {
                    last.len += 1;
                    self.last_x = x;
                    return;
                }
            }
        }
        self.spans.push(Span { x, len: 1, cover_idx: idx });
        self.last_x = x;
    }
    #[inline]
    fn add_span(&mut self, x: i32, len: u32, cover: u32) {
        if x == self.last_x.wrapping_add(1) {
            if let Some(last) = self.spans.last_mut() {
                if last.len < 0 && cover == self.covers[last.cover_idx as usize] as u32 {
                    last.len -= len as i32;
                    self.last_x = x + len as i32 - 1;
                    return;
                }
            }
        }
        let idx = self.covers.len() as u32;
        self.covers.push(cover as u8);
        self.spans.push(Span { x, len: -(len as i32), cover_idx: idx });
        self.last_x = x + len as i32 - 1;
    }
    #[inline]
    fn finalize(&mut self, y: i32) {
        self.y = y;
    }
    #[inline]
    fn num_spans(&self) -> usize {
        self.spans.len()
    }
}

impl ScanlineSpans for ScanlineP8 {
    #[inline]
    fn y(&self) -> i32 {
        self.y
    }
    #[inline]
    fn spans(&self) -> &[Span] {
        &self.spans
    }
    #[inline]
    fn covers(&self) -> &[u8] {
        &self.covers
    }
}

// ---------------------------------------------------------------------------
// scanline32_u8 (+ optional alpha mask, i.e. scanline32_u8_am)

pub struct ScanlineU8<'a> {
    min_x: i32,
    last_x: i32,
    y: i32,
    covers: Vec<u8>,
    spans: Vec<Span>,
    mask: Option<&'a AlphaMask>,
}

impl<'a> ScanlineU8<'a> {
    pub fn new(mask: Option<&'a AlphaMask>) -> Self {
        ScanlineU8 { min_x: 0, last_x: 0x7FFFFFF0, y: 0, covers: Vec::new(), spans: Vec::new(), mask }
    }
}

impl<'a> Scanline for ScanlineU8<'a> {
    fn reset(&mut self, min_x: i32, max_x: i32) {
        let max_len = (max_x - min_x + 2).max(0) as usize;
        if max_len > self.covers.len() {
            self.covers.resize(max_len, 0);
        }
        self.last_x = 0x7FFFFFF0;
        self.min_x = min_x;
        self.spans.clear();
    }
    fn reset_spans(&mut self) {
        self.last_x = 0x7FFFFFF0;
        self.spans.clear();
    }
    fn add_cell(&mut self, x: i32, cover: u32) {
        let xr = x - self.min_x;
        self.covers[xr as usize] = cover as u8;
        if xr == self.last_x.wrapping_add(1) {
            if let Some(last) = self.spans.last_mut() {
                last.len += 1;
                self.last_x = xr;
                return;
            }
        }
        self.spans.push(Span { x, len: 1, cover_idx: xr as u32 });
        self.last_x = xr;
    }
    fn add_span(&mut self, x: i32, len: u32, cover: u32) {
        let xr = x - self.min_x;
        for c in &mut self.covers[xr as usize..xr as usize + len as usize] {
            *c = cover as u8;
        }
        if xr == self.last_x.wrapping_add(1) {
            if let Some(last) = self.spans.last_mut() {
                last.len += len as i32;
                self.last_x = xr + len as i32 - 1;
                return;
            }
        }
        self.spans.push(Span { x, len: len as i32, cover_idx: xr as u32 });
        self.last_x = xr + len as i32 - 1;
    }
    fn finalize(&mut self, y: i32) {
        self.y = y;
        if let Some(mask) = self.mask {
            for sp in &self.spans {
                let s = sp.cover_idx as usize;
                mask.combine_hspan(sp.x, y, &mut self.covers[s..s + sp.len as usize]);
            }
        }
    }
    fn num_spans(&self) -> usize {
        self.spans.len()
    }
}

impl<'a> ScanlineSpans for ScanlineU8<'a> {
    fn y(&self) -> i32 {
        self.y
    }
    fn spans(&self) -> &[Span] {
        &self.spans
    }
    fn covers(&self) -> &[u8] {
        &self.covers
    }
}

// ---------------------------------------------------------------------------
// scanline32_bin

#[derive(Default)]
pub struct ScanlineBin {
    last_x: i32,
    y: i32,
    spans: Vec<Span>,
}

impl ScanlineBin {
    pub fn new() -> Self {
        ScanlineBin { last_x: 0x7FFFFFF0, y: 0, spans: Vec::new() }
    }
}

impl Scanline for ScanlineBin {
    fn reset(&mut self, _min_x: i32, _max_x: i32) {
        self.last_x = 0x7FFFFFF0;
        self.spans.clear();
    }
    fn reset_spans(&mut self) {
        self.last_x = 0x7FFFFFF0;
        self.spans.clear();
    }
    fn add_cell(&mut self, x: i32, _cover: u32) {
        if x == self.last_x.wrapping_add(1) {
            if let Some(last) = self.spans.last_mut() {
                last.len += 1;
                self.last_x = x;
                return;
            }
        }
        self.spans.push(Span { x, len: 1, cover_idx: 0 });
        self.last_x = x;
    }
    fn add_span(&mut self, x: i32, len: u32, _cover: u32) {
        if x == self.last_x.wrapping_add(1) {
            if let Some(last) = self.spans.last_mut() {
                last.len += len as i32;
                self.last_x = x + len as i32 - 1;
                return;
            }
        }
        self.spans.push(Span { x, len: len as i32, cover_idx: 0 });
        self.last_x = x + len as i32 - 1;
    }
    fn finalize(&mut self, y: i32) {
        self.y = y;
    }
    fn num_spans(&self) -> usize {
        self.spans.len()
    }
}

impl ScanlineSpans for ScanlineBin {
    fn y(&self) -> i32 {
        self.y
    }
    fn spans(&self) -> &[Span] {
        &self.spans
    }
    fn covers(&self) -> &[u8] {
        &[]
    }
}

// ---------------------------------------------------------------------------
// scanline_storage_aa8 and replay with an integer offset

#[derive(Clone, Default)]
pub struct StoredScanline {
    pub y: i32,
    pub span_start: u32,
    pub num_spans: u32,
}

/// Equivalent of `agg::scanline_storage_aa8` followed by `serialize()` and
/// replay through `serialized_scanlines_adaptor_aa8`.
#[derive(Clone)]
pub struct ScanlineStorage {
    pub scanlines: Vec<StoredScanline>,
    pub spans: Vec<Span>,
    pub covers: Vec<u8>,
    pub min_x: i32,
    pub min_y: i32,
    pub max_x: i32,
    pub max_y: i32,
}

impl Default for ScanlineStorage {
    fn default() -> Self {
        ScanlineStorage {
            scanlines: Vec::new(),
            spans: Vec::new(),
            covers: Vec::new(),
            min_x: 0x7FFFFFFF,
            min_y: 0x7FFFFFFF,
            max_x: -0x7FFFFFFF,
            max_y: -0x7FFFFFFF,
        }
    }
}

impl ScanlineStorage {
    pub fn prepare(&mut self) {
        self.scanlines.clear();
        self.spans.clear();
        self.covers.clear();
        self.min_x = 0x7FFFFFFF;
        self.min_y = 0x7FFFFFFF;
        self.max_x = -0x7FFFFFFF;
        self.max_y = -0x7FFFFFFF;
    }

    pub fn render<S: ScanlineSpans>(&mut self, sl: &S) {
        let y = sl.y();
        if y < self.min_y {
            self.min_y = y;
        }
        if y > self.max_y {
            self.max_y = y;
        }
        let start = self.spans.len() as u32;
        let covers = sl.covers();
        for sp in sl.spans() {
            let len = sp.len.abs();
            let cid = self.covers.len() as u32;
            if sp.len < 0 {
                self.covers.push(covers[sp.cover_idx as usize]);
            } else {
                let s = sp.cover_idx as usize;
                self.covers.extend_from_slice(&covers[s..s + len as usize]);
            }
            self.spans.push(Span { x: sp.x, len: sp.len, cover_idx: cid });
            let x1 = sp.x;
            let x2 = sp.x + len - 1;
            if x1 < self.min_x {
                self.min_x = x1;
            }
            if x2 > self.max_x {
                self.max_x = x2;
            }
        }
        self.scanlines.push(StoredScanline { y, span_start: start, num_spans: sl.spans().len() as u32 });
    }
}
