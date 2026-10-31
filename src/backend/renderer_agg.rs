//! Port of matplotlib's `RendererAgg` (`src/_backend_agg.h/.cpp`).

use crate::agg::basics::*;
use crate::agg::curves::ConvCurve;
use crate::agg::pixfmt::*;
use crate::agg::rasterizer::RasterizerScanlineAa;
use crate::agg::renderer::*;
use crate::agg::scanline::*;
use crate::agg::stroke::*;
use crate::agg::trans_affine::{ConvTransform, TransAffine};

use super::path_converters::*;

/// A matplotlib `Path`: vertices, optional codes and simplification info.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MplPath {
    pub vertices: Vec<[f64; 2]>,
    pub codes: Option<Vec<u8>>,
    pub should_simplify: bool,
    pub simplify_threshold: f64,
}

impl MplPath {
    pub fn new(vertices: Vec<[f64; 2]>, codes: Option<Vec<u8>>) -> Self {
        MplPath { vertices, codes, should_simplify: false, simplify_threshold: 1.0 / 9.0 }
    }
    #[inline]
    pub fn total_vertices(&self) -> usize {
        self.vertices.len()
    }
    #[inline]
    pub fn has_codes(&self) -> bool {
        self.codes.is_some()
    }
    pub fn iter(&self) -> PathIter<'_> {
        PathIter { path: self, i: 0 }
    }
}

/// `mpl::PathIterator`.
pub struct PathIter<'a> {
    path: &'a MplPath,
    i: usize,
}

impl<'a> VertexSource for PathIter<'a> {
    #[inline]
    fn rewind(&mut self, path_id: u32) {
        self.i = path_id as usize;
    }
    #[inline]
    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let n = self.path.vertices.len();
        if self.i >= n {
            *x = 0.0;
            *y = 0.0;
            return PATH_CMD_STOP;
        }
        let idx = self.i;
        self.i += 1;
        let v = self.path.vertices[idx];
        *x = v[0];
        *y = v[1];
        match &self.path.codes {
            Some(c) => c[idx] as u32,
            None => {
                if idx == 0 {
                    PATH_CMD_MOVE_TO
                } else {
                    PATH_CMD_LINE_TO
                }
            }
        }
    }
}

/// Dash pattern in points: offset and (on, off) pairs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Dashes {
    pub offset: f64,
    pub dashes: Vec<(f64, f64)>,
}

impl Dashes {
    /// Build from a matplotlib dash sequence (odd lengths repeat twice).
    pub fn from_seq(offset: f64, seq: &[f64]) -> Self {
        let n = seq.len();
        let mut d = Dashes { offset, dashes: Vec::new() };
        if n == 0 {
            return d;
        }
        let len = if n % 2 == 1 { 2 * n } else { n };
        let mut i = 0;
        while i < len {
            d.dashes.push((seq[i % n], seq[(i + 1) % n]));
            i += 2;
        }
        d
    }

    fn dash_to_stroke(&self, dash: &mut VcgenDash, dpi: f64, isaa: bool) {
        let scaleddpi = dpi / 72.0;
        for &(v0, v1) in &self.dashes {
            let mut val0 = v0 * scaleddpi;
            let mut val1 = v1 * scaleddpi;
            if !isaa {
                val0 = (val0 as i32) as f64 + 0.5;
                val1 = (val1 as i32) as f64 + 0.5;
            }
            dash.add_dash(val0, val1);
        }
        dash.dash_start(self.offset * scaleddpi);
    }
}

#[derive(Clone, Debug)]
pub struct ClipPath {
    pub path: std::sync::Arc<MplPath>,
    pub trans: TransAffine,
}

#[derive(Clone, Debug)]
pub struct Hatch {
    pub path: MplPath,
    pub color: Rgba,
    pub linewidth: f64,
}

/// `GCAgg`.
#[derive(Clone, Debug)]
pub struct Gc {
    pub linewidth: f64,
    pub alpha: f64,
    pub forced_alpha: bool,
    pub color: Rgba,
    pub isaa: bool,
    pub cap: LineCap,
    pub join: LineJoin,
    /// (x1, y1, x2, y2) in display coordinates (origin bottom-left), or None.
    pub cliprect: Option<[f64; 4]>,
    pub clippath: Option<ClipPath>,
    pub dashes: Dashes,
    pub snap_mode: SnapMode,
    pub hatch: Option<Hatch>,
    pub sketch: Option<(f64, f64, f64)>,
}

impl Default for Gc {
    fn default() -> Self {
        Gc {
            linewidth: 1.0,
            alpha: 1.0,
            forced_alpha: false,
            color: Rgba::new(0.0, 0.0, 0.0, 1.0),
            isaa: true,
            cap: LineCap::Butt,
            join: LineJoin::Round,
            cliprect: None,
            clippath: None,
            dashes: Dashes::default(),
            snap_mode: SnapMode::Auto,
            hatch: None,
            sketch: None,
        }
    }
}

impl Gc {
    fn cliprect_d(&self) -> [f64; 4] {
        self.cliprect.unwrap_or([0.0; 4])
    }
}

impl<P: PixFmt> PixFmt for &mut P {
    #[inline(always)]
    fn width(&self) -> u32 {
        (**self).width()
    }
    #[inline(always)]
    fn height(&self) -> u32 {
        (**self).height()
    }
    #[inline(always)]
    fn copy_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8) {
        (**self).copy_hline(x, y, len, c)
    }
    #[inline(always)]
    fn blend_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8, cover: u8) {
        (**self).blend_hline(x, y, len, c, cover)
    }
    #[inline(always)]
    fn blend_solid_hspan(&mut self, x: i32, y: i32, len: u32, c: &Rgba8, covers: &[u8]) {
        (**self).blend_solid_hspan(x, y, len, c, covers)
    }
    #[inline(always)]
    fn blend_color_hspan(&mut self, x: i32, y: i32, len: u32, colors: &[Rgba8], covers: Option<&[u8]>, cover: u8) {
        (**self).blend_color_hspan(x, y, len, colors, covers, cover)
    }
}

/// Key identifying the most recently rendered clip path.
#[derive(Clone, PartialEq)]
struct ClipKey {
    ptr: usize,
    trans: TransAffine,
}

pub struct RendererAgg {
    pub width: u32,
    pub height: u32,
    pub dpi: f64,
    pub pixf: PixfmtRgba32,
    ras: RasterizerScanlineAa,
    alpha_mask: Option<AlphaMask>,
    last_clip: Option<ClipKey>,
    hatch_size: u32,
    fill_color: Rgba,
    /// scanline buffers reused by `render_current` (each use resets them)
    sl_p8: ScanlineP8,
    sl_bin: ScanlineBin,
}

#[derive(Clone, Copy)]
enum Face {
    None,
    Some(Rgba8),
}

impl RendererAgg {
    pub fn new(width: u32, height: u32, dpi: f64) -> Self {
        let mut r = RendererAgg {
            width,
            height,
            dpi,
            pixf: PixfmtRgba32::new(width, height),
            ras: RasterizerScanlineAa::new(),
            alpha_mask: None,
            last_clip: None,
            hatch_size: dpi as u32,
            fill_color: Rgba::new(1.0, 1.0, 1.0, 0.0),
            sl_p8: ScanlineP8::new(),
            sl_bin: ScanlineBin::new(),
        };
        r.clear();
        r
    }

    pub fn clear(&mut self) {
        let c = self.fill_color.to_rgba8();
        let mut rb = RendererBase::new(&mut self.pixf);
        rb.clear(&c);
    }

    #[inline]
    pub fn points_to_pixels(&self, points: f64) -> f64 {
        points * self.dpi / 72.0
    }

    /// RGBA buffer (top row first).
    pub fn buffer(&self) -> &[u8] {
        &self.pixf.buf
    }

    fn set_clipbox_ras(ras: &mut RasterizerScanlineAa, cliprect: [f64; 4], width: u32, height: u32) {
        let [x1, y1, x2, y2] = cliprect;
        if x1 != 0.0 || y1 != 0.0 || x2 != 0.0 || y2 != 0.0 {
            ras.clip_box(
                ((x1 + 0.5).floor() as i32).max(0) as f64,
                ((height as f64 - y1 + 0.5).floor() as i32).max(0) as f64,
                ((x2 + 0.5).floor() as i32).min(width as i32) as f64,
                ((height as f64 - y2 + 0.5).floor() as i32).min(height as i32) as f64,
            );
        } else {
            ras.clip_box(0.0, 0.0, width as f64, height as f64);
        }
    }

    fn set_clipbox_rb<P: PixFmt>(rb: &mut RendererBase<P>, cliprect: [f64; 4], width: u32, height: u32) {
        let [x1, y1, x2, y2] = cliprect;
        if x1 != 0.0 || y1 != 0.0 || x2 != 0.0 || y2 != 0.0 {
            rb.clip_box(
                ((x1 + 0.5).floor() as i32).max(0),
                ((height as f64 - y1 + 0.5).floor() as i32).max(0),
                ((x2 + 0.5).floor() as i32).min(width as i32),
                ((height as f64 - y2 + 0.5).floor() as i32).min(height as i32),
            );
        } else {
            rb.clip_box(0, 0, width as i32, height as i32);
        }
    }

    /// `render_clippath`: rasterize the clip path into the alpha mask.
    fn render_clippath(&mut self, clippath: &Option<ClipPath>, snap_mode: SnapMode) -> bool {
        let cp = match clippath {
            Some(cp) if cp.path.total_vertices() != 0 => cp,
            _ => return false,
        };
        let key = ClipKey { ptr: std::sync::Arc::as_ptr(&cp.path) as usize, trans: cp.trans };
        if self.last_clip.as_ref() != Some(&key) {
            if self.alpha_mask.is_none() {
                self.alpha_mask = Some(AlphaMask::new(self.width, self.height));
            }
            let mut trans = cp.trans;
            trans.multiply(&TransAffine::scaling(1.0, -1.0));
            trans.multiply(&TransAffine::translation(0.0, self.height as f64));

            let mask = self.alpha_mask.as_mut().unwrap();
            mask.clear();

            let path = &*cp.path;
            let transformed = ConvTransform::new(path.iter(), trans);
            let nan_removed = PathNanRemover::new(transformed, true, path.has_codes());
            let snapped = PathSnapper::new(nan_removed, snap_mode, path.total_vertices(), 0.0);
            let simplified =
                PathSimplifier::new(snapped, path.should_simplify && !path.has_codes(), path.simplify_threshold);
            let mut curved = ConvCurve::new(simplified);
            self.ras.add_path(&mut curved);
            let mut rb = RendererBase::new(PixfmtGray8 { mask });
            let mut sl = ScanlineU8::new(None);
            render_scanlines_aa_solid(&mut self.ras, &mut sl, &mut rb, &Rgba8::new(255, 255, 255, 255));
            self.last_clip = Some(key);
        }
        true
    }

    /// `_draw_path`: render face (+hatch) and stroke of a converted path.
    fn draw_path_inner<V: VertexSource>(&mut self, path: &mut V, has_clippath: bool, face: Face, gc: &Gc) {
        let width = self.width;
        let height = self.height;

        // Render face
        if let Face::Some(fc) = face {
            self.ras.add_path(path);
            self.render_current(gc.isaa, has_clippath, &fc, false);
        }

        // Render hatch
        if let Some(hatch) = &gc.hatch {
            if hatch.path.total_vertices() != 0 {
                self.render_hatch(path, has_clippath, hatch, gc, width, height);
            }
        }

        // Render stroke
        if gc.linewidth != 0.0 {
            let mut linewidth = self.points_to_pixels(gc.linewidth);
            if !gc.isaa {
                linewidth = if linewidth < 0.5 { 0.5 } else { mpl_round(linewidth) };
            }
            if gc.dashes.dashes.is_empty() {
                let mut stroke = conv_stroke(&mut *path);
                stroke.generator.width(self.points_to_pixels(gc.linewidth));
                stroke.generator.line_cap(gc.cap);
                stroke.generator.line_join(gc.join);
                stroke.generator.miter_limit(self.points_to_pixels(gc.linewidth));
                self.ras.add_path(&mut stroke);
            } else {
                let mut dash = conv_dash(&mut *path);
                gc.dashes.dash_to_stroke(&mut dash.generator, self.dpi, gc.isaa);
                let mut stroke = conv_stroke(dash);
                stroke.generator.line_cap(gc.cap);
                stroke.generator.line_join(gc.join);
                stroke.generator.width(linewidth);
                stroke.generator.miter_limit(self.points_to_pixels(gc.linewidth));
                self.ras.add_path(&mut stroke);
            }
            let c = gc.color.to_rgba8();
            self.render_current(gc.isaa, has_clippath, &c, true);
        }
    }

    /// Render whatever is in the rasterizer with a solid colour.
    fn render_current(&mut self, isaa: bool, has_clippath: bool, c: &Rgba8, is_stroke: bool) {
        if isaa {
            if has_clippath {
                let mask = self.alpha_mask.as_ref().unwrap();
                let pfa = PixfmtAmaskAdaptor::new(&mut self.pixf, mask);
                let mut rb = RendererBase::new(pfa);
                let mut sl = ScanlineU8::new(None);
                render_scanlines_aa_solid(&mut self.ras, &mut sl, &mut rb, c);
            } else {
                let mut rb = RendererBase::new(&mut self.pixf);
                render_scanlines_aa_solid(&mut self.ras, &mut self.sl_p8, &mut rb, c);
            }
        } else if has_clippath {
            let mask = self.alpha_mask.as_ref().unwrap();
            let pfa = PixfmtAmaskAdaptor::new(&mut self.pixf, mask);
            let mut rb = RendererBase::new(pfa);
            let mut sl = ScanlineU8::new(None);
            render_scanlines_bin_solid(&mut self.ras, &mut sl, &mut rb, c);
        } else {
            let mut rb = RendererBase::new(&mut self.pixf);
            if is_stroke {
                render_scanlines_bin_solid(&mut self.ras, &mut self.sl_bin, &mut rb, c);
            } else {
                render_scanlines_bin_solid(&mut self.ras, &mut self.sl_p8, &mut rb, c);
            }
        }
    }

    fn render_hatch<V: VertexSource>(
        &mut self,
        path: &mut V,
        has_clippath: bool,
        hatch: &Hatch,
        gc: &Gc,
        _width: u32,
        _height: u32,
    ) {
        // Reset any clipping that may be in effect, since we'll be
        // drawing the hatch in a scratch buffer at origin (0, 0)
        self.ras.reset_clipping();

        let hs = self.hatch_size;
        let mut hatch_trans = TransAffine::identity();
        hatch_trans.multiply(&TransAffine::scaling(1.0, -1.0));
        hatch_trans.multiply(&TransAffine::translation(0.0, 1.0));
        hatch_trans.multiply(&TransAffine::scaling(hs as f64, hs as f64));

        let mut hatch_buf = PixfmtRgba32::new(hs, hs);
        {
            let mut rb = RendererBase::new(&mut hatch_buf);
            rb.clear(&self.fill_color.to_rgba8());
            let hcolor = hatch.color.to_rgba8();
            let mut sl = ScanlineP8::new();
            {
                let t = ConvTransform::new(hatch.path.iter(), hatch_trans);
                let mut curve = ConvCurve::new(t);
                self.ras.add_path(&mut curve);
            }
            render_scanlines_aa_solid(&mut self.ras, &mut sl, &mut rb, &hcolor);
            {
                let t = ConvTransform::new(hatch.path.iter(), hatch_trans);
                let curve = ConvCurve::new(t);
                let mut stroke = conv_stroke(curve);
                stroke.generator.width(self.points_to_pixels(hatch.linewidth));
                stroke.generator.line_cap(LineCap::Square);
                self.ras.add_path(&mut stroke);
            }
            render_scanlines_aa_solid(&mut self.ras, &mut sl, &mut rb, &hcolor);
        }

        // Put clipping back on, if originally set on entry to this function
        Self::set_clipbox_ras(&mut self.ras, gc.cliprect_d(), self.width, self.height);
        if has_clippath {
            self.last_clip = None;
            self.render_clippath(&gc.clippath, gc.snap_mode);
        }

        // Transfer the hatch to the main image buffer
        let mut gen = PatternSpan { img: &hatch_buf };
        self.ras.add_path(path);
        if has_clippath {
            let mask = self.alpha_mask.as_ref().unwrap();
            let pfa = PixfmtAmaskAdaptor::new(&mut self.pixf, mask);
            let mut rb = RendererBase::new(pfa);
            let mut sl = ScanlineP8::new();
            render_scanlines_aa(&mut self.ras, &mut sl, &mut rb, &mut gen);
        } else {
            let mut rb = RendererBase::new(&mut self.pixf);
            let mut sl = ScanlineP8::new();
            render_scanlines_aa(&mut self.ras, &mut sl, &mut rb, &mut gen);
        }
    }

    /// `draw_path(gc, path, trans, face)`.  `face` must already include the
    /// alpha adjustments made by the Python wrapper.
    pub fn draw_path(&mut self, gc: &Gc, path: &MplPath, trans: &TransAffine, face: Option<Rgba>) {
        let face = match face {
            Some(f) if f.a != 0.0 => Face::Some(f.to_rgba8()),
            _ => Face::None,
        };

        self.ras.reset_clipping();
        Self::set_clipbox_ras(&mut self.ras, gc.cliprect_d(), self.width, self.height);
        let has_clippath = self.render_clippath(&gc.clippath, gc.snap_mode);

        let mut trans = *trans;
        trans.multiply(&TransAffine::scaling(1.0, -1.0));
        trans.multiply(&TransAffine::translation(0.0, self.height as f64));
        let has_hatch = gc.hatch.as_ref().map(|h| h.path.total_vertices() != 0).unwrap_or(false);
        let clip = matches!(face, Face::None) && !has_hatch;
        let simplify = path.should_simplify && clip;
        let mut snapping_linewidth = self.points_to_pixels(gc.linewidth);
        if gc.color.a == 0.0 {
            snapping_linewidth = 0.0;
        }

        let tpath = ConvTransform::new(path.iter(), trans);
        let nan_removed = PathNanRemover::new(tpath, true, path.has_codes());
        let clipped = PathClipper::new(nan_removed, clip, self.width as f64, self.height as f64);
        let snapped = PathSnapper::new(clipped, gc.snap_mode, path.total_vertices(), snapping_linewidth);
        let simplified = PathSimplifier::new(snapped, simplify, path.simplify_threshold);
        let curve = ConvCurve::new(simplified);
        match gc.sketch {
            Some((scale, length, randomness)) => {
                let mut sketch = Sketch::new(curve, scale, length, randomness);
                self.draw_path_inner(&mut sketch, has_clippath, face, gc);
            }
            None => {
                let mut curve = curve;
                self.draw_path_inner(&mut curve, has_clippath, face, gc);
            }
        }
    }

    /// `draw_markers`.
    pub fn draw_markers(
        &mut self,
        gc: &Gc,
        marker_path: &MplPath,
        marker_trans: &TransAffine,
        path: &MplPath,
        trans: &TransAffine,
        face: Option<Rgba>,
    ) {
        // Deal with the difference in y-axis direction
        let mut marker_trans = *marker_trans;
        marker_trans.multiply(&TransAffine::scaling(1.0, -1.0));

        let mut trans = *trans;
        trans.multiply(&TransAffine::scaling(1.0, -1.0));
        trans.multiply(&TransAffine::translation(0.5, self.height as f64 + 0.5));

        let mpt = ConvTransform::new(marker_path.iter(), marker_trans);
        let mpnr = PathNanRemover::new(mpt, true, marker_path.has_codes());
        let mut marker_path_snapped =
            PathSnapper::new(mpnr, gc.snap_mode, marker_path.total_vertices(), self.points_to_pixels(gc.linewidth));
        if !marker_path_snapped.is_snapping() {
            // If the path snapper isn't in effect, at least make sure the marker
            // at (0, 0) is in the center of a pixel.  This, importantly, makes
            // the circle markers look centered around the point they refer to.
            marker_path_snapped.source.source.trans.multiply(&TransAffine::translation(0.5, 0.5));
        }
        let mut marker_path_curve = ConvCurve::new(marker_path_snapped);

        let pt = ConvTransform::new(path.iter(), trans);
        let pnr = PathNanRemover::new(pt, false, false);
        let psn = PathSnapper::new(pnr, SnapMode::False, path.total_vertices(), 0.0);
        let mut path_curve = ConvCurve::new(psn);
        path_curve.rewind(0);

        let face = match face {
            Some(f) if f.a != 0.0 => Some(f.to_rgba8()),
            _ => None,
        };

        // maxim's suggestions for cached scanlines
        let mut scanlines = ScanlineStorage::default();
        self.ras.reset();
        self.ras.reset_clipping();
        let mut marker_size = RectI::new(0x7FFFFFFF, 0x7FFFFFFF, -0x7FFFFFFF, -0x7FFFFFFF);

        let mut sl = ScanlineP8::new();
        let mut fill_buffer: Option<ScanlineStorage> = None;
        if face.is_some() {
            self.ras.add_path(&mut marker_path_curve);
            render_scanlines_to_storage(&mut self.ras, &mut sl, &mut scanlines);
            fill_buffer = Some(scanlines.clone());
            marker_size = RectI::new(scanlines.min_x, scanlines.min_y, scanlines.max_x, scanlines.max_y);
        }

        {
            let mut stroke = conv_stroke(&mut marker_path_curve);
            stroke.generator.width(self.points_to_pixels(gc.linewidth));
            stroke.generator.line_cap(gc.cap);
            stroke.generator.line_join(gc.join);
            stroke.generator.miter_limit(self.points_to_pixels(gc.linewidth));
            self.ras.reset();
            self.ras.add_path(&mut stroke);
        }
        render_scanlines_to_storage(&mut self.ras, &mut sl, &mut scanlines);
        let stroke_buffer = scanlines;
        marker_size = RectI::new(
            marker_size.x1.min(stroke_buffer.min_x),
            marker_size.y1.min(stroke_buffer.min_y),
            marker_size.x2.max(stroke_buffer.max_x),
            marker_size.y2.max(stroke_buffer.max_y),
        );

        self.ras.reset_clipping();
        let has_clippath = self.render_clippath(&gc.clippath, gc.snap_mode);

        let clipping_rect = RectD::new(
            -1.0 - marker_size.x2 as f64,
            -1.0 - marker_size.y2 as f64,
            1.0 + self.width as f64 - marker_size.x1 as f64,
            1.0 + self.height as f64 - marker_size.y1 as f64,
        );

        let edge = gc.color.to_rgba8();
        let (w, h) = (self.width, self.height);
        let cliprect = gc.cliprect_d();
        let (mut x, mut y) = (0.0, 0.0);

        if has_clippath {
            let mask = self.alpha_mask.as_ref().unwrap();
            let pfa = PixfmtAmaskAdaptor::new(&mut self.pixf, mask);
            // A fresh renderer_base over the adaptor: the clip box set on the
            // shared renderer_base does not apply here (as in matplotlib).
            let mut rb = RendererBase::new(pfa);
            while path_curve.vertex(&mut x, &mut y) != PATH_CMD_STOP {
                if !(x.is_finite() && y.is_finite()) {
                    continue;
                }
                // These values are correctly snapped above -- so we don't want
                // to round here, we really only want to truncate
                x = x.floor();
                y = y.floor();
                // Cull points outside the boundary of the image.
                if !clipping_rect.hit_test(x, y) {
                    continue;
                }
                let (dx, dy) = (iround(x), iround(y));
                if let (Some(fc), Some(fb)) = (face, &fill_buffer) {
                    render_storage_aa_solid(fb, dx, dy, &mut rb, &fc);
                }
                render_storage_aa_solid(&stroke_buffer, dx, dy, &mut rb, &edge);
            }
        } else {
            let mut rb = RendererBase::new(&mut self.pixf);
            Self::set_clipbox_rb(&mut rb, cliprect, w, h);
            while path_curve.vertex(&mut x, &mut y) != PATH_CMD_STOP {
                if !(x.is_finite() && y.is_finite()) {
                    continue;
                }
                x = x.floor();
                y = y.floor();
                if !clipping_rect.hit_test(x, y) {
                    continue;
                }
                let (dx, dy) = (iround(x), iround(y));
                if let (Some(fc), Some(fb)) = (face, &fill_buffer) {
                    render_storage_aa_solid(fb, dx, dy, &mut rb, &fc);
                }
                render_storage_aa_solid(&stroke_buffer, dx, dy, &mut rb, &edge);
            }
        }

        self.ras.reset_clipping();
    }

    /// `draw_text_image` for angle == 0: blend a gray8 coverage image whose
    /// bottom-left corner is at (x, y) (y measured from the top).
    pub fn draw_text_image(&mut self, gc: &Gc, image: &[u8], img_w: usize, img_h: usize, x: i32, y: i32) {
        self.ras.reset_clipping();
        let deltay = y - img_h as i32;
        let mut text = RectI::new(x, deltay, x + img_w as i32, y);
        let fig = RectI::new(0, 0, self.width as i32, self.height as i32);
        text.clip(&fig);

        if let Some([cx1, cy1, cx2, cy2]) = gc.cliprect {
            if cx1 != 0.0 || cy1 != 0.0 || cx2 != 0.0 || cy2 != 0.0 {
                let clip = RectI::new(
                    mpl_round_to_int(cx1),
                    mpl_round_to_int(self.height as f64 - cy2),
                    mpl_round_to_int(cx2),
                    mpl_round_to_int(self.height as f64 - cy1),
                );
                text.clip(&clip);
            }
        }

        if text.x2 > text.x1 {
            let deltax = text.x2 - text.x1;
            let deltax2 = text.x1 - x;
            let c = gc.color.to_rgba8();
            for yi in text.y1..text.y2 {
                let row = (yi - deltay) as usize * img_w + deltax2 as usize;
                self.pixf.blend_solid_hspan(text.x1, yi, deltax as u32, &c, &image[row..row + deltax as usize]);
            }
        }
    }

    /// `draw_image`: blend an RGBA image at display position (x, y) of its
    /// bottom-left corner.  The array is passed exactly as matplotlib passes
    /// it (Agg attaches it with a negative stride).  Clip paths are not yet
    /// supported for images (they use the plain clip box).
    pub fn draw_image(&mut self, gc: &Gc, x: f64, y: f64, image: &[u8], img_w: usize, img_h: usize) {
        let x = mpl_round(x);
        let y = mpl_round(y);
        // (the Python wrapper sets gc.alpha to 1)
        let alpha = 1.0f64;
        let (w, h) = (self.width, self.height);
        let cliprect = gc.cliprect_d();
        self.ras.reset_clipping();
        Self::set_clipbox_ras(&mut self.ras, cliprect, w, h);
        let has_clippath = self.render_clippath(&gc.clippath, gc.snap_mode);
        let x0 = x as i32;
        let y0 = (h as f64 - (y + img_h as f64)) as i32;
        if has_clippath {
            // the image rectangle, rendered through the clip mask with a
            // nearest-neighbour image span (an integer translation)
            let mut rect = crate::agg::basics::PathStorage::new();
            let (iw, ih) = (img_w as f64, img_h as f64);
            for (k, (px, py)) in [(0.0, 0.0), (iw, 0.0), (iw, ih), (0.0, ih), (0.0, 0.0)].into_iter().enumerate() {
                let (px, py) = (px + x0 as f64, py + y0 as f64);
                if k == 0 {
                    rect.move_to(px, py);
                } else {
                    rect.line_to(px, py);
                }
            }
            self.ras.add_path(&mut rect);
            let mut gen = ImageNnSpan { img: image, w: img_w as i32, h: img_h as i32, tx: x0, ty: y0 };
            let mask = self.alpha_mask.as_ref().unwrap();
            let pfa = PixfmtAmaskAdaptor::new(&mut self.pixf, mask);
            let mut rb = RendererBase::new(pfa);
            let mut sl = ScanlineU8::new(None);
            render_scanlines_aa(&mut self.ras, &mut sl, &mut rb, &mut gen);
        } else {
            let cover = (alpha * 255.0) as u8;
            let mut rb = RendererBase::new(&mut self.pixf);
            Self::set_clipbox_rb(&mut rb, cliprect, w, h);
            blend_from_rgba(&mut rb, image, img_w, img_h, x0, y0, cover);
        }
    }

    /// `draw_path_collection` / `_draw_path_collection_generic`.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_path_collection(
        &mut self,
        gc: &Gc,
        master_transform: &TransAffine,
        paths: &[MplPath],
        transforms: &[TransAffine],
        offsets: &[[f64; 2]],
        offset_trans: &TransAffine,
        facecolors: &[Rgba],
        edgecolors: &[Rgba],
        linewidths: &[f64],
        linestyles: &[Dashes],
        antialiaseds: &[bool],
        check_snap: bool,
        has_codes: bool,
        hatchcolors: &[Rgba],
    ) {
        let npaths = paths.len();
        let noffsets = offsets.len();
        let n = npaths.max(noffsets);
        let ntransforms = transforms.len();
        let nfacecolors = facecolors.len();
        let nedgecolors = edgecolors.len();
        let nlinewidths = linewidths.len();
        let nlinestyles = linestyles.len().min(n);
        let naa = antialiaseds.len();
        let nhatchcolors = hatchcolors.len();

        if (nfacecolors == 0 && nedgecolors == 0 && nhatchcolors == 0) || npaths == 0 {
            return;
        }

        // Handle any clipping globally
        self.ras.reset_clipping();
        Self::set_clipbox_ras(&mut self.ras, gc.cliprect_d(), self.width, self.height);
        let has_clippath = self.render_clippath(&gc.clippath, gc.snap_mode);

        // Set some defaults, assuming no face or edge
        let mut gc = gc.clone();
        gc.linewidth = 0.0;
        let mut face = Face::None;
        let has_hatch = gc.hatch.as_ref().map(|h| h.path.total_vertices() != 0).unwrap_or(false);
        let do_clip = nfacecolors == 0 && !has_hatch;

        if !has_clippath && !has_hatch && gc.sketch.is_none() && n >= PARALLEL_MIN_PATHS {
            let threads = collection_threads();
            if threads > 1 {
                let c = Collection {
                    master_transform,
                    paths,
                    transforms,
                    offsets,
                    offset_trans,
                    facecolors,
                    edgecolors,
                    linewidths,
                    linestyles,
                    antialiaseds,
                    check_snap,
                    has_codes,
                    do_clip,
                };
                self.draw_path_collection_parallel(&gc, &c, n, threads);
                return;
            }
        }

        for i in 0..n {
            let path = &paths[i % npaths];
            let mut trans = if ntransforms > 0 {
                let mut t = transforms[i % ntransforms];
                t.multiply(master_transform);
                t
            } else {
                *master_transform
            };

            if noffsets > 0 {
                let mut xo = offsets[i % noffsets][0];
                let mut yo = offsets[i % noffsets][1];
                offset_trans.transform(&mut xo, &mut yo);
                trans.multiply(&TransAffine::translation(xo, yo));
            }

            // These transformations must be done post-offsets
            trans.multiply(&TransAffine::scaling(1.0, -1.0));
            trans.multiply(&TransAffine::translation(0.0, self.height as f64));

            if nfacecolors > 0 {
                face = Face::Some(facecolors[i % nfacecolors].to_rgba8());
            }

            if nedgecolors > 0 {
                gc.color = edgecolors[i % nedgecolors];
                gc.linewidth = if nlinewidths > 0 { linewidths[i % nlinewidths] } else { 1.0 };
                if nlinestyles > 0 {
                    gc.dashes = linestyles[i % nlinestyles].clone();
                }
            }

            if nhatchcolors > 0 {
                if let Some(h) = gc.hatch.as_mut() {
                    h.color = hatchcolors[i % nhatchcolors];
                }
            }

            gc.isaa = antialiaseds[i % naa];

            let tpath = ConvTransform::new(path.iter(), trans);
            let nan_removed = PathNanRemover::new(tpath, true, has_codes);
            let clipped = PathClipper::new(nan_removed, do_clip, self.width as f64, self.height as f64);
            if check_snap {
                let snapped =
                    PathSnapper::new(clipped, gc.snap_mode, path.total_vertices(), self.points_to_pixels(gc.linewidth));
                if has_codes {
                    let mut curve = ConvCurve::new(snapped);
                    self.draw_with_sketch(&mut curve, has_clippath, face, &gc);
                } else {
                    let mut s = snapped;
                    self.draw_with_sketch(&mut s, has_clippath, face, &gc);
                }
            } else if has_codes {
                let mut curve = ConvCurve::new(clipped);
                self.draw_with_sketch(&mut curve, has_clippath, face, &gc);
            } else {
                let mut c = clipped;
                self.draw_with_sketch(&mut c, has_clippath, face, &gc);
            }
        }
    }

    /// `draw_gouraud_triangles(gc, points, colors, trans)`.
    pub fn draw_gouraud_triangles(&mut self, gc: &Gc, points: &[[[f64; 2]; 3]], colors: &[[[f64; 4]; 3]], trans: &TransAffine) {
        self.ras.reset_clipping();
        Self::set_clipbox_ras(&mut self.ras, gc.cliprect_d(), self.width, self.height);
        let has_clippath = self.render_clippath(&gc.clippath, gc.snap_mode);
        for (p, c) in points.iter().zip(colors) {
            self.draw_gouraud_triangle(p, c, trans, has_clippath);
        }
    }

    fn draw_gouraud_triangle(&mut self, points: &[[f64; 2]; 3], colors: &[[f64; 4]; 3], trans: &TransAffine, has_clippath: bool) {
        let mut trans = *trans;
        trans.multiply(&TransAffine::scaling(1.0, -1.0));
        trans.multiply(&TransAffine::translation(0.0, self.height as f64));
        let mut tp = [[0.0; 2]; 3];
        for (t, p) in tp.iter_mut().zip(points) {
            let (mut x, mut y) = (p[0], p[1]);
            trans.transform(&mut x, &mut y);
            if x.is_nan() || y.is_nan() {
                return;
            }
            *t = [x, y];
        }
        let col = |c: &[f64; 4]| Rgba::new(c[0], c[1], c[2], c[3]).to_rgba8();
        let mut gen = crate::agg::span_gouraud::SpanGouraudRgba::new(
            col(&colors[0]),
            col(&colors[1]),
            col(&colors[2]),
            tp[0][0],
            tp[0][1],
            tp[1][0],
            tp[1][1],
            tp[2][0],
            tp[2][1],
            0.5,
        );
        self.ras.add_path(&mut gen);
        if has_clippath {
            let mask = self.alpha_mask.as_ref().unwrap();
            let pfa = PixfmtAmaskAdaptor::new(&mut self.pixf, mask);
            let mut rb = RendererBase::new(pfa);
            let mut sl = ScanlineU8::new(None);
            render_scanlines_aa(&mut self.ras, &mut sl, &mut rb, &mut gen);
        } else {
            let mut rb = RendererBase::new(&mut self.pixf);
            let mut sl = ScanlineP8::new();
            render_scanlines_aa(&mut self.ras, &mut sl, &mut rb, &mut gen);
        }
    }

    fn draw_with_sketch<V: VertexSource>(&mut self, path: &mut V, has_clippath: bool, face: Face, gc: &Gc) {
        match gc.sketch {
            Some((scale, length, randomness)) => {
                let mut sketch = Sketch::new(path, scale, length, randomness);
                self.draw_path_inner(&mut sketch, has_clippath, face, gc);
            }
            None => self.draw_path_inner(path, has_clippath, face, gc),
        }
    }
}

/// `span_image_filter_rgba_nn` over an `image_accessor_clip` (transparent
/// outside) of an image attached with a negative stride, for the integer
/// translation (tx, ty) `draw_image` uses.
struct ImageNnSpan<'a> {
    img: &'a [u8],
    w: i32,
    h: i32,
    tx: i32,
    ty: i32,
}

impl SpanGenerator for ImageNnSpan<'_> {
    fn generate(&mut self, span: &mut [Rgba8], x: i32, y: i32) {
        let sy = y - self.ty;
        for (i, s) in span.iter_mut().enumerate() {
            let sx = x + i as i32 - self.tx;
            *s = if sx >= 0 && sx < self.w && sy >= 0 && sy < self.h {
                let o = (((self.h - 1 - sy) * self.w + sx) * 4) as usize;
                Rgba8::new(self.img[o], self.img[o + 1], self.img[o + 2], self.img[o + 3])
            } else {
                Rgba8::new(0, 0, 0, 0)
            };
        }
    }
}

/// Span generator replicating `span_pattern_rgba` over an
/// `image_accessor_wrap<wrap_mode_repeat_auto_pow2>` source.
struct PatternSpan<'a> {
    img: &'a PixfmtRgba32,
}

impl<'a> SpanGenerator for PatternSpan<'a> {
    fn generate(&mut self, span: &mut [Rgba8], x: i32, y: i32) {
        let w = self.img.width as i32;
        let h = self.img.height as i32;
        let wrap = |v: i32, size: i32| -> i32 {
            // wrap_mode_repeat_auto_pow2
            if size & (size - 1) == 0 {
                v & (size - 1)
            } else {
                // wrap_mode_repeat: m_add = size * (0x3FFFFFFF / size)
                let add = size * (0x3FFFFFFF / size);
                ((v as u32).wrapping_add(add as u32) % size as u32) as i32
            }
        };
        let yy = wrap(y, h);
        let mut xx = x;
        for s in span.iter_mut() {
            let px = wrap(xx, w);
            *s = self.img.pixel(px, yy);
            xx += 1;
        }
    }
}

/// `renderer_base::blend_from(pixfmt, nullptr, dx, dy, cover)` for an RGBA
/// source image.
fn blend_from_rgba<P: PixFmt>(
    rb: &mut RendererBase<P>,
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dx: i32,
    dy: i32,
    cover: u8,
) {
    // rect_i rsrc(0, 0, src.width(), src.height()); rdst(dx, dy, ...)
    let mut rsrc = RectI::new(0, 0, src_w as i32, src_h as i32);
    let mut rdst = RectI::new(rsrc.x1 + dx, rsrc.y1 + dy, rsrc.x2 + dx, rsrc.y2 + dy);
    // clip_rect_area(rdst, rsrc, src.width(), src.height())
    let cb = RectI::new(rb.xmin(), rb.ymin(), rb.xmax() + 1, rb.ymax() + 1);
    if rsrc.x1 < 0 {
        rdst.x1 -= rsrc.x1;
        rsrc.x1 = 0;
    }
    if rsrc.y1 < 0 {
        rdst.y1 -= rsrc.y1;
        rsrc.y1 = 0;
    }
    if rsrc.x2 > src_w as i32 {
        rsrc.x2 = src_w as i32;
    }
    if rsrc.y2 > src_h as i32 {
        rsrc.y2 = src_h as i32;
    }
    if rdst.x1 < cb.x1 {
        rsrc.x1 += cb.x1 - rdst.x1;
        rdst.x1 = cb.x1;
    }
    if rdst.y1 < cb.y1 {
        rsrc.y1 += cb.y1 - rdst.y1;
        rdst.y1 = cb.y1;
    }
    if rdst.x2 > cb.x2 {
        rdst.x2 = cb.x2;
    }
    if rdst.y2 > cb.y2 {
        rdst.y2 = cb.y2;
    }
    let width = (rdst.x2 - rdst.x1).min(rsrc.x2 - rsrc.x1);
    let height = (rdst.y2 - rdst.y1).min(rsrc.y2 - rsrc.y1);
    if width <= 0 || height <= 0 {
        return;
    }
    let mut colors = vec![Rgba8::default(); width as usize];
    for row in 0..height {
        // matplotlib attaches the image with a negative stride, so row 0 of
        // the Agg rendering buffer is the last row of the array.
        let sy = src_h - 1 - (rsrc.y1 + row) as usize;
        let ty = rdst.y1 + row;
        for (i, c) in colors.iter_mut().enumerate() {
            let o = (sy * src_w + rsrc.x1 as usize + i) * 4;
            *c = Rgba8::new(src[o], src[o + 1], src[o + 2], src[o + 3]);
        }
        rb.ren.blend_color_hspan(rdst.x1, ty, width as u32, &colors, None, cover);
    }
}

/// Below this many paths `draw_path_collection` draws on one thread.
const PARALLEL_MIN_PATHS: usize = 256;

/// The threads of `draw_path_collection_parallel`: the available
/// parallelism (at most 8), or `MPLRS_THREADS` (1 draws on one thread).
fn collection_threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("MPLRS_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&t| t >= 1)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |t| t.get()).min(8))
    })
}

/// The arguments of a `draw_path_collection` call.
struct Collection<'a> {
    master_transform: &'a TransAffine,
    paths: &'a [MplPath],
    transforms: &'a [TransAffine],
    offsets: &'a [[f64; 2]],
    offset_trans: &'a TransAffine,
    facecolors: &'a [Rgba],
    edgecolors: &'a [Rgba],
    linewidths: &'a [f64],
    linestyles: &'a [Dashes],
    antialiaseds: &'a [bool],
    check_snap: bool,
    has_codes: bool,
    do_clip: bool,
}

/// The scanlines a rasterization produced, as the renderer received them.
#[derive(Default)]
struct Recorded {
    /// (y, first span, number of spans)
    rows: Vec<(i32, u32, u32)>,
    spans: Vec<Span>,
    covers: Vec<u8>,
}

impl Recorded {
    fn clear(&mut self) {
        self.rows.clear();
        self.spans.clear();
        self.covers.clear();
    }

    /// `render_scanlines` of *ras* with *sl*, keeping the scanlines.
    fn record<S: Scanline + ScanlineSpans>(&mut self, ras: &mut RasterizerScanlineAa, sl: &mut S) {
        self.clear();
        if ras.rewind_scanlines() {
            sl.reset(ras.min_x(), ras.max_x());
            while ras.sweep_scanline(sl) {
                let base = self.covers.len() as u32;
                self.covers.extend_from_slice(sl.covers());
                let first = self.spans.len() as u32;
                self.spans.extend(sl.spans().iter().map(|sp| Span { cover_idx: sp.cover_idx + base, ..*sp }));
                self.rows.push((sl.y(), first, self.spans.len() as u32 - first));
            }
        }
    }

    /// The recorded scanlines through `render_scanline_aa_solid` (*aa*) or
    /// `render_scanline_bin_solid`.
    fn replay<P: PixFmt>(&self, rb: &mut RendererBase<P>, c: &Rgba8, aa: bool) {
        for &(y, first, n) in &self.rows {
            let row = RowView { y, spans: &self.spans[first as usize..(first + n) as usize], covers: &self.covers };
            if aa {
                render_scanline_aa_solid(&row, rb, c);
            } else {
                render_scanline_bin_solid(&row, rb, c);
            }
        }
    }
}

struct RowView<'a> {
    y: i32,
    spans: &'a [Span],
    covers: &'a [u8],
}

impl ScanlineSpans for RowView<'_> {
    fn y(&self) -> i32 {
        self.y
    }
    fn spans(&self) -> &[Span] {
        self.spans
    }
    fn covers(&self) -> &[u8] {
        self.covers
    }
}

/// One path of a collection, rasterized: its face and its stroke.
#[derive(Default)]
struct RecordedPath {
    face: Option<Rgba8>,
    face_rows: Recorded,
    stroke: Option<Rgba8>,
    stroke_rows: Recorded,
    isaa: bool,
}

/// The rasterizer and scanlines of a thread of `draw_path_collection_parallel`.
struct PathWorker {
    ras: RasterizerScanlineAa,
    sl_p8: ScanlineP8,
    sl_bin: ScanlineBin,
}

impl PathWorker {
    /// `draw_path_inner` without clip path, hatch or sketch, recording what
    /// `render_current` would render.
    fn record<V: VertexSource>(&mut self, path: &mut V, face: Face, gc: &Gc, dpi: f64, out: &mut RecordedPath) {
        let ptp = |points: f64| points * dpi / 72.0;
        out.isaa = gc.isaa;
        out.face = None;
        out.stroke = None;
        if let Face::Some(fc) = face {
            self.ras.add_path(path);
            out.face_rows.record(&mut self.ras, &mut self.sl_p8);
            out.face = Some(fc);
        }
        if gc.linewidth != 0.0 {
            let mut linewidth = ptp(gc.linewidth);
            if !gc.isaa {
                linewidth = if linewidth < 0.5 { 0.5 } else { mpl_round(linewidth) };
            }
            if gc.dashes.dashes.is_empty() {
                let mut stroke = conv_stroke(&mut *path);
                stroke.generator.width(ptp(gc.linewidth));
                stroke.generator.line_cap(gc.cap);
                stroke.generator.line_join(gc.join);
                stroke.generator.miter_limit(ptp(gc.linewidth));
                self.ras.add_path(&mut stroke);
            } else {
                let mut dash = conv_dash(&mut *path);
                gc.dashes.dash_to_stroke(&mut dash.generator, dpi, gc.isaa);
                let mut stroke = conv_stroke(dash);
                stroke.generator.line_cap(gc.cap);
                stroke.generator.line_join(gc.join);
                stroke.generator.width(linewidth);
                stroke.generator.miter_limit(ptp(gc.linewidth));
                self.ras.add_path(&mut stroke);
            }
            if gc.isaa {
                out.stroke_rows.record(&mut self.ras, &mut self.sl_p8);
            } else {
                out.stroke_rows.record(&mut self.ras, &mut self.sl_bin);
            }
            out.stroke = Some(gc.color.to_rgba8());
        }
    }
}

impl RendererAgg {
    /// `draw_path_collection` without clip path, hatch or sketch: the paths
    /// are rasterized on *threads* threads (each with a rasterizer set up as
    /// the shared one) and their scanlines rendered in order, so the image is
    /// the same as when drawing them one after the other.
    fn draw_path_collection_parallel(&mut self, gc0: &Gc, c: &Collection<'_>, n: usize, threads: usize) {
        let (w, h) = (self.width, self.height);
        let (dpi, height) = (self.dpi, self.height as f64);
        let cliprect = gc0.cliprect_d();
        let npaths = c.paths.len();
        let (ntransforms, noffsets) = (c.transforms.len(), c.offsets.len());
        let (nfacecolors, nedgecolors, nlinewidths) = (c.facecolors.len(), c.edgecolors.len(), c.linewidths.len());
        let nlinestyles = c.linestyles.len().min(n);
        let naa = c.antialiaseds.len();

        // The state of the sequential loop's graphics context for path i (the
        // edge properties keep their previous values when not given).
        let gc_for = |i: usize, gc: &mut Gc| -> (Face, TransAffine) {
            let mut trans = if ntransforms > 0 {
                let mut t = c.transforms[i % ntransforms];
                t.multiply(c.master_transform);
                t
            } else {
                *c.master_transform
            };
            if noffsets > 0 {
                let mut xo = c.offsets[i % noffsets][0];
                let mut yo = c.offsets[i % noffsets][1];
                c.offset_trans.transform(&mut xo, &mut yo);
                trans.multiply(&TransAffine::translation(xo, yo));
            }
            trans.multiply(&TransAffine::scaling(1.0, -1.0));
            trans.multiply(&TransAffine::translation(0.0, height));
            let face = if nfacecolors > 0 { Face::Some(c.facecolors[i % nfacecolors].to_rgba8()) } else { Face::None };
            if nedgecolors > 0 {
                gc.color = c.edgecolors[i % nedgecolors];
                gc.linewidth = if nlinewidths > 0 { c.linewidths[i % nlinewidths] } else { 1.0 };
                if nlinestyles > 0 {
                    gc.dashes = c.linestyles[i % nlinestyles].clone();
                }
            }
            gc.isaa = c.antialiaseds[i % naa];
            (face, trans)
        };

        let rasterize = |worker: &mut PathWorker, gc: &mut Gc, i: usize, out: &mut RecordedPath| {
            let (face, trans) = gc_for(i, gc);
            let path = &c.paths[i % npaths];
            let tpath = ConvTransform::new(path.iter(), trans);
            let nan_removed = PathNanRemover::new(tpath, true, c.has_codes);
            let clipped = PathClipper::new(nan_removed, c.do_clip, w as f64, h as f64);
            if c.check_snap {
                let snapped = PathSnapper::new(clipped, gc.snap_mode, path.total_vertices(), gc.linewidth * dpi / 72.0);
                if c.has_codes {
                    worker.record(&mut ConvCurve::new(snapped), face, gc, dpi, out);
                } else {
                    let mut s = snapped;
                    worker.record(&mut s, face, gc, dpi, out);
                }
            } else if c.has_codes {
                worker.record(&mut ConvCurve::new(clipped), face, gc, dpi, out);
            } else {
                let mut cl = clipped;
                worker.record(&mut cl, face, gc, dpi, out);
            }
        };

        let new_worker = || {
            let mut ras = RasterizerScanlineAa::new();
            ras.reset_clipping();
            Self::set_clipbox_ras(&mut ras, cliprect, w, h);
            PathWorker { ras, sl_p8: ScanlineP8::new(), sl_bin: ScanlineBin::new() }
        };

        // Batches: the threads (started once) rasterize their share of a
        // batch into their own buffers, then this thread renders the batch
        // in order.  Small batches keep the recorded scanlines in cache.
        const PER_THREAD: usize = 512;
        let batch = PER_THREAD * threads;
        let share = |start: usize, t: usize| {
            let end = (start + batch).min(n);
            let per = (end - start).div_ceil(threads);
            let a = (start + t * per).min(end);
            (a, (a + per).min(end))
        };
        let slots: Vec<std::sync::Mutex<(PathWorker, Gc, Vec<RecordedPath>)>> = (0..threads)
            .map(|_| {
                let mut gc = gc0.clone();
                gc.linewidth = 0.0;
                std::sync::Mutex::new((new_worker(), gc, Vec::new()))
            })
            .collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let finished = std::sync::atomic::AtomicBool::new(false);
        let go = std::sync::Barrier::new(threads + 1);
        let done = std::sync::Barrier::new(threads + 1);
        let pixf = &mut self.pixf;
        std::thread::scope(|s| {
            for (t, slot) in slots.iter().enumerate() {
                let (rasterize, share, next, finished, go, done) = (&rasterize, &share, &next, &finished, &go, &done);
                s.spawn(move || loop {
                    go.wait();
                    if finished.load(std::sync::atomic::Ordering::Acquire) {
                        break;
                    }
                    let (a, b) = share(next.load(std::sync::atomic::Ordering::Acquire), t);
                    {
                        let mut g = slot.lock().unwrap();
                        let (worker, gc, out) = &mut *g;
                        if out.len() < b - a {
                            out.resize_with(b - a, RecordedPath::default);
                        }
                        // (the state gc_for gives path i depends on i only)
                        for (j, i) in (a..b).enumerate() {
                            rasterize(worker, gc, i, &mut out[j]);
                        }
                    }
                    done.wait();
                });
            }
            let mut start = 0;
            while start < n {
                next.store(start, std::sync::atomic::Ordering::Release);
                go.wait();
                done.wait();
                // (as render_current: clipped by the rasterizer, not here)
                let mut rb = RendererBase::new(&mut *pixf);
                for (t, slot) in slots.iter().enumerate() {
                    let (a, b) = share(start, t);
                    let g = slot.lock().unwrap();
                    for r in &g.2[..b - a] {
                        if let Some(fc) = &r.face {
                            r.face_rows.replay(&mut rb, fc, r.isaa);
                        }
                        if let Some(ec) = &r.stroke {
                            r.stroke_rows.replay(&mut rb, ec, r.isaa);
                        }
                    }
                }
                start = (start + batch).min(n);
            }
            finished.store(true, std::sync::atomic::Ordering::Release);
            go.wait();
        });
    }
}
