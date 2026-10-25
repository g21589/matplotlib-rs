//! Port of the Python half of `matplotlib.backends.backend_agg.RendererAgg`
//! (text drawing, metrics) on top of the Rust Agg renderer.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use super::artist::GcState;
use super::fonts::{resolve_font, FontProperties};
use super::path::Path;
use super::transforms::{Mat, PyRadians};
use crate::agg::pixfmt::Rgba;
use crate::agg::TransAffine;
use crate::backend::renderer_agg::{Dashes, RendererAgg};
use crate::font::outline::FtMatrix;
use crate::font::FtFont;

#[derive(Clone, PartialEq, Eq, Hash)]
struct MetricsKey {
    text: String,
    prop: FontProperties,
    ismath: bool,
    dpi: u64,
    hinting: String,
}

/// A renderer implemented outside the Rust core (matplotlib's vector
/// backends): drawing calls, text metrics and groups are forwarded to it.
pub trait ForeignRenderer: Send + Sync {
    fn draw_path(&self, gc: &GcState, path: &Path, trans: &Mat, face: Option<Rgba>);
    fn draw_markers(&self, gc: &GcState, marker: &Path, marker_trans: &Mat, path: &Path, trans: &Mat, face: Option<Rgba>);
    #[allow(clippy::too_many_arguments)]
    fn draw_path_collection(
        &self,
        gc: &GcState,
        master_transform: &Mat,
        paths: &[Path],
        transforms: &[Mat],
        offsets: &[[f64; 2]],
        offset_trans: &Mat,
        facecolors: &[Rgba],
        edgecolors: &[Rgba],
        linewidths: &[f64],
        linestyles: &[(f64, Option<Vec<f64>>)],
        antialiaseds: &[bool],
    );
    fn draw_image(&self, gc: &GcState, x: f64, y: f64, rgba: &[u8], w: usize, h: usize);
    #[allow(clippy::too_many_arguments)]
    fn draw_text(&self, gc: &GcState, x: f64, y: f64, s: &str, prop: &FontProperties, angle: f64, ismath: bool, mtext: Option<&MText<'_>>);
    fn text_metrics(&self, s: &str, prop: &FontProperties, ismath: bool) -> (f64, f64, f64);
    fn open_group(&self, name: &str, gid: Option<&str>);
    fn close_group(&self, name: &str);
    /// What matplotlib's `allow_rasterization` does before drawing a
    /// non-rasterized artist (stop a pending rasterization).
    fn artist_start(&self);
    /// `flipy()` and the canvas height (for text positions).
    fn flipy(&self) -> bool;
    fn canvas_height(&self) -> f64;
    fn as_any(&self) -> &dyn std::any::Any;
}

/// The `mtext` argument of `draw_text`: the text's mirrored object, if
/// any, and what renderers use of it (anchor in display space, alignment).
pub struct MText<'a> {
    pub owner: Option<&'a OwnerRef>,
    pub x: f64,
    pub y: f64,
    pub ha: &'a str,
    pub va: &'a str,
    pub rotation_mode: &'a str,
}

/// The object an artist mirrors (e.g. the Python `Text` passed as `mtext`).
#[derive(Clone)]
pub struct OwnerRef(pub Arc<dyn std::any::Any + Send + Sync>);

impl std::fmt::Debug for OwnerRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OwnerRef")
    }
}

thread_local! {
    static FOREIGN: RefCell<Vec<Arc<dyn ForeignRenderer>>> = const { RefCell::new(Vec::new()) };
}

/// Make *f* the renderer whose metrics layout uses until [`pop_foreign`].
pub fn push_foreign(f: Arc<dyn ForeignRenderer>) {
    FOREIGN.with(|s| s.borrow_mut().push(f));
}

pub fn pop_foreign() {
    FOREIGN.with(|s| {
        s.borrow_mut().pop();
    });
}

fn current_foreign() -> Option<Arc<dyn ForeignRenderer>> {
    FOREIGN.with(|s| s.borrow().last().cloned())
}

pub struct Renderer {
    pub agg: RendererAgg,
    /// drawing and text metrics go to this renderer instead of `agg`
    pub foreign: Option<Arc<dyn ForeignRenderer>>,
    /// float canvas size (`RendererAgg.width/height` in Python)
    pub width: f64,
    pub height: f64,
    pub dpi: f64,
    metrics: HashMap<MetricsKey, (f64, f64, f64)>,
    /// drawing calls are ignored (`_draw_disabled`)
    pub disabled: bool,
}

/// `np.round(v).astype(int)` / Python `round()`.
#[inline]
pub fn py_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

impl Renderer {
    pub fn new(width: f64, height: f64, dpi: f64) -> Renderer {
        Renderer {
            agg: RendererAgg::new(width as u32, height as u32, dpi),
            foreign: None,
            width,
            height,
            dpi,
            metrics: HashMap::new(),
            disabled: false,
        }
    }

    /// A renderer used only for text metrics (no pixel buffer); the metrics
    /// are those of the current foreign renderer, if any.
    pub fn metrics_only(width: f64, height: f64, dpi: f64) -> Renderer {
        Renderer {
            agg: RendererAgg::new(1, 1, dpi),
            foreign: current_foreign(),
            width,
            height,
            dpi,
            metrics: HashMap::new(),
            disabled: true,
        }
    }

    /// A renderer forwarding everything to *f*.
    pub fn foreign(f: Arc<dyn ForeignRenderer>, width: f64, height: f64, dpi: f64) -> Renderer {
        Renderer { agg: RendererAgg::new(1, 1, dpi), foreign: Some(f), width, height, dpi, metrics: HashMap::new(), disabled: false }
    }

    /// `open_group` (a no-op for Agg).
    pub fn open_group(&mut self, name: &str, gid: Option<&str>) {
        if self.disabled {
            return;
        }
        if let Some(f) = &self.foreign {
            f.open_group(name, gid);
        }
    }

    /// Called when a (non-rasterized) artist starts drawing.
    pub fn artist_start(&mut self) {
        if self.disabled {
            return;
        }
        if let Some(f) = &self.foreign {
            f.artist_start();
        }
    }

    pub fn close_group(&mut self, name: &str) {
        if self.disabled {
            return;
        }
        if let Some(f) = &self.foreign {
            f.close_group(name);
        }
    }

    /// `flipy()` and `get_canvas_width_height()[1]`, for text positions.
    pub fn text_flip(&self) -> Option<f64> {
        match &self.foreign {
            Some(f) => f.flipy().then(|| f.canvas_height()),
            None => Some(self.height),
        }
    }

    pub fn points_to_pixels(&self, points: f64) -> f64 {
        points * self.dpi / 72.0
    }

    pub fn clear(&mut self) {
        self.agg.clear();
    }

    pub fn draw_path(&mut self, gc: &GcState, path: &Path, trans: &Mat, face: Option<Rgba>) {
        if self.disabled {
            return;
        }
        if let Some(f) = &self.foreign {
            f.draw_path(gc, path, trans, face);
            return;
        }
        let g = gc.to_agg();
        let face = gc.face(face, false);
        self.agg.draw_path(&g, path, &TransAffine::from_matrix(trans), face);
    }

    pub fn draw_markers(&mut self, gc: &GcState, marker: &Path, marker_trans: &Mat, path: &Path, trans: &Mat, face: Option<Rgba>) {
        if self.disabled {
            return;
        }
        if let Some(f) = &self.foreign {
            f.draw_markers(gc, marker, marker_trans, path, trans, face);
            return;
        }
        let g = gc.to_agg();
        let face = gc.face(face, false);
        self.agg.draw_markers(
            &g,
            marker,
            &TransAffine::from_matrix(marker_trans),
            path,
            &TransAffine::from_matrix(trans),
            face,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw_path_collection(
        &mut self,
        gc: &GcState,
        master_transform: &Mat,
        paths: &[Path],
        transforms: &[Mat],
        offsets: &[[f64; 2]],
        offset_trans: &Mat,
        facecolors: &[Rgba],
        edgecolors: &[Rgba],
        linewidths: &[f64],
        linestyles: &[(f64, Option<Vec<f64>>)],
        antialiaseds: &[bool],
    ) {
        if self.disabled {
            return;
        }
        if let Some(f) = &self.foreign {
            f.draw_path_collection(
                gc,
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
            );
            return;
        }
        let g = gc.to_agg();
        let trs: Vec<TransAffine> = transforms.iter().map(TransAffine::from_matrix).collect();
        let ls: Vec<Dashes> = linestyles
            .iter()
            .map(|(o, d)| match d {
                Some(seq) if !seq.is_empty() => Dashes::from_seq(*o, seq),
                _ => Dashes::default(),
            })
            .collect();
        let has_codes = paths.iter().any(|p| p.codes.is_some());
        self.agg.draw_path_collection(
            &g,
            &TransAffine::from_matrix(master_transform),
            paths,
            &trs,
            offsets,
            &TransAffine::from_matrix(offset_trans),
            facecolors,
            edgecolors,
            linewidths,
            &ls,
            antialiaseds,
            true,
            has_codes,
            &[],
        );
    }

    pub fn draw_image(&mut self, gc: &GcState, x: f64, y: f64, rgba: &[u8], w: usize, h: usize) {
        if self.disabled {
            return;
        }
        if let Some(f) = &self.foreign {
            f.draw_image(gc, x, y, rgba, w, h);
            return;
        }
        let g = gc.to_agg();
        self.agg.draw_image(&g, x, y, rgba, w, h);
    }

    /// `get_text_width_height_descent` (cached like `_get_text_metrics_with_cache`).
    pub fn get_text_width_height_descent(&mut self, s: &str, prop: &FontProperties, ismath: bool) -> (f64, f64, f64) {
        let key = MetricsKey {
            text: s.to_string(),
            prop: prop.clone(),
            ismath,
            dpi: self.dpi.to_bits(),
            hinting: super::rc::rc().string("text.hinting"),
        };
        if let Some(v) = self.metrics.get(&key) {
            return *v;
        }
        let v = match &self.foreign {
            Some(f) => f.text_metrics(s, prop, ismath),
            None => text_metrics(s, prop, ismath, self.dpi),
        };
        self.metrics.insert(key, v);
        v
    }

    /// `draw_text` for plain (non-math) text.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_text(&mut self, gc: &GcState, x: f64, y: f64, s: &str, prop: &FontProperties, angle: f64, ismath: bool) {
        self.draw_text_for(gc, x, y, s, prop, angle, ismath, None);
    }

    /// `draw_text` with the `mtext` argument of matplotlib's renderers.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_text_for(&mut self, gc: &GcState, x: f64, y: f64, s: &str, prop: &FontProperties, angle: f64, ismath: bool, mtext: Option<&MText<'_>>) {
        if self.disabled {
            return;
        }
        if let Some(f) = &self.foreign {
            f.draw_text(gc, x, y, s, prop, angle, ismath, mtext);
            return;
        }
        if ismath {
            super::mathtext::draw_mathtext(self, gc, x, y, s, prop, angle);
            return;
        }
        let Some(font) = resolve_font(prop) else { return };
        let mut ft = FtFont::new(font.fonts[0].clone());
        ft.set_size(prop.size, self.dpi);
        let cps: Vec<u32> = s.chars().map(|c| c as u32).collect();
        let items = ft.layout(&cps);
        let glyphs: Vec<GlyphDraw> = items
            .into_iter()
            .map(|(gid, dx, dy)| GlyphDraw { font: 0, size: prop.size, gid, slant: 0.0, extend: 1.0, dx, dy })
            .collect();
        self.draw_text_glyphs_and_boxes(gc, x, y, angle, &font.fonts, &glyphs, &[]);
    }

    /// `_draw_text_glyphs_and_boxes`.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_text_glyphs_and_boxes(
        &mut self,
        gc: &GcState,
        x: f64,
        y: f64,
        angle: f64,
        fonts: &[std::sync::Arc<crate::font::FontData>],
        glyphs: &[GlyphDraw],
        boxes: &[(f64, f64, f64, f64)],
    ) {
        let rad = angle.to_radians_py();
        let cos = rad.cos();
        let sin = rad.sin();
        let g = gc.to_agg();
        let height_i = self.height as i64;
        let mut ft: Option<(usize, FtFont)> = None;
        for gl in glyphs {
            let font = match &mut ft {
                Some((idx, f)) if *idx == gl.font => f,
                _ => {
                    ft = Some((gl.font, FtFont::new(fonts[gl.font].clone())));
                    &mut ft.as_mut().unwrap().1
                }
            };
            font.set_size(gl.size, self.dpi);
            // (0x10000 * [[cos, -sin], [sin, cos]]) @ [[extend, extend*slant], [0, 1]]
            let a = [[65536.0 * cos, 65536.0 * -sin], [65536.0 * sin, 65536.0 * cos]];
            let b = [[gl.extend, gl.extend * gl.slant], [0.0, 1.0]];
            let m = |i: usize, j: usize| a[i][0] * b[0][j] + a[i][1] * b[1][j];
            let matrix = FtMatrix { xx: py_round(m(0, 0)), xy: py_round(m(0, 1)), yx: py_round(m(1, 0)), yy: py_round(m(1, 1)) };
            let delta = (
                py_round(64.0 * (x + gl.dx * cos - gl.dy * sin)),
                py_round(64.0 * (self.height - y + gl.dx * sin + gl.dy * cos)),
            );
            font.set_transform(matrix, delta);
            let Ok(bm) = font.render_glyph(gl.gid) else { continue };
            let mut buffer = bm.buffer;
            if !gc.antialiased {
                // RenderMode.MONO is approximated by thresholding.
                for v in buffer.iter_mut() {
                    *v = if *v >= 128 { 255 } else { 0 };
                }
            }
            let yy = height_i - bm.top as i64 + bm.rows as i64;
            self.agg.draw_text_image(&g, &buffer, bm.width as usize, bm.rows as usize, bm.left, yy as i32);
        }
        if boxes.is_empty() {
            return;
        }
        let mut rgba = gc.rgb;
        if gc.forced_alpha {
            rgba.a = gc.alpha;
        }
        let mut gc1 = GcState { linewidth: 0.0, snap: gc.snap, ..GcState::default() };
        gc1.snap = gc.snap;
        for &(mut dx, mut dy, mut w, mut h) in boxes {
            if gc1.snap.is_none() || gc1.snap == Some(true) {
                if w < 1.0 {
                    dx -= (1.0 - w) / 2.0;
                    w = 1.0;
                }
                if h < 1.0 {
                    dy -= (1.0 - h) / 2.0;
                    h = 1.0;
                }
            }
            let path = super::path::closed_path(
                vec![[dx, dy], [dx + w, dy], [dx + w, dy + h], [dx, dy + h], [dx, dy]],
                super::path::SimplifyParams::default(),
            );
            let t = super::transforms::Affine2D::identity().rotate_deg(angle).translate(x, self.height - y);
            self.draw_path(&gc1, &path, &t.0, Some(rgba));
        }
    }
}

/// One glyph to draw: font index into the font list, size, glyph id, the
/// slant/extend of the font and the offset of the glyph from the origin.
#[derive(Clone, Copy, Debug)]
pub struct GlyphDraw {
    pub font: usize,
    pub size: f64,
    pub gid: u32,
    pub slant: f64,
    pub extend: f64,
    pub dx: f64,
    pub dy: f64,
}

/// Uncached `get_text_width_height_descent`.
pub fn text_metrics(s: &str, prop: &FontProperties, ismath: bool, dpi: f64) -> (f64, f64, f64) {
    if ismath {
        return super::mathtext::mathtext_metrics(s, prop, dpi);
    }
    let Some(font) = resolve_font(prop) else { return (0.0, 0.0, 0.0) };
    let mut ft = FtFont::new(font.fonts[0].clone());
    ft.set_size(prop.size, dpi);
    let cps: Vec<u32> = s.chars().map(|c| c as u32).collect();
    ft.text_width_height_descent(&cps)
}
