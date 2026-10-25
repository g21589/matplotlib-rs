//! Mathtext (`$...$`).  The layout is computed by matplotlib's mathtext
//! engine (registered by the Python layer) and cached; the glyphs are drawn
//! by the Rust renderer like `RendererAgg.draw_mathtext`.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use super::artist::GcState;
use super::fonts::{load_font_data, FontProperties};
use super::renderer::{GlyphDraw, Renderer};

/// A mathtext layout (`VectorParse`): glyphs are (font file, face index,
/// size, glyph index, x, y).
#[derive(Debug, Default)]
pub struct MathLayout {
    pub width: f64,
    pub height: f64,
    pub depth: f64,
    pub glyphs: Vec<(String, u32, f64, u32, f64, f64)>,
    pub rects: Vec<(f64, f64, f64, f64)>,
}

pub trait MathtextEngine: Send + Sync {
    fn layout(&self, s: &str, prop: &FontProperties, dpi: f64) -> Option<MathLayout>;
}

static ENGINE: RwLock<Option<Arc<dyn MathtextEngine>>> = RwLock::new(None);

type Key = (String, FontProperties, u64, u64);
static CACHE: LazyLock<Mutex<HashMap<Key, Arc<MathLayout>>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn set_engine(e: Option<Arc<dyn MathtextEngine>>) {
    *ENGINE.write().unwrap() = e;
    CACHE.lock().unwrap().clear();
}

fn layout(s: &str, prop: &FontProperties, dpi: f64) -> Option<Arc<MathLayout>> {
    let key = (s.to_string(), prop.clone(), dpi.to_bits(), super::rc::rc_version());
    if let Some(l) = CACHE.lock().unwrap().get(&key) {
        return Some(l.clone());
    }
    let engine = ENGINE.read().unwrap().clone()?;
    let l = Arc::new(engine.layout(s, prop, dpi)?);
    let mut c = CACHE.lock().unwrap();
    if c.len() > 4096 {
        c.clear();
    }
    c.insert(key, l.clone());
    Some(l)
}

pub fn mathtext_metrics(s: &str, prop: &FontProperties, dpi: f64) -> (f64, f64, f64) {
    match layout(s, prop, dpi) {
        Some(l) => (l.width, l.height, l.depth),
        None => super::renderer::text_metrics(s, prop, false, dpi),
    }
}

pub fn draw_mathtext(r: &mut Renderer, gc: &GcState, x: f64, y: f64, s: &str, prop: &FontProperties, angle: f64) {
    let Some(l) = layout(s, prop, r.dpi) else {
        r.draw_text(gc, x, y, s, prop, angle, false);
        return;
    };
    let mut fonts = Vec::new();
    let mut keys: Vec<(&str, u32)> = Vec::new();
    let mut glyphs = Vec::with_capacity(l.glyphs.len());
    for (path, index, size, gid, dx, dy) in &l.glyphs {
        let k = (path.as_str(), *index);
        let fi = match keys.iter().position(|x| *x == k) {
            Some(i) => i,
            None => {
                let Some(data) = load_font_data(path, *index) else { continue };
                fonts.push(data);
                keys.push(k);
                fonts.len() - 1
            }
        };
        glyphs.push(GlyphDraw { font: fi, size: *size, gid: *gid, slant: 0.0, extend: 1.0, dx: *dx, dy: *dy });
    }
    r.draw_text_glyphs_and_boxes(gc, x, y, angle, &fonts, &glyphs, &l.rects);
}
