//! Mathtext (`$...$`) layout.  Placeholder: renders the raw string as plain
//! text until the mathtext engine is ported.

use super::artist::GcState;
use super::fonts::FontProperties;
use super::renderer::Renderer;

pub fn mathtext_metrics(s: &str, prop: &FontProperties, dpi: f64) -> (f64, f64, f64) {
    super::renderer::text_metrics(s, prop, false, dpi)
}

pub fn draw_mathtext(r: &mut Renderer, gc: &GcState, x: f64, y: f64, s: &str, prop: &FontProperties, angle: f64) {
    r.draw_text(gc, x, y, s, prop, angle, false);
}
