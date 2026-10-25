//! Port of `matplotlib._tight_layout` (filled in progressively).

use super::figure::Figure;
use super::renderer::Renderer;

pub fn tight_layout(_fig: &mut Figure, _renderer: &mut Renderer, _pad: f64, _h_pad: Option<f64>, _w_pad: Option<f64>, _rect: Option<(f64, f64, f64, f64)>) {}
