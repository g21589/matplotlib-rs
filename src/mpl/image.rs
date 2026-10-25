//! Port of `matplotlib.image.AxesImage` (filled in progressively).

use super::artist::{ArtistBase, ClipCtx, TransCtx};
use super::renderer::Renderer;

#[derive(Clone, Debug, Default)]
pub struct AxesImage {
    pub base: ArtistBase,
    pub sticky_x: Vec<f64>,
    pub sticky_y: Vec<f64>,
    pub extent: (f64, f64, f64, f64),
}

impl AxesImage {
    pub fn draw(&self, _renderer: &mut Renderer, _tc: &TransCtx, _clip: &ClipCtx) {}

    pub fn get_extent(&self) -> (f64, f64, f64, f64) {
        self.extent
    }
}
