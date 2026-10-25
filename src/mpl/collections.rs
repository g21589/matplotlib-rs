//! Port of `matplotlib.collections` (filled in progressively).

use super::artist::{ArtistBase, ClipCtx, TransCtx};
use super::renderer::Renderer;
use super::transforms::Bbox;

#[derive(Clone, Debug, Default)]
pub struct Collection {
    pub base: ArtistBase,
    pub sticky_x: Vec<f64>,
    pub sticky_y: Vec<f64>,
}

impl Collection {
    pub fn draw(&self, _renderer: &mut Renderer, _tc: &TransCtx, _clip: &ClipCtx) {}

    /// (datalim, minpos, updatex, updatey)
    pub fn get_datalim(&self, _tc: &TransCtx) -> (Bbox, [f64; 2], bool, bool) {
        (Bbox::null(), [f64::INFINITY, f64::INFINITY], false, false)
    }
}
