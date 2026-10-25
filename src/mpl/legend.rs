//! Port of `matplotlib.legend.Legend` (filled in progressively).

use super::artist::{ClipCtx, TransCtx};
use super::axes::Axes;
use super::renderer::Renderer;
use super::transforms::Bbox;

#[derive(Clone, Debug)]
pub struct Legend {
    pub zorder: f64,
}

impl Legend {
    pub fn draw(&mut self, _renderer: &mut Renderer, _tc: &TransCtx, _clip: &ClipCtx, _parent_bbox: Bbox, _ax: &Axes) {}

    pub fn get_window_extent(&mut self, _renderer: &mut Renderer, _tc: &TransCtx, _parent_bbox: Bbox, _ax: &Axes) -> Option<Bbox> {
        None
    }
}
