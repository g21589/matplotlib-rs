//! Port of `matplotlib.text.Text` (layout, drawing, extents).

use super::artist::{ArtistBase, ClipCtx, GcState, TSpec, TransCtx};
use super::fonts::{resolve_font, FontProperties};
use super::renderer::Renderer;
use super::transforms::{Bbox, PyRadians};
use crate::agg::pixfmt::Rgba;

/// Python's float `%` (result has the sign of the divisor).
pub fn py_mod(a: f64, b: f64) -> f64 {
    let m = a % b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m + b
        } else {
            m
        }
    } else {
        0.0f64.copysign(b)
    }
}

/// Python `max(a, b)` for floats.
#[inline]
pub fn pymax(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python `min(a, b)` for floats.
#[inline]
pub fn pymin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// `cbook.is_math_text`.
pub fn is_math_text(s: &str) -> bool {
    let dollars = s.matches('$').count() as i64 - s.matches("\\$").count() as i64;
    dollars > 0 && dollars % 2 == 0
}

pub fn rotate_point(angle: f64, x: f64, y: f64) -> (f64, f64) {
    if angle == 0.0 {
        return (x, y);
    }
    let r = angle.to_radians_py();
    let (c, s) = (r.cos(), r.sin());
    (c * x - s * y, s * x + c * y)
}

/// A laid-out line: text, (width, ascent, descent), offset.
#[derive(Clone, Debug)]
pub struct LayoutLine {
    pub text: String,
    pub wad: (f64, f64, f64),
    pub xy: (f64, f64),
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub bbox: Bbox,
    pub lines: Vec<LayoutLine>,
    pub corner: (f64, f64),
    pub size_horiz: (f64, f64),
}

/// The `bbox=` patch of a text (FancyBboxPatch around the text).
#[derive(Clone, Debug)]
pub struct TextBbox {
    pub patch: super::patches::Patch,
    pub pad: f64,
}

#[derive(Clone, Debug)]
pub struct Text {
    pub base: ArtistBase,
    pub x: f64,
    pub y: f64,
    pub text: String,
    pub color: Rgba,
    pub fontproperties: FontProperties,
    pub usetex: bool,
    pub parse_math: bool,
    pub wrap: bool,
    pub va: String,
    pub ha: String,
    pub multialignment: Option<String>,
    pub rotation: f64,
    pub transform_rotates_text: bool,
    /// `None` for 'normal'
    pub linespacing: Option<f64>,
    pub rotation_mode: String,
    pub antialiased: bool,
    pub bbox_patch: Option<Box<TextBbox>>,
    pub url: Option<String>,
    /// the object this text mirrors (the `mtext` of `draw_text`)
    pub owner: Option<super::renderer::OwnerRef>,
}

impl Text {
    /// `Text(x, y, text)` with rc defaults.
    pub fn new(x: f64, y: f64, text: &str) -> Text {
        let rc = super::rc::rc();
        Text {
            base: ArtistBase { zorder: 3.0, transform: TSpec::Identity, ..ArtistBase::default() },
            x,
            y,
            text: text.to_string(),
            color: rc.color("text.color"),
            fontproperties: FontProperties::from_rc(),
            usetex: false,
            parse_math: rc.boolean("text.parse_math"),
            wrap: false,
            va: "baseline".into(),
            ha: "left".into(),
            multialignment: None,
            rotation: 0.0,
            transform_rotates_text: false,
            linespacing: None,
            rotation_mode: "default".into(),
            antialiased: rc.boolean("text.antialiased"),
            bbox_patch: None,
            url: None,
            owner: None,
        }
    }

    /// `Text._reset_visual_defaults()` (keeps position and transform).
    pub fn reset_visual_defaults(&mut self) {
        let fresh = Text::new(self.x, self.y, "");
        self.text = String::new();
        self.color = fresh.color;
        self.fontproperties = fresh.fontproperties;
        self.usetex = false;
        self.parse_math = fresh.parse_math;
        self.wrap = false;
        self.va = "baseline".into();
        self.ha = "left".into();
        self.multialignment = None;
        self.rotation = 0.0;
        self.transform_rotates_text = false;
        self.bbox_patch = None;
        self.linespacing = None;
        self.rotation_mode = "default".into();
        self.antialiased = fresh.antialiased;
    }

    pub fn set_rotation_deg(&mut self, r: f64) {
        self.rotation = py_mod(r, 360.0);
    }

    pub fn get_rotation(&self, _ctx: &TransCtx) -> f64 {
        // transform_rotates_text is rarely used; angles are not transformed.
        self.rotation
    }

    pub fn multialignment(&self) -> &str {
        self.multialignment.as_deref().unwrap_or(&self.ha)
    }

    /// `_preprocess_math`.
    pub fn preprocess_math(&self, s: &str) -> (String, bool) {
        if !self.parse_math {
            (s.to_string(), false)
        } else if is_math_text(s) {
            (s.to_string(), true)
        } else {
            (s.replace("\\$", "$"), false)
        }
    }

    fn ha_for_angle(&self, angle: f64) -> &'static str {
        let anchor_at_bottom = self.va == "bottom";
        if angle <= 10.0
            || (85.0..=95.0).contains(&angle)
            || 350.0 <= angle
            || (170.0..=190.0).contains(&angle)
            || (265.0..=275.0).contains(&angle)
        {
            "center"
        } else if (10.0 < angle && angle < 85.0) || (190.0 < angle && angle < 265.0) {
            if anchor_at_bottom {
                "left"
            } else {
                "right"
            }
        } else if anchor_at_bottom {
            "right"
        } else {
            "left"
        }
    }

    fn va_for_angle(&self, angle: f64) -> &'static str {
        let anchor_at_left = self.ha == "left";
        if angle <= 10.0
            || 350.0 <= angle
            || (170.0..=190.0).contains(&angle)
            || (80.0..=100.0).contains(&angle)
            || (260.0..=280.0).contains(&angle)
        {
            "center"
        } else if (190.0 < angle && angle < 260.0) || (10.0 < angle && angle < 80.0) {
            if anchor_at_left {
                "baseline"
            } else {
                "top"
            }
        } else if anchor_at_left {
            "top"
        } else {
            "baseline"
        }
    }

    /// Font height metrics from the OS/2 (or hhea) table.
    fn font_height_metrics(&self, dpi: f64) -> Option<(f64, f64, f64)> {
        let font = resolve_font(&self.fontproperties)?;
        let sf = &font.fonts[0].sfnt;
        let fontsize = self.fontproperties.size;
        let upem = sf.units_per_em as f64;
        let scale = 1.0 / upem * fontsize * dpi / 72.0;
        if let Some(os2) = sf.os2.filter(|o| o.version != 0xffff) {
            let line_gap = os2.s_typo_line_gap as f64 * scale;
            let asc = os2.s_typo_ascender as f64 * scale;
            let desc = -(os2.s_typo_descender as f64) * scale;
            return Some((asc, desc, line_gap));
        }
        Some((sf.ascender as f64 * scale, -(sf.descender as f64) * scale, sf.line_gap as f64 * scale))
    }

    /// `_get_wrapped_text` (wrapping is not supported yet).
    pub fn wrapped_text(&self) -> String {
        self.text.clone()
    }

    /// `Text._get_layout`.
    pub fn get_layout(&self, renderer: &mut Renderer, ctx: &TransCtx) -> Layout {
        let dpi = ctx.dpi;
        let thisx = 0.0;
        let mut thisy = 0.0;
        let text = self.wrapped_text();
        let lines: Vec<&str> = text.split('\n').collect();
        let mut wads: Vec<(f64, f64, f64)> = Vec::with_capacity(lines.len());
        let mut xs = Vec::with_capacity(lines.len());
        let mut ys = Vec::with_capacity(lines.len());
        let (min_ascent, min_descent, mut line_gap) = match self.font_height_metrics(dpi) {
            Some(m) => m,
            None => {
                let (_, h, d) = renderer.get_text_width_height_descent("lp", &self.fontproperties, false);
                (h - d, d, 0.0)
            }
        };
        if lines.len() == 1 {
            line_gap = 0.0;
        }
        let mut baseline = 0.0;
        let mut d = 0.0;
        for line in &lines {
            let (clean, ismath) = self.preprocess_math(line);
            let (w, h, dd) = if !clean.is_empty() {
                renderer.get_text_width_height_descent(&clean, &self.fontproperties, ismath)
            } else {
                (0.0, 0.0, 0.0)
            };
            d = dd;
            let mut a = h - d;
            match self.linespacing {
                None => {
                    a = pymax(a, min_ascent) + line_gap / 2.0;
                    d = pymax(d, min_descent) + line_gap / 2.0;
                }
                Some(ls) => {
                    let line_height = ls * (min_ascent + min_descent);
                    let leading = line_height - (a + d);
                    a += leading / 2.0;
                    d += leading / 2.0;
                }
            }
            baseline = a - thisy;
            thisy -= a;
            wads.push((w, a, d));
            xs.push(thisx);
            ys.push(thisy);
            thisy -= d;
        }
        let descent = d;
        let ws: Vec<f64> = wads.iter().map(|w| w.0).collect();
        let width = ws.iter().copied().fold(f64::NEG_INFINITY, pymax);
        let xmin0 = 0.0;
        let xmax0 = width;
        let ymax0 = 0.0;
        let ymin0 = ys[ys.len() - 1] - descent;
        let malign = self.multialignment().to_string();
        let offset_layout: Vec<(f64, f64)> = xs
            .iter()
            .zip(ys.iter())
            .zip(ws.iter())
            .map(|((&x, &y), &w)| match malign.as_str() {
                "center" => (x + width / 2.0 - w / 2.0, y),
                "right" => (x + width - w, y),
                _ => (x, y),
            })
            .collect();
        let corners_horiz = [(xmin0, ymin0), (xmin0, ymax0), (xmax0, ymax0), (xmax0, ymin0)];
        let size_horiz = (xmax0 - xmin0, ymax0 - ymin0);
        let angle = self.get_rotation(ctx);
        let corners_rotated: Vec<(f64, f64)> = corners_horiz.iter().map(|&(x, y)| rotate_point(angle, x, y)).collect();
        let cx: Vec<f64> = corners_rotated.iter().map(|c| c.0).collect();
        let cy: Vec<f64> = corners_rotated.iter().map(|c| c.1).collect();
        let mut xmin = cx[1..].iter().copied().fold(cx[0], pymin);
        let xmax = cx[1..].iter().copied().fold(cx[0], pymax);
        let mut ymin = cy[1..].iter().copied().fold(cy[0], pymin);
        let ymax = cy[1..].iter().copied().fold(cy[0], pymax);
        let width_rot = xmax - xmin;
        let height_rot = ymax - ymin;
        let mut halign: &str = &self.ha;
        let mut valign: &str = &self.va;
        let (offsetx, offsety);
        if self.rotation_mode != "anchor" {
            if self.rotation_mode == "xtick" {
                halign = self.ha_for_angle(angle);
            } else if self.rotation_mode == "ytick" {
                valign = self.va_for_angle(angle);
            }
            offsetx = match halign {
                "left" => xmin,
                "right" => xmax,
                _ => (xmin + xmax) / 2.0,
            };
            offsety = match valign {
                "bottom" => ymin,
                "top" => ymax,
                "center" => (ymin + ymax) / 2.0,
                "baseline" => ymin + descent,
                _ => ymin + height_rot - baseline / 2.0,
            };
        } else {
            let (xmin1, ymin1) = corners_horiz[0];
            let (xmax1, ymax1) = corners_horiz[2];
            let ox = match halign {
                "left" => xmin1,
                "right" => xmax1,
                _ => (xmin1 + xmax1) / 2.0,
            };
            let oy = match valign {
                "bottom" => ymin1,
                "top" => ymax1,
                "center" => (ymin1 + ymax1) / 2.0,
                "baseline" => ymax1 - baseline,
                _ => ymax1 - baseline / 2.0,
            };
            let (a, b) = rotate_point(angle, ox, oy);
            offsetx = a;
            offsety = b;
        }
        xmin -= offsetx;
        ymin -= offsety;
        let bbox = Bbox::from_bounds(xmin, ymin, width_rot, height_rot);
        let out_lines = lines
            .iter()
            .zip(wads.iter())
            .zip(offset_layout.iter())
            .map(|((l, w), &(x, y))| {
                let (rx, ry) = rotate_point(angle, x, y);
                LayoutLine { text: l.to_string(), wad: *w, xy: (rx - offsetx, ry - offsety) }
            })
            .collect();
        let (x, y) = corners_rotated[0];
        Layout { bbox, lines: out_lines, corner: (x - offsetx, y - offsety), size_horiz }
    }

    /// Position in display coordinates.
    pub fn xy_display(&self, ctx: &TransCtx) -> (f64, f64) {
        ctx.resolve(&self.base.transform).transform_point(self.x, self.y)
    }

    /// `Text.get_window_extent`.
    pub fn get_window_extent(&self, renderer: &mut Renderer, ctx: &TransCtx) -> Bbox {
        if !self.base.visible {
            return Bbox::unit();
        }
        if self.text.is_empty() {
            let (tx, ty) = self.xy_display(ctx);
            return Bbox::from_bounds(tx, ty, 0.0, 0.0);
        }
        let layout = self.get_layout(renderer, ctx);
        let (x, y) = self.xy_display(ctx);
        layout.bbox.translated(x, y)
    }

    /// `update_bbox_position_size`.
    fn update_bbox_position_size(&mut self, renderer: &mut Renderer, ctx: &TransCtx) {
        if self.bbox_patch.is_none() {
            return;
        }
        let (posx, posy) = self.xy_display(ctx);
        let layout = self.get_layout(renderer, ctx);
        let ((x_box, y_box), (w_box, h_box)) = (layout.corner, layout.size_horiz);
        let rot = self.get_rotation(ctx);
        let fs = renderer.points_to_pixels(self.fontproperties.size);
        let bp = self.bbox_patch.as_mut().unwrap();
        bp.patch.set_bounds(0.0, 0.0, w_box, h_box);
        let t = super::transforms::Affine2D::identity().rotate_deg(rot).translate(posx + x_box, posy + y_box);
        bp.patch.base.transform = TSpec::Affine(t.0);
        bp.patch.mutation_scale = fs;
    }

    pub fn draw(&mut self, renderer: &mut Renderer, ctx: &TransCtx, clip: &ClipCtx) {
        renderer.artist_start();
        if !self.base.visible || self.text.is_empty() {
            return;
        }
        renderer.open_group("text", self.base.gid.as_deref());
        let layout = self.get_layout(renderer, ctx);
        let trans = ctx.resolve(&self.base.transform);
        let (posx, posy) = trans.transform_point(self.x, self.y);
        if !posx.is_finite() || !posy.is_finite() {
            // (matplotlib returns without closing the group)
            return;
        }
        let flip = renderer.text_flip();
        if self.bbox_patch.is_some() {
            self.update_bbox_position_size(renderer, ctx);
            if let Some(bp) = &self.bbox_patch {
                bp.patch.draw(renderer, ctx, clip);
            }
        }
        let mut gc = GcState::default();
        gc.set_foreground(self.color);
        gc.set_alpha(self.base.alpha);
        gc.antialiased = self.antialiased;
        gc.snap = self.base.snap;
        gc.set_clip_from(&self.base, clip);
        let angle = self.get_rotation(ctx);
        let mtext = super::renderer::MText {
            owner: self.owner.as_ref(),
            x: posx,
            y: posy,
            ha: &self.ha,
            va: &self.va,
            rotation_mode: &self.rotation_mode,
        };
        let mtext = if layout.lines.len() == 1 { Some(&mtext) } else { None };
        for line in &layout.lines {
            let x = line.xy.0 + posx;
            let mut y = line.xy.1 + posy;
            if let Some(canvash) = flip {
                y = canvash - y;
            }
            let (clean, ismath) = self.preprocess_math(&line.text);
            renderer.draw_text_for(&gc, x, y, &clean, &self.fontproperties, angle, ismath, mtext);
        }
        renderer.close_group("text");
    }
}
