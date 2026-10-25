//! Colour specifications as stored by artists.  Parsing of user colour
//! strings happens on the Python side; Rust keeps the resolved values plus
//! the special `'none'` / `'auto'` markers whose semantics matter.

use crate::agg::pixfmt::Rgba;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorSpec {
    Rgba(Rgba),
    /// `'none'`: fully transparent, ignores the artist alpha.
    None,
    /// `'auto'` (marker colours following the line colour).
    Auto,
}

impl ColorSpec {
    pub fn rgba(r: f64, g: f64, b: f64, a: f64) -> ColorSpec {
        ColorSpec::Rgba(Rgba::new(r, g, b, a))
    }

    /// `to_rgba(c, alpha)`.
    pub fn to_rgba(&self, alpha: Option<f64>) -> Rgba {
        match self {
            ColorSpec::Rgba(c) => match alpha {
                Some(a) => Rgba::new(c.r, c.g, c.b, a),
                None => *c,
            },
            _ => Rgba::new(0.0, 0.0, 0.0, 0.0),
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(self, ColorSpec::None)
    }

    pub fn is_auto(&self) -> bool {
        matches!(self, ColorSpec::Auto)
    }
}

impl From<Rgba> for ColorSpec {
    fn from(c: Rgba) -> Self {
        ColorSpec::Rgba(c)
    }
}

/// `to_rgba(c, alpha)` for a plain colour.
pub fn to_rgba(c: Rgba, alpha: Option<f64>) -> Rgba {
    match alpha {
        Some(a) => Rgba::new(c.r, c.g, c.b, a),
        None => c,
    }
}
