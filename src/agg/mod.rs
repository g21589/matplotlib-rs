//! A faithful Rust port of the subset of Anti-Grain Geometry 2.4 used by
//! matplotlib's Agg backend.
//!
//! Anti-Grain Geometry - Version 2.4
//! Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
//! Permission to copy, use, modify, sell and distribute this software is
//! granted provided this copyright notice appears in all copies.  This
//! software is provided "as is" without express or implied warranty, and
//! with no claim as to its suitability for any purpose.

pub mod basics;
pub mod clip;
pub mod curves;
pub mod pixfmt;
pub mod rasterizer;
pub mod renderer;
pub mod scanline;
pub mod span_gouraud;
pub mod stroke;
pub mod trans_affine;

pub use basics::*;
pub use trans_affine::{ConvTransform, TransAffine};
