//! Port of `matplotlib.scale` (linear, log, symlog, logit, asinh).

use super::logticker::{LogFormatter, LogFormatterKind, LogLocator, Subs};
use super::rc::rc;
use super::ticker::{Formatter, Locator, MaxNLocator, MinorNDivs, ScalarFormatter};
use super::transforms::ScaleTransform;

#[derive(Clone, Debug, PartialEq)]
pub enum Scale {
    Linear,
    Log { base: f64, subs: Option<Vec<f64>>, clip: bool },
    Symlog { base: f64, linthresh: f64, linscale: f64, subs: Option<Vec<f64>> },
    Logit { clip: bool },
    Asinh { linear_width: f64 },
}

impl Scale {
    pub fn name(&self) -> &'static str {
        match self {
            Scale::Linear => "linear",
            Scale::Log { .. } => "log",
            Scale::Symlog { .. } => "symlog",
            Scale::Logit { .. } => "logit",
            Scale::Asinh { .. } => "asinh",
        }
    }

    pub fn transform(&self) -> Option<ScaleTransform> {
        match self {
            Scale::Linear => None,
            Scale::Log { base, clip, .. } => Some(ScaleTransform::Log { base: *base, clip: *clip }),
            Scale::Symlog { base, linthresh, linscale, .. } => {
                Some(ScaleTransform::Symlog { base: *base, linthresh: *linthresh, linscale: *linscale })
            }
            Scale::Logit { clip } => Some(ScaleTransform::Logit { clip: *clip }),
            Scale::Asinh { linear_width } => Some(ScaleTransform::AsinH { a: *linear_width }),
        }
    }

    pub fn default_locators_and_formatters(&self, is_x: bool) -> (Locator, Locator, Formatter, Formatter) {
        let r = rc();
        let minor_visible = r.boolean(if is_x { "xtick.minor.visible" } else { "ytick.minor.visible" });
        match self {
            Scale::Log { base, subs, .. } => (
                Locator::Log(LogLocator::new(*base, Subs::List(vec![1.0]), None)),
                Locator::Log(LogLocator::new(*base, subs.clone().map(Subs::List).unwrap_or(Subs::Auto), None)),
                Formatter::Log(LogFormatter::new(LogFormatterKind::SciNotation, *base, false)),
                Formatter::Log(LogFormatter::new(LogFormatterKind::SciNotation, *base, subs.is_some())),
            ),
            _ => (
                Locator::MaxN(MaxNLocator::auto()),
                if minor_visible { Locator::AutoMinor { ndivs: None::<MinorNDivs> } } else { Locator::Null },
                Formatter::Scalar(ScalarFormatter::new()),
                Formatter::Null,
            ),
        }
    }

    pub fn limit_range_for_scale(&self, vmin: f64, vmax: f64, minpos: f64) -> (f64, f64) {
        match self {
            Scale::Log { .. } => {
                let mp = if minpos.is_finite() { minpos } else { 1e-300 };
                (if vmin <= 0.0 { mp } else { vmin }, if vmax <= 0.0 { mp } else { vmax })
            }
            Scale::Logit { .. } => {
                let mp = if minpos.is_finite() { minpos } else { 1e-7 };
                (if vmin <= 0.0 { mp } else { vmin }, if vmax >= 1.0 { 1.0 - mp } else { vmax })
            }
            _ => (vmin, vmax),
        }
    }

    pub fn val_in_range(&self, v: f64) -> bool {
        match self {
            Scale::Log { .. } => v.is_finite() && v > 0.0,
            Scale::Logit { .. } => v.is_finite() && v > 0.0 && v < 1.0,
            _ => true,
        }
    }
}
