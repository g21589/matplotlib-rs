//! Typed snapshot of the rcParams used by the Rust model.  Python resolves
//! values (colors to RGBA, font sizes to points, ...) and passes them in.

use std::collections::HashMap;

use crate::agg::pixfmt::Rgba;

#[derive(Clone, Debug)]
pub enum RcValue {
    Bool(bool),
    Num(f64),
    Str(String),
    Color(Rgba),
    NumList(Vec<f64>),
    StrList(Vec<String>),
    ColorList(Vec<Rgba>),
    None,
}

/// rcParams snapshot.  Values are looked up by their matplotlib key.
#[derive(Clone, Debug, Default)]
pub struct Rc {
    pub map: HashMap<String, RcValue>,
}

impl Rc {
    pub fn get(&self, key: &str) -> Option<&RcValue> {
        self.map.get(key)
    }

    pub fn num(&self, key: &str) -> f64 {
        match self.map.get(key) {
            Some(RcValue::Num(v)) => *v,
            Some(RcValue::Bool(b)) => *b as i32 as f64,
            _ => default_num(key),
        }
    }

    /// A font size entry, resolving named sizes ('large', ...) against
    /// `font.size` like `FontProperties.set_size`.
    pub fn font_size(&self, key: &str) -> f64 {
        match self.map.get(key) {
            Some(RcValue::Num(v)) => *v,
            Some(RcValue::Str(s)) => super::fonts::font_scaling(s).unwrap_or(1.0) * self.num("font.size"),
            _ => {
                let d = default_num(key);
                if d.is_nan() {
                    self.num("font.size")
                } else {
                    d
                }
            }
        }
    }

    pub fn opt_num(&self, key: &str) -> Option<f64> {
        match self.map.get(key) {
            Some(RcValue::Num(v)) => Some(*v),
            Some(RcValue::None) => None,
            None => {
                let d = default_num(key);
                if d.is_nan() {
                    None
                } else {
                    Some(d)
                }
            }
            _ => None,
        }
    }

    pub fn boolean(&self, key: &str) -> bool {
        match self.map.get(key) {
            Some(RcValue::Bool(b)) => *b,
            Some(RcValue::Num(v)) => *v != 0.0,
            Some(RcValue::Str(s)) => s == "True" || s == "true",
            _ => default_bool(key),
        }
    }

    pub fn string(&self, key: &str) -> String {
        match self.map.get(key) {
            Some(RcValue::Str(s)) => s.clone(),
            Some(RcValue::Bool(b)) => if *b { "True".into() } else { "False".into() },
            _ => default_str(key).to_string(),
        }
    }

    pub fn color(&self, key: &str) -> Rgba {
        match self.map.get(key) {
            Some(RcValue::Color(c)) => *c,
            _ => default_color(key),
        }
    }

    pub fn opt_color(&self, key: &str) -> Option<Rgba> {
        match self.map.get(key) {
            Some(RcValue::Color(c)) => Some(*c),
            _ => None,
        }
    }

    pub fn num_list(&self, key: &str) -> Vec<f64> {
        match self.map.get(key) {
            Some(RcValue::NumList(v)) => v.clone(),
            _ => default_num_list(key),
        }
    }

    pub fn str_list(&self, key: &str) -> Vec<String> {
        match self.map.get(key) {
            Some(RcValue::StrList(v)) => v.clone(),
            Some(RcValue::Str(s)) => vec![s.clone()],
            _ => default_str_list(key),
        }
    }

    pub fn color_list(&self, key: &str) -> Vec<Rgba> {
        match self.map.get(key) {
            Some(RcValue::ColorList(v)) => v.clone(),
            _ => tab10(),
        }
    }
}

pub fn tab10() -> Vec<Rgba> {
    [
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
        (1.0, 0.4980392156862745, 0.054901960784313725),
        (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
        (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
        (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
        (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
        (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
        (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
        (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
        (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
    ]
    .iter()
    .map(|&(r, g, b)| Rgba::new(r, g, b, 1.0))
    .collect()
}

fn default_num(key: &str) -> f64 {
    match key {
        "figure.dpi" => 100.0,
        "font.size" => 10.0,
        "lines.linewidth" => 1.5,
        "lines.markersize" => 6.0,
        "lines.markeredgewidth" => 1.0,
        "axes.linewidth" => 0.8,
        "axes.xmargin" | "axes.ymargin" => 0.05,
        "axes.zmargin" => 0.05,
        "axes.titlepad" => 6.0,
        "axes.labelpad" => 4.0,
        "axes.titlesize" => 12.0,
        "axes.labelsize" => 10.0,
        "axes.formatter.offset_threshold" => 4.0,
        "xtick.major.size" | "ytick.major.size" => 3.5,
        "xtick.minor.size" | "ytick.minor.size" => 2.0,
        "xtick.major.width" | "ytick.major.width" => 0.8,
        "xtick.minor.width" | "ytick.minor.width" => 0.6,
        "xtick.major.pad" | "ytick.major.pad" => 3.5,
        "xtick.minor.pad" | "ytick.minor.pad" => 3.4,
        "xtick.labelsize" | "ytick.labelsize" => 10.0,
        "grid.linewidth" => 0.8,
        "grid.alpha" => 1.0,
        "patch.linewidth" => 1.0,
        "hatch.linewidth" => 1.0,
        "path.simplify_threshold" => 0.111111111111,
        "legend.fontsize" => 10.0,
        "legend.title_fontsize" => 10.0,
        "legend.framealpha" => 0.8,
        "legend.borderpad" => 0.4,
        "legend.labelspacing" => 0.5,
        "legend.handlelength" => 2.0,
        "legend.handleheight" => 0.7,
        "legend.handletextpad" => 0.8,
        "legend.borderaxespad" => 0.5,
        "legend.columnspacing" => 2.0,
        "legend.markerscale" => 1.0,
        "legend.scatterpoints" => 1.0,
        "legend.numpoints" => 1.0,
        "scatter.edgecolors_lw" => 1.0,
        "figure.subplot.left" => 0.125,
        "figure.subplot.right" => 0.9,
        "figure.subplot.bottom" => 0.11,
        "figure.subplot.top" => 0.88,
        "figure.subplot.wspace" => 0.2,
        "figure.subplot.hspace" => 0.2,
        "figure.titlesize" => 12.0,
        "figure.labelsize" => 12.0,
        "errorbar.capsize" => 0.0,
        "boxplot.flierprops.markersize" => 6.0,
        "image.resample" => 1.0,
        "savefig.pad_inches" => 0.1,
        "axes.titley" => f64::NAN,
        _ => f64::NAN,
    }
}

fn default_bool(key: &str) -> bool {
    matches!(
        key,
        "axes.formatter.useoffset"
            | "font.enable_last_resort"
            | "axes.unicode_minus"
            | "path.simplify"
            | "lines.scale_dashes"
            | "text.antialiased"
            | "lines.antialiased"
            | "patch.antialiased"
            | "lines.dash_capstyle_dummy"
            | "axes.spines.left"
            | "axes.spines.right"
            | "axes.spines.bottom"
            | "axes.spines.top"
            | "xtick.bottom"
            | "xtick.labelbottom"
            | "ytick.left"
            | "ytick.labelleft"
            | "xtick.major.bottom"
            | "xtick.major.top"
            | "xtick.minor.bottom"
            | "xtick.minor.top"
            | "ytick.major.left"
            | "ytick.major.right"
            | "ytick.minor.left"
            | "ytick.minor.right"
            | "legend.frameon"
            | "legend.fancybox"
            | "legend.shadow_dummy"
            | "figure.frameon"
            | "text.parse_math"
            | "axes.autolimit_dummy"
            | "legend.scatteryoffsets_dummy"
            | "scatter.marker_dummy"
            | "axes.formatter.min_exponent_dummy"
    )
}

fn default_str(key: &str) -> &'static str {
    match key {
        "axes.autolimit_mode" => "data",
        "axes.axisbelow" => "line",
        "xtick.direction" | "ytick.direction" => "out",
        "xtick.alignment" => "center",
        "ytick.alignment" => "center_baseline",
        "lines.linestyle" => "-",
        "lines.marker" => "None",
        "lines.solid_capstyle" => "projecting",
        "lines.solid_joinstyle" => "round",
        "lines.dash_capstyle" => "butt",
        "lines.dash_joinstyle" => "round",
        "lines.drawstyle" => "default",
        "markers.fillstyle" => "full",
        "text.hinting" => "default",
        "axes.titleweight" => "normal",
        "axes.titlelocation" => "center",
        "axes.labelweight" => "normal",
        "font.style" => "normal",
        "font.variant" => "normal",
        "font.weight" => "normal",
        "font.stretch" => "normal",
        "grid.linestyle" => "-",
        "axes.grid.which" => "major",
        "axes.grid.axis" => "both",
        "legend.loc" => "best",
        "legend.edgecolor" => "0.8",
        "legend.facecolor" => "inherit",
        "legend.labelcolor" => "None",
        "image.interpolation" => "auto",
        "image.origin" => "upper",
        "image.cmap" => "viridis",
        "image.aspect" => "equal",
        "scatter.marker" => "o",
        "hist.bins" => "10",
        "mathtext.fontset" => "dejavusans",
        "mathtext.default" => "it",
        "xaxis.labellocation" => "center",
        "yaxis.labellocation" => "center",
        "savefig.bbox" => "None",
        "figure.titleweight" => "normal",
        "figure.labelweight" => "normal",
        "errorbar.capsize_dummy" => "",
        _ => "",
    }
}

fn default_color(key: &str) -> Rgba {
    match key {
        "figure.facecolor" | "axes.facecolor" | "savefig.facecolor" | "legend.facecolor" => {
            Rgba::new(1.0, 1.0, 1.0, 1.0)
        }
        "figure.edgecolor" | "savefig.edgecolor" => Rgba::new(1.0, 1.0, 1.0, 1.0),
        "grid.color" => Rgba::new(0.6901960784313725, 0.6901960784313725, 0.6901960784313725, 1.0),
        "legend.edgecolor" => Rgba::new(0.8, 0.8, 0.8, 1.0),
        "patch.facecolor" => Rgba::new(0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
        _ => Rgba::new(0.0, 0.0, 0.0, 1.0),
    }
}

fn default_num_list(key: &str) -> Vec<f64> {
    match key {
        "figure.figsize" => vec![6.4, 4.8],
        "axes.formatter.limits" => vec![-5.0, 6.0],
        "lines.dashed_pattern" => vec![3.7, 1.6],
        "lines.dashdot_pattern" => vec![6.4, 1.6, 1.0, 1.6],
        "lines.dotted_pattern" => vec![1.0, 1.65],
        _ => vec![],
    }
}

fn default_str_list(key: &str) -> Vec<String> {
    match key {
        "font.family" => vec!["sans-serif".into()],
        "font.sans-serif" => ["DejaVu Sans", "Bitstream Vera Sans", "Computer Modern Sans Serif", "Lucida Grande",
            "Verdana", "Geneva", "Lucid", "Arial", "Helvetica", "Avant Garde", "sans-serif"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        _ => vec![],
    }
}

static RC: std::sync::LazyLock<std::sync::RwLock<std::sync::Arc<Rc>>> =
    std::sync::LazyLock::new(|| std::sync::RwLock::new(std::sync::Arc::new(Rc::default())));
static RC_VERSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The current rcParams snapshot (cheap to clone).
pub fn rc() -> std::sync::Arc<Rc> {
    RC.read().unwrap().clone()
}

/// Version counter bumped on every rc change (for caches keyed on rc).
pub fn rc_version() -> u64 {
    RC_VERSION.load(std::sync::atomic::Ordering::Relaxed)
}

/// Update rcParams (copy-on-write).
pub fn rc_update<F: FnOnce(&mut Rc)>(f: F) {
    let mut guard = RC.write().unwrap();
    f(std::sync::Arc::make_mut(&mut *guard));
    RC_VERSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Replace the whole rc snapshot.
pub fn rc_replace(new: Rc) {
    *RC.write().unwrap() = std::sync::Arc::new(new);
    RC_VERSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
