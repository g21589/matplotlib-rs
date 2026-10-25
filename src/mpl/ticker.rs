//! Port of `matplotlib.ticker` (the default locators and formatters; other
//! tick helpers can be supplied from Python through the `External*` traits).

use std::sync::Arc;

use super::pyfmt::{self, divmod, floordiv, np_round};
use super::rc::rc;

/// What a locator/formatter may ask of its axis.
#[derive(Clone, Debug, Default)]
pub struct AxisInfo {
    pub view: (f64, f64),
    pub data: (f64, f64),
    pub minpos: f64,
    pub tick_space: i64,
    /// 'x' or 'y'
    pub axis_name: char,
    pub scale: String,
    pub major_locs: Vec<f64>,
}

pub trait ExternalLocator: Send + Sync + std::fmt::Debug {
    fn call(&self, info: &AxisInfo) -> Vec<f64>;
    fn tick_values(&self, vmin: f64, vmax: f64, info: &AxisInfo) -> Vec<f64>;
    fn nonsingular(&self, v0: f64, v1: f64) -> (f64, f64) {
        nonsingular(v0, v1, 0.05, 1e-15, true)
    }
    fn view_limits(&self, v0: f64, v1: f64) -> (f64, f64) {
        nonsingular(v0, v1, 0.001, 1e-15, true)
    }
    fn is_null(&self) -> bool {
        false
    }
    fn as_any(&self) -> &dyn std::any::Any;
}

pub trait ExternalFormatter: Send + Sync + std::fmt::Debug {
    fn format_ticks(&self, values: &[f64], info: &AxisInfo) -> Vec<String>;
    fn get_offset(&self) -> String;
    fn as_any(&self) -> &dyn std::any::Any;
}

/// `transforms._nonsingular`.
pub fn nonsingular(vmin: f64, vmax: f64, expander: f64, tiny: f64, increasing: bool) -> (f64, f64) {
    if !vmin.is_finite() || !vmax.is_finite() {
        return (-expander, expander);
    }
    let (mut vmin, mut vmax) = (vmin, vmax);
    let mut swapped = false;
    if vmax < vmin {
        std::mem::swap(&mut vmin, &mut vmax);
        swapped = true;
    }
    let maxabsvalue = super::text::pymax(vmin.abs(), vmax.abs());
    if maxabsvalue < (1e6 / tiny) * f64::MIN_POSITIVE {
        vmin = -expander;
        vmax = expander;
    } else if vmax - vmin <= maxabsvalue * tiny {
        if vmax == 0.0 && vmin == 0.0 {
            vmin = -expander;
            vmax = expander;
        } else {
            vmin -= expander * vmin.abs();
            vmax += expander * vmax.abs();
        }
    }
    if swapped && !increasing {
        std::mem::swap(&mut vmin, &mut vmax);
    }
    (vmin, vmax)
}

fn sorted2(a: (f64, f64)) -> (f64, f64) {
    if a.1 < a.0 {
        (a.1, a.0)
    } else {
        a
    }
}

/// `np.arange(start, stop)` for float bounds with unit step.
fn arange_f(start: f64, stop: f64) -> Vec<f64> {
    let n = (stop - start).ceil();
    if !(n > 0.0) {
        return Vec::new();
    }
    let n = n as usize;
    let mut v = Vec::with_capacity(n);
    if n >= 1 {
        v.push(start);
    }
    if n >= 2 {
        v.push(start + 1.0);
        let delta = v[1] - start;
        for i in 2..n {
            v.push(start + i as f64 * delta);
        }
    }
    v
}

#[derive(Clone, Debug, PartialEq)]
pub enum NBins {
    Auto,
    N(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaxNLocator {
    pub nbins: NBins,
    pub steps: Vec<f64>,
    pub extended_steps: Vec<f64>,
    pub integer: bool,
    pub symmetric: bool,
    /// None, "lower", "upper", "both"
    pub prune: Option<String>,
    pub min_n_ticks: usize,
}

struct EdgeInteger {
    step: f64,
    offset: f64,
}

impl EdgeInteger {
    fn new(step: f64, offset: f64) -> Self {
        EdgeInteger { step, offset: offset.abs() }
    }
    fn closeto(&self, ms: f64, edge: f64) -> bool {
        let tol = if self.offset > 0.0 {
            let digits = (self.offset / self.step).log10();
            let tol = super::text::pymax(1e-10, 10f64.powf(digits - 12.0));
            super::text::pymin(0.4999, tol)
        } else {
            1e-10
        };
        (ms - edge).abs() < tol
    }
    fn le(&self, x: f64) -> f64 {
        let (d, m) = divmod(x, self.step);
        if self.closeto(m / self.step, 1.0) {
            d + 1.0
        } else {
            d
        }
    }
    fn ge(&self, x: f64) -> f64 {
        let (d, m) = divmod(x, self.step);
        if self.closeto(m / self.step, 0.0) {
            d
        } else {
            d + 1.0
        }
    }
}

/// `ticker.scale_range`.
pub fn scale_range(vmin: f64, vmax: f64, n: f64, threshold: f64) -> (f64, f64) {
    let dv = (vmax - vmin).abs();
    let meanv = (vmax + vmin) / 2.0;
    let offset = if meanv.abs() / dv < threshold {
        0.0
    } else {
        10f64.powf(floordiv(meanv.abs().log10(), 1.0)).copysign(meanv)
    };
    let scale = 10f64.powf(floordiv((dv / n).log10(), 1.0));
    (scale, offset)
}

impl MaxNLocator {
    pub fn new(nbins: NBins, steps: Option<Vec<f64>>) -> MaxNLocator {
        let mut l = MaxNLocator {
            nbins,
            steps: Vec::new(),
            extended_steps: Vec::new(),
            integer: false,
            symmetric: false,
            prune: None,
            min_n_ticks: 2,
        };
        l.set_steps(steps);
        l
    }

    /// `AutoLocator()`.
    pub fn auto() -> MaxNLocator {
        MaxNLocator::new(NBins::Auto, Some(vec![1.0, 2.0, 2.5, 5.0, 10.0]))
    }

    pub fn set_steps(&mut self, steps: Option<Vec<f64>>) {
        let mut s = match steps {
            None => vec![1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
            Some(s) => s,
        };
        if s[0] != 1.0 {
            s.insert(0, 1.0);
        }
        if s[s.len() - 1] != 10.0 {
            s.push(10.0);
        }
        let mut ext: Vec<f64> = s[..s.len() - 1].iter().map(|v| 0.1 * v).collect();
        ext.extend_from_slice(&s);
        ext.push(10.0 * s[1]);
        self.steps = s;
        self.extended_steps = ext;
    }

    fn raw_ticks(&self, vmin: f64, vmax: f64, info: &AxisInfo) -> Vec<f64> {
        let nbins = match self.nbins {
            NBins::Auto => {
                let lo = (self.min_n_ticks as i64 - 1).max(1);
                info.tick_space.clamp(lo, 9) as f64
            }
            NBins::N(n) => n as f64,
        };
        let (scale, offset) = scale_range(vmin, vmax, nbins, 100.0);
        let vmin_ = vmin - offset;
        let vmax_ = vmax - offset;
        let mut steps: Vec<f64> = self.extended_steps.iter().map(|s| s * scale).collect();
        if self.integer {
            steps.retain(|&s| s < 1.0 || (s - s.round_ties_even()).abs() < 0.001);
        }
        let raw_step = (vmax_ - vmin_) / nbins;
        let round_numbers = rc().string("axes.autolimit_mode") == "round_numbers";
        let large: Vec<bool> = steps
            .iter()
            .map(|&s| {
                let mut ok = s >= raw_step;
                if round_numbers {
                    let fv = floordiv(vmin_, s) * s;
                    let fvmax = fv + s * nbins;
                    ok = ok && fvmax >= vmax_;
                }
                ok
            })
            .collect();
        let istep = large.iter().position(|&b| b).unwrap_or(steps.len() - 1);
        let mut ticks: Vec<f64> = Vec::new();
        for &step0 in steps[..=istep].iter().rev() {
            let mut step = step0;
            if self.integer && vmax_.floor() - vmin_.ceil() >= self.min_n_ticks as f64 - 1.0 {
                step = super::text::pymax(1.0, step);
            }
            let best_vmin = floordiv(vmin_, step) * step;
            let edge = EdgeInteger::new(step, offset);
            let low = edge.le(vmin_ - best_vmin);
            let high = edge.ge(vmax_ - best_vmin);
            ticks = arange_f(low, high + 1.0).into_iter().map(|t| t * step + best_vmin).collect();
            let nticks = ticks.iter().filter(|&&t| t <= vmax_ && t >= vmin_).count();
            if nticks >= self.min_n_ticks {
                break;
            }
        }
        ticks.into_iter().map(|t| t + offset).collect()
    }

    pub fn tick_values(&self, vmin: f64, vmax: f64, info: &AxisInfo) -> Vec<f64> {
        let (mut vmin, mut vmax) = (vmin, vmax);
        if self.symmetric {
            vmax = super::text::pymax(vmin.abs(), vmax.abs());
            vmin = -vmax;
        }
        let (a, b) = nonsingular(vmin, vmax, 1e-13, 1e-14, true);
        vmin = a;
        vmax = b;
        let locs = self.raw_ticks(vmin, vmax, info);
        let n = locs.len();
        match self.prune.as_deref() {
            Some("lower") => locs[1.min(n)..].to_vec(),
            Some("upper") => locs[..n.saturating_sub(1)].to_vec(),
            Some("both") => {
                if n >= 2 {
                    locs[1..n - 1].to_vec()
                } else {
                    Vec::new()
                }
            }
            _ => locs,
        }
    }

    pub fn view_limits(&self, dmin: f64, dmax: f64, info: &AxisInfo) -> (f64, f64) {
        let (mut dmin, mut dmax) = (dmin, dmax);
        if self.symmetric {
            dmax = super::text::pymax(dmin.abs(), dmax.abs());
            dmin = -dmax;
        }
        let (a, b) = nonsingular(dmin, dmax, 1e-12, 1e-13, true);
        if rc().string("axes.autolimit_mode") == "round_numbers" {
            let t = self.raw_ticks(a, b, info);
            (t[0], t[t.len() - 1])
        } else {
            (a, b)
        }
    }
}

#[derive(Clone, Debug)]
pub enum MinorNDivs {
    Auto,
    N(usize),
}

#[derive(Clone, Debug)]
pub enum Locator {
    MaxN(MaxNLocator),
    Fixed { locs: Vec<f64>, nbins: Option<usize> },
    Null,
    Multiple { step: f64, offset: f64 },
    Linear { numticks: Option<usize> },
    Index { base: f64, offset: f64 },
    AutoMinor { ndivs: Option<MinorNDivs> },
    Log(super::logticker::LogLocator),
    External(Arc<dyn ExternalLocator>),
}

/// `np.isclose(a, b)` with default tolerances.
pub fn isclose(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

/// `np.unique` for floats (sorted, NaNs last).
pub fn np_unique(v: &[f64]) -> Vec<f64> {
    let mut u: Vec<f64> = v.to_vec();
    u.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
    u.dedup_by(|a, b| a == b);
    u
}

impl Locator {
    pub fn is_null(&self) -> bool {
        match self {
            Locator::Null => true,
            Locator::External(e) => e.is_null(),
            _ => false,
        }
    }

    /// `locator()`.
    pub fn call(&self, info: &AxisInfo) -> Vec<f64> {
        match self {
            Locator::MaxN(m) => m.tick_values(info.view.0, info.view.1, info),
            Locator::Fixed { .. } | Locator::Null => self.tick_values(f64::NAN, f64::NAN, info),
            Locator::Multiple { .. } | Locator::Linear { .. } => self.tick_values(info.view.0, info.view.1, info),
            Locator::Index { .. } => self.tick_values(info.data.0, info.data.1, info),
            Locator::AutoMinor { ndivs } => {
                if info.scale == "log" {
                    return Vec::new();
                }
                let majorlocs = np_unique(&info.major_locs);
                if majorlocs.len() < 2 {
                    return Vec::new();
                }
                let majorstep = majorlocs[1] - majorlocs[0];
                let nd = match ndivs {
                    Some(MinorNDivs::N(n)) => *n as f64,
                    _ => {
                        let key = if info.axis_name == 'y' { "ytick.minor.ndivs" } else { "xtick.minor.ndivs" };
                        let r = rc();
                        match r.get(key) {
                            Some(super::rc::RcValue::Num(n)) => *n,
                            _ => {
                                let mant = 10f64.powf(pyfmt::divmod(majorstep.log10(), 1.0).1);
                                if [1.0, 2.5, 5.0, 10.0].iter().any(|&c| isclose(mant, c)) {
                                    5.0
                                } else {
                                    4.0
                                }
                            }
                        }
                    }
                };
                let minorstep = majorstep / nd;
                let (vmin, vmax) = sorted2(info.view);
                let t0 = majorlocs[0];
                let tmin = ((vmin - t0) / minorstep).round_ties_even() as i64;
                let tmax = ((vmax - t0) / minorstep).round_ties_even() as i64 + 1;
                (tmin..tmax).map(|i| (i as f64 * minorstep) + t0).collect()
            }
            Locator::Log(l) => l.call(info),
            Locator::External(e) => e.call(info),
        }
    }

    pub fn tick_values(&self, vmin: f64, vmax: f64, info: &AxisInfo) -> Vec<f64> {
        match self {
            Locator::MaxN(m) => m.tick_values(vmin, vmax, info),
            Locator::Fixed { locs, nbins } => match nbins {
                None => locs.clone(),
                Some(nb) => {
                    let step = ((locs.len() as f64 / *nb as f64).ceil() as usize).max(1);
                    let mut ticks: Vec<f64> = locs.iter().step_by(step).copied().collect();
                    for i in 1..step {
                        let t1: Vec<f64> = locs.iter().skip(i).step_by(step).copied().collect();
                        let m1 = t1.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);
                        let m0 = ticks.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);
                        if m1 < m0 {
                            ticks = t1;
                        }
                    }
                    ticks
                }
            },
            Locator::Null => Vec::new(),
            Locator::Multiple { step, offset } => {
                let (mut vmin, mut vmax) = sorted2((vmin, vmax));
                let edge = EdgeInteger::new(*step, 0.0);
                vmin -= offset;
                vmax -= offset;
                let vmin = edge.ge(vmin) * step;
                let n = floordiv(vmax - vmin + 0.001 * step, *step);
                arange_f(0.0, n + 3.0).into_iter().map(|i| vmin - step + i * step + offset).collect()
            }
            Locator::Linear { numticks } => {
                let (vmin, vmax) = nonsingular(vmin, vmax, 0.05, 1e-15, true);
                let n = numticks.unwrap_or(11);
                if n == 0 {
                    return Vec::new();
                }
                super::path::linspace(vmin, vmax, n)
            }
            Locator::Index { base, offset } => {
                let start = vmin + offset;
                let stop = vmax + 1e-12;
                let n = ((stop - start) / base).ceil();
                if !(n > 0.0) {
                    return Vec::new();
                }
                let n = n as usize;
                let mut v = vec![start];
                if n >= 2 {
                    v.push(start + base);
                    let delta = v[1] - start;
                    for i in 2..n {
                        v.push(start + i as f64 * delta);
                    }
                }
                v
            }
            Locator::AutoMinor { .. } => self.call(info),
            Locator::Log(l) => l.tick_values(vmin, vmax, info),
            Locator::External(e) => e.tick_values(vmin, vmax, info),
        }
    }

    /// `nonsingular(v0, v1)`.
    pub fn nonsingular(&self, v0: f64, v1: f64, info: &AxisInfo) -> (f64, f64) {
        match self {
            Locator::Log(l) => l.nonsingular(v0, v1, info),
            Locator::External(e) => e.nonsingular(v0, v1),
            _ => nonsingular(v0, v1, 0.05, 1e-15, true),
        }
    }

    /// `view_limits(vmin, vmax)`.
    pub fn view_limits(&self, vmin: f64, vmax: f64, info: &AxisInfo) -> (f64, f64) {
        match self {
            Locator::MaxN(m) => m.view_limits(vmin, vmax, info),
            Locator::Linear { numticks } => {
                let (mut vmin, mut vmax) = sorted2((vmin, vmax));
                if vmin == vmax {
                    vmin -= 1.0;
                    vmax += 1.0;
                }
                if rc().string("axes.autolimit_mode") == "round_numbers" {
                    let n = numticks.unwrap_or(11);
                    let base = (n.saturating_sub(1)).max(1) as f64;
                    let (mut exponent, remainder) = divmod((vmax - vmin).log10(), base.log10());
                    if remainder < 0.5 {
                        exponent -= 1.0;
                    }
                    let scale = base.powf(-exponent);
                    vmin = (scale * vmin).floor() / scale;
                    vmax = (scale * vmax).ceil() / scale;
                }
                nonsingular(vmin, vmax, 0.001, 1e-15, true)
            }
            Locator::Multiple { step, offset } => {
                let (vmin, vmax) = if rc().string("axes.autolimit_mode") == "round_numbers" {
                    let edge = EdgeInteger::new(*step, 0.0);
                    let mut a = edge.le(vmin - offset) * step + offset;
                    let mut b = edge.ge(vmax - offset) * step + offset;
                    if a == b {
                        a -= 1.0;
                        b += 1.0;
                    }
                    (a, b)
                } else {
                    (vmin, vmax)
                };
                nonsingular(vmin, vmax, 0.001, 1e-15, true)
            }
            Locator::Log(l) => l.view_limits(vmin, vmax, info),
            Locator::External(e) => e.view_limits(vmin, vmax),
            _ => nonsingular(vmin, vmax, 0.001, 1e-15, true),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarFormatter {
    pub use_offset: bool,
    pub offset: f64,
    pub offset_threshold: f64,
    pub use_mathtext: bool,
    pub use_locale: bool,
    pub usetex: bool,
    pub order_of_magnitude: i64,
    /// decimal places of the `'%1.Nf'` format
    pub sigfigs: usize,
    pub scientific: bool,
    pub powerlimits: (i64, i64),
    pub locs: Vec<f64>,
}

impl ScalarFormatter {
    pub fn new() -> ScalarFormatter {
        let r = rc();
        let lims = r.num_list("axes.formatter.limits");
        ScalarFormatter {
            use_offset: r.boolean("axes.formatter.useoffset"),
            offset: 0.0,
            offset_threshold: r.num("axes.formatter.offset_threshold"),
            use_mathtext: r.boolean("axes.formatter.use_mathtext"),
            use_locale: r.boolean("axes.formatter.use_locale"),
            usetex: r.boolean("text.usetex"),
            order_of_magnitude: 0,
            sigfigs: 0,
            scientific: true,
            powerlimits: (lims.first().copied().unwrap_or(-5.0) as i64, lims.get(1).copied().unwrap_or(6.0) as i64),
            locs: Vec::new(),
        }
    }

    /// `set_useOffset(val)` for a numeric offset.
    pub fn set_offset_value(&mut self, v: f64) {
        self.use_offset = false;
        self.offset = v;
    }

    pub fn set_use_offset(&mut self, b: bool) {
        self.offset = 0.0;
        self.use_offset = b;
    }

    fn format_value(&self, xp: f64) -> String {
        let s = pyfmt::fmt_f(xp, self.sigfigs);
        let s = if self.usetex || self.use_mathtext { format!("$\\mathdefault{{{}}}$", s) } else { s };
        pyfmt::fix_minus(&s)
    }

    /// `__call__(x, pos)`.
    pub fn call(&self, x: f64) -> String {
        if self.locs.is_empty() {
            return String::new();
        }
        let mut xp = (x - self.offset) / 10f64.powf(self.order_of_magnitude as f64);
        if xp.abs() < 1e-8 {
            xp = 0.0;
        }
        self.format_value(xp)
    }

    pub fn set_locs(&mut self, locs: &[f64], info: &AxisInfo) {
        self.locs = locs.to_vec();
        if !self.locs.is_empty() {
            if self.use_offset {
                self.compute_offset(info);
            }
            self.set_order_of_magnitude(info);
            self.set_format(info);
        }
    }

    fn compute_offset(&mut self, info: &AxisInfo) {
        let (vmin, vmax) = sorted2(info.view);
        let locs: Vec<f64> = self.locs.iter().copied().filter(|&l| vmin <= l && l <= vmax).collect();
        if locs.is_empty() {
            self.offset = 0.0;
            return;
        }
        let lmin = locs.iter().copied().fold(f64::INFINITY, f64::min);
        let lmax = locs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if lmin == lmax || (lmin <= 0.0 && 0.0 <= lmax) {
            self.offset = 0.0;
            return;
        }
        let (abs_min, abs_max) = sorted2((lmin.abs(), lmax.abs()));
        let sign = 1f64.copysign(lmin);
        let oom_max = abs_max.log10().ceil();
        let p10 = |o: f64| 10f64.powf(o);
        let mut oom = oom_max;
        for _ in 0..2000 {
            if floordiv(abs_min, p10(oom)) != floordiv(abs_max, p10(oom)) {
                break;
            }
            oom -= 1.0;
        }
        let mut oom = 1.0 + oom;
        if (abs_max - abs_min) / p10(oom) <= 1e-2 {
            let mut o = oom_max;
            for _ in 0..2000 {
                if floordiv(abs_max, p10(o)) - floordiv(abs_min, p10(o)) > 1.0 {
                    break;
                }
                o -= 1.0;
            }
            oom = 1.0 + o;
        }
        let n = self.offset_threshold - 1.0;
        self.offset = if floordiv(abs_max, p10(oom)) >= pyfmt::pow10_int(n as i64) {
            sign * floordiv(abs_max, p10(oom)) * p10(oom)
        } else {
            0.0
        };
    }

    fn set_order_of_magnitude(&mut self, info: &AxisInfo) {
        if !self.scientific {
            self.order_of_magnitude = 0;
            return;
        }
        if self.powerlimits.0 == self.powerlimits.1 && self.powerlimits.0 != 0 {
            self.order_of_magnitude = self.powerlimits.0;
            return;
        }
        let (vmin, vmax) = sorted2(info.view);
        let locs: Vec<f64> = self.locs.iter().copied().filter(|&l| vmin <= l && l <= vmax).map(|l| l.abs()).collect();
        if locs.is_empty() {
            self.order_of_magnitude = 0;
            return;
        }
        let oom = if self.offset != 0.0 {
            (vmax - vmin).log10().floor() as i64
        } else {
            let val = locs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if val == 0.0 {
                0
            } else {
                val.log10().floor() as i64
            }
        };
        self.order_of_magnitude = if oom <= self.powerlimits.0 || oom >= self.powerlimits.1 { oom } else { 0 };
    }

    fn set_format(&mut self, info: &AxisInfo) {
        let few = self.locs.len() < 2;
        let mut src = self.locs.clone();
        if few {
            src.push(info.view.0);
            src.push(info.view.1);
        }
        let div = 10f64.powf(self.order_of_magnitude as f64);
        let mut locs: Vec<f64> = src.iter().map(|l| (l - self.offset) / div).collect();
        let mx = locs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mn = locs.iter().copied().fold(f64::INFINITY, f64::min);
        let mut loc_range = mx - mn;
        if loc_range == 0.0 {
            loc_range = locs.iter().map(|v| v.abs()).fold(f64::NEG_INFINITY, f64::max);
        }
        if loc_range == 0.0 {
            loc_range = 1.0;
        }
        if few {
            locs.truncate(locs.len() - 2);
        }
        let loc_range_oom = loc_range.log10().floor() as i64;
        let mut sigfigs: i64 = (3 - loc_range_oom).max(0);
        let thresh = 1e-3 * pyfmt::pow10_int(loc_range_oom);
        while sigfigs >= 0 {
            let m = locs.iter().map(|&l| (l - np_round(l, sigfigs as i32)).abs()).fold(f64::NEG_INFINITY, f64::max);
            if m < thresh {
                sigfigs -= 1;
            } else {
                break;
            }
        }
        sigfigs += 1;
        self.sigfigs = sigfigs as usize;
    }

    /// `format_data(value)`.
    pub fn format_data(&self, value: f64) -> String {
        let e = value.abs().log10().floor() as i64;
        let s = pyfmt::py_round_ndigits(value / pyfmt::pow10_int(e), 10);
        let significand = if pyfmt::divmod(s, 1.0).1 == 0.0 { pyfmt::fmt_d(s) } else { pyfmt::fmt_g(s, 10, false) };
        let significand = pyfmt::fix_minus(&significand);
        if e == 0 {
            return significand;
        }
        let exponent = pyfmt::fix_minus(&format!("{}", e));
        if self.use_mathtext || self.usetex {
            let exponent = format!("10^{{{}}}", exponent);
            if s == 1.0 {
                exponent
            } else {
                format!("{} \\times {}", significand, exponent)
            }
        } else {
            format!("{}e{}", significand, exponent)
        }
    }

    pub fn get_offset(&self) -> String {
        if self.locs.is_empty() {
            return String::new();
        }
        if self.order_of_magnitude != 0 || self.offset != 0.0 {
            let mut offset_str = String::new();
            let mut sci = String::new();
            if self.offset != 0.0 {
                offset_str = self.format_data(self.offset);
                if self.offset > 0.0 {
                    offset_str = format!("+{}", offset_str);
                }
            }
            if self.order_of_magnitude != 0 {
                if self.usetex || self.use_mathtext {
                    sci = self.format_data(pyfmt::pow10_int(self.order_of_magnitude));
                } else {
                    sci = format!("1e{}", self.order_of_magnitude);
                }
            }
            let s = if self.use_mathtext || self.usetex {
                if !sci.is_empty() {
                    sci = format!("\\times\\mathdefault{{{}}}", sci);
                }
                format!("${}\\mathdefault{{{}}}$", sci, offset_str)
            } else {
                format!("{}{}", sci, offset_str)
            };
            return pyfmt::fix_minus(&s);
        }
        String::new()
    }
}

impl Default for ScalarFormatter {
    fn default() -> Self {
        ScalarFormatter::new()
    }
}

#[derive(Clone, Debug)]
pub enum Formatter {
    Scalar(ScalarFormatter),
    Null,
    Fixed { seq: Vec<String>, offset_string: String },
    Log(super::logticker::LogFormatter),
    External(Arc<dyn ExternalFormatter>),
}

impl Formatter {
    /// `format_ticks(values)`.
    pub fn format_ticks(&mut self, values: &[f64], info: &AxisInfo) -> Vec<String> {
        match self {
            Formatter::Scalar(s) => {
                s.set_locs(values, info);
                values.iter().map(|&v| s.call(v)).collect()
            }
            Formatter::Null => vec![String::new(); values.len()],
            Formatter::Fixed { seq, .. } => (0..values.len()).map(|i| seq.get(i).cloned().unwrap_or_default()).collect(),
            Formatter::Log(l) => l.format_ticks(values, info),
            Formatter::External(e) => e.format_ticks(values, info),
        }
    }

    pub fn get_offset(&self) -> String {
        match self {
            Formatter::Scalar(s) => s.get_offset(),
            Formatter::Fixed { offset_string, .. } => offset_string.clone(),
            Formatter::External(e) => e.get_offset(),
            _ => String::new(),
        }
    }
}
