//! Port of `LogLocator` and the `LogFormatter` family.

use super::pyfmt;
use super::rc::rc;
use super::ticker::{nonsingular, AxisInfo, MaxNLocator};

#[derive(Clone, Debug, PartialEq)]
pub enum Subs {
    Auto,
    All,
    List(Vec<f64>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogLocator {
    pub base: f64,
    pub subs: Subs,
    /// None = 'auto'
    pub numticks: Option<i64>,
}

fn log_b(base: f64, x: f64) -> f64 {
    if base == 10.0 {
        x.log10()
    } else if base == 2.0 {
        x.log2()
    } else {
        x.ln() / base.ln()
    }
}

/// `_decade_less_equal`.
pub fn decade_less_equal(x: f64, base: f64) -> f64 {
    if x == 0.0 {
        x
    } else if x < 0.0 {
        -decade_greater_equal(-x, base)
    } else {
        base.powf((x.ln() / base.ln()).floor())
    }
}

/// `_decade_greater_equal`.
pub fn decade_greater_equal(x: f64, base: f64) -> f64 {
    if x == 0.0 {
        x
    } else if x < 0.0 {
        -decade_less_equal(-x, base)
    } else {
        base.powf((x.ln() / base.ln()).ceil())
    }
}

pub fn decade_less(x: f64, base: f64) -> f64 {
    if x < 0.0 {
        return -decade_greater(-x, base);
    }
    let mut less = decade_less_equal(x, base);
    if less == x {
        less /= base;
    }
    less
}

pub fn decade_greater(x: f64, base: f64) -> f64 {
    if x < 0.0 {
        return -decade_less(-x, base);
    }
    let mut g = decade_greater_equal(x, base);
    if g == x {
        g *= base;
    }
    g
}

/// `math.isclose(x, round(x))`.
pub fn is_close_to_int(x: f64) -> bool {
    let r = x.round_ties_even();
    let tol = 1e-9 * super::text::pymax(x.abs(), r.abs());
    (x - r).abs() <= tol || x == r
}

impl LogLocator {
    pub fn new(base: f64, subs: Subs, numticks: Option<i64>) -> LogLocator {
        LogLocator { base, subs, numticks }
    }

    pub fn call(&self, info: &AxisInfo) -> Vec<f64> {
        self.tick_values(info.view.0, info.view.1, info)
    }

    pub fn tick_values(&self, vmin: f64, vmax: f64, info: &AxisInfo) -> Vec<f64> {
        let mut n_request = match self.numticks {
            Some(n) => n,
            None => info.tick_space.clamp(2, 9),
        };
        let b = self.base;
        let mut vmin = vmin;
        let mut vmax = vmax;
        if vmin <= 0.0 {
            vmin = info.minpos;
            if vmin <= 0.0 || !vmin.is_finite() {
                return Vec::new();
            }
        }
        if vmax < vmin {
            std::mem::swap(&mut vmin, &mut vmax);
        }
        let efmin = log_b(b, vmin);
        let efmax = log_b(b, vmax);
        let emin = efmin.ceil() as i64;
        let emax = efmax.floor() as i64;
        let n_avail = emax - emin + 1;
        let subs: Vec<f64> = match &self.subs {
            Subs::Auto | Subs::All => {
                if n_avail >= 10 || b < 3.0 {
                    if self.subs == Subs::Auto {
                        return Vec::new();
                    }
                    vec![1.0]
                } else {
                    let first = if self.subs == Subs::Auto { 2.0 } else { 1.0 };
                    let mut v = Vec::new();
                    let mut s = first;
                    while s < b {
                        v.push(s);
                        s += 1.0;
                    }
                    v
                }
            }
            Subs::List(l) => l.clone(),
        };
        let mut stride = n_avail.div_euclid(n_request + 1) + 1;
        let nr = (n_avail as f64 / stride as f64).ceil() as i64;
        if nr <= n_request {
            n_request = nr;
        }
        let decades: Vec<i64> = if n_request == 0 {
            let d = vec![emin - 1, emax + 1];
            stride = d[1] - d[0];
            d
        } else if n_request == 1 {
            let mid = ((efmin + efmax) / 2.0).round_ties_even() as i64;
            stride = (mid - (emin - 1)).max((emax + 1) - mid);
            vec![mid - stride, mid, mid + stride]
        } else {
            stride = (n_avail - 1).div_euclid(n_request - 1);
            if (stride as f64) < n_avail as f64 / n_request as f64 {
                stride = n_avail.div_euclid(n_request);
            }
            let olo = (n_avail - stride * n_request).max(0);
            let ohi = (n_avail - stride * (n_request - 1)).min(stride);
            let mut offset = (-emin).rem_euclid(stride);
            if !(olo <= offset && offset < ohi) {
                offset = olo;
            }
            let mut d = Vec::new();
            let mut v = emin + offset - stride;
            while v < emax + stride + 1 {
                d.push(v);
                v += stride;
            }
            d
        };
        let is_minor = subs.len() > 1 || (subs.len() == 1 && subs[0] != 1.0);
        let ticklocs: Vec<f64> = if is_minor {
            if stride == 1 || n_avail <= 1 {
                let mut t = Vec::new();
                for decade in (emin - 1)..(emax + 1) {
                    let p = b.powf(decade as f64);
                    for s in &subs {
                        t.push(s * p);
                    }
                }
                t
            } else {
                Vec::new()
            }
        } else {
            decades.iter().map(|&d| b.powf(d as f64)).collect()
        };
        if subs.len() > 1 && stride == 1 {
            let inside = ticklocs.iter().filter(|&&t| vmin <= t && t <= vmax).count() as i64;
            if decades.len() as i64 - 2 + inside <= 1 {
                return MaxNLocator::auto().tick_values(vmin, vmax, info);
            }
        }
        ticklocs
    }

    pub fn view_limits(&self, vmin: f64, vmax: f64, info: &AxisInfo) -> (f64, f64) {
        let (mut vmin, mut vmax) = self.nonsingular(vmin, vmax, info);
        if rc().string("axes.autolimit_mode") == "round_numbers" {
            vmin = decade_less_equal(vmin, self.base);
            vmax = decade_greater_equal(vmax, self.base);
        }
        (vmin, vmax)
    }

    pub fn nonsingular(&self, vmin: f64, vmax: f64, info: &AxisInfo) -> (f64, f64) {
        let (mut vmin, mut vmax) = if vmin > vmax { (vmax, vmin) } else { (vmin, vmax) };
        if !vmin.is_finite() || !vmax.is_finite() || vmax <= 0.0 {
            return (1.0, 10.0);
        }
        let mut minpos = info.minpos;
        if !minpos.is_finite() {
            minpos = 1e-300;
        }
        if vmin <= 0.0 {
            vmin = minpos;
        }
        if vmin == vmax {
            vmin = decade_less(vmin, self.base);
            vmax = decade_greater(vmax, self.base);
        }
        (vmin, vmax)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogFormatterKind {
    Plain,
    Exponent,
    Mathtext,
    SciNotation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogFormatter {
    pub kind: LogFormatterKind,
    pub base: f64,
    pub label_only_base: bool,
    pub minor_thresholds: (f64, f64),
    pub sublabels: Option<Vec<f64>>,
    pub linthresh: Option<f64>,
    view: (f64, f64),
}

impl LogFormatter {
    pub fn new(kind: LogFormatterKind, base: f64, label_only_base: bool) -> LogFormatter {
        LogFormatter {
            kind,
            base,
            label_only_base,
            minor_thresholds: (1.0, 0.4),
            sublabels: None,
            linthresh: None,
            view: (1.0, 10.0),
        }
    }

    pub fn set_locs(&mut self, info: &AxisInfo) {
        self.view = info.view;
        if self.minor_thresholds.0.is_infinite() {
            self.sublabels = None;
            return;
        }
        let linthresh = self.linthresh;
        let (mut vmin, mut vmax) = info.view;
        if vmin > vmax {
            std::mem::swap(&mut vmin, &mut vmax);
        }
        if linthresh.is_none() && vmin <= 0.0 {
            self.sublabels = Some(vec![1.0]);
            return;
        }
        let b = self.base;
        let logb = |x: f64| x.ln() / b.ln();
        let (numticks, numdec) = if let Some(lt) = linthresh {
            let mut numdec = 0.0;
            let mut numticks = 0.0;
            if vmin < -lt {
                let rhs = super::text::pymin(vmax, -lt);
                numticks += logb(rhs.abs()).floor() - next_down(logb(vmin.abs())).floor();
                numdec += logb(vmin / rhs);
            }
            if vmax > lt {
                let lhs = super::text::pymax(vmin, lt);
                numticks += logb(vmax).floor() - next_down(logb(lhs)).floor();
                numdec += logb(vmax / lhs);
            }
            (numticks, numdec)
        } else {
            let lmin = logb(vmin);
            let lmax = logb(vmax);
            (lmax.floor() - next_down(lmin).floor(), (lmax - lmin).abs())
        };
        if numticks > self.minor_thresholds.0 {
            self.sublabels = Some(vec![1.0]);
        } else if numdec > self.minor_thresholds.1 {
            let n = (b as i64) / 2 + 1;
            // np.geomspace(1, b, n) rounded
            let c: Vec<f64> = (0..n)
                .map(|i| {
                    if n == 1 {
                        1.0
                    } else {
                        10f64.powf(b.log10() * i as f64 / (n - 1) as f64).round_ties_even()
                    }
                })
                .collect();
            self.sublabels = Some(c);
        } else {
            let mut v = Vec::new();
            let mut s = 1.0;
            while s < b + 1.0 {
                v.push(s);
                s += 1.0;
            }
            self.sublabels = Some(v);
        }
    }

    fn pprint_val(x: f64, d: f64) -> String {
        if x.abs() < 1e4 && x == x.trunc() {
            return pyfmt::fmt_d(x);
        }
        let s = if d < 1e-2 {
            pyfmt::fmt_e(x, 3)
        } else if d <= 1.0 {
            pyfmt::fmt_f(x, 3)
        } else if d <= 10.0 {
            pyfmt::fmt_f(x, 2)
        } else if d <= 1e5 {
            pyfmt::fmt_f(x, 1)
        } else {
            pyfmt::fmt_e(x, 1)
        };
        if let Some((m, e)) = s.split_once('e') {
            let mantissa = m.trim_end_matches('0').trim_end_matches('.');
            let exponent: i64 = e.parse().unwrap_or(0);
            if exponent != 0 {
                format!("{}e{}", mantissa, exponent)
            } else {
                mantissa.to_string()
            }
        } else {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        }
    }

    pub fn call(&self, x: f64) -> String {
        let b = self.base;
        match self.kind {
            LogFormatterKind::Plain | LogFormatterKind::Exponent => {
                if x == 0.0 {
                    return "0".into();
                }
                let x = x.abs();
                let fx = x.ln() / b.ln();
                let is_dec = is_close_to_int(fx);
                let exponent = if is_dec { fx.round_ties_even() } else { fx.floor() };
                let coeff = b.powf(fx - exponent).round_ties_even();
                if self.label_only_base && !is_dec {
                    return String::new();
                }
                if let Some(sl) = &self.sublabels {
                    if !sl.contains(&coeff) {
                        return String::new();
                    }
                }
                let (vmin, vmax) = nonsingular(self.view.0, self.view.1, 0.05, 1e-15, true);
                let s = if self.kind == LogFormatterKind::Plain {
                    if (1.0..=10000.0).contains(&x) {
                        Self::pprint_val(x, vmax - vmin)
                    } else {
                        pyfmt::fmt_e(x, 0)
                    }
                } else {
                    let fx = x.ln() / b.ln();
                    if (1.0..=10000.0).contains(&fx.abs()) {
                        let fd = (vmax - vmin).ln() / b.ln();
                        Self::pprint_val(fx, fd)
                    } else {
                        pyfmt::fmt_g(fx, 1, false)
                    }
                };
                pyfmt::fix_minus(&s)
            }
            LogFormatterKind::Mathtext | LogFormatterKind::SciNotation => {
                if x == 0.0 {
                    return "$\\mathdefault{0}$".into();
                }
                let sign = if x < 0.0 { "-" } else { "" };
                let x = x.abs();
                let mut fx = x.ln() / b.ln();
                let is_dec = is_close_to_int(fx);
                let exponent = if is_dec { fx.round_ties_even() } else { fx.floor() };
                let coeff = b.powf(fx - exponent).round_ties_even();
                if self.label_only_base && !is_dec {
                    return String::new();
                }
                if let Some(sl) = &self.sublabels {
                    if !sl.contains(&coeff) {
                        return String::new();
                    }
                }
                if is_dec {
                    fx = fx.round_ties_even();
                }
                let base = if pyfmt::divmod(b, 1.0).1 == 0.0 { pyfmt::fmt_d(b) } else { pyfmt::repr(b) };
                let min_exp = rc().num("axes.formatter.min_exponent");
                if fx.abs() < min_exp {
                    format!("$\\mathdefault{{{}{}}}$", sign, pyfmt::fmt_g(x, 6, false))
                } else if !is_dec {
                    if self.kind == LogFormatterKind::SciNotation {
                        let bf: f64 = b;
                        let e = fx.floor();
                        let mut c = bf.powf(fx - e);
                        if is_close_to_int(c) {
                            c = c.round_ties_even();
                        }
                        format!("$\\mathdefault{{{}{}\\times{}^{{{}}}}}$", sign, pyfmt::fmt_g(c, 6, false), base, e as i64)
                    } else {
                        format!("$\\mathdefault{{{}{}^{{{}}}}}$", sign, base, pyfmt::fmt_f(fx, 2))
                    }
                } else {
                    format!("$\\mathdefault{{{}{}^{{{}}}}}$", sign, base, fx as i64)
                }
            }
        }
    }

    pub fn format_ticks(&mut self, values: &[f64], info: &AxisInfo) -> Vec<String> {
        self.set_locs(info);
        values.iter().map(|&v| self.call(v)).collect()
    }
}

fn next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}
