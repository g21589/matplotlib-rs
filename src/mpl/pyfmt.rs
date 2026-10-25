//! Python-compatible number formatting and arithmetic helpers
//! (`'%f' % x`, `'%e'`, `'%g'`, `repr(float)`, float `//` and `divmod`,
//! `round(x, n)`, `np.round`).

/// `'%.{prec}f' % x`.
pub fn fmt_f(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    format!("{:.*}", prec, x)
}

/// Split Rust's `{:e}` output into mantissa and exponent.
fn rust_exp(x: f64, prec: usize) -> (String, i32) {
    let s = format!("{:.*e}", prec, x);
    let (m, e) = s.split_once('e').unwrap();
    (m.to_string(), e.parse().unwrap())
}

fn py_exp_suffix(e: i32) -> String {
    if e < 0 {
        format!("e-{:02}", -e)
    } else {
        format!("e+{:02}", e)
    }
}

/// `'%.{prec}e' % x`.
pub fn fmt_e(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let (m, e) = rust_exp(x, prec);
    format!("{}{}", m, py_exp_suffix(e))
}

fn strip_zeros(s: &str) -> String {
    if s.contains('.') {
        let t = s.trim_end_matches('0');
        t.trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

/// `'%.{prec}g' % x` (`alt` is the `#` flag).
pub fn fmt_g(x: f64, prec: usize, alt: bool) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let p = if prec == 0 { 1 } else { prec };
    if x == 0.0 {
        let base = if x.is_sign_negative() { "-0" } else { "0" };
        if alt {
            return format!("{}.{}", base, "0".repeat(p - 1)).trim_end_matches('.').to_string()
                + if p == 1 { "." } else { "" };
        }
        return base.to_string();
    }
    let (_, exp) = rust_exp(x, p - 1);
    if exp >= -4 && (exp as i64) < p as i64 {
        let digits = (p as i64 - 1 - exp as i64).max(0) as usize;
        let s = format!("{:.*}", digits, x);
        if alt {
            if !s.contains('.') {
                return s + ".";
            }
            s
        } else {
            strip_zeros(&s)
        }
    } else {
        let (m, e) = rust_exp(x, p - 1);
        let m = if alt {
            if m.contains('.') {
                m
            } else {
                m + "."
            }
        } else {
            strip_zeros(&m)
        };
        format!("{}{}", m, py_exp_suffix(e))
    }
}

/// `'%d' % x` for a float (truncation toward zero).
pub fn fmt_d(x: f64) -> String {
    format!("{}", x.trunc() as i64)
}

/// `repr(x)` / `str(x)` for a Python float.
pub fn repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // shortest round-trip digits
    let s = format!("{:e}", x);
    let (m, e) = s.split_once('e').unwrap();
    let e: i32 = e.parse().unwrap();
    let neg = m.starts_with('-');
    let digits: String = m.chars().filter(|c| c.is_ascii_digit()).collect();
    let n = digits.len() as i32;
    let sign = if neg { "-" } else { "" };
    if !(-4..16).contains(&e) {
        let mant = if n == 1 { digits.clone() } else { format!("{}.{}", &digits[..1], &digits[1..]) };
        format!("{}{}{}", sign, mant, py_exp_suffix(e))
    } else if e >= 0 {
        let int_len = (e + 1) as usize;
        if digits.len() <= int_len {
            format!("{}{}{}.0", sign, digits, "0".repeat(int_len - digits.len()))
        } else {
            format!("{}{}.{}", sign, &digits[..int_len], &digits[int_len..])
        }
    } else {
        format!("{}0.{}{}", sign, "0".repeat((-e - 1) as usize), digits)
    }
}

/// Python float floor division `a // b`.
pub fn floordiv(a: f64, b: f64) -> f64 {
    divmod(a, b).0
}

/// Python float `divmod(a, b)` (also numpy's `npy_divmod`).
pub fn divmod(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        return (a / b, f64::NAN);
    }
    let mut m = a % b;
    let mut div = (a - m) / b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m += b;
            div -= 1.0;
        }
    } else {
        m = 0.0f64.copysign(b);
    }
    let floordiv = if div != 0.0 {
        let mut f = div.floor();
        if div - f > 0.5 {
            f += 1.0;
        }
        f
    } else {
        0.0f64.copysign(a / b)
    };
    (floordiv, m)
}

/// numpy `power_of_ten`.
fn power_of_ten(n: u32) -> f64 {
    const P10: [f64; 9] = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8];
    if n < 9 {
        P10[n as usize]
    } else {
        let mut r = 1e9;
        let mut k = n;
        while k > 9 {
            r *= 10.0;
            k -= 1;
        }
        r
    }
}

/// `np.round(x, decimals)`.
pub fn np_round(x: f64, decimals: i32) -> f64 {
    if decimals >= 0 {
        let f = power_of_ten(decimals as u32);
        (x * f).round_ties_even() / f
    } else {
        let f = power_of_ten((-decimals) as u32);
        (x / f).round_ties_even() * f
    }
}

/// Python `round(x, ndigits)` for floats (correctly rounded).
pub fn py_round_ndigits(x: f64, ndigits: i32) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    if ndigits >= 0 {
        let s = format!("{:.*}", ndigits as usize, x);
        s.parse().unwrap_or(x)
    } else {
        let f = 10f64.powi(-ndigits);
        (x / f).round_ties_even() * f
    }
}

/// Python `10 ** e` for an int exponent (exact for e >= 0 up to 1e22).
pub fn pow10_int(e: i64) -> f64 {
    if (0..=22).contains(&e) {
        let mut v = 1.0;
        for _ in 0..e {
            v *= 10.0;
        }
        v
    } else {
        10f64.powf(e as f64)
    }
}

/// `axes.unicode_minus` handling (`Formatter.fix_minus`).
pub fn fix_minus(s: &str) -> String {
    if super::rc::rc().boolean("axes.unicode_minus") {
        s.replace('-', "\u{2212}")
    } else {
        s.to_string()
    }
}

/// `np.spacing(x)`: distance to the next float away from zero.
pub fn np_spacing(x: f64) -> f64 {
    let a = x.abs();
    if !a.is_finite() {
        return f64::NAN;
    }
    let next = f64::from_bits(a.to_bits() + 1);
    let d = next - a;
    if x < 0.0 {
        -d
    } else {
        d
    }
}

/// `cbook._g_sig_digits(value, delta)`.
pub fn g_sig_digits(value: f64, delta: f64) -> usize {
    if !value.is_finite() {
        return 0;
    }
    let mut delta = delta;
    if delta == 0.0 {
        if value == 0.0 {
            return 3;
        }
        delta = np_spacing(value).abs();
    }
    let a = if value != 0.0 { value.abs().log10().floor() as i64 + 1 } else { 1 };
    let b = delta.log10().floor() as i64;
    (a - b).max(0) as usize
}
