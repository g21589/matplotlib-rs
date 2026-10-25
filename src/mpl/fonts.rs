//! Port of `matplotlib.font_manager`: font properties, the font list and the
//! `findfont` scoring, plus a process-wide cache of parsed font files.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use super::rc::rc;
use crate::font::FontData;

pub const WEIGHT_DICT: &[(&str, i32)] = &[
    ("ultralight", 100),
    ("light", 200),
    ("normal", 400),
    ("regular", 400),
    ("book", 400),
    ("medium", 500),
    ("roman", 500),
    ("semibold", 600),
    ("demibold", 600),
    ("demi", 600),
    ("bold", 700),
    ("heavy", 800),
    ("extra bold", 800),
    ("black", 900),
];

pub const STRETCH_DICT: &[(&str, i32)] = &[
    ("ultra-condensed", 100),
    ("extra-condensed", 200),
    ("condensed", 300),
    ("semi-condensed", 400),
    ("normal", 500),
    ("semi-expanded", 600),
    ("semi-extended", 600),
    ("expanded", 700),
    ("extended", 700),
    ("extra-expanded", 800),
    ("extra-extended", 800),
    ("ultra-expanded", 900),
    ("ultra-extended", 900),
];

pub const FONT_SCALINGS: &[(&str, f64)] = &[
    ("xx-small", 0.579),
    ("x-small", 0.694),
    ("small", 0.833),
    ("medium", 1.0),
    ("large", 1.2),
    ("x-large", 1.44),
    ("xx-large", 1.728),
    ("larger", 1.2),
    ("smaller", 0.833),
];

const WEIGHT_REGEXES: &[(&str, i32)] = &[
    ("thin", 100),
    ("extralight", 200),
    ("ultralight", 200),
    ("demilight", 350),
    ("semilight", 350),
    ("light", 300),
    ("book", 380),
    ("regular", 400),
    ("normal", 400),
    ("medium", 500),
    ("demibold", 600),
    ("demi", 600),
    ("semibold", 600),
    ("extrabold", 800),
    ("superbold", 800),
    ("ultrabold", 800),
    ("bold", 700),
    ("ultrablack", 1000),
    ("superblack", 1000),
    ("extrablack", 1000),
    ("\\bultra", 1000),
    ("black", 900),
    ("heavy", 900),
];

const FAMILY_ALIASES: &[&str] = &["serif", "sans-serif", "sans serif", "cursive", "fantasy", "monospace", "sans"];

pub fn font_scaling(name: &str) -> Option<f64> {
    FONT_SCALINGS.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
}

pub fn weight_value(name: &str) -> Option<i32> {
    WEIGHT_DICT.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
}

pub fn stretch_value(name: &str) -> Option<i32> {
    STRETCH_DICT.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
}

/// A weight or stretch: either a name from the dicts above or a number.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NumOrName {
    Num(i32),
    Name(String),
}

impl NumOrName {
    pub fn weight(&self) -> i32 {
        match self {
            NumOrName::Num(v) => *v,
            NumOrName::Name(n) => weight_value(n).unwrap_or(400),
        }
    }
}

/// `matplotlib.font_manager.FontProperties`.
#[derive(Clone, Debug, PartialEq)]
pub struct FontProperties {
    pub family: Vec<String>,
    pub style: String,
    pub variant: String,
    pub weight: NumOrName,
    pub stretch: NumOrName,
    pub size: f64,
    pub file: Option<String>,
    pub math_fontfamily: String,
}

impl Eq for FontProperties {}

impl std::hash::Hash for FontProperties {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.family.hash(state);
        self.style.hash(state);
        self.variant.hash(state);
        self.weight.hash(state);
        self.stretch.hash(state);
        self.size.to_bits().hash(state);
        self.file.hash(state);
        self.math_fontfamily.hash(state);
    }
}

pub fn parse_weight(s: &str) -> NumOrName {
    if weight_value(s).is_some() {
        NumOrName::Name(s.to_string())
    } else if let Ok(v) = s.trim().parse::<i32>() {
        NumOrName::Num(v)
    } else {
        NumOrName::Name("normal".into())
    }
}

pub fn parse_stretch(s: &str) -> NumOrName {
    if stretch_value(s).is_some() {
        NumOrName::Name(s.to_string())
    } else if let Ok(v) = s.trim().parse::<i32>() {
        NumOrName::Num(v)
    } else {
        NumOrName::Name("normal".into())
    }
}

/// Resolve a font size (`FontProperties.set_size`).
pub fn resolve_size(size: Option<f64>, named: Option<&str>) -> f64 {
    let r = rc();
    let mut s = match (size, named) {
        (Some(v), _) => v,
        (None, Some(n)) => font_scaling(n).unwrap_or(1.0) * r.num("font.size"),
        (None, None) => r.num("font.size"),
    };
    if s < 1.0 {
        s = 1.0;
    }
    s
}

impl FontProperties {
    /// `FontProperties()` with every field from rcParams.
    pub fn from_rc() -> FontProperties {
        let r = rc();
        FontProperties {
            family: r.str_list("font.family"),
            style: r.string("font.style"),
            variant: r.string("font.variant"),
            weight: parse_weight(&r.string("font.weight")),
            stretch: parse_stretch(&r.string("font.stretch")),
            size: resolve_size(None, None),
            file: None,
            math_fontfamily: r.string("mathtext.fontset"),
        }
    }

    pub fn with_size(mut self, size: f64) -> FontProperties {
        self.size = if size < 1.0 { 1.0 } else { size };
        self
    }

    pub fn size_in_points(&self) -> f64 {
        self.size
    }
}

/// `FontEntry`.
#[derive(Clone, Debug)]
pub struct FontEntry {
    pub fname: String,
    pub index: u32,
    pub name: String,
    pub style: String,
    pub variant: String,
    pub weight: NumOrName,
    pub stretch: String,
    pub size: String,
}

fn decode_latin1(b: Option<&[u8]>) -> String {
    b.map(|b| b.iter().map(|&c| c as char).collect()).unwrap_or_default()
}

fn decode_utf16be(b: Option<&[u8]>) -> String {
    b.map(|b| {
        let u: Vec<u16> = b.chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
        String::from_utf16_lossy(&u)
    })
    .unwrap_or_default()
}

fn regex_search(pat: &str, s: &str) -> bool {
    let s = s.to_lowercase();
    if let Some(word) = pat.strip_prefix("\\b") {
        let bytes = s.as_bytes();
        let mut start = 0;
        while let Some(pos) = s[start..].find(word) {
            let p = start + pos;
            let before_ok = p == 0 || !(bytes[p - 1].is_ascii_alphanumeric() || bytes[p - 1] == b'_');
            if before_ok {
                return true;
            }
            start = p + 1;
        }
        false
    } else {
        s.contains(pat)
    }
}

fn regex_fullmatch(pat: &str, s: &str) -> bool {
    let s = s.to_lowercase();
    match pat.strip_prefix("\\b") {
        Some(w) => s == w,
        None => s == pat,
    }
}

/// `ttfFontProperty` plus `_get_font_alt_names`.
pub fn ttf_font_properties(fname: &str, index: u32, data: &FontData) -> Vec<FontEntry> {
    let sf = &data.sfnt;
    let name = sf.family_name.clone();
    let mac = |id: u16| sf.name_entry(1, 0, 0, id);
    let ms = |id: u16| sf.name_entry(3, 1, 0x409, id);
    let sfnt2 = {
        let a = decode_latin1(mac(2)).to_lowercase();
        if a.is_empty() {
            decode_utf16be(ms(2)).to_lowercase()
        } else {
            a
        }
    };
    let sfnt4 = {
        let a = decode_latin1(mac(4)).to_lowercase();
        if a.is_empty() {
            decode_utf16be(ms(4)).to_lowercase()
        } else {
            a
        }
    };
    let has_os2 = sf.os2.map(|o| o.version != 0xffff).unwrap_or(false);
    let fs_sel = sf.os2.map(|o| o.fs_selection).unwrap_or(0);
    let italic_flag = (has_os2 && (fs_sel & 512 != 0 || fs_sel & 1 != 0)) || (!has_os2 && sf.mac_style & 2 != 0);
    let bold_flag = (has_os2 && fs_sel & 32 != 0) || (!has_os2 && sf.mac_style & 1 != 0);
    let style = if sfnt4.contains("oblique") {
        "oblique"
    } else if sfnt4.contains("italic") {
        "italic"
    } else if sfnt2.contains("regular") {
        "normal"
    } else if italic_flag {
        "italic"
    } else {
        "normal"
    };
    let variant = if ["capitals", "small-caps"].contains(&name.to_lowercase().as_str()) { "small-caps" } else { "normal" };
    let mut styles: Vec<String> = vec![
        decode_latin1(mac(22)),
        decode_latin1(mac(16)),
        decode_latin1(mac(2)),
        decode_utf16be(ms(22)),
        decode_utf16be(ms(16)),
        decode_utf16be(ms(2)),
    ]
    .into_iter()
    .filter(|s| !s.is_empty())
    .collect();
    if styles.is_empty() {
        styles.push(sf.style_name.clone());
    }
    let weight = (|| {
        if let Some(o) = sf.os2 {
            if o.version != 0xffff {
                return o.us_weight_class as i32;
            }
        }
        for st in &styles {
            let st = st.replace(' ', "");
            for (re, w) in WEIGHT_REGEXES {
                if regex_search(re, &st) {
                    return *w;
                }
            }
        }
        if bold_flag {
            return 700;
        }
        500
    })();
    let _ = regex_fullmatch;
    let stretch = if ["narrow", "condensed", "cond"].iter().any(|w| sfnt4.contains(w)) {
        "condensed"
    } else if sfnt4.contains("demi cond") {
        "semi-condensed"
    } else if ["wide", "expanded", "extended"].iter().any(|w| sfnt4.contains(w)) {
        "expanded"
    } else {
        "normal"
    };
    let prop = FontEntry {
        fname: fname.to_string(),
        index,
        name: name.clone(),
        style: style.into(),
        variant: variant.into(),
        weight: NumOrName::Num(weight),
        stretch: stretch.into(),
        size: "scalable".into(),
    };
    let mut out = vec![prop.clone()];
    // _get_font_alt_names
    let mut seen = vec![name];
    let weight_from_subfam = |subfam: &str| {
        let subfam = subfam.replace(' ', "");
        for (re, w) in WEIGHT_REGEXES {
            if regex_search(re, &subfam) {
                return *w;
            }
        }
        400
    };
    for (fam_id, subfam_id) in [(1u16, 2u16), (16, 17), (21, 22)] {
        for (n, sub) in [
            (decode_latin1(mac(fam_id)), decode_latin1(mac(subfam_id))),
            (decode_utf16be(ms(fam_id)), decode_utf16be(ms(subfam_id))),
        ] {
            let n = n.trim().to_string();
            if n.is_empty() || seen.contains(&n) {
                continue;
            }
            seen.push(n.clone());
            let mut alt = prop.clone();
            alt.name = n;
            alt.weight = NumOrName::Num(weight_from_subfam(sub.trim()));
            out.push(alt);
        }
    }
    out
}

/// The font list and `findfont` cache.
pub struct FontManager {
    pub ttflist: Vec<FontEntry>,
    pub default_family: String,
    cache: HashMap<(FontProperties, Vec<Vec<String>>), Option<(String, u32)>>,
}

pub static FONT_MANAGER: LazyLock<RwLock<FontManager>> = LazyLock::new(|| {
    RwLock::new(FontManager { ttflist: Vec::new(), default_family: "DejaVu Sans".into(), cache: HashMap::new() })
});

pub fn set_font_list(list: Vec<FontEntry>) {
    let mut fm = FONT_MANAGER.write().unwrap();
    fm.ttflist = list;
    fm.cache.clear();
}

fn expand_aliases(family: &str) -> Vec<String> {
    let f = if family == "sans" || family == "sans serif" { "sans-serif" } else { family };
    rc().str_list(&format!("font.{f}"))
}

fn score_family(families: &[String], family2: &str) -> f64 {
    if families.is_empty() {
        return 1.0;
    }
    let family2 = family2.to_lowercase();
    let step = 1.0 / families.len() as f64;
    for (i, f1) in families.iter().enumerate() {
        let f1 = f1.to_lowercase();
        if FAMILY_ALIASES.contains(&f1.as_str()) {
            let options: Vec<String> = expand_aliases(&f1).iter().map(|s| s.to_lowercase()).collect();
            if let Some(idx) = options.iter().position(|o| *o == family2) {
                return (i as f64 + (idx as f64 / options.len() as f64)) * step;
            }
        } else if f1 == family2 {
            return i as f64 * step;
        }
    }
    1.0
}

fn score_style(s1: &str, s2: &str) -> f64 {
    if s1 == s2 {
        0.0
    } else if (s1 == "italic" || s1 == "oblique") && (s2 == "italic" || s2 == "oblique") {
        0.1
    } else {
        1.0
    }
}

fn score_weight(w1: &NumOrName, w2: &NumOrName) -> f64 {
    if let (NumOrName::Name(a), NumOrName::Name(b)) = (w1, w2) {
        if a == b {
            return 0.0;
        }
    }
    let a = w1.weight();
    let b = w2.weight();
    0.95 * ((a - b).abs() as f64 / 1000.0) + 0.05
}

fn stretch_num(s: &NumOrName) -> i32 {
    match s {
        NumOrName::Num(v) => *v,
        NumOrName::Name(n) => n.parse::<i32>().ok().or_else(|| stretch_value(n)).unwrap_or(500),
    }
}

fn score_stretch(s1: &NumOrName, s2: &str) -> f64 {
    let a = stretch_num(s1);
    let b = s2.parse::<i32>().ok().or_else(|| stretch_value(s2)).unwrap_or(500);
    (a - b).abs() as f64 / 1000.0
}

fn score_size(size1: f64, size2: &str) -> f64 {
    if size2 == "scalable" {
        return 0.0;
    }
    match size2.parse::<f64>() {
        Ok(v) => (size1 - v).abs() / 72.0,
        Err(_) => 1.0,
    }
}

impl FontManager {
    fn rc_key() -> Vec<Vec<String>> {
        let r = rc();
        ["font.serif", "font.sans-serif", "font.cursive", "font.fantasy", "font.monospace", "font.family"]
            .iter()
            .map(|k| r.str_list(k))
            .collect()
    }

    /// `findfont` (with or without fallback to the default family).
    pub fn findfont(&mut self, prop: &FontProperties, fallback_to_default: bool) -> Option<(String, u32)> {
        if let Some(f) = &prop.file {
            return Some((f.clone(), 0));
        }
        let key = (prop.clone(), Self::rc_key());
        let cached = if let Some(c) = self.cache.get(&key) {
            c.clone()
        } else {
            let mut best_score = 1e64;
            let mut best: Option<&FontEntry> = None;
            for font in &self.ttflist {
                let score = score_family(&prop.family, &font.name) * 10.0
                    + score_style(&prop.style, &font.style)
                    + if prop.variant == font.variant { 0.0 } else { 1.0 }
                    + score_weight(&prop.weight, &font.weight)
                    + score_stretch(&prop.stretch, &font.stretch)
                    + score_size(prop.size, &font.size);
                if score < best_score {
                    best_score = score;
                    best = Some(font);
                }
                if score == 0.0 {
                    break;
                }
            }
            let r = match best {
                Some(b) if best_score < 10.0 => Some((b.fname.clone(), b.index)),
                _ => None,
            };
            self.cache.insert(key, r.clone());
            r
        };
        if cached.is_none() && fallback_to_default {
            let mut p = prop.clone();
            p.family = vec![self.default_family.clone()];
            return self.findfont(&p, false);
        }
        cached
    }

    /// `_find_fonts_by_props`.
    pub fn find_fonts_by_props(&mut self, prop: &FontProperties) -> Vec<(String, u32)> {
        let mut out = Vec::new();
        for fam in &prop.family {
            let mut p = prop.clone();
            p.family = vec![fam.clone()];
            if let Some(f) = self.findfont(&p, false) {
                out.push(f);
            }
        }
        if out.is_empty() {
            let mut p = prop.clone();
            p.family = vec![self.default_family.clone()];
            if let Some(f) = self.findfont(&p, true) {
                out.push(f);
            }
        }
        out
    }
}

static FONT_FILES: LazyLock<Mutex<HashMap<(String, u32), Arc<FontData>>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Load (and cache) a font file.
pub fn load_font_data(path: &str, index: u32) -> Option<Arc<FontData>> {
    let key = (path.to_string(), index);
    if let Some(d) = FONT_FILES.lock().unwrap().get(&key) {
        return Some(d.clone());
    }
    let bytes = std::fs::read(path).ok()?;
    let data = Arc::new(FontData::from_bytes(bytes, index).ok()?);
    FONT_FILES.lock().unwrap().insert(key, data.clone());
    Some(data)
}

/// Resolved font for a `FontProperties` (primary plus fallbacks).
#[derive(Clone)]
pub struct ResolvedFont {
    pub fonts: Vec<Arc<FontData>>,
    pub paths: Vec<(String, u32)>,
    /// index of matplotlib's Last Resort font in `fonts` (used with its
    /// first charmap; glyphs from it are reported as missing)
    pub last_resort: Option<usize>,
}

impl ResolvedFont {
    /// FreeType-like font objects at `size` points and `dpi`, with the Last
    /// Resort font on its first charmap (as `font_manager._get_font` does).
    pub fn ft_fonts(&self, size: f64, dpi: f64) -> Vec<crate::font::FtFont> {
        self.fonts
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let mut f = crate::font::FtFont::new(d.clone());
                if Some(i) == self.last_resort && !d.cmaps.is_empty() {
                    f.charmap = Some(0);
                }
                f.set_size(size, dpi);
                f
            })
            .collect()
    }
}

static RESOLVED: LazyLock<Mutex<HashMap<(FontProperties, u64), Arc<ResolvedFont>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// `get_font(fontManager._find_fonts_by_props(prop))`.
pub fn resolve_font(prop: &FontProperties) -> Option<Arc<ResolvedFont>> {
    let key = (prop.clone(), super::rc::rc_version());
    if let Some(r) = RESOLVED.lock().unwrap().get(&key) {
        return Some(r.clone());
    }
    let paths = FONT_MANAGER.write().unwrap().find_fonts_by_props(prop);
    let mut fonts = Vec::new();
    let mut ok_paths = Vec::new();
    for (p, i) in paths {
        if let Some(d) = load_font_data(&p, i) {
            fonts.push(d);
            ok_paths.push((p, i));
        }
    }
    if fonts.is_empty() {
        return None;
    }
    // font_manager._get_font: the Last Resort font is always available
    let lr_path = FONT_MANAGER
        .read()
        .unwrap()
        .ttflist
        .iter()
        .find(|e| e.fname.ends_with("/LastResortHE-Regular.ttf") && e.fname.contains("mpl-data"))
        .map(|e| e.fname.clone());
    let mut last_resort = None;
    if let Some(lr) = lr_path {
        if let Some(i) = ok_paths.iter().position(|(p, i)| *p == lr && *i == 0) {
            last_resort = Some(i);
        } else if super::rc::rc().boolean("font.enable_last_resort") {
            if let Some(d) = load_font_data(&lr, 0) {
                fonts.push(d);
                ok_paths.push((lr, 0));
                last_resort = Some(fonts.len() - 1);
            }
        }
    }
    let r = Arc::new(ResolvedFont { fonts, paths: ok_paths, last_resort });
    let mut cache = RESOLVED.lock().unwrap();
    if cache.len() > 4096 {
        cache.clear();
    }
    cache.insert(key, r.clone());
    Some(r)
}
