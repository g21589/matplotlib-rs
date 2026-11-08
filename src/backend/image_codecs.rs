//! Image file formats besides PNG (`png_encode`), for `imread`, `imsave` and
//! `savefig` without Pillow: JPEG (jpeg-encoder, zune-jpeg), TIFF (tiff), GIF
//! (gif, with a median cut palette for images of more than 256 colors), WebP (the libwebp
//! that Pillow also uses, built from source by libwebp-sys) and AVIF
//! (writing only, ravif).
//!
//! The options mirror the defaults Pillow saves with: JPEG quality 75 with
//! 4:2:0 chroma subsampling, uncompressed TIFF, lossy WebP at quality 80 and
//! method 4, AVIF quality 75 and speed 6.

use std::io::Cursor;

/// Decoded samples, `channels` per pixel, row after row.
#[derive(Debug)]
pub enum Samples {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

/// A decoded image.
#[derive(Debug)]
pub struct Decoded {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub samples: Samples,
}

impl Decoded {
    fn u8(width: usize, height: usize, channels: usize, data: Vec<u8>) -> Result<Self, String> {
        if data.len() != width * height * channels {
            return Err(format!("decoded {} bytes for a {width}x{height}x{channels} image", data.len()));
        }
        Ok(Decoded { width, height, channels, samples: Samples::U8(data) })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Png,
    Jpeg,
    Tiff,
    Gif,
    Webp,
    Avif,
    Bmp,
}

impl Format {
    pub fn name(self) -> &'static str {
        match self {
            Format::Png => "png",
            Format::Jpeg => "jpeg",
            Format::Tiff => "tiff",
            Format::Gif => "gif",
            Format::Webp => "webp",
            Format::Avif => "avif",
            Format::Bmp => "bmp",
        }
    }
}

/// The format of a file, from its first bytes.
pub fn sniff(data: &[u8]) -> Option<Format> {
    let starts = |p: &[u8]| data.starts_with(p);
    if starts(b"\x89PNG\r\n\x1a\n") {
        Some(Format::Png)
    } else if starts(b"\xff\xd8\xff") {
        Some(Format::Jpeg)
    } else if starts(b"II*\0") || starts(b"MM\0*") || starts(b"II+\0") || starts(b"MM\0+") {
        Some(Format::Tiff)
    } else if starts(b"GIF87a") || starts(b"GIF89a") {
        Some(Format::Gif)
    } else if data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        Some(Format::Webp)
    } else if data.len() >= 12 && &data[4..8] == b"ftyp" && matches!(&data[8..12], b"avif" | b"avis") {
        Some(Format::Avif)
    } else if starts(b"BM") {
        Some(Format::Bmp)
    } else {
        None
    }
}

/// Decode a JPEG, TIFF, GIF or WebP file (PNG files are read by
/// `matplotlib_rs._png`).
pub fn decode(data: &[u8]) -> Result<(Format, Decoded), String> {
    let format = sniff(data).ok_or("cannot identify the image file format")?;
    let image = match format {
        Format::Jpeg => decode_jpeg(data)?,
        Format::Tiff => decode_tiff(data)?,
        Format::Gif => decode_gif(data)?,
        Format::Webp => decode_webp(data)?,
        _ => return Err(format!("cannot read {} files", format.name().to_uppercase())),
    };
    Ok((format, image))
}

// ---------------------------------------------------------------------------
// JPEG

/// Options of `encode_jpeg`, with Pillow's names.
#[derive(Clone, Debug)]
pub struct JpegOptions {
    /// 1 to 100 (Pillow's default: 75).
    pub quality: u8,
    /// Chroma subsampling: 0 is 4:4:4, 1 is 4:2:2, 2 is 4:2:0 (Pillow's default).
    pub subsampling: u8,
    /// Optimized Huffman tables.
    pub optimize: bool,
    pub progressive: bool,
    /// Pixel density in dots per inch for the JFIF header.
    pub dpi: Option<(u16, u16)>,
    pub icc_profile: Vec<u8>,
    pub exif: Vec<u8>,
}

impl Default for JpegOptions {
    fn default() -> Self {
        JpegOptions { quality: 75, subsampling: 2, optimize: false, progressive: false, dpi: None, icc_profile: vec![], exif: vec![] }
    }
}

/// Pixels per strip of a JPEG encoded on several threads.
const JPEG_STRIP_PIXELS: usize = 256 * 1024;

/// The JPEG file of gray (`channels` 1), RGB (3) or RGBA (4: composited
/// over the opaque `background`, as Pillow's ``paste`` with the image as
/// mask) pixels.
///
/// Large baseline images are encoded in strips of whole MCU rows on
/// `threads` threads and joined with restart markers: each strip is the
/// restart interval it would be in one encoding, so the pixels are those of
/// a single-threaded encoding, and the strips depend only on the image size.
pub fn encode_jpeg(
    pixels: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    background: [u8; 3],
    o: &JpegOptions,
    threads: usize,
) -> Result<Vec<u8>, String> {
    if !matches!(channels, 1 | 3 | 4) {
        return Err(format!("cannot write {channels}-channel JPEG images"));
    }
    check_size(pixels, width, height, channels, 65535)?;
    let gray = channels == 1;
    // MCU size: 8x8 for gray and 4:4:4, 16x8 for 4:2:2, 16x16 for 4:2:0.
    let (mcu_w, mcu_h) = match (gray, o.subsampling) {
        (true, _) | (false, 0) => (8, 8),
        (false, 1) => (16, 8),
        _ => (16, 16),
    };
    let mcus_per_row = width.div_ceil(mcu_w);
    let mcu_rows = (JPEG_STRIP_PIXELS / width / mcu_h).clamp(1, 65535 / mcus_per_row);
    let strip_rows = mcu_rows * mcu_h;
    let n = height.div_ceil(strip_rows);
    let stride = width * channels;
    // A strip's pixels as gray or RGB.
    let rows = |r0: usize, r1: usize| -> std::borrow::Cow<'_, [u8]> {
        let src = &pixels[r0 * stride..r1 * stride];
        if channels != 4 {
            return src.into();
        }
        src.chunks_exact(4).flat_map(|p| composite(p, background)).collect::<Vec<u8>>().into()
    };
    if n < 2 || o.optimize || o.progressive || threads < 2 {
        return encode_jpeg_plain(&rows(0, height), width, height, gray, o, true);
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let parts: std::sync::Mutex<Vec<Option<Result<Vec<u8>, String>>>> = std::sync::Mutex::new((0..n).map(|_| None).collect());
    let work = || loop {
        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if i >= n {
            break;
        }
        let (r0, r1) = (i * strip_rows, ((i + 1) * strip_rows).min(height));
        let part = encode_jpeg_plain(&rows(r0, r1), width, r1 - r0, gray, o, i == 0);
        parts.lock().unwrap()[i] = Some(part);
    };
    std::thread::scope(|s| {
        for _ in 1..threads.min(n) {
            s.spawn(work);
        }
        work();
    });
    let parts: Vec<Vec<u8>> = parts
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|p| p.unwrap_or_else(|| Err("a JPEG strip was not encoded".into())))
        .collect::<Result<_, _>>()?;
    join_jpeg_strips(&parts, height, (mcus_per_row * mcu_rows) as u16)
}

/// The offsets of the start of a JPEG file's SOF0 segment and of its SOS
/// segment, and the start and end of its entropy-coded data.
fn jpeg_layout(data: &[u8]) -> Result<(usize, usize, usize, usize), String> {
    let bad = || "cannot join JPEG strips: unexpected encoder output".to_string();
    let (mut pos, mut sof) = (2, None);
    while pos + 4 <= data.len() && data[pos] == 0xff {
        let marker = data[pos + 1];
        let len = usize::from(u16::from_be_bytes([data[pos + 2], data[pos + 3]]));
        match marker {
            0xc0 => sof = Some(pos),
            0xda => {
                let end = data.len().checked_sub(2).filter(|&e| data[e..] == [0xff, 0xd9]).ok_or_else(bad)?;
                return Ok((sof.ok_or_else(bad)?, pos, pos + 2 + len, end));
            }
            _ => {}
        }
        pos += 2 + len;
    }
    Err(bad())
}

/// One baseline JPEG of `parts`, the encodings of consecutive strips of a
/// `height`-row image of `interval` MCUs each (but the last).
fn join_jpeg_strips(parts: &[Vec<u8>], height: usize, interval: u16) -> Result<Vec<u8>, String> {
    let (sof, sos, start, end) = jpeg_layout(&parts[0])?;
    let first = &parts[0];
    let mut out = Vec::with_capacity(parts.iter().map(|p| p.len()).sum::<usize>() + 16);
    out.extend_from_slice(&first[..sos]);
    // SOF0: FFC0, length, precision, then the height.
    out[sof + 5..sof + 7].copy_from_slice(&(height as u16).to_be_bytes());
    out.extend_from_slice(&[0xff, 0xdd, 0, 4]);
    out.extend_from_slice(&interval.to_be_bytes());
    out.extend_from_slice(&first[sos..end]);
    for (i, part) in parts.iter().enumerate().skip(1) {
        let (_, _, start, end) = jpeg_layout(part)?;
        out.extend_from_slice(&[0xff, 0xd0 + ((i - 1) % 8) as u8]);
        out.extend_from_slice(&part[start..end]);
    }
    let _ = start;
    out.extend_from_slice(&[0xff, 0xd9]);
    Ok(out)
}

/// Pillow's alpha compositing of an RGBA pixel over an opaque color.
#[inline]
fn composite(p: &[u8], background: [u8; 3]) -> [u8; 3] {
    let a = u32::from(p[3]);
    let blend = |fg: u8, bg: u8| {
        let t = u32::from(fg) * a + u32::from(bg) * (255 - a) + 128;
        (((t >> 8) + t) >> 8) as u8
    };
    [blend(p[0], background[0]), blend(p[1], background[1]), blend(p[2], background[2])]
}

/// One encoder run; `metadata`: write the density, ICC profile and Exif data.
fn encode_jpeg_plain(pixels: &[u8], width: usize, height: usize, gray: bool, o: &JpegOptions, metadata: bool) -> Result<Vec<u8>, String> {
    use jpeg_encoder::{ChromaSubsamplingMethod, ColorType, Encoder, PixelDensity, PixelDensityUnit, SamplingFactor};
    let mut out = Vec::with_capacity(pixels.len() / 8 + 1024);
    let mut enc = Encoder::new(&mut out, o.quality.clamp(1, 100));
    enc.set_sampling_factor(match o.subsampling {
        0 => SamplingFactor::R_4_4_4,
        1 => SamplingFactor::R_4_2_2,
        _ => SamplingFactor::R_4_2_0,
    });
    // libjpeg's (and so Pillow's) box filter.
    enc.set_chroma_subsampling_method(ChromaSubsamplingMethod::Average);
    enc.set_optimized_huffman_tables(o.optimize);
    enc.set_progressive(o.progressive);
    let fail = |e: jpeg_encoder::EncodingError| format!("cannot write JPEG: {e}");
    if metadata {
        if let Some(density) = o.dpi {
            enc.set_density(PixelDensity { density, unit: PixelDensityUnit::Inches });
        }
        if !o.icc_profile.is_empty() {
            enc.add_icc_profile(&o.icc_profile).map_err(fail)?;
        }
        if !o.exif.is_empty() {
            enc.add_exif_metadata(&o.exif).map_err(fail)?;
        }
    }
    let color = if gray { ColorType::Luma } else { ColorType::Rgb };
    enc.encode(pixels, width as u16, height as u16, color).map_err(fail)?;
    Ok(out)
}

/// Gray JPEG files as gray, YCbCr and RGB ones as RGB, CMYK and YCCK ones as
/// RGBA (as Pillow's ``pil_to_array`` converts CMYK images).
fn decode_jpeg(data: &[u8]) -> Result<Decoded, String> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;
    let fail = |e: zune_jpeg::errors::DecodeErrors| format!("cannot decode JPEG: {e:?}");
    let options = DecoderOptions::default().set_max_width(1 << 20).set_max_height(1 << 20);
    let mut probe = JpegDecoder::new_with_options(ZCursor::new(data), options);
    probe.decode_headers().map_err(fail)?;
    let (out, channels) = match probe.input_colorspace() {
        Some(ColorSpace::Luma) => (ColorSpace::Luma, 1),
        Some(ColorSpace::CMYK | ColorSpace::YCCK) => (ColorSpace::CMYK, 4),
        _ => (ColorSpace::RGB, 3),
    };
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options.jpeg_set_out_colorspace(out));
    let mut pixels = decoder.decode().map_err(fail)?;
    let (w, h) = decoder.dimensions().ok_or("cannot decode JPEG: no image size")?;
    if out == ColorSpace::CMYK {
        // Adobe CMYK is stored inverted (Pillow's "CMYK;I"); then Pillow's
        // conversion to RGBA.
        for p in pixels.chunks_exact_mut(4) {
            let inverted = [255 - p[0], 255 - p[1], 255 - p[2], 255 - p[3]];
            p.copy_from_slice(&cmyk_to_rgba(&inverted));
        }
    }
    Decoded::u8(w, h, channels, pixels)
}

// ---------------------------------------------------------------------------
// TIFF

/// The TIFF file of gray (1), RGB (3) or RGBA (4, unassociated alpha) pixels;
/// `compression` is 0 (none, Pillow's default), 1 (LZW), 2 (Deflate) or 3
/// (PackBits).
pub fn encode_tiff(
    pixels: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    compression: u8,
    dpi: Option<(f64, f64)>,
) -> Result<Vec<u8>, String> {
    use tiff::encoder::{colortype, compression::DeflateLevel, Compression, TiffEncoder};
    check_size(pixels, width, height, channels, u32::MAX as usize)?;
    let compression = match compression {
        0 => Compression::Uncompressed,
        1 => Compression::Lzw,
        2 => Compression::Deflate(DeflateLevel::Balanced),
        3 => Compression::Packbits,
        _ => return Err(format!("invalid TIFF compression {compression}")),
    };
    let fail = |e: tiff::TiffError| format!("cannot write TIFF: {e}");
    let mut out = Cursor::new(Vec::with_capacity(pixels.len() + 1024));
    let mut enc = TiffEncoder::new(&mut out).map_err(fail)?.with_compression(compression);
    let (w, h) = (width as u32, height as u32);
    macro_rules! write {
        ($color:ty, $alpha:expr) => {{
            let mut image = enc.new_image::<$color>(w, h).map_err(fail)?;
            if $alpha {
                // Unassociated alpha, as Pillow writes RGBA images.
                image.encoder().write_tag(tiff::tags::Tag::ExtraSamples, &[2u16][..]).map_err(fail)?;
            }
            if let Some((x, y)) = dpi {
                image.resolution_unit(tiff::tags::ResolutionUnit::Inch);
                image.x_resolution(rational(x));
                image.y_resolution(rational(y));
            }
            image.write_data(pixels).map_err(fail)?;
        }};
    }
    match channels {
        1 => write!(colortype::Gray8, false),
        3 => write!(colortype::RGB8, false),
        4 => write!(colortype::RGBA8, true),
        _ => return Err(format!("cannot write {channels}-channel TIFF images")),
    }
    Ok(out.into_inner())
}

/// `v` as a fraction (exact for integers, else with denominator 10000).
fn rational(v: f64) -> tiff::encoder::Rational {
    let v = v.clamp(0.0, u32::MAX as f64);
    if v.fract() == 0.0 {
        return tiff::encoder::Rational { n: v as u32, d: 1 };
    }
    let d = 10_000u64;
    let n = (v * d as f64).round().min(u32::MAX as f64) as u64;
    let g = gcd(n, d);
    tiff::encoder::Rational { n: (n / g) as u32, d: (d / g) as u32 }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// The first image of a TIFF file, as Pillow's ``pil_to_array`` returns it:
/// 8-bit gray, 16-bit gray, RGB and RGBA as they are, 16-bit RGB(A) reduced
/// to their high bytes, and bilevel, gray + alpha, palette and CMYK images
/// converted to RGBA.
fn decode_tiff(data: &[u8]) -> Result<Decoded, String> {
    use tiff::decoder::{Decoder, DecodingResult, Limits};
    use tiff::ColorType;
    let fail = |e: tiff::TiffError| format!("cannot decode TIFF: {e}");
    let mut d = Decoder::new(Cursor::new(data)).map_err(fail)?.with_limits(Limits::unlimited());
    let (w, h) = d.dimensions().map_err(fail)?;
    let (w, h) = (w as usize, h as usize);
    let color = d.colortype().map_err(fail)?;
    let colormap = if matches!(color, ColorType::Palette(_)) {
        d.find_tag_unsigned_vec::<u16>(tiff::tags::Tag::ColorMap).map_err(fail)?
    } else {
        None
    };
    let image = d.read_image().map_err(fail)?;
    let n = w * h;
    let rgba = |f: &dyn Fn(usize) -> [u8; 4]| -> Vec<u8> { (0..n).flat_map(f).collect() };
    match (color, image) {
        (ColorType::Gray(8), DecodingResult::U8(v)) => Decoded::u8(w, h, 1, v),
        (ColorType::Gray(16), DecodingResult::U16(v)) if v.len() == n => {
            Ok(Decoded { width: w, height: h, channels: 1, samples: Samples::U16(v) })
        }
        (ColorType::Gray(bits @ (1 | 2 | 4)), DecodingResult::U8(v)) => {
            // Rows of packed samples; Pillow converts "1" images to 0/255.
            let stride = (w * bits as usize).div_ceil(8);
            let max = (1u16 << bits) - 1;
            let sample = |i: usize| {
                let (y, x) = (i / w, i % w);
                let bit = x * bits as usize;
                let byte = v.get(y * stride + bit / 8).copied().unwrap_or(0);
                let s = (byte >> (8 - bits as usize - bit % 8)) as u16 & max;
                (s * 255 / max) as u8
            };
            Decoded::u8(w, h, 4, rgba(&|i| {
                let g = sample(i);
                [g, g, g, 255]
            }))
        }
        (ColorType::GrayA(8) | ColorType::Multiband { bit_depth: 8, num_samples: 2 }, DecodingResult::U8(v)) if v.len() == 2 * n => {
            // Gray + alpha (Pillow's "LA"), converted to RGBA.
            Decoded::u8(w, h, 4, rgba(&|i| [v[2 * i], v[2 * i], v[2 * i], v[2 * i + 1]]))
        }
        (ColorType::RGB(8), DecodingResult::U8(v)) => Decoded::u8(w, h, 3, v),
        (ColorType::RGBA(8), DecodingResult::U8(v)) => Decoded::u8(w, h, 4, v),
        (ColorType::RGB(16), DecodingResult::U16(v)) => Decoded::u8(w, h, 3, v.iter().map(|&s| (s >> 8) as u8).collect()),
        (ColorType::RGBA(16), DecodingResult::U16(v)) => Decoded::u8(w, h, 4, v.iter().map(|&s| (s >> 8) as u8).collect()),
        (ColorType::Palette(8), DecodingResult::U8(v)) if v.len() == n => {
            let map = colormap.ok_or("cannot decode TIFF: palette image without a color map")?;
            let colors = map.len() / 3;
            Decoded::u8(w, h, 4, rgba(&|i| {
                let k = (v[i] as usize).min(colors.saturating_sub(1));
                let c = |j: usize| map.get(j * colors + k).map_or(0, |&s| (s >> 8) as u8);
                [c(0), c(1), c(2), 255]
            }))
        }
        (ColorType::CMYK(8), DecodingResult::U8(v)) if v.len() == 4 * n => {
            Decoded::u8(w, h, 4, rgba(&|i| cmyk_to_rgba(&v[4 * i..4 * i + 4])))
        }
        (color, _) => Err(format!("cannot read {color:?} TIFF images")),
    }
}

/// Pillow's CMYK to RGB conversion (``cmyk2rgb``).
fn cmyk_to_rgba(p: &[u8]) -> [u8; 4] {
    let nk = 255 - i32::from(p[3]);
    let channel = |c: u8| {
        let t = i32::from(c) * nk + 128;
        (nk - (((t >> 8) + t) >> 8)).clamp(0, 255) as u8
    };
    [channel(p[0]), channel(p[1]), channel(p[2]), 255]
}

// ---------------------------------------------------------------------------
// GIF

/// The GIF file of RGBA `frames` (one frame: a still image).  Pixels with
/// alpha below 128 are transparent; each frame gets its own palette, exact
/// for up to 256 colors, else by median cut.  `delay_ms` and `loops` (0:
/// forever) are written for animations.
pub fn encode_gif(frames: &[&[u8]], width: usize, height: usize, delay_ms: u32, loops: Option<u16>) -> Result<Vec<u8>, String> {
    use gif::{DisposalMethod, Encoder, Frame, Repeat};
    if frames.is_empty() {
        return Err("a GIF file needs a frame".into());
    }
    for f in frames {
        check_size(f, width, height, 4, 65535)?;
    }
    let fail = |e: gif::EncodingError| format!("cannot write GIF: {e}");
    let mut out = Vec::new();
    {
        let mut enc = Encoder::new(&mut out, width as u16, height as u16, &[]).map_err(fail)?;
        if let Some(n) = loops {
            enc.set_repeat(if n == 0 { Repeat::Infinite } else { Repeat::Finite(n) }).map_err(fail)?;
        }
        for rgba in frames {
            let (indices, palette, transparent) = quantize(rgba);
            let frame = Frame {
                width: width as u16,
                height: height as u16,
                delay: if frames.len() > 1 { (delay_ms / 10).min(u16::MAX as u32) as u16 } else { 0 },
                // Transparent pixels show the background, not the previous frame.
                dispose: if transparent.is_some() { DisposalMethod::Background } else { DisposalMethod::Keep },
                transparent,
                palette: Some(palette),
                buffer: indices.into(),
                ..Frame::default()
            };
            enc.write_frame(&frame).map_err(fail)?;
        }
    }
    Ok(out)
}

/// Palette indices, the RGB palette and the transparent index of RGBA pixels.
fn quantize(rgba: &[u8]) -> (Vec<u8>, Vec<u8>, Option<u8>) {
    use std::collections::HashMap;
    let opaque = |p: &[u8]| p[3] >= 128;
    let key = |p: &[u8]| u32::from_be_bytes([0, p[0], p[1], p[2]]);
    let transparent = rgba.chunks_exact(4).any(|p| !opaque(p));
    let max_colors = if transparent { 255 } else { 256 };
    let mut colors: HashMap<u32, u8> = HashMap::new();
    let mut exact = true;
    let mut last = None;
    for p in rgba.chunks_exact(4).filter(|p| opaque(p)) {
        let k = key(p);
        if last == Some(k) {
            continue;
        }
        last = Some(k);
        colors.entry(k).or_insert(0);
        if colors.len() > max_colors {
            exact = false;
            break;
        }
    }
    let (mut palette, indices) = if exact {
        let mut sorted: Vec<u32> = colors.into_keys().collect();
        sorted.sort_unstable();
        let mut palette = Vec::with_capacity(768);
        let mut index: HashMap<u32, u8> = HashMap::new();
        for (i, c) in sorted.iter().enumerate() {
            palette.extend_from_slice(&c.to_be_bytes()[1..]);
            index.insert(*c, i as u8);
        }
        let tindex = (palette.len() / 3) as u8;
        let indices = rgba
            .chunks_exact(4)
            .map(|p| if opaque(p) { index.get(&key(p)).copied().unwrap_or(0) } else { tindex })
            .collect();
        (palette, indices)
    } else {
        median_cut(rgba, max_colors)
    };
    let tindex = palette.len() / 3;
    if transparent {
        palette.extend_from_slice(&[0, 0, 0]);
    }
    // GIF palettes have a power of two entries.
    let entries = (palette.len() / 3).max(2).next_power_of_two();
    palette.resize(entries * 3, 0);
    (indices, palette, transparent.then_some(tindex as u8))
}

/// A palette of at most `max_colors` for the opaque pixels of `rgba` by
/// median cut of their 5-bit-per-channel histogram (each palette color the
/// mean of its pixels), and the index of every pixel (`max_colors` for the
/// transparent ones).
fn median_cut(rgba: &[u8], max_colors: usize) -> (Vec<u8>, Vec<u8>) {
    let bin = |p: &[u8]| (usize::from(p[0] >> 3) << 10) | (usize::from(p[1] >> 3) << 5) | usize::from(p[2] >> 3);
    let mut count = vec![0u32; 1 << 15];
    let mut sum = vec![[0u64; 3]; 1 << 15];
    for p in rgba.chunks_exact(4).filter(|p| p[3] >= 128) {
        let b = bin(p);
        count[b] += 1;
        for c in 0..3 {
            sum[b][c] += u64::from(p[c]);
        }
    }
    let channel = |b: u16, c: usize| (b >> (10 - 5 * c)) & 31;
    let mut bins: Vec<u16> = (0..1u16 << 15).filter(|&b| count[b as usize] > 0).collect();
    // Boxes: ranges of `bins`; split the one with the most pixels times
    // extent along its widest channel, at the pixel median of that channel.
    let mut boxes: Vec<std::ops::Range<usize>> = vec![0..bins.len()];
    let extent = |bins: &[u16]| -> (u32, usize) {
        (0..3)
            .map(|c| {
                let (lo, hi) = bins.iter().fold((31, 0), |(lo, hi), &b| (lo.min(channel(b, c)), hi.max(channel(b, c))));
                (u32::from(hi - lo), c)
            })
            .max()
            .unwrap()
    };
    while boxes.len() < max_colors {
        let best = boxes
            .iter()
            .enumerate()
            .filter(|(_, r)| r.len() > 1)
            .map(|(i, r)| {
                let pixels: u64 = bins[r.clone()].iter().map(|&b| u64::from(count[b as usize])).sum();
                (pixels * u64::from(extent(&bins[r.clone()]).0 + 1), i)
            })
            .max();
        let Some((_, i)) = best else { break };
        let r = boxes[i].clone();
        let (_, c) = extent(&bins[r.clone()]);
        bins[r.clone()].sort_unstable_by_key(|&b| (channel(b, c), b));
        let total: u64 = bins[r.clone()].iter().map(|&b| u64::from(count[b as usize])).sum();
        let mut acc = 0u64;
        let mut cut = r.start + 1;
        for (j, &b) in bins[r.clone()].iter().enumerate() {
            acc += u64::from(count[b as usize]);
            if 2 * acc >= total {
                cut = (r.start + j + 1).clamp(r.start + 1, r.end - 1);
                break;
            }
        }
        boxes[i] = r.start..cut;
        boxes.push(cut..r.end);
    }
    let mut lut = vec![0u8; 1 << 15];
    let mut palette = Vec::with_capacity(boxes.len() * 3);
    for (i, r) in boxes.iter().enumerate() {
        let (mut n, mut s) = (0u64, [0u64; 3]);
        for &b in &bins[r.clone()] {
            lut[b as usize] = i as u8;
            n += u64::from(count[b as usize]);
            for c in 0..3 {
                s[c] += sum[b as usize][c];
            }
        }
        palette.extend((0..3).map(|c| ((s[c] + n / 2) / n.max(1)) as u8));
    }
    let tindex = max_colors as u8;
    let indices = rgba.chunks_exact(4).map(|p| if p[3] >= 128 { lut[bin(p)] } else { tindex }).collect();
    (palette, indices)
}

/// The colors of RGB `pixels` in increasing order (as 0xRRGGBB) and the
/// index of each pixel, if there are at most `max` colors.
pub fn indexed_colors(pixels: &[u8], max: usize) -> Option<(Vec<u8>, Vec<u8>)> {
    use std::collections::HashMap;
    let mut colors: HashMap<u32, u8> = HashMap::new();
    let key = |p: &[u8]| u32::from_be_bytes([0, p[0], p[1], p[2]]);
    let mut last = None;
    for p in pixels.chunks_exact(3) {
        let k = key(p);
        if last == Some(k) {
            continue;
        }
        last = Some(k);
        colors.entry(k).or_insert(0);
        if colors.len() > max.min(256) {
            return None;
        }
    }
    let mut sorted: Vec<u32> = colors.keys().copied().collect();
    sorted.sort_unstable();
    let mut palette = Vec::with_capacity(sorted.len() * 3);
    for (i, c) in sorted.iter().enumerate() {
        palette.extend_from_slice(&c.to_be_bytes()[1..]);
        colors.insert(*c, i as u8);
    }
    Some((palette, pixels.chunks_exact(3).map(|p| colors[&key(p)]).collect()))
}

/// The first frame of a GIF file as RGBA, on a transparent canvas of the
/// logical screen size.
fn decode_gif(data: &[u8]) -> Result<Decoded, String> {
    let fail = |e: gif::DecodingError| format!("cannot decode GIF: {e}");
    let mut options = gif::DecodeOptions::new();
    options.set_color_output(gif::ColorOutput::RGBA);
    let mut d = options.read_info(Cursor::new(data)).map_err(fail)?;
    let (w, h) = (usize::from(d.width()), usize::from(d.height()));
    let frame = d.read_next_frame().map_err(fail)?.ok_or("cannot decode GIF: no image")?;
    let mut canvas = vec![0u8; w * h * 4];
    let (fw, fh) = (usize::from(frame.width), usize::from(frame.height));
    let (left, top) = (usize::from(frame.left), usize::from(frame.top));
    for y in 0..fh.min(h.saturating_sub(top)) {
        let cols = fw.min(w.saturating_sub(left));
        let src = &frame.buffer[y * fw * 4..(y * fw + cols) * 4];
        let start = ((top + y) * w + left) * 4;
        canvas[start..start + cols * 4].copy_from_slice(src);
    }
    Decoded::u8(w, h, 4, canvas)
}

// ---------------------------------------------------------------------------
// WebP

/// Options of `encode_webp`, with Pillow's names and defaults.
#[derive(Clone, Copy, Debug)]
pub struct WebpOptions {
    pub lossless: bool,
    /// 0 to 100 (default 80).
    pub quality: f32,
    /// 0 to 100 (default 100).
    pub alpha_quality: f32,
    /// 0 (fast) to 6 (slow, small); default 4.
    pub method: i32,
    /// Keep the RGB values of transparent pixels.
    pub exact: bool,
}

impl Default for WebpOptions {
    fn default() -> Self {
        WebpOptions { lossless: false, quality: 80.0, alpha_quality: 100.0, method: 4, exact: false }
    }
}

/// A libwebp picture holding RGB (3 channels) or RGBA (4) pixels as ARGB,
/// as Pillow imports them.
struct Picture(libwebp_sys::WebPPicture);

impl Picture {
    fn new(pixels: &[u8], width: usize, height: usize, channels: usize) -> Result<Self, String> {
        use libwebp_sys as w;
        // SAFETY: the picture is initialized before use and freed on drop.
        unsafe {
            let mut pic: w::WebPPicture = std::mem::zeroed();
            if !w::WebPPictureInit(&mut pic) {
                return Err("cannot initialize a WebP picture".into());
            }
            pic.width = width as i32;
            pic.height = height as i32;
            pic.use_argb = 1;
            if w::WebPPictureAlloc(&mut pic) == 0 {
                return Err("cannot allocate a WebP picture".into());
            }
            let picture = Picture(pic);
            let stride = picture.0.argb_stride as usize;
            let argb = std::slice::from_raw_parts_mut(picture.0.argb, stride * height);
            for y in 0..height {
                let row = &pixels[y * width * channels..(y + 1) * width * channels];
                for (x, p) in row.chunks_exact(channels).enumerate() {
                    let a = if channels == 4 { p[3] } else { 255 };
                    argb[y * stride + x] = u32::from_be_bytes([a, p[0], p[1], p[2]]);
                }
            }
            Ok(picture)
        }
    }
}

impl Drop for Picture {
    fn drop(&mut self) {
        // SAFETY: allocated by WebPPictureAlloc.
        unsafe { libwebp_sys::WebPPictureFree(&mut self.0) }
    }
}

fn webp_config(o: &WebpOptions) -> Result<libwebp_sys::WebPConfig, String> {
    use libwebp_sys as w;
    // SAFETY: WebPConfigInit initializes every field.
    unsafe {
        let mut config: w::WebPConfig = std::mem::zeroed();
        if !w::WebPInitConfig(&mut config) {
            return Err("cannot initialize the WebP configuration".into());
        }
        config.lossless = i32::from(o.lossless);
        config.quality = o.quality;
        config.alpha_quality = o.alpha_quality.round() as i32;
        config.method = o.method;
        config.exact = i32::from(o.exact);
        if w::WebPValidateConfig(&config) == 0 {
            return Err("invalid WebP configuration".into());
        }
        Ok(config)
    }
}

/// The WebP file of RGB (3 channels) or RGBA (4) pixels, written as Pillow
/// does (``WebPEncode`` with the default configuration, then ``WebPMux``).
pub fn encode_webp(pixels: &[u8], width: usize, height: usize, channels: usize, o: &WebpOptions) -> Result<Vec<u8>, String> {
    use libwebp_sys as w;
    check_webp(pixels, width, height, channels)?;
    let config = webp_config(o)?;
    let mut pic = Picture::new(pixels, width, height, channels)?;
    // SAFETY: the writer and mux follow libwebp's documented use.
    unsafe {
        let mut writer: w::WebPMemoryWriter = std::mem::zeroed();
        w::WebPMemoryWriterInit(&mut writer);
        pic.0.writer = Some(w::WebPMemoryWrite);
        pic.0.custom_ptr = &mut writer as *mut _ as *mut std::ffi::c_void;
        let ok = w::WebPEncode(&config, &mut pic.0);
        let encoded = std::slice::from_raw_parts(writer.mem, writer.size).to_vec();
        w::WebPMemoryWriterClear(&mut writer);
        if ok == 0 {
            return Err(format!("cannot write WebP: encoding error {}", pic.0.error_code as i32));
        }
        let mux = w::WebPMuxNew();
        if mux.is_null() {
            return Err("cannot create a WebP mux".into());
        }
        let image = w::WebPData { bytes: encoded.as_ptr(), size: encoded.len() };
        let mut assembled: w::WebPData = std::mem::zeroed();
        let ok = w::WebPMuxSetImage(mux, &image, 0) == w::WebPMuxError::WEBP_MUX_OK
            && w::WebPMuxAssemble(mux, &mut assembled) == w::WebPMuxError::WEBP_MUX_OK;
        w::WebPMuxDelete(mux);
        if !ok {
            w::WebPDataClear(&mut assembled);
            return Err("cannot assemble the WebP file".into());
        }
        let out = std::slice::from_raw_parts(assembled.bytes, assembled.size).to_vec();
        w::WebPDataClear(&mut assembled);
        Ok(out)
    }
}

/// An animated WebP file of RGB or RGBA `frames`, each shown `delay_ms`,
/// played `loops` times (0: forever).
pub fn encode_webp_animation(
    frames: &[&[u8]],
    width: usize,
    height: usize,
    channels: usize,
    delay_ms: u32,
    loops: u32,
    o: &WebpOptions,
) -> Result<Vec<u8>, String> {
    use libwebp_sys as w;
    if frames.is_empty() {
        return Err("a WebP animation needs a frame".into());
    }
    for f in frames {
        check_webp(f, width, height, channels)?;
    }
    let config = webp_config(o)?;
    // SAFETY: the encoder follows libwebp's documented use and is deleted.
    unsafe {
        let mut options: w::WebPAnimEncoderOptions = std::mem::zeroed();
        if w::WebPAnimEncoderOptionsInitInternal(&mut options, w::WebPGetMuxABIVersion()) == 0 {
            return Err("cannot initialize the WebP animation options".into());
        }
        options.anim_params.loop_count = loops as i32;
        options.anim_params.bgcolor = 0;
        let enc = w::WebPAnimEncoderNewInternal(width as i32, height as i32, &options, w::WebPGetMuxABIVersion());
        if enc.is_null() {
            return Err("cannot create a WebP animation encoder".into());
        }
        let result = (|| {
            let mut timestamp = 0i32;
            for f in frames {
                let mut pic = Picture::new(f, width, height, channels)?;
                if w::WebPAnimEncoderAdd(enc, &mut pic.0, timestamp, &config) == 0 {
                    return Err("cannot add a WebP animation frame".to_string());
                }
                timestamp += delay_ms as i32;
            }
            if w::WebPAnimEncoderAdd(enc, std::ptr::null_mut(), timestamp, std::ptr::null()) == 0 {
                return Err("cannot finish the WebP animation".to_string());
            }
            let mut data: w::WebPData = std::mem::zeroed();
            if w::WebPAnimEncoderAssemble(enc, &mut data) == 0 {
                return Err("cannot assemble the WebP animation".to_string());
            }
            let out = std::slice::from_raw_parts(data.bytes, data.size).to_vec();
            w::WebPDataClear(&mut data);
            Ok(out)
        })();
        w::WebPAnimEncoderDelete(enc);
        result
    }
}

fn check_webp(pixels: &[u8], width: usize, height: usize, channels: usize) -> Result<(), String> {
    if !matches!(channels, 3 | 4) {
        return Err(format!("cannot write {channels}-channel WebP images"));
    }
    check_size(pixels, width, height, channels, 16383)
}

/// A WebP file (the first frame of an animation) as RGB, or RGBA if it has
/// alpha.
fn decode_webp(data: &[u8]) -> Result<Decoded, String> {
    use libwebp_sys as w;
    // SAFETY: libwebp reads `data` and returns buffers freed here.
    unsafe {
        let mut features: w::WebPBitstreamFeatures = std::mem::zeroed();
        if w::WebPGetFeatures(data.as_ptr(), data.len(), &mut features) != w::VP8StatusCode::VP8_STATUS_OK {
            return Err("cannot decode WebP: invalid file".into());
        }
        let alpha = features.has_alpha != 0;
        if features.has_animation != 0 {
            let mut options: w::WebPAnimDecoderOptions = std::mem::zeroed();
            if w::WebPAnimDecoderOptionsInit(&mut options) == 0 {
                return Err("cannot decode WebP animation".into());
            }
            options.color_mode = w::WEBP_CSP_MODE::MODE_RGBA;
            let wd = w::WebPData { bytes: data.as_ptr(), size: data.len() };
            let dec = w::WebPAnimDecoderNew(&wd, &options);
            if dec.is_null() {
                return Err("cannot decode WebP animation".into());
            }
            let mut info: w::WebPAnimInfo = std::mem::zeroed();
            let mut buf: *mut u8 = std::ptr::null_mut();
            let mut ts = 0;
            let ok = w::WebPAnimDecoderGetInfo(dec, &mut info) != 0 && w::WebPAnimDecoderGetNext(dec, &mut buf, &mut ts) != 0;
            let (wd_, ht) = (info.canvas_width as usize, info.canvas_height as usize);
            let result = if ok {
                let rgba = std::slice::from_raw_parts(buf, wd_ * ht * 4);
                if alpha {
                    Decoded::u8(wd_, ht, 4, rgba.to_vec())
                } else {
                    Decoded::u8(wd_, ht, 3, rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect())
                }
            } else {
                Err("cannot decode WebP animation".into())
            };
            w::WebPAnimDecoderDelete(dec);
            return result;
        }
        let (mut width, mut height) = (0, 0);
        let buf = if alpha {
            w::WebPDecodeRGBA(data.as_ptr(), data.len(), &mut width, &mut height)
        } else {
            w::WebPDecodeRGB(data.as_ptr(), data.len(), &mut width, &mut height)
        };
        if buf.is_null() {
            return Err("cannot decode WebP".into());
        }
        let channels = if alpha { 4 } else { 3 };
        let (width, height) = (width as usize, height as usize);
        let pixels = std::slice::from_raw_parts(buf, width * height * channels).to_vec();
        w::WebPFree(buf as *mut std::ffi::c_void);
        Decoded::u8(width, height, channels, pixels)
    }
}

// ---------------------------------------------------------------------------
// AVIF

/// The AVIF file of RGBA pixels (alpha dropped when every pixel is opaque);
/// `quality` 0 to 100 and `speed` 0 (slow) to 10, as Pillow's options.
pub fn encode_avif(pixels: &[u8], width: usize, height: usize, quality: f32, speed: u8, threads: usize) -> Result<Vec<u8>, String> {
    check_size(pixels, width, height, 4, u32::MAX as usize)?;
    let rgba: Vec<ravif::RGBA8> = pixels.chunks_exact(4).map(|p| ravif::RGBA8::new(p[0], p[1], p[2], p[3])).collect();
    let encoded = ravif::Encoder::new()
        .with_quality(quality.clamp(0.0, 100.0))
        .with_alpha_quality(quality.clamp(0.0, 100.0))
        .with_speed(speed.clamp(1, 10))
        .with_num_threads(Some(threads.max(1)))
        .encode_rgba(ravif::Img::new(&rgba[..], width, height))
        .map_err(|e| format!("cannot write AVIF: {e}"))?;
    Ok(encoded.avif_file)
}

fn check_size(pixels: &[u8], width: usize, height: usize, channels: usize, max: usize) -> Result<(), String> {
    if width == 0 || height == 0 || width > max || height > max {
        return Err(format!("cannot write a {width}x{height} image in this format (at most {max}x{max})"));
    }
    if width.checked_mul(height).and_then(|n| n.checked_mul(channels)) != Some(pixels.len()) {
        return Err(format!("{} bytes are not {height} rows of {width}x{channels}", pixels.len()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: usize, h: usize, channels: usize) -> Vec<u8> {
        (0..w * h * channels)
            .map(|i| {
                let (x, y, c) = ((i / channels) % w, i / channels / w, i % channels);
                if c == 3 {
                    if x < 4 { 0 } else { 255 }
                } else {
                    ((x * 5 + y * 3 + c * 70) % 256) as u8
                }
            })
            .collect()
    }

    fn pixels(d: &Decoded) -> &[u8] {
        match &d.samples {
            Samples::U8(v) => v,
            Samples::U16(_) => panic!("16-bit"),
        }
    }

    #[test]
    fn sniffing() {
        assert_eq!(sniff(b"\x89PNG\r\n\x1a\n...."), Some(Format::Png));
        assert_eq!(sniff(b"\xff\xd8\xff\xe0"), Some(Format::Jpeg));
        assert_eq!(sniff(b"MM\0*...."), Some(Format::Tiff));
        assert_eq!(sniff(b"GIF89a"), Some(Format::Gif));
        assert_eq!(sniff(b"RIFF\0\0\0\0WEBPVP8 "), Some(Format::Webp));
        assert_eq!(sniff(b"\0\0\0\x1cftypavif"), Some(Format::Avif));
        assert_eq!(sniff(b"hello"), None);
    }

    #[test]
    fn jpeg_round_trip() {
        let rgb: Vec<u8> = (0..40 * 30 * 3).map(|i| ((i / 3 % 40) * 4 + (i / 120) * 2 + (i % 3) * 30) as u8).collect();
        for (q, sub) in [(75, 2), (95, 0), (50, 1)] {
            let o = JpegOptions { quality: q, subsampling: sub, dpi: Some((100, 100)), ..JpegOptions::default() };
            let data = encode_jpeg(&rgb, 40, 30, 3, [255; 3], &o, 1).unwrap();
            assert_eq!(&data[6..11], b"JFIF\0");
            assert_eq!(&data[13..18], &[1, 0, 100, 0, 100]);
            let (format, d) = decode(&data).unwrap();
            assert_eq!((format, d.width, d.height, d.channels), (Format::Jpeg, 40, 30, 3));
            let err: f64 = pixels(&d).iter().zip(&rgb).map(|(&a, &b)| f64::from((a as i32 - b as i32).abs())).sum::<f64>()
                / rgb.len() as f64;
            assert!(err < 3.0, "quality {q}: mean error {err}");
        }
        let gray = image(17, 9, 1);
        let d = decode_jpeg(&encode_jpeg(&gray, 17, 9, 1, [255; 3], &JpegOptions::default(), 1).unwrap()).unwrap();
        assert_eq!(d.channels, 1);
    }

    #[test]
    fn jpeg_strips() {
        // Encoded in strips joined by restart markers: the same pixels as one
        // encoding, whatever the number of threads.
        for (w, h, channels, sub) in [(700, 900, 4, 2), (1000, 500, 3, 1), (333, 1500, 1, 2), (64, 5000, 3, 0)] {
            let px = image(w, h, channels);
            let o = JpegOptions { dpi: Some((72, 72)), ..JpegOptions { subsampling: sub, ..JpegOptions::default() } };
            let one = encode_jpeg(&px, w, h, channels, [10, 20, 30], &o, 1).unwrap();
            let four = encode_jpeg(&px, w, h, channels, [10, 20, 30], &o, 4).unwrap();
            assert!(four.windows(2).any(|m| m == [0xff, 0xdd]), "{w}x{h}: no restart interval");
            assert_eq!(four, encode_jpeg(&px, w, h, channels, [10, 20, 30], &o, 3).unwrap());
            let (a, b) = (decode(&one).unwrap().1, decode(&four).unwrap().1);
            assert_eq!((b.width, b.height), (w, h));
            assert_eq!(pixels(&a), pixels(&b), "{w}x{h}x{channels}");
        }
    }

    #[test]
    fn composite_like_pillow() {
        // Pillow's paste: DIV255(fg * a + bg * (255 - a)).
        assert_eq!(composite(&[200, 100, 0, 255], [1, 2, 3]), [200, 100, 0]);
        assert_eq!(composite(&[200, 100, 0, 0], [1, 2, 3]), [1, 2, 3]);
        assert_eq!(composite(&[255, 0, 128, 128], [0, 255, 0]), [128, 127, 64]);
    }

    #[test]
    fn tiff_round_trip() {
        for channels in [1, 3, 4] {
            let px = image(33, 21, channels);
            for compression in 0..4 {
                let data = encode_tiff(&px, 33, 21, channels, compression, Some((100.0, 72.5))).unwrap();
                let (format, d) = decode(&data).unwrap();
                assert_eq!((format, d.width, d.height, d.channels), (Format::Tiff, 33, 21, channels));
                assert_eq!(pixels(&d), &px[..]);
            }
        }
    }

    #[test]
    fn gif_round_trip() {
        // Few colors: exact palette.
        let mut px = image(20, 10, 4);
        for p in px.chunks_exact_mut(4) {
            p[0] &= 0xc0;
            p[1] &= 0xc0;
            p[2] &= 0xc0;
        }
        let data = encode_gif(&[&px], 20, 10, 0, None).unwrap();
        let (format, d) = decode(&data).unwrap();
        assert_eq!((format, d.channels), (Format::Gif, 4));
        for (a, b) in pixels(&d).chunks_exact(4).zip(px.chunks_exact(4)) {
            if b[3] >= 128 {
                assert_eq!(&a[..3], &b[..3]);
                assert_eq!(a[3], 255);
            } else {
                assert_eq!(a[3], 0);
            }
        }
        // Many colors: median cut, animated.
        let px = image(64, 64, 4);
        let data = encode_gif(&[&px, &px], 64, 64, 100, Some(0)).unwrap();
        let d = decode(&data).unwrap().1;
        let err = pixels(&d)
            .chunks_exact(4)
            .zip(px.chunks_exact(4))
            .filter(|(_, b)| b[3] >= 128)
            .map(|(a, b)| (0..3).map(|c| (a[c] as i32 - b[c] as i32).abs()).max().unwrap())
            .max()
            .unwrap();
        assert!(err < 40, "error {err}");
    }

    #[test]
    fn webp_round_trip() {
        let px = image(30, 20, 4);
        let lossless = WebpOptions { lossless: true, ..WebpOptions::default() };
        let d = decode(&encode_webp(&px, 30, 20, 4, &lossless).unwrap()).unwrap().1;
        assert_eq!((d.width, d.height, d.channels), (30, 20, 4));
        // Lossless, but transparent pixels lose their color unless `exact`.
        for (a, b) in pixels(&d).chunks_exact(4).zip(px.chunks_exact(4)) {
            assert_eq!(a[3], b[3]);
            if b[3] > 0 {
                assert_eq!(a, b);
            }
        }
        let rgb = image(30, 20, 3);
        let d = decode(&encode_webp(&rgb, 30, 20, 3, &WebpOptions::default()).unwrap()).unwrap().1;
        assert_eq!(d.channels, 3);
        let anim = encode_webp_animation(&[&px, &px], 30, 20, 4, 100, 0, &lossless).unwrap();
        let d = decode(&anim).unwrap().1;
        assert_eq!((d.width, d.height, d.channels), (30, 20, 4));
    }

    #[test]
    fn avif_writes() {
        let px = image(32, 16, 4);
        let data = encode_avif(&px, 32, 16, 75.0, 10, 1).unwrap();
        assert_eq!(sniff(&data), Some(Format::Avif));
        assert!(decode(&data).is_err());
    }

    #[test]
    fn indexed() {
        let px = [9, 9, 9, 1, 2, 3, 9, 9, 9, 0, 0, 0];
        let (palette, idx) = indexed_colors(&px, 256).unwrap();
        assert_eq!(palette, vec![0, 0, 0, 1, 2, 3, 9, 9, 9]);
        assert_eq!(idx, vec![2, 1, 2, 0]);
        assert!(indexed_colors(&px, 2).is_none());
        let many: Vec<u8> = (0..300u32).flat_map(|i| [i as u8, (i >> 8) as u8, 0]).collect();
        assert!(indexed_colors(&many, 256).is_none());
    }

    #[test]
    fn cmyk() {
        assert_eq!(cmyk_to_rgba(&[0, 0, 0, 0]), [255, 255, 255, 255]);
        assert_eq!(cmyk_to_rgba(&[255, 0, 0, 0]), [0, 255, 255, 255]);
        assert_eq!(cmyk_to_rgba(&[0, 0, 0, 255]), [0, 0, 0, 255]);
    }
}
