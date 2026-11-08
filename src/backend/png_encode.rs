//! PNG writer for 8-bit gray, gray + alpha, RGB and RGBA images.
//!
//! Each row is filtered with the filter type whose output has the smallest
//! sum of absolute (signed byte) values, the heuristic of libpng and Pillow.
//! The filtered rows are deflated by zlib-rs in strips of about
//! [`STRIP_BYTES`]: every strip is primed with the 32 KiB of filtered data
//! before it and ends on a byte boundary (a sync flush), so the compressed
//! strips concatenate into one zlib stream, as in pigz.  Strips are filtered
//! and compressed on several threads; their boundaries depend only on the
//! image size, so the file is the same whatever the number of threads.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use zlib_rs::{Deflate, DeflateConfig, DeflateFlush, Status, Strategy};

/// The zlib level Pillow (and so matplotlib) writes PNG files with.
pub const DEFAULT_LEVEL: u8 = 6;

/// Target amount of filtered data per independently compressed strip.
pub const STRIP_BYTES: usize = 256 * 1024;

/// The deflate window: how much data before a strip primes its compressor.
const WINDOW: usize = 32 * 1024;

/// The largest chunk the PNG format allows.
const MAX_CHUNK: usize = i32::MAX as usize;

/// What to write besides the pixels.
#[derive(Clone, Copy, Debug)]
pub struct Options<'a> {
    /// zlib compression level, 0 (stored) to 9; Pillow's default is 6.
    pub level: u8,
    /// zlib strategy (Pillow's `compress_type`): 0 default, 1 filtered,
    /// 2 Huffman only, 3 run-length, 4 fixed codes.
    pub strategy: u8,
    /// Horizontal and vertical pixel density for the pHYs chunk, in pixels
    /// per inch (none: no pHYs chunk).
    pub dpi: Option<(f64, f64)>,
    /// (keyword, text) pairs, written as tEXt chunks, or as iTXt chunks if the
    /// text is not Latin-1 (as Pillow's `PngInfo.add_text` does).
    pub text: &'a [(String, String)],
    /// Threads to filter and compress on (1: only the calling thread).
    pub threads: usize,
}

/// The PNG file of the `height` rows of `width` pixels of `channels` (1 to 4)
/// bytes each in `pixels`.
pub fn encode(
    pixels: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    opts: &Options<'_>,
) -> Result<Vec<u8>, String> {
    let color_type = match channels {
        1 => 0u8,
        2 => 4,
        3 => 2,
        4 => 6,
        _ => return Err(format!("cannot write {channels}-channel PNG images")),
    };
    if width == 0 || height == 0 || width > i32::MAX as usize || height > i32::MAX as usize {
        return Err(format!("cannot write a {width}x{height} PNG image"));
    }
    let stride = width
        .checked_mul(channels)
        .filter(|s| s.checked_mul(height) == Some(pixels.len()))
        .ok_or_else(|| format!("{} bytes are not {height} rows of {width}x{channels}", pixels.len()))?;
    let level = opts.level.min(9);
    let strategy = Strategy::try_from(i32::from(opts.strategy))
        .map_err(|()| format!("invalid zlib strategy {}", opts.strategy))?;

    let mut out = Vec::with_capacity(pixels.len() / 8 + 1024);
    out.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    let mut ihdr = [0u8; 13];
    ihdr[..4].copy_from_slice(&(width as u32).to_be_bytes());
    ihdr[4..8].copy_from_slice(&(height as u32).to_be_bytes());
    ihdr[8] = 8;
    ihdr[9] = color_type;
    write_chunk(&mut out, b"IHDR", &[&ihdr]);
    for (key, value) in opts.text {
        let key = latin1(key).ok_or_else(|| format!("PNG text keywords must be Latin-1: {key:?}"))?;
        match latin1(value) {
            Some(text) => write_chunk(&mut out, b"tEXt", &[&key, b"\0", &text]),
            // Uncompressed, no language tag or translated keyword.
            None => write_chunk(&mut out, b"iTXt", &[&key, b"\0\0\0\0\0", value.as_bytes()]),
        }
    }
    if let Some((x, y)) = opts.dpi {
        // Pixels per meter, rounded as by Pillow: int(dpi / 0.0254 + 0.5).
        let ppm = |dpi: f64| (dpi / 0.0254 + 0.5).clamp(0.0, u32::MAX as f64) as u32;
        let mut phys = [1u8; 9];
        phys[..4].copy_from_slice(&ppm(x).to_be_bytes());
        phys[4..8].copy_from_slice(&ppm(y).to_be_bytes());
        write_chunk(&mut out, b"pHYs", &[&phys]);
    }

    let image = Image { pixels, stride, bpp: channels };
    let header = zlib_header(level, strategy);
    let config = DeflateConfig { window_bits: -15, strategy, ..DeflateConfig::new(i32::from(level)) };
    let strips = compress_strips(&image, height, config, &header, opts.threads.max(1))?;
    // One IDAT chunk per strip: the zlib header goes in front of the first,
    // the Adler-32 of all the filtered data after the last.
    let mut adler = 1u32;
    for s in &strips {
        adler = zlib_rs::adler32::adler32_combine(adler, s.adler, s.raw_len as u64);
    }
    let trailer = adler.to_be_bytes();
    let last = strips.len() - 1;
    for (i, s) in strips.iter().enumerate() {
        let head: &[u8] = if i == 0 { &header } else { &[] };
        let tail: &[u8] = if i == last { &trailer } else { &[] };
        let len = head.len() + s.data.len() + tail.len();
        if len > MAX_CHUNK {
            let data = [head, &s.data, tail].concat();
            for part in data.chunks(MAX_CHUNK) {
                write_chunk(&mut out, b"IDAT", &[part]);
            }
            continue;
        }
        out.reserve(len + 12);
        out.extend_from_slice(&(len as u32).to_be_bytes());
        out.extend_from_slice(b"IDAT");
        out.extend_from_slice(head);
        out.extend_from_slice(&s.data);
        out.extend_from_slice(tail);
        // The strip's worker computed the CRC of "IDAT", the header and its data.
        let crc = zlib_rs::crc32::crc32(s.crc, tail);
        out.extend_from_slice(&crc.to_be_bytes());
    }
    write_chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

/// `s` encoded as Latin-1, if it can be.
fn latin1(s: &str) -> Option<Vec<u8>> {
    s.chars().map(|c| u8::try_from(u32::from(c)).ok()).collect()
}

fn write_chunk(out: &mut Vec<u8>, kind: &[u8; 4], parts: &[&[u8]]) {
    let len: usize = parts.iter().map(|p| p.len()).sum();
    out.extend_from_slice(&(len as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    for p in parts {
        out.extend_from_slice(p);
    }
    let crc = zlib_rs::crc32::crc32(0, &out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// The two-byte zlib header (32 KiB window, no dictionary) zlib writes for
/// `level` and `strategy`.
fn zlib_header(level: u8, strategy: Strategy) -> [u8; 2] {
    let flevel = match level {
        _ if strategy as i32 >= Strategy::HuffmanOnly as i32 => 0u16,
        0 | 1 => 0,
        2..=5 => 1,
        6 => 2,
        _ => 3,
    };
    let mut h = 0x7800u16 | (flevel << 6);
    h += 31 - h % 31;
    h.to_be_bytes()
}

struct Image<'a> {
    pixels: &'a [u8],
    stride: usize,
    bpp: usize,
}

impl Image<'_> {
    fn row(&self, y: usize) -> &[u8] {
        &self.pixels[y * self.stride..(y + 1) * self.stride]
    }

    /// Append the filter type byte and the filtered bytes of rows `rows`.
    fn filter_rows(&self, filter: FilterRow, rows: std::ops::Range<usize>, zeros: &[u8], out: &mut Vec<u8>) {
        for y in rows {
            let prior = if y == 0 { zeros } else { self.row(y - 1) };
            let start = out.len();
            out.resize(start + 1 + self.stride, 0);
            filter(self.row(y), prior, self.bpp, &mut out[start..]);
        }
    }
}

/// A compressed strip.
struct Strip {
    /// Raw deflate data ending with a sync flush (or the final block).
    data: Vec<u8>,
    /// CRC-32 of "IDAT", the zlib header if first, and `data`.
    crc: u32,
    /// Adler-32 and length of the filtered data.
    adler: u32,
    raw_len: usize,
}

fn compress_strips(
    image: &Image<'_>,
    height: usize,
    config: DeflateConfig,
    header: &[u8; 2],
    threads: usize,
) -> Result<Vec<Strip>, String> {
    let rows_per_strip = (STRIP_BYTES / (image.stride + 1)).max(1);
    let n = height.div_ceil(rows_per_strip);
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Result<Strip, String>>>> = Mutex::new((0..n).map(|_| None).collect());
    let filter = row_filter();
    let work = || {
        let zeros = vec![0u8; image.stride];
        let mut filtered = Vec::new();
        let mut dict = Vec::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            let r0 = i * rows_per_strip;
            let r1 = (r0 + rows_per_strip).min(height);
            filtered.clear();
            image.filter_rows(filter, r0..r1, &zeros, &mut filtered);
            // The window before the strip: the filtered rows it ends with.
            dict.clear();
            if r0 > 0 {
                let rows = WINDOW.div_ceil(image.stride + 1).min(r0);
                image.filter_rows(filter, r0 - rows..r0, &zeros, &mut dict);
            }
            let dict = &dict[dict.len().saturating_sub(WINDOW)..];
            // A new compressor for every strip: one reset after use can
            // compress differently, which would make the file depend on how
            // the strips were shared between the threads.
            let d = &mut Deflate::new_with_config(config);
            let head: &[u8] = if i == 0 { header } else { &[] };
            let result = compress_strip(d, &filtered, dict, head, i + 1 == n);
            results.lock().unwrap()[i] = Some(result);
        }
    };
    let workers = threads.min(n);
    if workers <= 1 {
        work();
    } else {
        std::thread::scope(|s| {
            for _ in 1..workers {
                s.spawn(work);
            }
            work();
        });
    }
    results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|r| r.unwrap_or_else(|| Err("a PNG strip was not compressed".into())))
        .collect()
}

/// Deflate `input` after `dict`; `head` is what the IDAT chunk has before the
/// deflate data (the zlib header for the first strip).
fn compress_strip(d: &mut Deflate, input: &[u8], dict: &[u8], head: &[u8], last: bool) -> Result<Strip, String> {
    let fail = |e: zlib_rs::DeflateError| format!("PNG compression failed: {}", e.as_str());
    if !dict.is_empty() {
        d.set_dictionary(dict).map_err(fail)?;
    }
    let flush = if last { DeflateFlush::Finish } else { DeflateFlush::SyncFlush };
    // Enough for stored blocks; grown if needed.
    let mut data = vec![0u8; input.len() + input.len() / 8 + 64];
    let (mut read, mut written) = (0usize, 0usize);
    loop {
        let (in0, out0) = (d.total_in(), d.total_out());
        let status = d.compress(&input[read..], &mut data[written..], flush).map_err(fail)?;
        let (r, w) = ((d.total_in() - in0) as usize, (d.total_out() - out0) as usize);
        read += r;
        written += w;
        let done = match status {
            Status::StreamEnd => true,
            _ => !last && read == input.len() && written < data.len(),
        };
        if done {
            break;
        }
        if written == data.len() {
            data.resize(data.len() * 2, 0);
        } else if r == 0 && w == 0 {
            return Err("PNG compression made no progress".into());
        }
    }
    data.truncate(written);
    let crc = zlib_rs::crc32::crc32(zlib_rs::crc32::crc32(zlib_rs::crc32::crc32(0, b"IDAT"), head), &data);
    Ok(Strip { data, crc, adler: zlib_rs::adler32::adler32(1, input), raw_len: input.len() })
}

// ---------------------------------------------------------------------------
// Row filters

const NONE: u8 = 0;
const SUB: u8 = 1;
const UP: u8 = 2;
const AVERAGE: u8 = 3;
const PAETH: u8 = 4;

/// `filter_row`, compiled for some CPU features.
type FilterRow = fn(&[u8], &[u8], usize, &mut [u8]);

/// The fastest `filter_row` for this CPU (all give the same bytes).
fn row_filter() -> FilterRow {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2.
            return |row, prior, bpp, out| unsafe { filter_row_avx2(row, prior, bpp, out) };
        }
    }
    filter_row
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn filter_row_avx2(row: &[u8], prior: &[u8], bpp: usize, out: &mut [u8]) {
    filter_row_inline(row, prior, bpp, out)
}

/// Write to `out` the filter type and the bytes of `row` filtered with the
/// type whose bytes have the least sum of absolute values (ties go to the
/// lower type), `prior` being the row above (zeros for the first row).
fn filter_row(row: &[u8], prior: &[u8], bpp: usize, out: &mut [u8]) {
    filter_row_inline(row, prior, bpp, out)
}

#[inline(always)]
fn filter_row_inline(row: &[u8], prior: &[u8], bpp: usize, out: &mut [u8]) {
    match bpp {
        1 => filter_row_bpp::<1>(row, prior, out),
        2 => filter_row_bpp::<2>(row, prior, out),
        3 => filter_row_bpp::<3>(row, prior, out),
        _ => filter_row_bpp::<4>(row, prior, out),
    }
}

#[inline(always)]
fn filter_row_bpp<const BPP: usize>(row: &[u8], prior: &[u8], out: &mut [u8]) {
    let (kind, dst) = out.split_first_mut().unwrap();
    if row == prior {
        // A repeated row (or an all-zero first row): Up gives only zeros.
        *kind = if prior.iter().all(|&b| b == 0) { NONE } else { UP };
        dst.fill(0);
        return;
    }
    let cost = sums::<BPP>(row, prior);
    let mut best = NONE;
    for f in [SUB, UP, AVERAGE, PAETH] {
        if cost[f as usize] < cost[best as usize] {
            best = f;
        }
    }
    *kind = best;
    apply::<BPP>(best, row, prior, dst);
}

/// |v| of the byte read as signed (0 to 128): min(v, -v) as unsigned bytes.
#[inline(always)]
fn abs8(v: u8) -> u8 {
    v.min(v.wrapping_neg())
}

#[inline(always)]
fn average(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b)) >> 1) as u8
}

#[inline(always)]
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (ia, ib, ic) = (i16::from(a), i16::from(b), i16::from(c));
    let pa = (ib - ic).abs();
    let pb = (ia - ic).abs();
    let pc = (ia + ib - 2 * ic).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Bytes handled per step of the vectorized loops.
const LANES: usize = 32;

/// Steps after which the 16-bit per-lane costs (at most 128 a step) are
/// added to the totals.
const STEPS_PER_FLUSH: usize = 256;

/// The cost (sum of absolute values) of each filter type for `row`.
#[inline(always)]
fn sums<const BPP: usize>(row: &[u8], prior: &[u8]) -> [u64; 5] {
    let n = row.len();
    let head = BPP.min(n);
    let mut total = [0u64; 5];
    // The first pixel has no left neighbor: Sub is None, Paeth is Up.
    for (&x, &b) in row[..head].iter().zip(&prior[..head]) {
        total[NONE as usize] += u64::from(abs8(x));
        total[SUB as usize] += u64::from(abs8(x));
        total[UP as usize] += u64::from(abs8(x.wrapping_sub(b)));
        total[AVERAGE as usize] += u64::from(abs8(x.wrapping_sub(b >> 1)));
        total[PAETH as usize] += u64::from(abs8(x.wrapping_sub(b)));
    }
    if n <= BPP {
        return total;
    }
    let (x, a) = (&row[BPP..], &row[..n - BPP]);
    let (b, c) = (&prior[BPP..], &prior[..n - BPP]);
    let m = x.len();
    let full = m - m % LANES;
    let mut acc = [[0u16; LANES]; 5];
    let mut steps = 0;
    let flush = |acc: &mut [[u16; LANES]; 5], total: &mut [u64; 5]| {
        for (t, lanes) in total.iter_mut().zip(acc.iter_mut()) {
            *t += lanes.iter().map(|&v| u64::from(v)).sum::<u64>();
            *lanes = [0; LANES];
        }
    };
    let mut i = 0;
    while i < full {
        let x: &[u8; LANES] = x[i..i + LANES].try_into().unwrap();
        let a: &[u8; LANES] = a[i..i + LANES].try_into().unwrap();
        let b: &[u8; LANES] = b[i..i + LANES].try_into().unwrap();
        let c: &[u8; LANES] = c[i..i + LANES].try_into().unwrap();
        for l in 0..LANES {
            acc[0][l] += u16::from(abs8(x[l]));
            acc[1][l] += u16::from(abs8(x[l].wrapping_sub(a[l])));
            acc[2][l] += u16::from(abs8(x[l].wrapping_sub(b[l])));
            acc[3][l] += u16::from(abs8(x[l].wrapping_sub(average(a[l], b[l]))));
            acc[4][l] += u16::from(abs8(x[l].wrapping_sub(paeth(a[l], b[l], c[l]))));
        }
        steps += 1;
        if steps == STEPS_PER_FLUSH {
            flush(&mut acc, &mut total);
            steps = 0;
        }
        i += LANES;
    }
    flush(&mut acc, &mut total);
    for i in full..m {
        let (x, a, b, c) = (x[i], a[i], b[i], c[i]);
        total[NONE as usize] += u64::from(abs8(x));
        total[SUB as usize] += u64::from(abs8(x.wrapping_sub(a)));
        total[UP as usize] += u64::from(abs8(x.wrapping_sub(b)));
        total[AVERAGE as usize] += u64::from(abs8(x.wrapping_sub(average(a, b))));
        total[PAETH as usize] += u64::from(abs8(x.wrapping_sub(paeth(a, b, c))));
    }
    total
}

/// Write `row` filtered with `kind` to `dst`.
#[inline(always)]
fn apply<const BPP: usize>(kind: u8, row: &[u8], prior: &[u8], dst: &mut [u8]) {
    let n = row.len();
    let head = BPP.min(n);
    match kind {
        NONE => dst.copy_from_slice(row),
        UP => {
            for ((d, &x), &b) in dst.iter_mut().zip(row).zip(prior) {
                *d = x.wrapping_sub(b);
            }
        }
        _ => {
            for ((d, &x), &b) in dst[..head].iter_mut().zip(&row[..head]).zip(&prior[..head]) {
                *d = match kind {
                    SUB => x,
                    AVERAGE => x.wrapping_sub(b >> 1),
                    _ => x.wrapping_sub(b),
                };
            }
            if n <= BPP {
                return;
            }
            let (x, a) = (&row[BPP..], &row[..n - BPP]);
            let (b, c) = (&prior[BPP..], &prior[..n - BPP]);
            let d = &mut dst[BPP..];
            match kind {
                SUB => {
                    for ((d, &x), &a) in d.iter_mut().zip(x).zip(a) {
                        *d = x.wrapping_sub(a);
                    }
                }
                AVERAGE => {
                    for (((d, &x), &a), &b) in d.iter_mut().zip(x).zip(a).zip(b) {
                        *d = x.wrapping_sub(average(a, b));
                    }
                }
                _ => {
                    let m = x.len();
                    let full = m - m % LANES;
                    let mut i = 0;
                    while i < full {
                        let d: &mut [u8; LANES] = (&mut d[i..i + LANES]).try_into().unwrap();
                        let x: &[u8; LANES] = x[i..i + LANES].try_into().unwrap();
                        let a: &[u8; LANES] = a[i..i + LANES].try_into().unwrap();
                        let b: &[u8; LANES] = b[i..i + LANES].try_into().unwrap();
                        let c: &[u8; LANES] = c[i..i + LANES].try_into().unwrap();
                        for l in 0..LANES {
                            d[l] = x[l].wrapping_sub(paeth(a[l], b[l], c[l]));
                        }
                        i += LANES;
                    }
                    for i in full..m {
                        d[i] = x[i].wrapping_sub(paeth(a[i], b[i], c[i]));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Undo the filters and inflate, checking every chunk's CRC.
    fn decode(png: &[u8]) -> (usize, usize, u8, Vec<u8>, Vec<(Vec<u8>, Vec<u8>)>) {
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        let mut pos = 8;
        let (mut w, mut h, mut color, mut idat, mut chunks) = (0, 0, 0, Vec::new(), Vec::new());
        while pos < png.len() {
            let len = u32::from_be_bytes(png[pos..pos + 4].try_into().unwrap()) as usize;
            let kind = &png[pos + 4..pos + 8];
            let data = &png[pos + 8..pos + 8 + len];
            let crc = u32::from_be_bytes(png[pos + 8 + len..pos + 12 + len].try_into().unwrap());
            assert_eq!(crc, zlib_rs::crc32::crc32(0, &png[pos + 4..pos + 8 + len]));
            match kind {
                b"IHDR" => {
                    w = u32::from_be_bytes(data[..4].try_into().unwrap()) as usize;
                    h = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
                    color = data[9];
                }
                b"IDAT" => idat.extend_from_slice(data),
                b"IEND" => assert_eq!(pos + 12, png.len()),
                _ => chunks.push((kind.to_vec(), data.to_vec())),
            }
            pos += 12 + len;
        }
        let bpp = match color {
            0 => 1,
            4 => 2,
            2 => 3,
            _ => 4,
        };
        let mut raw = vec![0u8; h * (w * bpp + 1)];
        let (out, rc) = zlib_rs::decompress_slice(&mut raw, &idat, zlib_rs::InflateConfig::default());
        assert_eq!(rc, zlib_rs::ReturnCode::Ok);
        assert_eq!(out.len(), raw.len());
        let stride = w * bpp;
        let mut pixels = vec![0u8; h * stride];
        for y in 0..h {
            let (kind, line) = (raw[y * (stride + 1)], &raw[y * (stride + 1) + 1..(y + 1) * (stride + 1)]);
            for i in 0..stride {
                let a = if i >= bpp { pixels[y * stride + i - bpp] } else { 0 };
                let b = if y > 0 { pixels[(y - 1) * stride + i] } else { 0 };
                let c = if y > 0 && i >= bpp { pixels[(y - 1) * stride + i - bpp] } else { 0 };
                let p = match kind {
                    0 => 0,
                    1 => a,
                    2 => b,
                    3 => average(a, b),
                    4 => paeth(a, b, c),
                    _ => panic!("bad filter type {kind}"),
                };
                pixels[y * stride + i] = line[i].wrapping_add(p);
            }
        }
        (w, h, color, pixels, chunks)
    }

    fn image(w: usize, h: usize, channels: usize, seed: u32) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..w * h * channels)
            .map(|i| {
                let (x, y) = ((i / channels) % w, i / channels / w);
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                // Smooth areas, edges, repeated rows and noise.
                match (y / 7) % 4 {
                    0 => (x / 3) as u8,
                    1 => if (x + y) % 11 < 5 { 255 } else { 0 },
                    2 => ((x * y) >> 4) as u8,
                    _ => (state >> 24) as u8,
                }
            })
            .collect()
    }

    #[test]
    fn round_trip() {
        let text = [("Software".to_string(), "test".to_string()), ("Title".to_string(), "\u{3b1} \u{e9}".to_string())];
        for &(w, h) in &[(1, 1), (2, 3), (37, 19), (300, 900), (1500, 200)] {
            for channels in 1..=4 {
                let pixels = image(w, h, channels, (w * h + channels) as u32);
                let mut first = None;
                for (level, strategy, threads) in [(0, 0, 1), (1, 0, 3), (6, 0, 1), (6, 0, 4), (9, 0, 2), (6, 3, 2), (4, 1, 2)] {
                    let opts = Options { level, strategy, dpi: Some((100.0, 100.0)), text: &text, threads };
                    let png = encode(&pixels, w, h, channels, &opts).unwrap();
                    let (dw, dh, color, decoded, chunks) = decode(&png);
                    assert_eq!((dw, dh, color), (w, h, [0, 4, 2, 6][channels - 1]));
                    assert!(decoded == pixels, "{w}x{h}x{channels} level {level}");
                    assert_eq!(chunks[0], (b"tEXt".to_vec(), b"Software\0test".to_vec()));
                    assert_eq!(chunks[1], (b"iTXt".to_vec(), "Title\0\0\0\0\0\u{3b1} \u{e9}".as_bytes().to_vec()));
                    assert_eq!(chunks[2], (b"pHYs".to_vec(), vec![0, 0, 0x0f, 0x61, 0, 0, 0x0f, 0x61, 1]));
                    if (level, strategy) == (6, 0) {
                        // The file does not depend on the number of threads.
                        assert!(first.get_or_insert_with(|| png.clone()) == &png);
                    }
                }
            }
        }
    }

    #[test]
    fn latin1_text() {
        let text = [("Author".to_string(), "Andr\u{e9}".to_string())];
        let opts = Options { level: 6, strategy: 0, dpi: None, text: &text, threads: 1 };
        let png = encode(&[0; 4], 1, 1, 4, &opts).unwrap();
        let (.., chunks) = decode(&png);
        assert_eq!(chunks, vec![(b"tEXt".to_vec(), b"Author\0Andr\xe9".to_vec())]);
        let bad = [("\u{3b1}".to_string(), String::new())];
        assert!(encode(&[0; 4], 1, 1, 4, &Options { text: &bad, ..opts }).is_err());
    }

    #[test]
    fn zlib_headers() {
        for level in 0..=9 {
            let h = zlib_header(level, Strategy::Default);
            assert_eq!(u16::from_be_bytes(h) % 31, 0);
        }
        assert_eq!(zlib_header(6, Strategy::Default), [0x78, 0x9c]);
        assert_eq!(zlib_header(9, Strategy::Filtered), [0x78, 0xda]);
        assert_eq!(zlib_header(1, Strategy::Default), [0x78, 0x01]);
        assert_eq!(zlib_header(6, Strategy::Rle), [0x78, 0x01]);
    }
}
