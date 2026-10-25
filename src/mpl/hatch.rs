//! Port of `matplotlib.hatch`: hatch patterns as unit-square paths.

use super::path::{self as mpath, linspace, Path, CLOSEPOLY, LINETO, MOVETO};

/// `np.linspace(start, stop, num, endpoint=False)` and its step.
fn linspace_noend(start: f64, stop: f64, num: usize) -> (Vec<f64>, f64) {
    let delta = stop - start;
    let step = delta / num as f64;
    ((0..num).map(|i| i as f64 * step + start).collect(), step)
}

fn count(h: &str, chars: &[char]) -> usize {
    h.chars().filter(|c| chars.contains(c)).count()
}

struct Shapes {
    num_rows: usize,
    size: f64,
    filled: bool,
    verts: Vec<[f64; 2]>,
    codes: Vec<u8>,
}

impl Shapes {
    fn emit(&self, v: &mut Vec<[f64; 2]>, c: &mut Vec<u8>) {
        if self.num_rows == 0 {
            return;
        }
        let offset = 1.0 / self.num_rows as f64;
        let mut sv: Vec<[f64; 2]> = self.verts.iter().map(|p| [p[0] * offset * self.size, p[1] * offset * self.size]).collect();
        let mut sc = self.codes.clone();
        if !self.filled {
            let rev: Vec<[f64; 2]> = sv.iter().rev().map(|p| [p[0] * 0.9, p[1] * 0.9]).collect();
            sv.extend(rev);
            let c2 = sc.clone();
            sc.extend(c2);
        }
        for row in 0..=self.num_rows {
            let cols = if row % 2 == 0 {
                linspace(0.0, 1.0, self.num_rows + 1)
            } else {
                linspace(offset / 2.0, 1.0 - offset / 2.0, self.num_rows)
            };
            let row_pos = row as f64 * offset;
            for col in cols {
                for p in &sv {
                    v.push([p[0] + col, p[1] + row_pos]);
                }
                c.extend_from_slice(&sc);
            }
        }
    }
}

/// `hatch.get_path(hatchpattern, density)`; `None` for an empty pattern.
pub fn get_path(h: &str, density: usize) -> Option<Path> {
    let mut v: Vec<[f64; 2]> = Vec::new();
    let mut c: Vec<u8> = Vec::new();
    // HorizontalHatch
    let n = count(h, &['-', '+']) * density;
    if n > 0 {
        let (steps, step) = linspace_noend(0.0, 1.0, n);
        for s in steps {
            let s = s + step / 2.0;
            v.push([0.0, s]);
            v.push([1.0, s]);
            c.push(MOVETO);
            c.push(LINETO);
        }
    }
    // VerticalHatch
    let n = count(h, &['|', '+']) * density;
    if n > 0 {
        let (steps, step) = linspace_noend(0.0, 1.0, n);
        for s in steps {
            let s = s + step / 2.0;
            v.push([s, 0.0]);
            v.push([s, 1.0]);
            c.push(MOVETO);
            c.push(LINETO);
        }
    }
    // NorthEastHatch
    let n = count(h, &['/', 'x', 'X']) * density;
    if n > 0 {
        for s in linspace(-0.5, 0.5, n + 1) {
            v.push([0.0 + s, 0.0 - s]);
            v.push([1.0 + s, 1.0 - s]);
            c.push(MOVETO);
            c.push(LINETO);
        }
    }
    // SouthEastHatch
    let n = count(h, &['\\', 'x', 'X']) * density;
    if n > 0 {
        for s in linspace(-0.5, 0.5, n + 1) {
            v.push([0.0 + s, 1.0 + s]);
            v.push([1.0 + s, 0.0 + s]);
            c.push(MOVETO);
            c.push(LINETO);
        }
    }
    let circle = mpath::unit_circle();
    let ccodes = circle.codes.clone().unwrap();
    for (ch, size, filled) in [('o', 0.2, false), ('O', 0.35, false), ('.', 0.1, true)] {
        Shapes { num_rows: count(h, &[ch]) * density, size, filled, verts: circle.vertices.clone(), codes: ccodes.clone() }
            .emit(&mut v, &mut c);
    }
    let star = mpath::unit_regular_star(5, 0.5);
    let mut scodes = vec![LINETO; star.vertices.len()];
    scodes[0] = MOVETO;
    let last = scodes.len() - 1;
    scodes[last] = CLOSEPOLY;
    Shapes { num_rows: count(h, &['*']) * density, size: 1.0 / 3.0, filled: true, verts: star.vertices.clone(), codes: scodes }
        .emit(&mut v, &mut c);
    if v.is_empty() {
        return None;
    }
    Some(Path::new(v, Some(c)))
}
