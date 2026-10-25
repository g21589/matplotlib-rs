//! Port of `agg::curve3_div`, `agg::curve4_div` and `agg::conv_curve`.

use super::basics::*;

const CURVE_COLLINEARITY_EPSILON: f64 = 1e-30;
const CURVE_ANGLE_TOLERANCE_EPSILON: f64 = 0.01;
const CURVE_RECURSION_LIMIT: u32 = 32;

#[derive(Default)]
pub struct Curve3Div {
    approximation_scale: f64,
    distance_tolerance_square: f64,
    angle_tolerance: f64,
    count: usize,
    points: Vec<[f64; 2]>,
}

impl Curve3Div {
    pub fn new() -> Self {
        Curve3Div { approximation_scale: 1.0, ..Default::default() }
    }

    pub fn reset(&mut self) {
        self.points.clear();
        self.count = 0;
    }

    pub fn init(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64) {
        self.points.clear();
        self.distance_tolerance_square = 0.5 / self.approximation_scale;
        self.distance_tolerance_square *= self.distance_tolerance_square;
        self.points.push([x1, y1]);
        self.recursive_bezier(x1, y1, x2, y2, x3, y3, 0);
        self.points.push([x3, y3]);
        self.count = 0;
    }

    #[allow(clippy::too_many_arguments)]
    fn recursive_bezier(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, level: u32) {
        if level > CURVE_RECURSION_LIMIT {
            return;
        }

        let x12 = (x1 + x2) / 2.0;
        let y12 = (y1 + y2) / 2.0;
        let x23 = (x2 + x3) / 2.0;
        let y23 = (y2 + y3) / 2.0;
        let x123 = (x12 + x23) / 2.0;
        let y123 = (y12 + y23) / 2.0;

        let dx = x3 - x1;
        let dy = y3 - y1;
        let mut d = ((x2 - x3) * dy - (y2 - y3) * dx).abs();

        if d > CURVE_COLLINEARITY_EPSILON {
            // Regular case
            if d * d <= self.distance_tolerance_square * (dx * dx + dy * dy) {
                if self.angle_tolerance < CURVE_ANGLE_TOLERANCE_EPSILON {
                    self.points.push([x123, y123]);
                    return;
                }
                let mut da = ((y3 - y2).atan2(x3 - x2) - (y2 - y1).atan2(x2 - x1)).abs();
                if da >= PI {
                    da = 2.0 * PI - da;
                }
                if da < self.angle_tolerance {
                    self.points.push([x123, y123]);
                    return;
                }
            }
        } else {
            // Collinear case
            let da = dx * dx + dy * dy;
            if da == 0.0 {
                d = calc_sq_distance(x1, y1, x2, y2);
            } else {
                d = ((x2 - x1) * dx + (y2 - y1) * dy) / da;
                if d > 0.0 && d < 1.0 {
                    // Simple collinear case, 1---2---3
                    return;
                }
                if d <= 0.0 {
                    d = calc_sq_distance(x2, y2, x1, y1);
                } else if d >= 1.0 {
                    d = calc_sq_distance(x2, y2, x3, y3);
                } else {
                    d = calc_sq_distance(x2, y2, x1 + d * dx, y1 + d * dy);
                }
            }
            if d < self.distance_tolerance_square {
                self.points.push([x2, y2]);
                return;
            }
        }

        self.recursive_bezier(x1, y1, x12, y12, x123, y123, level + 1);
        self.recursive_bezier(x123, y123, x23, y23, x3, y3, level + 1);
    }

    #[inline]
    pub fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        if self.count >= self.points.len() {
            return PATH_CMD_STOP;
        }
        let p = self.points[self.count];
        self.count += 1;
        *x = p[0];
        *y = p[1];
        if self.count == 1 {
            PATH_CMD_MOVE_TO
        } else {
            PATH_CMD_LINE_TO
        }
    }
}

#[derive(Default)]
pub struct Curve4Div {
    approximation_scale: f64,
    distance_tolerance_square: f64,
    angle_tolerance: f64,
    cusp_limit: f64,
    count: usize,
    points: Vec<[f64; 2]>,
}

impl Curve4Div {
    pub fn new() -> Self {
        Curve4Div { approximation_scale: 1.0, ..Default::default() }
    }

    pub fn reset(&mut self) {
        self.points.clear();
        self.count = 0;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn init(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, x4: f64, y4: f64) {
        self.points.clear();
        self.distance_tolerance_square = 0.5 / self.approximation_scale;
        self.distance_tolerance_square *= self.distance_tolerance_square;
        self.points.push([x1, y1]);
        self.recursive_bezier(x1, y1, x2, y2, x3, y3, x4, y4, 0);
        self.points.push([x4, y4]);
        self.count = 0;
    }

    #[allow(clippy::too_many_arguments)]
    fn recursive_bezier(
        &mut self,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        x3: f64,
        y3: f64,
        x4: f64,
        y4: f64,
        level: u32,
    ) {
        if level > CURVE_RECURSION_LIMIT {
            return;
        }

        let x12 = (x1 + x2) / 2.0;
        let y12 = (y1 + y2) / 2.0;
        let x23 = (x2 + x3) / 2.0;
        let y23 = (y2 + y3) / 2.0;
        let x34 = (x3 + x4) / 2.0;
        let y34 = (y3 + y4) / 2.0;
        let x123 = (x12 + x23) / 2.0;
        let y123 = (y12 + y23) / 2.0;
        let x234 = (x23 + x34) / 2.0;
        let y234 = (y23 + y34) / 2.0;
        let x1234 = (x123 + x234) / 2.0;
        let y1234 = (y123 + y234) / 2.0;

        let dx = x4 - x1;
        let dy = y4 - y1;

        let mut d2 = ((x2 - x4) * dy - (y2 - y4) * dx).abs();
        let mut d3 = ((x3 - x4) * dy - (y3 - y4) * dx).abs();
        let mut da1;
        let mut da2;
        let mut k;

        match (((d2 > CURVE_COLLINEARITY_EPSILON) as u32) << 1) + (d3 > CURVE_COLLINEARITY_EPSILON) as u32 {
            0 => {
                // All collinear OR p1==p4
                k = dx * dx + dy * dy;
                if k == 0.0 {
                    d2 = calc_sq_distance(x1, y1, x2, y2);
                    d3 = calc_sq_distance(x4, y4, x3, y3);
                } else {
                    k = 1.0 / k;
                    da1 = x2 - x1;
                    da2 = y2 - y1;
                    d2 = k * (da1 * dx + da2 * dy);
                    da1 = x3 - x1;
                    da2 = y3 - y1;
                    d3 = k * (da1 * dx + da2 * dy);
                    if d2 > 0.0 && d2 < 1.0 && d3 > 0.0 && d3 < 1.0 {
                        // Simple collinear case, 1---2---3---4
                        return;
                    }
                    if d2 <= 0.0 {
                        d2 = calc_sq_distance(x2, y2, x1, y1);
                    } else if d2 >= 1.0 {
                        d2 = calc_sq_distance(x2, y2, x4, y4);
                    } else {
                        d2 = calc_sq_distance(x2, y2, x1 + d2 * dx, y1 + d2 * dy);
                    }

                    if d3 <= 0.0 {
                        d3 = calc_sq_distance(x3, y3, x1, y1);
                    } else if d3 >= 1.0 {
                        d3 = calc_sq_distance(x3, y3, x4, y4);
                    } else {
                        d3 = calc_sq_distance(x3, y3, x1 + d3 * dx, y1 + d3 * dy);
                    }
                }
                if d2 > d3 {
                    if d2 < self.distance_tolerance_square {
                        self.points.push([x2, y2]);
                        return;
                    }
                } else if d3 < self.distance_tolerance_square {
                    self.points.push([x3, y3]);
                    return;
                }
            }
            1 => {
                // p1,p2,p4 are collinear, p3 is significant
                if d3 * d3 <= self.distance_tolerance_square * (dx * dx + dy * dy) {
                    if self.angle_tolerance < CURVE_ANGLE_TOLERANCE_EPSILON {
                        self.points.push([x23, y23]);
                        return;
                    }
                    da1 = ((y4 - y3).atan2(x4 - x3) - (y3 - y2).atan2(x3 - x2)).abs();
                    if da1 >= PI {
                        da1 = 2.0 * PI - da1;
                    }
                    if da1 < self.angle_tolerance {
                        self.points.push([x2, y2]);
                        self.points.push([x3, y3]);
                        return;
                    }
                    if self.cusp_limit != 0.0 && da1 > self.cusp_limit {
                        self.points.push([x3, y3]);
                        return;
                    }
                }
            }
            2 => {
                // p1,p3,p4 are collinear, p2 is significant
                if d2 * d2 <= self.distance_tolerance_square * (dx * dx + dy * dy) {
                    if self.angle_tolerance < CURVE_ANGLE_TOLERANCE_EPSILON {
                        self.points.push([x23, y23]);
                        return;
                    }
                    da1 = ((y3 - y2).atan2(x3 - x2) - (y2 - y1).atan2(x2 - x1)).abs();
                    if da1 >= PI {
                        da1 = 2.0 * PI - da1;
                    }
                    if da1 < self.angle_tolerance {
                        self.points.push([x2, y2]);
                        self.points.push([x3, y3]);
                        return;
                    }
                    if self.cusp_limit != 0.0 && da1 > self.cusp_limit {
                        self.points.push([x2, y2]);
                        return;
                    }
                }
            }
            _ => {
                // Regular case
                if (d2 + d3) * (d2 + d3) <= self.distance_tolerance_square * (dx * dx + dy * dy) {
                    if self.angle_tolerance < CURVE_ANGLE_TOLERANCE_EPSILON {
                        self.points.push([x23, y23]);
                        return;
                    }
                    k = (y3 - y2).atan2(x3 - x2);
                    da1 = (k - (y2 - y1).atan2(x2 - x1)).abs();
                    da2 = ((y4 - y3).atan2(x4 - x3) - k).abs();
                    if da1 >= PI {
                        da1 = 2.0 * PI - da1;
                    }
                    if da2 >= PI {
                        da2 = 2.0 * PI - da2;
                    }
                    if da1 + da2 < self.angle_tolerance {
                        self.points.push([x23, y23]);
                        return;
                    }
                    if self.cusp_limit != 0.0 {
                        if da1 > self.cusp_limit {
                            self.points.push([x2, y2]);
                            return;
                        }
                        if da2 > self.cusp_limit {
                            self.points.push([x3, y3]);
                            return;
                        }
                    }
                }
            }
        }

        self.recursive_bezier(x1, y1, x12, y12, x123, y123, x1234, y1234, level + 1);
        self.recursive_bezier(x1234, y1234, x234, y234, x34, y34, x4, y4, level + 1);
    }

    #[inline]
    pub fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        if self.count >= self.points.len() {
            return PATH_CMD_STOP;
        }
        let p = self.points[self.count];
        self.count += 1;
        *x = p[0];
        *y = p[1];
        if self.count == 1 {
            PATH_CMD_MOVE_TO
        } else {
            PATH_CMD_LINE_TO
        }
    }
}

/// `agg::conv_curve` with the default (subdivision) curve approximations.
pub struct ConvCurve<S: VertexSource> {
    pub source: S,
    last_x: f64,
    last_y: f64,
    curve3: Curve3Div,
    curve4: Curve4Div,
}

impl<S: VertexSource> ConvCurve<S> {
    pub fn new(source: S) -> Self {
        ConvCurve { source, last_x: 0.0, last_y: 0.0, curve3: Curve3Div::new(), curve4: Curve4Div::new() }
    }
}

impl<S: VertexSource> VertexSource for ConvCurve<S> {
    fn rewind(&mut self, path_id: u32) {
        self.source.rewind(path_id);
        self.last_x = 0.0;
        self.last_y = 0.0;
        self.curve3.reset();
        self.curve4.reset();
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        if !is_stop(self.curve3.vertex(x, y)) {
            self.last_x = *x;
            self.last_y = *y;
            return PATH_CMD_LINE_TO;
        }

        if !is_stop(self.curve4.vertex(x, y)) {
            self.last_x = *x;
            self.last_y = *y;
            return PATH_CMD_LINE_TO;
        }

        let (mut ct2_x, mut ct2_y, mut end_x, mut end_y) = (0.0, 0.0, 0.0, 0.0);

        let mut cmd = self.source.vertex(x, y);
        match cmd {
            PATH_CMD_CURVE3 => {
                self.source.vertex(&mut end_x, &mut end_y);
                self.curve3.init(self.last_x, self.last_y, *x, *y, end_x, end_y);
                self.curve3.vertex(x, y); // First call returns path_cmd_move_to
                self.curve3.vertex(x, y); // This is the first vertex of the curve
                cmd = PATH_CMD_LINE_TO;
            }
            PATH_CMD_CURVE4 => {
                self.source.vertex(&mut ct2_x, &mut ct2_y);
                self.source.vertex(&mut end_x, &mut end_y);
                self.curve4.init(self.last_x, self.last_y, *x, *y, ct2_x, ct2_y, end_x, end_y);
                self.curve4.vertex(x, y); // First call returns path_cmd_move_to
                self.curve4.vertex(x, y); // This is the first vertex of the curve
                cmd = PATH_CMD_LINE_TO;
            }
            _ => {}
        }
        self.last_x = *x;
        self.last_y = *y;
        cmd
    }
}
