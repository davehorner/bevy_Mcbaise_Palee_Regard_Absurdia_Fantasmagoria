/// Invert a rigid transform (rotation + translation) stored as a 4x4 row-major matrix.
///
/// Assumes the last row is `[0, 0, 0, 1]` and the upper-left 3x3 is orthonormal.
pub fn invert_rigid_mat4(m: &[f64; 16]) -> [f64; 16] {
    // Rotation transpose
    let r = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
    let t = [m[3], m[7], m[11]];

    // r^T
    let rt = [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]];

    // -r^T * t
    let tx = -(rt[0] * t[0] + rt[1] * t[1] + rt[2] * t[2]);
    let ty = -(rt[3] * t[0] + rt[4] * t[1] + rt[5] * t[2]);
    let tz = -(rt[6] * t[0] + rt[7] * t[1] + rt[8] * t[2]);

    [
        rt[0], rt[1], rt[2], tx, rt[3], rt[4], rt[5], ty, rt[6], rt[7], rt[8], tz, 0.0, 0.0, 0.0,
        1.0,
    ]
}

/// Multiply two 4x4 matrices (row-major).
pub fn mat4_mul(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut out = [0.0; 16];
    for row in 0..4 {
        for col in 0..4 {
            let mut acc = 0.0;
            for k in 0..4 {
                acc += a[row * 4 + k] * b[k * 4 + col];
            }
            out[row * 4 + col] = acc;
        }
    }
    out
}

/// Multiply a 4x4 matrix (row-major) by a vec4.
pub fn mat4_mul_vec4(m: &[f64; 16], v: [f64; 4]) -> [f64; 4] {
    let mut out = [0.0; 4];
    for (row, slot) in out.iter_mut().enumerate() {
        let base = row * 4;
        *slot = m[base] * v[0] + m[base + 1] * v[1] + m[base + 2] * v[2] + m[base + 3] * v[3];
    }
    out
}

pub fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

pub fn normalize3(a: [f64; 3]) -> [f64; 3] {
    let n = norm3(a);
    if n == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        [a[0] / n, a[1] / n, a[2] / n]
    }
}

/// Rodrigues' rotation formula from rotation vector (axis * angle).
pub fn rotvec_to_rotmat(rv: [f64; 3]) -> [f64; 9] {
    let theta = norm3(rv);
    if theta == 0.0 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let k = [rv[0] / theta, rv[1] / theta, rv[2] / theta];
    let (s, c) = theta.sin_cos();
    let v = 1.0 - c;
    let (kx, ky, kz) = (k[0], k[1], k[2]);
    [
        kx * kx * v + c,
        kx * ky * v - kz * s,
        kx * kz * v + ky * s,
        ky * kx * v + kz * s,
        ky * ky * v + c,
        ky * kz * v - kx * s,
        kz * kx * v - ky * s,
        kz * ky * v + kx * s,
        kz * kz * v + c,
    ]
}
