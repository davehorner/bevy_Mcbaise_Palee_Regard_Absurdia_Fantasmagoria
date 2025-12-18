//! Forward kinematics over the skeleton tree.

use anyhow::{Result, bail};

use crate::data::reference::TensorData;
use crate::util::math::{cross3, dot3, invert_rigid_mat4, mat4_mul, normalize3, rotvec_to_rotmat};

/// Compute posed bone transforms using the `root_relative_world` parameterization
/// used in the Python reference (`RiggedModelWithLinearBlendShapes.forward`).
///
/// Returns `(bone_poses, bone_transforms)` where:
/// - `bone_poses` are global posed matrices [B,J,4,4]
/// - `bone_transforms` are posed * rest_inv [B,J,4,4] (used by skinning)
pub fn forward_root_relative_world(
    rest_bone_poses: &TensorData<f64>,
    pose_parameters: &TensorData<f64>,
    bone_parents: &[i64],
) -> Result<(TensorData<f64>, TensorData<f64>)> {
    let batch = rest_bone_poses.shape[0];
    let bones = rest_bone_poses.shape[1];
    if pose_parameters.shape.len() != 4
        || rest_bone_poses.shape.len() != 4
        || pose_parameters.shape[0] != batch
        || pose_parameters.shape[1] != bones
    {
        bail!("rest_bone_poses and pose_parameters must be [B,J,4,4]");
    }
    if bone_parents.len() != bones {
        bail!("bone_parents length mismatch");
    }

    // Precompute rest inverses.
    let mut rest_inv = vec![[0.0f64; 16]; batch * bones];
    for b in 0..batch {
        for j in 0..bones {
            let rest = slice_mat4(rest_bone_poses, b, j)?;
            rest_inv[b * bones + j] = invert_rigid_mat4(&rest);
        }
    }

    let mut bone_poses = TensorData {
        shape: vec![batch, bones, 4, 4],
        data: vec![0.0; batch * bones * 16],
    };
    let mut bone_transforms = TensorData {
        shape: vec![batch, bones, 4, 4],
        data: vec![0.0; batch * bones * 16],
    };

    for b in 0..batch {
        // Clone deltas for in-place adjustment of root.
        let mut deltas = vec![[0.0f64; 16]; bones];
        for (j, delta) in deltas.iter_mut().enumerate().take(bones) {
            *delta = slice_mat4(pose_parameters, b, j)?;
        }

        // root_relative_world adjustment
        let rest_root = slice_mat4(rest_bone_poses, b, 0)?;
        let rest_root_rot = [
            rest_root[0],
            rest_root[1],
            rest_root[2],
            0.0,
            rest_root[4],
            rest_root[5],
            rest_root[6],
            0.0,
            rest_root[8],
            rest_root[9],
            rest_root[10],
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ];
        deltas[0] = mat4_mul(&deltas[0], &rest_root_rot);
        let base_transform = invert_rigid_mat4(&rest_root);

        for j in 0..bones {
            let t = slice_mat4(rest_bone_poses, b, j)?;
            let td = mat4_mul(&t, &deltas[j]);
            let pose = if bone_parents[j] < 0 {
                mat4_mul(&base_transform, &td)
            } else {
                let parent = bone_parents[j] as usize;
                let parent_tf = slice_mat4(&bone_transforms, b, parent)?;
                mat4_mul(&parent_tf, &td)
            };
            let transform = mat4_mul(&pose, &rest_inv[b * bones + j]);
            write_mat4(&mut bone_poses, b, j, &pose)?;
            write_mat4(&mut bone_transforms, b, j, &transform)?;
        }
    }

    Ok((bone_poses, bone_transforms))
}

/// Compute rest bone poses from heads/tails and roll matrices (Blender convention).
pub fn rest_bone_poses_from_heads_tails(
    heads: &TensorData<f64>,             // [B,J,3]
    tails: &TensorData<f64>,             // [B,J,3]
    bone_rolls_rotmat: &TensorData<f64>, // [1,J,3,3]
) -> Result<TensorData<f64>> {
    let batch = heads.shape[0];
    let bones = heads.shape[1];
    if heads.shape != tails.shape || heads.shape.len() != 3 || heads.shape[2] != 3 {
        bail!("heads/tails must be [B,J,3]");
    }
    if bone_rolls_rotmat.shape.len() != 4
        || bone_rolls_rotmat.shape[1] != bones
        || bone_rolls_rotmat.shape[2] != 3
        || bone_rolls_rotmat.shape[3] != 3
    {
        bail!("bone_rolls_rotmat must be [1,J,3,3]");
    }
    let y_axis = [0.0, 1.0, 0.0];
    let degenerate = [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0];
    let epsilon = 0.1;

    let mut out = TensorData {
        shape: vec![batch, bones, 4, 4],
        data: vec![0.0; batch * bones * 16],
    };

    for b in 0..batch {
        for j in 0..bones {
            let head = read_vec3(heads, b, j)?;
            let tail = read_vec3(tails, b, j)?;
            let vec = [tail[0] - head[0], tail[1] - head[1], tail[2] - head[2]];
            let y = normalize3(vec);
            let cross = cross3(y, y_axis);
            let dot = dot3(y, y_axis);
            let cross_norm = (dot3(cross, cross)).sqrt();
            let angle = cross_norm.atan2(dot);
            let axis = if cross_norm == 0.0 {
                [0.0, 0.0, 0.0]
            } else {
                [
                    cross[0] / cross_norm,
                    cross[1] / cross_norm,
                    cross[2] / cross_norm,
                ]
            };
            let mut r = rotvec_to_rotmat([-angle * axis[0], -angle * axis[1], -angle * axis[2]]);
            let valid = (dot3(axis, axis) - 1.0).abs() < epsilon;
            if !valid {
                r = degenerate;
            }
            let roll = read_mat3(bone_rolls_rotmat, 0, j)?;
            let r_final = mat3_mul(&r, &roll);
            let mut h = [0.0f64; 16];
            h[0] = r_final[0];
            h[1] = r_final[1];
            h[2] = r_final[2];
            h[4] = r_final[3];
            h[5] = r_final[4];
            h[6] = r_final[5];
            h[8] = r_final[6];
            h[9] = r_final[7];
            h[10] = r_final[8];
            h[3] = head[0];
            h[7] = head[1];
            h[11] = head[2];
            h[15] = 1.0;
            write_mat4(&mut out, b, j, &h)?;
        }
    }
    Ok(out)
}

fn slice_mat4(t: &TensorData<f64>, b: usize, j: usize) -> Result<[f64; 16]> {
    let idx = (b * t.shape[1] + j) * 16;
    let slice = t
        .data
        .get(idx..idx + 16)
        .ok_or_else(|| anyhow::anyhow!("mat4 slice OOB"))?;
    let mut out = [0.0; 16];
    out.copy_from_slice(slice);
    Ok(out)
}

fn write_mat4(t: &mut TensorData<f64>, b: usize, j: usize, m: &[f64; 16]) -> Result<()> {
    let idx = (b * t.shape[1] + j) * 16;
    let slice = t
        .data
        .get_mut(idx..idx + 16)
        .ok_or_else(|| anyhow::anyhow!("mat4 write OOB"))?;
    slice.copy_from_slice(m);
    Ok(())
}

fn read_vec3(t: &TensorData<f64>, b: usize, j: usize) -> Result<[f64; 3]> {
    let idx = (b * t.shape[1] + j) * 3;
    let slice = t
        .data
        .get(idx..idx + 3)
        .ok_or_else(|| anyhow::anyhow!("vec3 slice OOB"))?;
    Ok([slice[0], slice[1], slice[2]])
}

fn read_mat3(t: &TensorData<f64>, b: usize, j: usize) -> Result<[f64; 9]> {
    let idx = (b * t.shape[1] + j) * 9;
    let slice = t
        .data
        .get(idx..idx + 9)
        .ok_or_else(|| anyhow::anyhow!("mat3 slice OOB"))?;
    let mut out = [0.0; 9];
    out.copy_from_slice(slice);
    Ok(out)
}

fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut out = [0.0; 9];
    for row in 0..3 {
        for col in 0..3 {
            let a0 = a[row * 3];
            let a1 = a[row * 3 + 1];
            let a2 = a[row * 3 + 2];
            let b0 = b[col];
            let b1 = b[3 + col];
            let b2 = b[6 + col];
            out[row * 3 + col] = a0 * b0 + a1 * b1 + a2 * b2;
        }
    }
    out
}
