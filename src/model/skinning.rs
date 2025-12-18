//! Linear blend skinning helpers.

use anyhow::{Result, bail};

use crate::data::reference::TensorData;
use crate::util::math::{invert_rigid_mat4, mat4_mul, mat4_mul_vec4};

/// Apply linear blend skinning to rest vertices.
///
/// - `rest_vertices`: shape [B, V, 3]
/// - `bone_poses`: shape [B, J, 4, 4] (global posed)
/// - `rest_bone_poses`: shape [B, J, 4, 4] (global rest)
/// - `vertex_indices`: shape [V, K]
/// - `vertex_weights`: shape [V, K]
pub fn linear_blend_skinning(
    rest_vertices: &TensorData<f64>,
    bone_poses: &TensorData<f64>,
    rest_bone_poses: &TensorData<f64>,
    vertex_indices: &TensorData<i64>,
    vertex_weights: &TensorData<f64>,
) -> Result<TensorData<f64>> {
    // Shapes
    if rest_vertices.shape.len() != 3 {
        bail!("rest_vertices must be [B,V,3]");
    }
    if bone_poses.shape.len() != 4 || rest_bone_poses.shape.len() != 4 {
        bail!("bone pose tensors must be [B,J,4,4]");
    }
    if vertex_indices.shape != vertex_weights.shape {
        bail!("vertex index/weight shapes must match");
    }
    let batch = rest_vertices.shape[0];
    let verts = rest_vertices.shape[1];
    let bones = bone_poses.shape[1];
    let k = vertex_indices.shape[1];

    if bone_poses.shape[0] != batch || rest_bone_poses.shape[0] != batch {
        bail!("bone pose batch mismatches rest vertices batch");
    }
    if bone_poses.shape[1] != rest_bone_poses.shape[1] {
        bail!("bone pose count mismatch");
    }
    if bone_poses.shape[2] != 4
        || bone_poses.shape[3] != 4
        || rest_bone_poses.shape[2] != 4
        || rest_bone_poses.shape[3] != 4
    {
        bail!("bone matrices must be 4x4");
    }

    // Precompute bone transforms = posed * inv(rest) per batch/bone.
    let mut bone_transforms = vec![[0.0f64; 16]; batch * bones];
    for b in 0..batch {
        for j in 0..bones {
            let idx = b * bones + j;
            let rest = slice_mat4(rest_bone_poses, b, j)?;
            let posed = slice_mat4(bone_poses, b, j)?;
            let rest_inv = invert_rigid_mat4(&rest);
            bone_transforms[idx] = mat4_mul(&posed, &rest_inv);
        }
    }

    let mut out = TensorData {
        shape: vec![batch, verts, 3],
        data: vec![0.0; batch * verts * 3],
    };

    for b in 0..batch {
        for v in 0..verts {
            let pos = get_vertex(rest_vertices, b, v)?;
            let mut accum = [0.0f64; 3];
            for ki in 0..k {
                let weight = vertex_weights.data[v * k + ki];
                if weight == 0.0 {
                    continue;
                }
                let bone_idx = vertex_indices.data[v * k + ki];
                if bone_idx < 0 || bone_idx as usize >= bones {
                    bail!("bone index out of range: {}", bone_idx);
                }
                let tf = bone_transforms[b * bones + bone_idx as usize];
                let res = mat4_mul_vec4(&tf, [pos[0], pos[1], pos[2], 1.0]);
                accum[0] += weight * res[0];
                accum[1] += weight * res[1];
                accum[2] += weight * res[2];
            }
            let out_idx = (b * verts + v) * 3;
            out.data[out_idx] = accum[0];
            out.data[out_idx + 1] = accum[1];
            out.data[out_idx + 2] = accum[2];
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

fn get_vertex(t: &TensorData<f64>, b: usize, v: usize) -> Result<[f64; 3]> {
    let idx = (b * t.shape[1] + v) * 3;
    let slice = t
        .data
        .get(idx..idx + 3)
        .ok_or_else(|| anyhow::anyhow!("vertex slice OOB"))?;
    let mut out = [0.0; 3];
    out.copy_from_slice(slice);
    Ok(out)
}
