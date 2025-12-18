//! Blendshape application in the rest pose.

use anyhow::{Result, bail};

use crate::data::reference::TensorData;

/// Apply blendshapes to base vertices.
pub fn apply_blendshapes(
    template_vertices: &TensorData<f64>, // [V,3]
    blendshapes: &TensorData<f64>,       // [N,V,3]
    weights: &TensorData<f64>,           // [B,N]
) -> Result<TensorData<f64>> {
    if template_vertices.shape.len() != 2 || template_vertices.shape[1] != 3 {
        bail!("template_vertices must be [V,3]");
    }
    if blendshapes.shape.len() != 3 || blendshapes.shape[1] != template_vertices.shape[0] {
        bail!("blendshapes must be [N,V,3]");
    }
    if weights.shape.len() != 2 || weights.shape[1] != blendshapes.shape[0] {
        bail!("weights must be [B,N]");
    }
    let batch = weights.shape[0];
    let verts = template_vertices.shape[0];
    let blends = blendshapes.shape[0];
    let mut out = TensorData {
        shape: vec![batch, verts, 3],
        data: vec![0.0; batch * verts * 3],
    };
    for b in 0..batch {
        for v in 0..verts {
            let base = [
                template_vertices.data[v * 3],
                template_vertices.data[v * 3 + 1],
                template_vertices.data[v * 3 + 2],
            ];
            let mut acc = base;
            for n in 0..blends {
                let w = weights.data[b * blends + n];
                if w == 0.0 {
                    continue;
                }
                let idx = (n * verts + v) * 3;
                acc[0] += w * blendshapes.data[idx];
                acc[1] += w * blendshapes.data[idx + 1];
                acc[2] += w * blendshapes.data[idx + 2];
            }
            let out_idx = (b * verts + v) * 3;
            out.data[out_idx] = acc[0];
            out.data[out_idx + 1] = acc[1];
            out.data[out_idx + 2] = acc[2];
        }
    }
    Ok(out)
}

/// Apply blendshapes to bone heads/tails.
pub fn apply_bone_blendshapes(
    template: &TensorData<f64>, // [J,3]
    deltas: &TensorData<f64>,   // [N,J,3]
    weights: &TensorData<f64>,  // [B,N]
) -> Result<TensorData<f64>> {
    if template.shape.len() != 2 || template.shape[1] != 3 {
        bail!("template bone points must be [J,3]");
    }
    if deltas.shape.len() != 3 || deltas.shape[1] != template.shape[0] {
        bail!("bone deltas must be [N,J,3]");
    }
    if weights.shape.len() != 2 || weights.shape[1] != deltas.shape[0] {
        bail!("weights must be [B,N]");
    }
    let batch = weights.shape[0];
    let joints = template.shape[0];
    let blends = deltas.shape[0];
    let mut out = TensorData {
        shape: vec![batch, joints, 3],
        data: vec![0.0; batch * joints * 3],
    };
    for b in 0..batch {
        for j in 0..joints {
            let base = [
                template.data[j * 3],
                template.data[j * 3 + 1],
                template.data[j * 3 + 2],
            ];
            let mut acc = base;
            for n in 0..blends {
                let w = weights.data[b * blends + n];
                if w == 0.0 {
                    continue;
                }
                let idx = (n * joints + j) * 3;
                acc[0] += w * deltas.data[idx];
                acc[1] += w * deltas.data[idx + 1];
                acc[2] += w * deltas.data[idx + 2];
            }
            let out_idx = (b * joints + j) * 3;
            out.data[out_idx] = acc[0];
            out.data[out_idx + 1] = acc[1];
            out.data[out_idx + 2] = acc[2];
        }
    }
    Ok(out)
}
