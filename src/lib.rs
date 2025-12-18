//! burn_human: parametric human forward model (Anny) for Burn.
//!
//! This crate is currently a scaffold. It mirrors the module layout needed
//! to port the Python implementation stage by stage while keeping tests and
//! assets aligned with burn_depth / burn_dino conventions.

pub mod data;
pub mod model;
pub mod util;

use std::collections::HashMap;

use crate::data::reference::{
    load_reference_bundle, load_reference_bundle_from_bytes, ReferenceBundle, ReferenceCase,
    TensorData,
};
use anyhow::{bail, Context, Result};
use model::{kinematics, skinning};

/// Lightweight placeholder for the eventual loaded model.
#[derive(Debug, Clone, Default)]
pub struct AnnyBodyPlaceholder;

impl AnnyBodyPlaceholder {
    /// Construct an empty placeholder instance.
    pub fn new() -> Self {
        Self
    }

    /// Stub forward pass. This will be replaced by the full pipeline once the
    /// data format and reference tests are in place.
    pub fn forward(&self) {
        // no-op placeholder
    }
}

/// Inference-only model backed by safetensors reference data.
#[derive(Debug, Clone)]
pub struct AnnyReference {
    bundle: ReferenceBundle,
    cases_by_name: HashMap<String, ReferenceCase>,
}

#[derive(Debug, Clone)]
pub struct AnnyOutput {
    pub rest_vertices: TensorData<f64>,
    pub posed_vertices: TensorData<f64>,
    pub rest_bone_poses: TensorData<f64>,
    pub bone_poses: TensorData<f64>,
    pub bone_heads: TensorData<f64>,
    pub bone_tails: TensorData<f64>,
}

impl AnnyReference {
    /// Load reference safetensors + metadata from disk.
    pub fn from_paths(
        tensor_path: impl AsRef<std::path::Path>,
        meta_path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        let bundle = load_reference_bundle(tensor_path, meta_path)?;
        let mut cases_by_name = HashMap::new();
        for case in bundle.cases.iter() {
            cases_by_name.insert(case.name.clone(), case.clone());
        }
        Ok(Self {
            bundle,
            cases_by_name,
        })
    }

    /// Load reference safetensors + metadata from in-memory bytes (useful for wasm).
    pub fn from_bytes(tensor_bytes: &'static [u8], meta_bytes: &'static [u8]) -> Result<Self> {
        let bundle = load_reference_bundle_from_bytes(tensor_bytes, meta_bytes)?;
        let mut cases_by_name = HashMap::new();
        for case in bundle.cases.iter() {
            cases_by_name.insert(case.name.clone(), case.clone());
        }
        Ok(Self {
            bundle,
            cases_by_name,
        })
    }

    /// Run a forward pass for a known reference case by name.
    pub fn forward_case(&self, name: &str) -> Result<AnnyOutput> {
        self.forward_with_offsets(name, None, None, None)
    }

    /// Forward pass with optional blendshape and root translation offsets.
    /// Offsets are added to the stored reference values and clamped to [0, 1] for blendshapes.
    pub fn forward_with_offsets(
        &self,
        name: &str,
        blendshape_delta: Option<&[f64]>,
        root_translation_delta: Option<[f64; 3]>,
        pose_parameters_delta: Option<&[f64]>,
    ) -> Result<AnnyOutput> {
        let case = self
            .cases_by_name
            .get(name)
            .context("reference case not found")?;
        let weights = if let Some(delta) = blendshape_delta {
            if delta.len() != case.blendshape_coeffs.data.len() {
                bail!("blendshape delta len mismatch");
            }
            let mut w = case.blendshape_coeffs.data.clone();
            for (w_i, d) in w.iter_mut().zip(delta.iter()) {
                *w_i = (*w_i + d).clamp(0.0, 1.0);
            }
            TensorData {
                shape: case.blendshape_coeffs.shape.clone(),
                data: w,
            }
        } else {
            case.blendshape_coeffs.clone()
        };

        let rest_vertices = model::blendshape::apply_blendshapes(
            &self.bundle.static_data.template_vertices,
            &self.bundle.static_data.blendshapes,
            &weights,
        )?;
        let rest_bone_heads = model::blendshape::apply_bone_blendshapes(
            &self.bundle.static_data.template_bone_heads,
            &self.bundle.static_data.bone_heads_blendshapes,
            &weights,
        )?;
        let rest_bone_tails = model::blendshape::apply_bone_blendshapes(
            &self.bundle.static_data.template_bone_tails,
            &self.bundle.static_data.bone_tails_blendshapes,
            &weights,
        )?;
        let rest_bone_poses = kinematics::rest_bone_poses_from_heads_tails(
            &rest_bone_heads,
            &rest_bone_tails,
            &self.bundle.static_data.bone_rolls_rotmat,
        )?;
        let mut pose = case.pose_parameters.data.clone();
        if let Some(delta) = pose_parameters_delta {
            if delta.len() != pose.len() {
                bail!("pose_parameters delta len mismatch");
            }
            for (p, d) in pose.iter_mut().zip(delta.iter()) {
                *p += d;
            }
        }
        if let Some(delta_t) = root_translation_delta {
            // indices 3,7,11 in row-major 4x4 for translation
            let base = 0;
            if pose.len() >= 12 {
                pose[base + 3] += delta_t[0];
                pose[base + 7] += delta_t[1];
                pose[base + 11] += delta_t[2];
            }
        }
        let pose_parameters = TensorData {
            shape: case.pose_parameters.shape.clone(),
            data: pose,
        };
        let (bone_poses, _) = kinematics::forward_root_relative_world(
            &rest_bone_poses,
            &pose_parameters,
            &self.bundle.metadata.bone_parents,
        )?;
        let posed_vertices = skinning::linear_blend_skinning(
            &rest_vertices,
            &bone_poses,
            &rest_bone_poses,
            &self.bundle.static_data.vertex_bone_indices,
            &self.bundle.static_data.vertex_bone_weights,
        )?;
        Ok(AnnyOutput {
            rest_vertices,
            posed_vertices,
            rest_bone_poses,
            bone_poses,
            bone_heads: case.bone_heads.clone(),
            bone_tails: case.bone_tails.clone(),
        })
    }

    /// Access metadata (e.g., case names) for driving tests.
    pub fn case_names(&self) -> impl Iterator<Item = &str> {
        self.bundle.metadata.case_names.iter().map(|s| s.as_str())
    }

    pub fn metadata(&self) -> &ReferenceBundle {
        &self.bundle
    }
}

/// Public-facing inference-only model (backed by reference data for now).
#[derive(Debug, Clone)]
pub struct AnnyBody {
    reference: AnnyReference,
}

#[derive(Debug, Clone)]
pub struct AnnyInput<'a> {
    /// Select a precomputed reference case by name (from metadata.case_names).
    pub case_name: &'a str,
}

impl AnnyBody {
    /// Load an inference-only model from reference safetensors + metadata.
    pub fn from_reference_paths(
        tensor_path: impl AsRef<std::path::Path>,
        meta_path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        Ok(Self {
            reference: AnnyReference::from_paths(tensor_path, meta_path)?,
        })
    }

    /// Load an inference-only model from reference bytes (for embedded/wasm usage).
    pub fn from_reference_bytes(
        tensor_bytes: &'static [u8],
        meta_bytes: &'static [u8],
    ) -> Result<Self> {
        Ok(Self {
            reference: AnnyReference::from_bytes(tensor_bytes, meta_bytes)?,
        })
    }

    /// Forward pass using reference outputs (placeholder until full pipeline).
    pub fn forward(&self, input: AnnyInput<'_>) -> Result<AnnyOutput> {
        self.reference.forward_case(input.case_name)
    }

    /// Forward pass with optional blendshape and root translation offsets.
    pub fn forward_with_offsets(
        &self,
        case_name: &str,
        blendshape_delta: Option<&[f64]>,
        root_translation_delta: Option<[f64; 3]>,
        pose_parameters_delta: Option<&[f64]>,
    ) -> Result<AnnyOutput> {
        self.reference
            .forward_with_offsets(
                case_name,
                blendshape_delta,
                root_translation_delta,
                pose_parameters_delta,
            )
    }

    /// Quad faces (topology helper).
    pub fn faces_quads(&self) -> &TensorData<i64> {
        &self.reference.bundle.static_data.faces_quads
    }

    /// Reference case names (for testing/benching).
    pub fn case_names(&self) -> impl Iterator<Item = &str> {
        self.reference.case_names()
    }

    /// Vertex bone indices/weights (skinning helper).
    pub fn skinning_bindings(&self) -> (&TensorData<i64>, &TensorData<f64>) {
        (
            &self.reference.bundle.static_data.vertex_bone_indices,
            &self.reference.bundle.static_data.vertex_bone_weights,
        )
    }

    /// Template/rest vertices from static data.
    pub fn template_vertices(&self) -> &TensorData<f64> {
        &self.reference.bundle.static_data.template_vertices
    }

    /// Full reference bundle (static data + metadata) access.
    pub fn metadata(&self) -> &ReferenceBundle {
        self.reference.metadata()
    }
}

#[cfg(test)]
mod tests {
    use super::AnnyBodyPlaceholder;

    #[test]
    fn placeholder_constructs() {
        let _model = AnnyBodyPlaceholder::new();
    }
}
