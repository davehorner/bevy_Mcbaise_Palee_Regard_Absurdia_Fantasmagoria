//! Phenotype parameter interpolation into blendshape weights.

use anyhow::{bail, Result};

use crate::data::reference::TensorData;

#[derive(Debug, Clone)]
pub struct PhenotypeEvaluator {
    pub phenotype_labels: Vec<String>,
    pub macrodetail_keys: Vec<String>,
    pub anchors: std::collections::HashMap<String, Vec<f64>>,
    pub variations: std::collections::HashMap<String, Vec<String>>,
    pub mask: TensorData<f64>, // [blend, macrodetail]
}

impl PhenotypeEvaluator {
    pub fn weights(&self, phenotype_inputs: &TensorData<f64>) -> Result<TensorData<f64>> {
        if phenotype_inputs.shape.len() != 2 {
            bail!("phenotype_inputs must be [B,P]");
        }
        if phenotype_inputs.shape[1] != self.phenotype_labels.len() {
            bail!("phenotype_inputs second dim must match phenotype labels");
        }
        let batch = phenotype_inputs.shape[0];

        // Build per-label arrays.
        let mut label_to_values = std::collections::HashMap::<String, Vec<f64>>::with_capacity(
            self.phenotype_labels.len(),
        );
        for (i, label) in self.phenotype_labels.iter().enumerate() {
            let mut vals = Vec::with_capacity(batch);
            for b in 0..batch {
                vals.push(phenotype_inputs.data[b * self.phenotype_labels.len() + i]);
            }
            label_to_values.insert(label.clone(), vals);
        }

        // Race handling: if race labels are missing, default to 1/3 each.
        let mut race_values = vec![
            label_to_values
                .remove("african")
                .unwrap_or_else(|| vec![1.0 / 3.0; batch]),
            label_to_values
                .remove("asian")
                .unwrap_or_else(|| vec![1.0 / 3.0; batch]),
            label_to_values
                .remove("caucasian")
                .unwrap_or_else(|| vec![1.0 / 3.0; batch]),
        ];
        // normalize
        for b in 0..batch {
            let sum = race_values[0][b] + race_values[1][b] + race_values[2][b];
            if sum != 0.0 {
                for r in 0..3 {
                    race_values[r][b] /= sum;
                }
            } else {
                race_values[0][b] = 1.0 / 3.0;
                race_values[1][b] = 1.0 / 3.0;
                race_values[2][b] = 1.0 / 3.0;
            }
        }

        // Interpolate per feature using anchors; produce per-variation weights.
        let mut variation_values = std::collections::HashMap::<String, Vec<f64>>::new();
        for (feature, variations) in &self.variations {
            if feature == "race" {
                let names = ["african", "asian", "caucasian"];
                for (i, name) in names.iter().enumerate() {
                    variation_values.insert((*name).to_string(), race_values[i].clone());
                }
                continue;
            }
            let anchors = self
                .anchors
                .get(feature)
                .ok_or_else(|| anyhow::anyhow!("missing anchors for feature {feature}"))?;
            let values = label_to_values
                .get(feature)
                .cloned()
                .unwrap_or_else(|| vec![0.5; batch]);
            let coeffs = linear_interpolation_coefficients(&values, anchors);
            if coeffs[0].len() != variations.len() {
                bail!("variation count mismatch for feature {feature}");
            }
            for (i, var_name) in variations.iter().enumerate() {
                let mut v = Vec::with_capacity(batch);
                for b in 0..batch {
                    v.push(coeffs[b][i]);
                }
                variation_values.insert(var_name.clone(), v);
            }
        }

        // Map to macrodetail order.
        let mut phens = vec![vec![0.0f64; self.macrodetail_keys.len()]; batch];
        for (macro_idx, key) in self.macrodetail_keys.iter().enumerate() {
            let vals = variation_values
                .get(key)
                .unwrap_or_else(|| panic!("missing macrodetail key {key}"));
            for b in 0..batch {
                phens[b][macro_idx] = vals[b];
            }
        }

        // Compute blendshape weights.
        let blend_count = self.mask.shape[0];
        let macro_len = self.mask.shape[1];
        let mut weights = TensorData {
            shape: vec![batch, blend_count],
            data: vec![0.0; batch * blend_count],
        };
        for b in 0..batch {
            for blend in 0..blend_count {
                let mut prod = 1.0;
                for m in 0..macro_len {
                    let mask = self.mask.data[blend * macro_len + m];
                    prod *= phens[b][m] * mask + (1.0 - mask);
                }
                weights.data[b * blend_count + blend] = prod;
            }
        }
        Ok(weights)
    }
}

fn linear_interpolation_coefficients(values: &[f64], anchors: &[f64]) -> Vec<Vec<f64>> {
    let n = anchors.len();
    let mut out = vec![vec![0.0; n]; values.len()];
    for (b, v) in values.iter().enumerate() {
        // find interval
        let mut idx = 0;
        while idx < n && anchors[idx] < *v {
            idx += 1;
        }
        if idx == 0 {
            out[b][0] = 1.0;
        } else if idx >= n {
            out[b][n - 1] = 1.0;
        } else {
            let low = anchors[idx - 1];
            let high = anchors[idx];
            let t = if high > low {
                (v - low) / (high - low)
            } else {
                0.0
            };
            out[b][idx - 1] = 1.0 - t;
            out[b][idx] = t;
        }
    }
    out
}
