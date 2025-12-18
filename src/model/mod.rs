//! Forward pipeline modules for Anny (phenotype -> shape -> pose -> skinning).
//!
//! Each submodule will be ported stage by stage with numerical parity tests.

pub mod blendshape;
pub mod kinematics;
pub mod mesh;
pub mod phenotype;
pub mod skinning;
