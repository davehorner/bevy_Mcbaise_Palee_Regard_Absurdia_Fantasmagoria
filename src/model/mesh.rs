//! Mesh topology helpers (quads/triangles/UVs).
//!
//! TODO: expose face buffers and triangulation helpers.

#[derive(Debug, Clone, Default)]
pub struct MeshTopology {
    pub vertices: usize,
    pub quads: usize,
}

impl MeshTopology {
    pub fn new(vertices: usize, quads: usize) -> Self {
        Self { vertices, quads }
    }
}
