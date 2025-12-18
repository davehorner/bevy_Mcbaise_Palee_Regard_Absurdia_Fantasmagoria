# burn_human

[![test](https://github.com/mosure/burn_human/workflows/test/badge.svg)](https://github.com/Mosure/burn_human/actions?query=workflow%3Atest)
[![crates.io](https://img.shields.io/crates/v/burn_human.svg)](https://crates.io/crates/burn_human)

Parametric 3D human ([anny](https://arxiv.org/abs/2511.03589)) model with a Rust/Burn forward pipeline and a ready-to-drop Bevy plugin. [View the demo](https://mosure.github.io/burn_human).

`tests/reference/fullbody_default.safetensors` is the deterministic reference export from the upstream Python anny code. It is ~200 MB because it stores double-precision template geometry, blendshapes, skinning bindings, and a handful of baked reference poses. The file is not published to crates.io; generate it locally (below) or grab it from the GitHub release artifact.

![Alt text](./assets/example.gif)

## Rust usage

```rust
use burn_human::{AnnyBody, AnnyInput};

fn main() -> anyhow::Result<()> {
    // Load deterministic reference data (see "generate reference data" below).
    let body = AnnyBody::from_reference_paths(
        "tests/reference/fullbody_default.safetensors",
        "tests/reference/fullbody_default.meta.json",
    )?;

    // 1) Reproduce a baked reference case by name.
    let neutral = body.forward_case("neutral_pose_random_phenotype")?;
    println!("posed vertices (case): {}", neutral.posed_vertices.shape[1]);

    // 2) Drive the model with your own phenotype vector (0..1 sliders).
    let custom = body.forward(AnnyInput {
        case_name: None,
        phenotype_inputs: Some(&[0.2, 0.15, 0.6, 0.65, 0.5, 0.55]), // gender, age, muscle, weight, height, proportions
        ..Default::default()
    })?;
    println!("posed vertices (custom phenotype): {}", custom.posed_vertices.shape[1]);

    Ok(())
}
```

- `AnnyInput` now accepts optional phenotype inputs, direct blendshape weights, pose parameter overrides, and deltas. Set `case_name` to `None` to use your own sliders; set it to a reference name to get deterministic parity.
- `AnnyBody::phenotype_evaluator()` exposes the same anchor/macrodetail mapping used by the Bevy demo if you want to precompute blendshape weights yourself.
- Rigging helpers: `AnnyBody::bone_hierarchy()` returns `(bone_labels, bone_parents)`; `skinning_bindings()` returns `(vertex_bone_indices, vertex_bone_weights)`. Every `AnnyOutput` carries posed/rest vertices plus posed bone heads/tails so you can attach your own rig logic or debug transforms.

## Bevy usage

The Bevy crate is now a plugin-first library. It loads the reference data once, caches generated meshes by input hash, and keeps your `Mesh3d` handles in sync automatically.

```rust
use bevy::prelude::*;
use bevy_burn_human::{BurnHumanInput, BurnHumanPlugin};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(BurnHumanPlugin::default()) // or BurnHumanPlugin::from_bytes(...) for wasm
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, mut materials: ResMut<Assets<StandardMaterial>>) {
    commands.spawn((
        BurnHumanInput {
            // Use your own phenotype values; omit to stick with reference cases.
            phenotype_inputs: Some(vec![0.5; 6]),
            ..Default::default()
        },
        MeshMaterial3d(materials.add(StandardMaterial::default())),
        Transform::from_scale(Vec3::splat(1.1)),
    ));
}
```

- Attach a `BurnHumanInput` component to control the body. Changing it (case name, phenotype vector, blendshape or pose deltas) triggers a cached mesh rebuild. The plugin will inject missing defaults for the mesh handle, cache key, and settings so you only need to add the input plus your material/transform.
- The included demo (`cargo run -p bevy_burn_human`) shows a lightweight UI: pick a baked reference case or dial phenotype sliders to quickly explore realistic body types. The plugin handles mesh generation and caching under the hood.
- Bones and rigging: `BurnHumanInput::pose_parameters` is `[B * J * 16]` row-major 4×4 transforms. If you supply `case_name`, the baked pose for that case is used unless you override it. The forward output contains posed bone heads/tails so you can drive your own rigs or debug transforms.
- Controls in the demo mirror the upstream anny sliders (phenotype, per-bone Euler overrides, pose/phenotype procedural noise). Press `R` to sample a plausible random pose/shape from the baked cases; the PanOrbit camera is automatically disabled when interacting with egui.

## generate reference data

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install torch safetensors roma numpy
python tool/scripts/export_reference.py --output tests/reference/fullbody_default.safetensors --seed 1234
```

The exporter runs against the vendored `anny/` codebase and writes `tests/reference/fullbody_default.safetensors` plus `tests/reference/fullbody_default.meta.json` consumed by the library/tests/demo. The output is deterministic (seeded) and cached assets live under `.cache/anny/`. The safetensors payload stays out of version control/crates.io; drop it into `tests/reference/` before running the demo or tests.

## run the demo
```bash
cargo run -p bevy_burn_human
```

## license
mit or apache-2.0 (anny stays under its original license)
