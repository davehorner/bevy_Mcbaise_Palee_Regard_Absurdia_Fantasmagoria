# project notes (for contributors)

This file keeps the scaffolding/protocol details that were previously in the README.

## highlights
- reference parity: safetensors + metadata exported directly from the official anny code (vendored under `anny/`).
- cpu pipeline in rust (phenotype → blendshapes → fk → lbs) targeting numerical parity.
- bevy demo (native & wasm) with procedural noise controls, lighting, and fps/mesh timing readouts.
- ci-ready workspace layout (library + demo + tools), mirroring the burn_depth pattern.

## layout
```
.
├─ anny/                      # upstream anny (python) reference
├─ src/                       # burn_human library
├─ crates/bevy_burn_human/    # bevy demo app (native + web)
├─ tool/                      # reference export scripts
└─ tests/reference/           # exported safetensors + metadata
```

## quickstart (native demo)
```bash
cargo run -p bevy_burn_human
```
use the in-app ui to tweak noise amplitudes (face/body/joints/phenotype) and see mesh update timings.

## web (wasm)
```bash
rustup target add wasm32-unknown-unknown
cargo build -p bevy_burn_human --target wasm32-unknown-unknown --release --no-default-features --features web
wasm-bindgen --target web --out-dir crates/bevy_burn_human/www/pkg \
  target/wasm32-unknown-unknown/release/bevy_burn_human.wasm
```
serve `crates/bevy_burn_human/www` with any static server.

## tests / lints / benches
```bash
cargo test --all
cargo clippy --all-targets -- -D warnings
cargo bench -p bevy_burn_human -- forward_all_cases
```

## reference export (python → safetensors)
```bash
python tool/scripts/export_reference.py \
  --output tests/reference/fullbody_default.safetensors \
  --seed 1234
```
this writes `fullbody_default.safetensors` and `fullbody_default.meta.json` consumed by the rust crate. the vendored anny lives under `anny/` and is untouched.

## status
- loader: ✅ (safetensors + metadata)
- fk + lbs: ✅ (root_relative_world parity for exported cases)
- phenotype ↔ blendshape noise: 🚧 tuned defaults for realistic, symmetric poses with configurable sliders.
- demo: ✅ (lighting, fps/mesh timing, extensive noise controls; wasm build supported)

## license
mit or apache-2.0 (same as upstream burn). anny remains under its original license.
