# burn_human (scaffold)

Parametric 3D human (Anny) forward pipeline port to Rust + Burn, following the
burn_depth / burn_dino patterns (workspace layout, wasm demo, numerical
equivalence tests).

## Layout (current)

```
.
├─ anny/                         # Python reference (vendored, unchanged)
├─ src/                          # burn_human library scaffold
├─ crates/bevy_burn_human/        # Bevy demo (native + wasm)
├─ tool/                         # import/export utilities (stub)
├─ assets/                       # runtime assets (placeholder)
└─ .github/workflows/            # CI + deploy pipelines
```

## Dev commands

- Build: `cargo build --all --all-targets`
- Lint: `cargo clippy --all-targets -- -D warnings`
- Test: `cargo test --all`
- Wasm (demo): `rustup target add wasm32-unknown-unknown` then  
  `cargo build -p bevy_burn_human --target wasm32-unknown-unknown --release --no-default-features --features web`  
  followed by  
  `wasm-bindgen --target web --out-dir crates/bevy_burn_human/www/pkg target/wasm32-unknown-unknown/release/bevy_burn_human.wasm`

## Reference export (Python oracle)

- Generate golden tensors from the vendored Python Anny (writes safetensors + metadata json):  
  `python tool/scripts/export_reference.py --output tests/reference/fullbody_default.safetensors --seed 1234`  
  (keeps cache under `.cache/anny` for reproducibility).

Rust tests include a safetensors loader sanity check to keep the oracle readable in Rust-only CI.

### Pipeline status

- Reference loader: ✅ (safetensors + metadata)
- Skinning: ✅ (CPU LBS reproduces reference posed vertices for exported cases)
- FK: ✅ (root_relative_world; matches reference bone poses)
- Phenotype interpolation, blendshape accumulation: TODO (rest vertices/rest bone poses still sourced from oracle outputs; export will need blendshape data)

## Roadmap snapshot

This repo will implement the Anny forward path (phenotype interpolation →
rest-shape blendshapes → FK → LBS) with a packed asset format, Python-exported
golden tests, and a Bevy/WebGPU demo deployed to GH Pages. Phase 0 provides the
workspace/CI scaffold; later phases lock reference outputs and port each stage
with numerical parity.
