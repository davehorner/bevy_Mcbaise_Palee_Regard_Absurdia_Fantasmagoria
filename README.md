# bevy_Mcbaise_Pale_Regard_Absurdia_Fantasmagoria

<https://davehorner.github.io/bevy_Mcbaise_Pale_Regard_Absurdia_Fantasmagoria/>

Bevy port of the CodePen <https://codepen.io/Dave-Horner/pen/zxqgOOo> “absurdia-stuck in an endless loop in mcdonalds playground slide” visualization, using `bevy_burn_human` as the moving subject.

Timeline youtube sync between the visualization and the video playback.

Forked from https://github.com/mosure/burn_human.git

<p align="center">
    <img alt="Tube ride screenshot" src="https://raw.githubusercontent.com/davehorner/bevy_Mcbaise_Pale_Regard_Absurdia_Fantasmagoria/main/screenshot.jpg" />
</p>

[![test](https://github.com/mosure/burn_human/workflows/test/badge.svg)](https://github.com/Mosure/burn_human/actions?query=workflow%3Atest)
[![crates.io](https://img.shields.io/crates/v/burn_human.svg)](https://crates.io/crates/burn_human)


parametric 3d human ([anny](https://arxiv.org/abs/2511.03589)) model, [view the demo](https://mosure.github.io/burn_human).


![Alt text](./docs/example.gif)


## usage

```rust
use bevy::prelude::*;
use bevy_burn_human::{BurnHumanAssets, BurnHumanInput, BurnHumanPlugin};

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, BurnHumanPlugin::default()))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut mats: ResMut<Assets<StandardMaterial>>,
    assets: Res<BurnHumanAssets>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 1.5, 4.0).looking_at(Vec3::Y, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(2.0, 4.0, 2.0).looking_at(Vec3::Y, Vec3::Y),
    ));

    let n = assets.body.metadata().metadata.phenotype_labels.len();
    commands.spawn((
        BurnHumanInput {
            phenotype_inputs: Some(vec![0.5; n]),
            ..default()
        },
        MeshMaterial3d(mats.add(StandardMaterial::default())),
    ));
}
```


## generate reference data

If you have Go Task installed, you can use the repo `Taskfile.yml`:

```bash
task export-reference
```

Or run the script directly:

```bash
python tool/scripts/export_reference.py --output assets/model/fullbody_default.safetensors --seed 1234
```

exporter runs against the vendored `anny/` codebase and writes `assets/model/fullbody_default.safetensors` plus `assets/model/fullbody_default.meta.json` consumed by the library/tests/demo. The output is deterministic (seeded) and cached assets live under `.cache/anny/`.


## run the demo
```bash
task demo:native
```


## tube ride demo (mcbaise)

Bevy port of a Three.js “tube ride” demo that drives the camera/animation from YouTube time when running on wasm.

- Crate: `crates/bevy_Mcbaise_Pale_Regard_Absurdia_Fantasmagoria`

Native:

```bash
task mcbaise:native
```

WASM (build + serve):

```bash
task mcbaise:wasm:serve
```

Then open the printed URL (serving from repo root keeps `/assets/...` available).

Notes:
- WASM build uses `wasm-bindgen-cli` (the Taskfile will install it automatically, or run `task mcbaise:wasm:bindgen:install`).
- Native has no YouTube panel; it auto-plays using an internal clock (controls: `Space` play/pause, `1`/`2` toggle scheme/pattern, `Up`/`Down` speed).


## GitHub Pages (wasm hosting)

This repo includes a workflow that builds and publishes the tube ride demo to GitHub Pages using the official Pages Actions.

- Workflow: `.github/workflows/deploy.yml`
- Pages URL (project pages): `https://davehorner.github.io/bevy_Mcbaise_Pale_Regard_Absurdia_Fantasmagoria/`

Enable Pages (create if missing; uses GitHub Actions as the build source):

```bash
gh api -X POST repos/davehorner/bevy_Mcbaise_Pale_Regard_Absurdia_Fantasmagoria/pages -f build_type=workflow
```

Or update an existing Pages config:

```bash
gh api -X PUT repos/davehorner/bevy_Mcbaise_Pale_Regard_Absurdia_Fantasmagoria/pages -f build_type=workflow
```

Deploy happens automatically on pushes to `main`, or you can trigger it manually in Actions (workflow `deploy-web`).


## taskfile

This repo includes a `Taskfile.yml` (https://taskfile.dev/) that wraps common commands:

```bash
# Python deps for the exporter
task install-python-modules

# Generate assets/model/fullbody_default.safetensors
task export-reference

# Run demos
task demo:native
task mcbaise:native
task mcbaise:native:youtube
task mcbaise:wasm:serve
```

## Tasks

This project exposes several convenient `task` invocations (see `Taskfile.yml`). Run tasks with the `task` command (go-task). Example: `task demo:native`.

Quick list of notable tasks and examples:

- demo:native — Run the bevy_burn_human demo natively
    - Example: `task demo:native`

- install-python-modules — Install Python deps used by tools and exporters
    - Example: `task install-python-modules`

- export-reference — Produce `assets/model/fullbody_default.safetensors` (depends on `install-python-modules`)
    - Example: `task export-reference`

- mcbaise:native — Run the main tube ride demo natively
    - Example: `task mcbaise:native`

- mcbaise:wasm:serve — Build and serve the WASM demo (opens URL shown in output)
    - Example: `task mcbaise:wasm:serve`

- mcbaise:native:youtube — Run the native tube ride + YouTube sync via WebDriver
    - Example: `task mcbaise:native:youtube`
    - Note: without a `cookies.txt` file playback can be interrupted by ads; export browser cookies and set `MCBAISE_CHROME_COOKIES_TXT` for longer playback during tests.

- update-chromedriver — Cross-platform helper to update chromedriver (dispatches to platform-specific subtasks)
    - Example: `task update-chromedriver`
    - macOS specific: `task update-chromedriver:mac:brew` (uses Homebrew) or `task update-chromedriver:mac:download` (direct download)
    - Linux: `task update-chromedriver:linux:download` (installs to `/usr/local/bin`)
    - Windows: `task update-chromedriver:windows:choco` (Chocolatey)

For the full list of tasks and details, see the `Taskfile.yml` at the repo root.


## YouTube (native) cookie note

If you run the native YouTube task without providing a `cookies.txt` file (see `MCBAISE_CHROME_COOKIES_TXT` / `MCBAISE_CHROME_USER_DATA_DIR` in `Taskfile.yml`), the demo will often show ads and, in our tests, playback stops after roughly one minute. To avoid ads and get longer playback during tests, export your browser cookies to `cookies.txt` and point `MCBAISE_CHROME_COOKIES_TXT` at that file before running `task mcbaise:native:youtube`.

mpv playback is yet to work.


## license
mit or apache-2.0 (anny stays under its original license)
