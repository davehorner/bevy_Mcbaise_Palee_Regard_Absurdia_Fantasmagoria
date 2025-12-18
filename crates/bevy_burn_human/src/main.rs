use bevy::asset::RenderAssetUsages;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_tasks::{AsyncComputeTaskPool, Task};
use futures_lite::future::block_on;
use burn_human::data::reference::TensorData;
use burn_human::{model::phenotype::PhenotypeEvaluator, AnnyBody, AnnyInput};
use noise::{NoiseFn, OpenSimplex};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

#[cfg(all(target_arch = "wasm32", feature = "web"))]
use wasm_bindgen::prelude::wasm_bindgen;

#[cfg_attr(all(target_arch = "wasm32", feature = "web"), wasm_bindgen(start))]
pub fn main() {
    #[cfg(all(target_arch = "wasm32", feature = "web"))]
    console_error_panic_hook::set_once();

    App::new()
        .insert_resource(ClearColor(Color::srgb(0.04, 0.05, 0.08)))
        .insert_resource(TimeScale { scale: 1.0 })
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "burn_human demo".to_string(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(EguiPlugin::default())
        .add_plugins(PanOrbitCameraPlugin)
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (queue_mesh_update, apply_mesh_update))
        .add_systems(EguiPrimaryContextPass, ui_time_scale)
        .run();
}

#[derive(Resource)]
struct DemoModel {
    body: Arc<AnnyBody>,
    case_name: String,
    mesh: Handle<Mesh>,
    noise: OpenSimplex,
    blendshape_len: usize,
    pose_param_len: usize,
    bone_rot_scales: Vec<f32>,
    bone_trans_scales: Vec<f32>,
    bone_count: usize,
    phenotype_len: usize,
    phenotype_labels: Vec<String>,
    phenotype_base_inputs: Vec<f64>,
    phenotype_base_weights: Vec<f64>,
    phenotype_eval: Arc<PhenotypeEvaluator>,
    bone_major_flags: Vec<bool>,
    bone_neck_flags: Vec<bool>,
    bone_symmetry_map: Vec<usize>,
    faces: Arc<TensorData<i64>>,
}

#[derive(Resource)]
struct TimeScale {
    scale: f32,
}

#[derive(Resource, Clone)]
struct NoiseControls {
    global_amp: f32,
    face_blend_amp: f32,
    face_fast_amp: f32,
    face_slow_amp: f32,
    body_blend_amp: f32,
    body_fast_amp: f32,
    body_slow_amp: f32,
    phenotype_amp: f32,
    phenotype_freq: f32,
    bone_major_amp: f32,
    bone_neck_amp: f32,
    bone_other_amp: f32,
    bone_rot_amp: f32,
    bone_trans_amp: f32,
}

#[derive(Resource)]
struct MeshUpdateState {
    task: Option<Task<MeshUpdate>>,
    frame_counter: u32,
    timer: Timer,
    recompute_normals_every: u32,
    start_time: Option<Instant>,
    last_update_ms: f32,
}

struct MeshUpdate {
    positions: Vec<[f32; 3]>,
    normals: Option<Vec<[f32; 3]>>,
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    #[cfg(target_arch = "wasm32")]
    let body = AnnyBody::from_reference_bytes(
        include_bytes!("../../tests/reference/fullbody_default.safetensors"),
        include_bytes!("../../tests/reference/fullbody_default.meta.json"),
    )
    .expect("load embedded reference");

    #[cfg(not(target_arch = "wasm32"))]
    let body = AnnyBody::from_reference_paths(
        "tests/reference/fullbody_default.safetensors",
        "tests/reference/fullbody_default.meta.json",
    )
    .expect("load reference bundle");
    let body = Arc::new(body);
    let case_name = body
        .case_names()
        .next()
        .expect("at least one reference case")
        .to_string();
    let initial = body
        .forward(AnnyInput { case_name: &case_name })
        .expect("forward");
    let pose_param_len = body
        .metadata()
        .cases
        .iter()
        .find(|c| c.name == case_name)
        .map(|c| c.pose_parameters.data.len())
        .expect("case pose parameters length");
    let bone_count = pose_param_len / 16;
    let (bone_rot_scales, bone_trans_scales, bone_major_flags, bone_neck_flags, bone_symmetry_map) =
    {
        let labels = &body.metadata().metadata.bone_labels;
        let mut rot = Vec::with_capacity(labels.len());
        let mut trans = Vec::with_capacity(labels.len());
        let mut major_flags = Vec::with_capacity(labels.len());
        let mut neck_flags = Vec::with_capacity(labels.len());
        let mut normalized_to_index: HashMap<String, usize> = HashMap::new();
        let mut symmetry_map = Vec::with_capacity(labels.len());
        for (i, label) in labels.iter().enumerate() {
            let l: String = label.to_lowercase();
            let is_root = i == 0 || l.contains("root");
            let is_shoulder = l.contains("shoulder") || l.contains("clavicle");
            let is_elbow = l.contains("elbow");
            let is_knee = l.contains("knee");
            let is_wrist = l.contains("wrist") || l.contains("hand");
            let is_upper_arm = l.contains("upperarm") || l.contains("upper_arm");
            let is_lower_arm = l.contains("lowerarm") || l.contains("lower_arm") || l.contains("forearm");
            let is_thigh = l.contains("thigh") || l.contains("upperleg") || l.contains("upper_leg");
            let is_calf = l.contains("calf") || l.contains("shin") || l.contains("lowerleg") || l.contains("lower_leg");
            let is_arm = l.contains("arm") || is_shoulder || is_elbow || is_wrist || is_upper_arm || is_lower_arm;
            let is_hip = l.contains("hip");
            let is_leg = l.contains("leg") || is_knee || l.contains("foot") || is_hip || is_thigh || is_calf || l.contains("ankle");
            let is_spine = l.contains("spine") || l.contains("back") || l.contains("chest") || l.contains("pelvis");
            let is_neck = l.contains("neck") || l.contains("head");
            let is_major_joint = is_shoulder || is_elbow || is_knee || is_wrist || l.contains("ankle");
            let rot_scale = if is_root {
                0.03
            } else if is_major_joint {
                0.5
            } else if is_arm || is_leg || is_spine || is_neck {
                0.38
            } else {
                0.17
            };
            let trans_scale = if is_root {
                0.0
            } else if is_major_joint {
                0.005
            } else if is_arm || is_leg || is_spine || is_neck {
                0.0045
            } else {
                0.0035
            };
            rot.push(rot_scale);
            trans.push(trans_scale);
            major_flags.push(is_major_joint);
            neck_flags.push(is_neck);
            let normalized = l
                .replace("left", "")
                .replace("right", "")
                .replace("l_", "")
                .replace("r_", "")
                .replace("-", "")
                .replace(" ", "");
            if let Some(&other) = normalized_to_index.get(&normalized) {
                symmetry_map.push(other.min(i));
            } else {
                normalized_to_index.insert(normalized, i);
                symmetry_map.push(i);
            }
        }
        (rot, trans, major_flags, neck_flags, symmetry_map)
    };
    let phenotype_labels = body.metadata().metadata.phenotype_labels.clone();
    let phenotype_len = phenotype_labels.len();
    let phenotype_eval = Arc::new(PhenotypeEvaluator {
        phenotype_labels: phenotype_labels.clone(),
        macrodetail_keys: body.metadata().metadata.macrodetail_keys.clone(),
        anchors: body.metadata().metadata.phenotype_anchors.clone(),
        variations: body.metadata().metadata.phenotype_variations.clone(),
        mask: body.metadata().static_data.blendshape_mask.clone(),
    });
    let phenotype_base_inputs = body
        .metadata()
        .cases
        .iter()
        .find(|c| c.name == case_name)
        .map(|c| c.phenotype_inputs.data.clone())
        .unwrap_or_else(|| vec![0.5; phenotype_len]);
    let phenotype_base_weights = phenotype_eval
        .as_ref()
        .weights(&TensorData {
            shape: vec![1, phenotype_len],
            data: phenotype_base_inputs.clone(),
        })
        .expect("phenotype weights")
        .data;

    let faces = Arc::new(body.faces_quads().clone());
    let positions = tensor_to_vec3(&initial.posed_vertices);
    let blendshape_len = body.metadata().static_data.blendshapes.shape[0];
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_indices(Indices::U32(triangulate_quads(faces.as_ref())));
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, to_position_attribute(&positions));
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, compute_normals(&positions, faces.as_ref()));
    if let Some(uvs) =
        tensor_to_uv_attribute(&body.metadata().static_data.texture_coordinates, positions.len())
    {
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    }
    let mesh_handle = meshes.add(mesh);
    commands.insert_resource(DemoModel {
        body: body.clone(),
        case_name: case_name.clone(),
        mesh: mesh_handle.clone(),
        noise: OpenSimplex::new(42),
        blendshape_len,
        pose_param_len,
        bone_rot_scales,
        bone_trans_scales,
        bone_count,
        phenotype_len,
        phenotype_labels,
        phenotype_base_inputs,
        phenotype_base_weights,
        phenotype_eval,
        bone_major_flags,
        bone_neck_flags,
        bone_symmetry_map,
        faces,
    });
    commands.insert_resource(NoiseControls {
        global_amp: 1.0,
        face_blend_amp: 0.6,
        face_fast_amp: 0.6,
        face_slow_amp: 0.7,
        body_blend_amp: 0.55,
        body_fast_amp: 0.55,
        body_slow_amp: 0.65,
        phenotype_amp: 0.45,
        phenotype_freq: 0.85,
        bone_major_amp: 1.0,
        bone_neck_amp: 0.4,
        bone_other_amp: 0.55,
        bone_rot_amp: 1.0,
        bone_trans_amp: 0.75,
    });
    commands.insert_resource(MeshUpdateState {
        task: None,
        frame_counter: 0,
        timer: Timer::from_seconds(1.0 / 24.0, TimerMode::Repeating),
        recompute_normals_every: 0,
        start_time: None,
        last_update_ms: 0.0,
    });

    commands.spawn((
        Mesh3d(mesh_handle.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.72, 0.7, 0.68),
            metallic: 0.0,
            reflectance: 0.5,
            perceptual_roughness: 0.55,
            ..default()
        })),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .with_scale(Vec3::splat(1.15)),
    ));

    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.85, 0.85, 0.9),
        brightness: 1.05,
        affects_lightmapped_meshes: true,
    });

    commands.spawn((
        DirectionalLight {
            illuminance: 3_000.0,
            shadows_enabled: false,
            color: Color::srgb(0.97, 0.97, 1.0),
            ..default()
        },
        Transform::from_xyz(5.0, 7.0, 4.5).looking_at(Vec3::new(0.0, 1.2, 0.0), Vec3::Y),
    ));

    // Opposing cool fill to reduce dark backsides.
    commands.spawn((
        DirectionalLight {
            illuminance: 1_800.0,
            shadows_enabled: false,
            color: Color::srgb(0.9, 0.94, 1.0),
            ..default()
        },
        Transform::from_xyz(-4.0, 5.5, -4.5).looking_at(Vec3::new(0.0, 1.1, 0.0), Vec3::Y),
    ));

    commands.spawn((
        PointLight {
            intensity: 620.0,
            shadows_enabled: false,
            range: 18.0,
            color: Color::srgb(0.94, 0.95, 0.99),
            ..default()
        },
        Transform::from_xyz(-3.0, 3.2, 2.4),
    ));

    commands.spawn((
        PointLight {
            intensity: 520.0,
            range: 16.0,
            color: Color::srgb(0.7, 0.75, 0.9),
            ..default()
        },
        Transform::from_xyz(3.2, 2.4, -2.8),
    ));

    // Low-front left fill to lift remaining shadows on face.
    commands.spawn((
        DirectionalLight {
            illuminance: 1_200.0,
            shadows_enabled: false,
            color: Color::srgb(0.98, 0.98, 1.0),
            ..default()
        },
        Transform::from_xyz(-2.5, 1.5, 4.0).looking_at(Vec3::new(0.0, 1.3, 0.0), Vec3::Y),
    ));

    // Soft back fill to lift rear shadows without harsh contrast.
    commands.spawn((
        PointLight {
            intensity: 520.0,
            range: 24.0,
            color: Color::srgb(0.92, 0.95, 1.0),
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(0.0, 2.8, -6.0),
    ));

    // Overhead fill to even top lighting.
    commands.spawn((
        PointLight {
            intensity: 450.0,
            range: 20.0,
            color: Color::srgb(0.98, 0.99, 1.0),
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(0.0, 6.5, 0.5),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(2.8, 1.6, 4.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        PanOrbitCamera::default(),
    ));
}

fn queue_mesh_update(
    time: Res<Time>,
    time_scale: Res<TimeScale>,
    noise_controls: Res<NoiseControls>,
    model: Res<DemoModel>,
    mut state: ResMut<MeshUpdateState>,
) {
    if state.task.is_some() || !state.timer.tick(time.delta()).just_finished() {
        return;
    }

    let t = time.elapsed_secs_f64() * time_scale.scale as f64;
    let noise = model.noise.clone();
    let body = model.body.clone();
    let case_name = model.case_name.clone();
    let blendshape_len = model.blendshape_len;
    let pose_param_len = model.pose_param_len;
    let faces = model.faces.clone();
    let bone_rot_scales = model.bone_rot_scales.clone();
    let bone_trans_scales = model.bone_trans_scales.clone();
    let bone_count = model.bone_count;
    let phenotype_len = model.phenotype_len;
    let phenotype_labels = model.phenotype_labels.clone();
    let phenotype_base_inputs = model.phenotype_base_inputs.clone();
    let phenotype_base_weights = model.phenotype_base_weights.clone();
    let phenotype_eval = model.phenotype_eval.clone();
    let bone_major_flags = model.bone_major_flags.clone();
    let bone_neck_flags = model.bone_neck_flags.clone();
    let bone_symmetry_map = model.bone_symmetry_map.clone();
    let nc = noise_controls.clone();
    let do_normals = state.recompute_normals_every != 0
        && state.frame_counter % state.recompute_normals_every == 0;
    state.frame_counter = state.frame_counter.wrapping_add(1);

    state.start_time = Some(Instant::now());

    let task = AsyncComputeTaskPool::get().spawn(async move {
        let mut blend_delta = vec![0.0f64; blendshape_len];
        for (i, value) in blend_delta.iter_mut().enumerate() {
            let is_face = i < 48;
            let fast_scale = if is_face {
                0.006 * nc.face_blend_amp as f64 * nc.face_fast_amp as f64
            } else {
                0.032 * nc.body_blend_amp as f64 * nc.body_fast_amp as f64
            } * nc.global_amp as f64;
            let slow_scale = if is_face {
                0.012 * nc.face_blend_amp as f64 * nc.face_slow_amp as f64
            } else {
                0.05 * nc.body_blend_amp as f64 * nc.body_slow_amp as f64
            } * nc.global_amp as f64;
            let fast = noise.get([t * 0.32, i as f64 * 0.29]) * fast_scale;
            let slow = noise.get([t * 0.05, i as f64 * 0.04]) * slow_scale;
            let raw = fast + slow;
            let clamp_limit = if is_face { 0.12 } else { 0.18 };
            *value = raw.clamp(-clamp_limit, clamp_limit);
        }
        let mut phenotype_inputs = phenotype_base_inputs.clone();
        for (i, val) in phenotype_inputs.iter_mut().enumerate() {
            let label = phenotype_labels.get(i).map(|s| s.as_str()).unwrap_or("");
            let (freq, amp) = match label {
                "gender" => (0.04, 0.35),
                "muscle" | "weight" => (0.06, 0.28),
                "height" => (0.04, 0.2),
                "proportions" => (0.07, 0.28),
                _ => (0.06, 0.22),
            };
            let amp = amp * nc.phenotype_amp as f64 * nc.global_amp as f64;
            let n = noise.get([t * freq * nc.phenotype_freq as f64, (500.0 + i as f64) * 0.17]);
            *val = (*val + n * amp).clamp(0.0, 1.0);
        }
        let phenotype_weights = phenotype_eval
            .as_ref()
            .weights(&TensorData {
                shape: vec![1, phenotype_len],
                data: phenotype_inputs.clone(),
            })
            .unwrap_or_else(|_| TensorData {
                shape: vec![1, phenotype_len],
                data: phenotype_base_weights.clone(),
            });
        for (dst, w) in blend_delta
            .iter_mut()
            .zip(phenotype_weights.data.iter().zip(phenotype_base_weights.iter()))
        {
            let delta = (w.0 - w.1) * 0.22 * nc.global_amp as f64;
            *dst += delta;
        }

        let mut pose_delta = vec![0.0f64; pose_param_len];
        let rot_indices = [0usize, 1, 2, 4, 5, 6, 8, 9, 10];
        for bone in 0..bone_count {
            let base = bone * 16;
            let sym = *bone_symmetry_map.get(bone).unwrap_or(&bone);
            let sym_seed = (sym as f64) * 0.19;
            for (j, idx) in rot_indices.iter().enumerate() {
                let n = noise.get([t * 0.26, sym_seed + (idx + j) as f64 * 0.021]);
                let base_scale = *bone_rot_scales.get(bone).unwrap_or(&0.2) as f64;
                let joint_mult = if *bone_neck_flags.get(bone).unwrap_or(&false) {
                    nc.bone_neck_amp
                } else if *bone_major_flags.get(bone).unwrap_or(&false) {
                    nc.bone_major_amp
                } else {
                    nc.bone_other_amp * 0.5
                };
                let scale = base_scale * joint_mult as f64 * nc.bone_rot_amp as f64 * nc.global_amp as f64;
                let max_rot = if *bone_major_flags.get(bone).unwrap_or(&false) {
                    0.34
                } else if *bone_neck_flags.get(bone).unwrap_or(&false) {
                    0.18
                } else {
                    0.24
                };
                pose_delta[base + idx] = (n * scale).clamp(-max_rot, max_rot);
            }
            if bone > 0 {
                let tscale = *bone_trans_scales.get(bone).unwrap_or(&0.018) as f64
                    * nc.bone_trans_amp as f64
                    * nc.global_amp as f64;
                pose_delta[base + 3] =
                    (noise.get([t * 0.14, sym_seed + 0.17]) * tscale).clamp(-0.02, 0.02);
                pose_delta[base + 7] =
                    (noise.get([t * 0.15, sym_seed + 0.15]) * tscale).clamp(-0.02, 0.02);
                pose_delta[base + 11] =
                    (noise.get([t * 0.13, sym_seed + 0.13]) * tscale).clamp(-0.02, 0.02);
            }
        }

        let out = body
            .forward_with_offsets(&case_name, Some(&blend_delta), None, Some(&pose_delta))
            .expect("forward with offsets");
        let positions = tensor_to_vec3(&out.posed_vertices);
        let normals = if do_normals {
            Some(compute_normals(&positions, faces.as_ref()))
        } else {
            None
        };

        MeshUpdate {
            positions: to_position_attribute(&positions),
            normals,
        }
    });

    state.task = Some(task);
}

fn apply_mesh_update(
    mut meshes: ResMut<Assets<Mesh>>,
    model: Res<DemoModel>,
    mut state: ResMut<MeshUpdateState>,
) {
    if let Some(task) = state.task.take() {
        if task.is_finished() {
            let update = block_on(task);
            if let Some(mesh) = meshes.get_mut(&model.mesh) {
                mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, update.positions);
                if let Some(normals) = update.normals {
                    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
                }
            }
            if let Some(start) = state.start_time.take() {
                state.last_update_ms = start.elapsed().as_secs_f32() * 1000.0;
            }
        } else {
            state.task = Some(task);
        }
    }
}

fn ui_time_scale(
    mut contexts: EguiContexts,
    mut time_scale: ResMut<TimeScale>,
    mut noise_controls: ResMut<NoiseControls>,
    diagnostics: Res<DiagnosticsStore>,
    state: Res<MeshUpdateState>,
) {
    let ctx = contexts.ctx_mut().expect("primary Egui context");
    egui::Window::new("Playback").show(ctx, |ui| {
        ui.label("Time scale");
        ui.add(
            egui::Slider::new(&mut time_scale.scale, 0.0..=3.0)
                .logarithmic(false)
                .text("speed"),
        );
        ui.label("Lower to slow motion; raise to speed up.");
        ui.separator();
        ui.label("Noise amplitudes");
        ui.add(egui::Slider::new(&mut noise_controls.global_amp, 0.0..=2.0).text("global"));
        ui.add(egui::Slider::new(&mut noise_controls.face_blend_amp, 0.0..=2.0).text("face blend"));
        ui.add(egui::Slider::new(&mut noise_controls.face_fast_amp, 0.0..=2.0).text("face fast noise"));
        ui.add(egui::Slider::new(&mut noise_controls.face_slow_amp, 0.0..=2.0).text("face slow noise"));
        ui.add(egui::Slider::new(&mut noise_controls.body_blend_amp, 0.0..=2.0).text("body blend"));
        ui.add(egui::Slider::new(&mut noise_controls.body_fast_amp, 0.0..=2.0).text("body fast noise"));
        ui.add(egui::Slider::new(&mut noise_controls.body_slow_amp, 0.0..=2.0).text("body slow noise"));
        ui.add(
            egui::Slider::new(&mut noise_controls.phenotype_amp, 0.0..=2.0)
                .text("phenotype/body-type"),
        );
        ui.add(
            egui::Slider::new(&mut noise_controls.phenotype_freq, 0.25..=3.0)
                .logarithmic(true)
                .text("phenotype noise freq"),
        );
        ui.add(
            egui::Slider::new(&mut noise_controls.bone_major_amp, 0.0..=2.0)
                .text("joints (shoulder/elbow/wrist/knee)"),
        );
        ui.add(
            egui::Slider::new(&mut noise_controls.bone_neck_amp, 0.0..=2.0).text("neck/head joints"),
        );
        ui.add(
            egui::Slider::new(&mut noise_controls.bone_rot_amp, 0.0..=2.0).text("joint rotation scale"),
        );
        ui.add(
            egui::Slider::new(&mut noise_controls.bone_trans_amp, 0.0..=2.0).text("joint translation scale"),
        );
        ui.add(
            egui::Slider::new(&mut noise_controls.bone_other_amp, 0.0..=2.0)
                .text("other joints"),
        );
        ui.separator();
        if let Some(fps) = diagnostics
            .get(&FrameTimeDiagnosticsPlugin::FPS)
            .and_then(|d| d.smoothed())
        {
            ui.label(format!("FPS: {:.1}", fps));
        }
        ui.label(format!("Last mesh update: {:.1} ms", state.last_update_ms));
    });
}

fn tensor_to_vec3(data: &TensorData<f64>) -> Vec<Vec3> {
    match data.shape.as_slice() {
        // legacy shape [N,3]
        [n, 3] => data
            .data
            .chunks_exact(3)
            .take(*n)
            .map(|c| Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32))
            .collect(),
        // batched shape [1,N,3]
        [1, n, 3] => data
            .data
            .chunks_exact(3)
            .take(*n)
            .map(|c| Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32))
            .collect(),
        other => panic!("expected [N,3] or [1,N,3] tensor, got shape {:?}", other),
    }
}

fn to_position_attribute(points: &[Vec3]) -> Vec<[f32; 3]> {
    points.iter().map(|p| [p.x, p.y, p.z]).collect()
}

fn tensor_to_uv_attribute(data: &TensorData<f64>, vertex_count: usize) -> Option<Vec<[f32; 2]>> {
    let to_uvs = |n: usize, data: &[f64]| -> Vec<[f32; 2]> {
        data.chunks_exact(2)
            .take(n.min(vertex_count))
            .map(|c| [c[0] as f32, c[1] as f32])
            .collect()
    };
    match data.shape.as_slice() {
        [n, 2] => Some(to_uvs(*n, &data.data)),
        [1, n, 2] => Some(to_uvs(*n, &data.data)),
        _ => None,
    }
}

fn triangulate_quads(quads: &TensorData<i64>) -> Vec<u32> {
    assert_eq!(quads.shape.len(), 2, "faces tensor should be [F,4]");
    assert_eq!(quads.shape[1], 4, "faces tensor should be [F,4]");
    let mut indices = Vec::with_capacity(quads.shape[0] * 6);
    for face in quads.data.chunks_exact(4) {
        let (a, b, c, d) = (
            face[0] as u32,
            face[1] as u32,
            face[2] as u32,
            face[3] as u32,
        );
        indices.extend_from_slice(&[a, b, c, a, c, d]);
    }
    indices
}

fn compute_normals(positions: &[Vec3], quads: &TensorData<i64>) -> Vec<[f32; 3]> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for face in quads.data.chunks_exact(4) {
        let a = face[0] as usize;
        let b = face[1] as usize;
        let c = face[2] as usize;
        let d = face[3] as usize;
        let pa = positions[a];
        let pb = positions[b];
        let pc = positions[c];
        let pd = positions[d];
        let n0 = (pb - pa).cross(pc - pa);
        let n1 = (pc - pa).cross(pd - pa);
        let normal = (n0 + n1).normalize_or_zero();
        for idx in [a, b, c, d] {
            normals[idx] += normal;
        }
    }
    normals
        .into_iter()
        .map(|n| n.normalize_or_zero())
        .map(|n| [n.x, n.y, n.z])
        .collect()
}
