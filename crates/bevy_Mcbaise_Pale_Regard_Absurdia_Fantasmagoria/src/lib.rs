#![cfg_attr(target_arch = "wasm32", no_main)]

use bevy::asset::RenderAssetUsages;
use bevy::asset::embedded_asset;
use bevy::asset::io::embedded::EmbeddedAssetRegistry;
use bevy::ecs::system::SystemParam;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::pbr::{DistanceFog, FogFalloff, Material, MaterialPlugin};
use bevy::prelude::*;
// Re-export the UI prelude to ensure `Node`, `Button`, `ImageNode`,
// `FocusPolicy`, etc. are available.
// `FocusPolicy` is now imported from its full path or moved out of the prelude.
// use bevy::ui::FocusPolicy;
// UI capture/preview is optional; gated behind `capture_ui` feature when enabled.
use bevy::reflect::TypePath;
use bevy::render::render_resource::AsBindGroup;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use bevy::render::texture::GpuImage;
use bevy::shader::ShaderRef;
use bevy::window::{PrimaryWindow, Window};
use bevy::camera::{RenderTarget, Viewport};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use bevy_async_task::TaskPool;
use futures::channel::oneshot;
use bevy::asset::AssetId;
use bevy_render::texture::ManualTextureViews;
// UI capture/preview is optional; gated behind `capture_ui` feature when enabled.

#[cfg(feature = "burn_human")]
use bevy_burn_human::BurnHumanSource;
#[cfg(feature = "burn_human")]
use bevy_burn_human::{BurnHumanAssets, BurnHumanInput, BurnHumanPlugin};

// Provide a placeholder `BurnHumanAssets` type when the feature is disabled
// so code that references the type (behind runtime gates) can still compile.
#[cfg(not(feature = "burn_human"))]
#[derive(Resource)]
pub struct BurnHumanAssets;

#[cfg(all(not(target_arch = "wasm32"), feature = "burn_human"))]
mod native_assets;
#[cfg(not(target_arch = "wasm32"))]
use open;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;
// wasm: no direct `Windows` import; use ECS `Query<&mut Window, With<PrimaryWindow>>` below.

// Render-scale API is implemented later in the file as `render_scale_api`.

// Global readback request counter observed by the render/ECS systems.
static READBACK_REQUEST_SEQ: AtomicU32 = AtomicU32::new(0);

// GPU readback phase:
// 0 = idle (do not copy)
// 1 = armed this frame (wait for extraction/target to propagate)
// 2 = wait one full render into the target
// 3 = capture this frame
static READBACK_GPU_PHASE: AtomicU32 = AtomicU32::new(0);

// Whether a staging buffer is currently mapped (or mapping) for GPU readback.
// When set, we poll the device each render tick to help drive mapping callbacks.
static READBACK_MAP_IN_FLIGHT: AtomicU32 = AtomicU32::new(0);

// Debug: remember the last request sequence we logged about to avoid spamming.
static READBACK_DEBUG_LAST_SEQ_ARM: AtomicU32 = AtomicU32::new(0);

static READBACK_DEBUG_LAST_SEQ_RENDER: AtomicU32 = AtomicU32::new(0);

// Strongly wait for the device to become idle. This aggressively polls
// the render device with blocking waits and short sleeps to give drivers
// (especially Vulkan) time to drain all in-flight work before we destroy
// surface or swapchain-backed resources.
#[cfg(not(target_arch = "wasm32"))]
fn wait_for_device_idle_strong(render_device: &bevy::render::renderer::RenderDevice) {
    const MAX_TRIES: usize = 64;
    for i in 0..MAX_TRIES {
        if render_device.poll(wgpu::PollType::Wait).is_ok() {
            return;
        }
        if i < 4 {
            sleep(Duration::from_millis(4));
        } else {
            sleep(Duration::from_millis(8));
        }
    }
    let _ = render_device.poll(wgpu::PollType::Wait);
}

#[cfg(target_arch = "wasm32")]
fn wait_for_device_idle_strong(_render_device: &bevy::render::renderer::RenderDevice) {
    // No-op on wasm: cannot block/sleep.
}

// Global sequence counter for tracking submits in logs.
static SUBMIT_SEQ_COUNTER: AtomicU64 = AtomicU64::new(0);
// Global sequence counter for surface acquires (instrumentation only).
static ACQUIRE_SEQ_COUNTER: AtomicU64 = AtomicU64::new(0);

#[allow(dead_code)]
fn wasm_dbg(msg: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::console::warn_1(&wasm_bindgen::JsValue::from_str(msg));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        eprintln!("{}", msg);
    }
}

#[derive(Resource, Default)]
struct BurnHumanSpawned(pub bool);

#[derive(Resource, Clone, Copy)]
struct BurnHumanEnabled(pub bool);

impl Default for BurnHumanEnabled {
    fn default() -> Self {
        BurnHumanEnabled(true)
    }
}

#[cfg(feature = "burn_human")]
fn spawn_burn_human_when_ready(
    mut commands: Commands,
    assets_opt: Option<Res<BurnHumanAssets>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut spawned: ResMut<BurnHumanSpawned>,
) {
    if spawned.0 {
        return;
    }

    let Some(assets) = assets_opt else {
        return;
    };

    let phenotype_len = assets.body.metadata().metadata.phenotype_labels.len();
    let selected_case = assets
        .body
        .metadata()
        .cases
        .iter()
        .position(|c| c.pose_parameters.shape[0] == 1)
        .unwrap_or(0usize);

    commands.spawn((
        BurnHumanInput {
            case_name: assets
                .body
                .metadata()
                .metadata
                .case_names
                .get(selected_case)
                .cloned(),
            phenotype_inputs: Some(vec![0.5; phenotype_len]),
            ..Default::default()
        },
        MeshMaterial3d(std_materials.add(StandardMaterial {
            base_color: Color::srgb(0.95, 0.95, 0.95),
            metallic: 0.0,
            reflectance: 0.5,
            perceptual_roughness: 0.6,
            cull_mode: None,
            ..default()
        })),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .with_scale(Vec3::splat(HUMAN_SCALE)),
        Visibility::default(),
        SubjectTag,
        Name::new("burn_human_subject"),
    ));

    spawned.0 = true;
}

fn enforce_burn_human_subject_mode(
    mut subject_mode: ResMut<SubjectMode>,
    burn_enabled: Option<Res<BurnHumanEnabled>>,
) {
    let enabled = burn_enabled.map(|r| r.0).unwrap_or(true);
    if !enabled && *subject_mode == SubjectMode::Human {
        *subject_mode = SubjectMode::Doughnut;
    }
}

// Render-subapp cleanup monitor: run in the render app Cleanup stage. When
// a teardown entry exists in the global table we perform a blocking device
// poll on the render thread and then signal the corresponding Arc flag so
// the main app may safely destroy GPU-backed resources.
fn render_teardown_monitor_system(render_device: Res<bevy::render::renderer::RenderDevice>) {
    let map = match GLOBAL_RENDER_TEARDOWNS.get() {
        Some(m) => m,
        None => return,
    };
    let mut guard = match map.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if guard.is_empty() {
        return;
    }
    let count = guard.len();
    let ids: Vec<u64> = guard.keys().copied().collect();
    eprintln!("native(render): render cleanup monitor found {} teardown(s) ids={:?} -> polling device and signalling", count, ids);
    // Use a stronger idle-wait helper that aggressively polls the device
    // to give drivers extra time to finish in-flight work before we signal
    // the main world it's safe to destroy GPU-backed resources.
    wait_for_device_idle_strong(&*render_device);
    // Drain and signal all pending flags.
    let drained: Vec<_> = guard.drain().collect();
    for (_id, flag) in drained {
        flag.store(true, Ordering::SeqCst);
    }
}

// Render-side consumer: remove image assets from the render world's `Assets<Image>`.
// This forces the GPU-backed image resources to be dropped on the render thread
// where wgpu expects them to be freed, avoiding cross-thread swapchain races.
fn render_asset_drop_consume_system(
    images: Option<ResMut<Assets<Image>>>,
    mut pending_opt: Option<ResMut<PendingRenderAssetDrops>>,
    mut staging_res_opt: Option<ResMut<ReadbackStaging>>,
    render_device: Res<bevy::render::renderer::RenderDevice>,
    render_queue: Res<bevy::render::renderer::RenderQueue>,
    mut delayed_opt: Option<ResMut<DelayedAttachmentDrops>>,
    mut task_pool: TaskPool<'_, bevy::asset::AssetId<Image>>,
    frame: Res<GlobalFrameCount>,
) {
    // If the render world does not have `Assets<Image>` yet, skip safely.
    let mut images = match images {
        Some(i) => i,
        None => {
            // Nothing to drop on the render side right now.
            return;
        }
    };
    // If the render world does not have a pending drops table, nothing to do.
    let mut pending = match pending_opt {
        Some(p) => p,
        None => return,
    };
    // Only consume pending drops once the scheduled_frame (if any)
    // has been reached. This adds a small frame-delay safety window to
    // avoid removing attachments while the render graph may still be
    // recording passes that reference them.
    if pending.images.is_empty() {
        return;
    }
    // If main set a placeholder (Some(0)), set a render-side scheduled frame
    // based on the render world's `GlobalFrameCount`. This avoids cross-world
    // frame counter mismatch which could otherwise allow premature drops.
    if let Some(sched) = pending.scheduled_frame {
            if sched == 0 {
            // Record a submit-frame hint (proxy for the submission point)
            // and schedule immediate finalization — we rely on per-handle
            // oneshot callbacks (registered below) rather than an extra
            // multi-frame safety window.
            pending.submit_frame_hint = Some(frame.0);
            pending.scheduled_frame = Some(frame.0);
            // New schedule set; return so the next frame can consume.
            return;
        }
        if frame.0 < sched {
            // Not safe yet; wait another frame.
            return;
        }
    }
    let count = pending.images.len();
    eprintln!(
        "native(render): consuming {} pending render asset drop(s) scheduled_frame={:?} frame={}",
        count,
        pending.scheduled_frame,
        frame.0
    );
    // Give the device a conservative drain before removing images.
    // Additionally, if we have a submit_frame_hint, prefer to wait until a
    // few frames have advanced beyond it as a proxy for the render
    // submission completing — this adds determinism on platforms where
    // `poll(Wait)` alone may be insufficiently synchronized with encoder
    // finish timing.
    fn wait_for_device_idle_with_hint(
        render_device: &bevy::render::renderer::RenderDevice,
        frame: u64,
        submit_hint: Option<u64>,
    ) {
        const MAX_TRIES: usize = 64;
        for i in 0..MAX_TRIES {
            let _ = render_device.poll(wgpu::PollType::Wait);
            // If we have a submit hint, prefer at least one frame beyond it
            // as a conservative proxy that the render world advanced; do
            // not rely on large frame delays here since oneshots provide
            // per-submit guarantees.
            if let Some(hint) = submit_hint {
                if frame.saturating_sub(hint) > 0 {
                    break;
                }
            } else if i > 8 {
                break;
            }
            let ms = if i < 4 { 4 } else { 8 };
            sleep(Duration::from_millis(ms));
        }
        let _ = render_device.poll(wgpu::PollType::Wait);
    }
    wait_for_device_idle_with_hint(&*render_device, frame.0, pending.submit_frame_hint);

    // As an extra safety net, perform a strong idle wait before performing
    // any destructive drops of GPU-backed image resources. This helps ensure
    // surface-acquire semaphores and other transient objects are no longer
    // in use on drivers with delayed teardown semantics.
    wait_for_device_idle_strong(&*render_device);

    // If a readback staging buffer exists in the render world, drop it here
    // so the GPU-backed staging memory is released on the render thread.
    if let Some(staging_res) = staging_res_opt.as_mut() {
        if staging_res.buffer.is_some() {
            eprintln!(
                "native(render): dropping readback staging buffer ({} bytes)",
                staging_res.size
            );
            staging_res.buffer = None;
            staging_res.size = 0;
        }
    }

    // Log a snapshot of image handles currently present in the render world's
    // `Assets<Image>` to help correlate which attachments exist at drop-time.
    let mut snapshot: Vec<String> = Vec::new();
    for (h, _) in images.iter().take(64) {
        snapshot.push(format!("{:?}", h));
    }
    eprintln!(
        "native(render): render Assets<Image> count={} sample_ids={:?}",
        images.len(),
        snapshot
    );

    // Queue final destructive drops into the delayed drop table. Move handles
    // into `DelayedAttachmentDrops` with an optional submission-complete flag
    // and spawn a small waiter that sets the flag once the queue reports the
    // submitted work is done. This provides a deterministic gate for final
    // destructive removal instead of relying on time/frame heuristics alone.
    if !pending.images.is_empty() {
        // No coarse frame delays; schedule finalization as soon as the
        // render frame has advanced. Precise ordering is enforced by the
        // per-handle oneshot callbacks registered below.
        let mut safe_frame = frame.0;
        if let Some(hint) = pending.submit_frame_hint {
            safe_frame = std::cmp::max(safe_frame, hint);
        }
        let queued: Vec<Handle<Image>> = pending.images.drain(..).collect();
        eprintln!(
            "native(render): queued {} pending image drops for finalization at frame={} (current={})",
            queued.len(),
            safe_frame,
            frame.0
        );

        // Perform two explicit noop submissions here so drivers that delay
        // acquire-semaphore teardown have a clear submission/fence to
        // observe before we register per-handle completion callbacks and
        // allow final destructive drops. The double-noop provides an
        // additional fence so drivers with delayed internal maintenance
        // are more likely to have cleared references.
        submit_noop_and_wait(&*render_device, &*render_queue);
        submit_noop_and_wait(&*render_device, &*render_queue);

        // For each queued handle create a completion waiter and push into
        // the delayed list. Instead of spawning a blocking thread or using
        // an Arc<AtomicBool>, create a oneshot channel and spawn an async
        // waiter task via `bevy_async_task::TaskPool`. When the queue
        // invokes the callback we send on the oneshot and the async task
        // completes; a dedicated pump system will collect completed ids and
        // make them visible to finalizers.
        let rq = render_queue.clone();
        for handle in queued.into_iter() {
            let id = handle.id();
            // Register placeholder in delayed list with no flag; we'll use
            // the CompletedDropIds resource to observe completion.
            if let Some(delayed) = delayed_opt.as_mut() {
                delayed.entries.push((handle.clone(), safe_frame, None));
            }

            // Create a oneshot pair; the closure will send on `s` when the
            // queue reports completion. The async task awaits the receiver
            // and returns the handle id when done. To also provide
            // per-acquire notification, create a small Arc<Mutex<Option<>>>
            // wrapper so either the queue callback or an acquire-notifier
            // thread can fulfill the oneshot exactly once.
            let (s, rcv) = oneshot::channel::<()>();
            let shared_s = Arc::new(Mutex::new(Some(s)));
            let rq_local = rq.clone();
            eprintln!(
                "native(render): registering on_submitted_work_done (async) for drop id={} scheduled_frame={} (current={})",
                id,
                safe_frame,
                frame.0
            );
            let my_submit_seq = SUBMIT_SEQ_COUNTER.fetch_add(1, Ordering::SeqCst) + 1;
            let acq_snapshot = ACQUIRE_SEQ_COUNTER.load(Ordering::SeqCst);
            let shared_s_cb = shared_s.clone();
            rq_local.on_submitted_work_done(move || {
                eprintln!("native(render): async-drop on_submitted_work_done my_submit_seq={} acquire_seq={}", my_submit_seq, acq_snapshot);
                if let Some(tx) = shared_s_cb.lock().unwrap().take() {
                    let _ = tx.send(());
                }
            });

            // Also register a crossbeam waiter keyed by the current
            // `acquire_seq` snapshot so per-acquire notifications can wake
            // this oneshot when a submit that referenced the acquire
            // completes. This complements the queue callback and ensures
            // the drop won't wait for an unrelated submit.
            let acq_snapshot = ACQUIRE_SEQ_COUNTER.load(Ordering::SeqCst);
            if acq_snapshot > 0 {
                if let Some(map_lock) = ACQUIRE_WAITS.get() {
                    if let Ok(mut map) = map_lock.lock() {
                        map.entry(acq_snapshot).or_insert_with(Vec::new).push(shared_s.clone());
                    }
                } else {
                    let _ = ACQUIRE_WAITS.get_or_init(|| Mutex::new(HashMap::new()));
                    if let Some(map_lock) = ACQUIRE_WAITS.get() {
                        if let Ok(mut map) = map_lock.lock() {
                            map.entry(acq_snapshot).or_insert_with(Vec::new).push(shared_s.clone());
                        }
                    }
                }
            }

            // Spawn an async waiter that completes when the oneshot fires.
            // The TaskPool will run futures on a background pool, and the
            // pump system will observe completions on the render thread.
            task_pool.spawn(async move {
                let _ = rcv.await;
                id
            });
        }
        pending.scheduled_frame = Some(safe_frame);
    }
}

fn wasm_dbg_kv(prefix: &str, value: &str) {
    let s = format!("{prefix}{value}");
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::console::warn_1(&wasm_bindgen::JsValue::from_str(&s));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        eprintln!("{}", s);
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(Resource, Default)]
struct WasmDebugFrameOnce {
    did_log: bool,
}

#[cfg(target_arch = "wasm32")]
fn wasm_debug_first_update_tick(mut st: ResMut<WasmDebugFrameOnce>) {
    if st.did_log {
        return;
    }
    st.did_log = true;
    wasm_dbg("wasm: first Update tick");
}

// --- GPU readback (Bevy render target -> wgpu buffer map) ---
// We render the scene into an offscreen Image with COPY_SRC enabled, then in
// the RenderApp we copy that texture into a MAP_READ buffer and post the bytes
// to the parent window.

#[derive(Resource, Clone, Default)]
struct GpuReadbackImage(pub Handle<Image>);

impl bevy::render::extract_resource::ExtractResource for GpuReadbackImage {
    type Source = GpuReadbackImage;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

#[derive(Resource, Default)]
struct GpuReadbackPending {
    active: bool,
    just_armed: bool,
    frames_left: u8,
}

use bevy::render::render_resource::Buffer as BevyBuffer;
use std::sync::{Arc, Mutex, atomic::AtomicBool, OnceLock};
use crossbeam_channel::{unbounded, Sender, Receiver};
use std::any::Any;
use bevy::render::view::window::WindowSurfaces;
use std::collections::HashMap;
use std::thread::sleep;
use std::time::Duration;

fn mcbaise_render_noop_logs_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Treat probes as a superset of render-thread verbose logging.
        std::env::var("MCBAISE_RENDER_PROBES").as_deref().ok() == Some("1")
            || std::env::var("MCBAISE_RENDER_NOOP_LOGS").as_deref().ok() == Some("1")
    })
}

// NOTE: per-acquire oneshot notifications are used instead of frame-based
// safety delays. Remove the coarse frame delay logic and prefer explicit
// submit-completion gates (oneshot + on_submitted_work_done) to determine
// when it is safe to finalize GPU-backed resources.


// Submit a no-op command buffer and wait for its completion. Extracted
// into a helper so it can be invoked from multiple places (e.g. the
// dedicated system and inline where we queue pending drops) to ensure
// a clear fence/submission boundary before destructive surface/swapchain
// operations occur.
fn submit_noop_and_wait(
    render_device: &bevy::render::renderer::RenderDevice,
    render_queue: &bevy::render::renderer::RenderQueue,
) {
    let log_enabled = mcbaise_render_noop_logs_enabled();

    // Ensure the global teardown helper exists
    let rt = RENDER_TEARDOWN.get_or_init(|| Arc::new(RenderTeardown::default()));

    // If a teardown is already in progress, avoid adding new submits. Wait
    // for the last recorded submit to complete instead of issuing another.
    if rt.in_progress.load(Ordering::SeqCst) {
        let target = rt.last_submit_seq.load(Ordering::SeqCst);
        let mut tries = 0u32;
        // Be more patient when a teardown is already in-progress; some
        // drivers require additional time to retire semaphore/fence state.
        while rt.last_complete_seq.load(Ordering::SeqCst) < target && tries < 60_000 {
            let _ = render_device.poll(wgpu::PollType::Wait);
            std::thread::sleep(std::time::Duration::from_millis(1));
            tries = tries.saturating_add(1);
        }
        if rt.last_complete_seq.load(Ordering::SeqCst) < target {
            if log_enabled {
                eprintln!(
                    "native(render): warning: teardown in progress and last-submit did not complete in time"
                );
            }
        } else {
            if log_enabled {
                eprintln!("native(render): teardown in progress; last-submit already completed");
            }
        }
        return;
    }

    if log_enabled {
        eprintln!("native(render): creating encoder label=mcbaise_pre_unconfigure_noop");
    }
    let mut encoder = render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mcbaise_pre_unconfigure_noop"),
    });
    let finished = encoder.finish();

    // Record a lightweight submit sequence id and register a completion
    // callback that records when that sequence completed. Also tag a
    // global submit sequence so we can correlate with acquires.
    let seq = rt.last_submit_seq.fetch_add(1, Ordering::SeqCst) + 1;
    let my_submit_seq = SUBMIT_SEQ_COUNTER.fetch_add(1, Ordering::SeqCst) + 1;
    let flag = Arc::new(AtomicBool::new(false));
    let flag_cb = flag.clone();
    let rt_cb = rt.clone();
    let acq_seq = ACQUIRE_SEQ_COUNTER.load(Ordering::SeqCst);
    render_queue.on_submitted_work_done(move || {
        flag_cb.store(true, Ordering::SeqCst);
        rt_cb.last_complete_seq.store(seq, Ordering::SeqCst);
        if log_enabled {
            eprintln!(
                "native(render): noop on_submitted_work_done seq={} my_submit_seq={} acquire_seq={}",
                seq,
                my_submit_seq,
                acq_seq
            );
        }
    });
    render_queue.submit(std::iter::once(finished));
    if log_enabled {
        eprintln!(
            "native(render): submitted noop submit seq={} my_submit_seq={} acquire_seq={}",
            seq,
            my_submit_seq,
            acq_seq
        );
    }

    let mut tries = 0u32;
    while !flag.load(Ordering::SeqCst) && tries < 60_000 {
        let _ = render_device.poll(wgpu::PollType::Wait);
        std::thread::sleep(std::time::Duration::from_millis(1));
        tries = tries.saturating_add(1);
    }
    if !flag.load(Ordering::SeqCst) {
        if log_enabled {
            eprintln!("native(render): warning: noop submit did not complete within wait limit");
        }
    } else {
        if log_enabled {
            eprintln!(
                "native(render): noop submit completed; safe to proceed with surface reconfigure/drop"
            );
        }

        // Extra defensive wait: even after the on_submitted_work_done
        // callback signals completion, wgpu may still be performing
        // internal "maintain" work that can reference submission indices.
        // Give maintain more tries before we proceed with destructive ops.
        const MAINTAIN_EXTRA_TRIES: usize = 256;
        let mut maintain_tries = 0usize;
        if rt.last_complete_seq.load(Ordering::SeqCst) < seq {
            while rt.last_complete_seq.load(Ordering::SeqCst) < seq && maintain_tries < MAINTAIN_EXTRA_TRIES {
                let _ = render_device.poll(wgpu::PollType::Wait);
                std::thread::sleep(std::time::Duration::from_millis(1));
                maintain_tries += 1;
            }
        }
        // Do a few extra polls to give Device::maintain() time to finish
        for _ in 0..16 {
            let _ = render_device.poll(wgpu::PollType::Wait);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        if maintain_tries >= MAINTAIN_EXTRA_TRIES {
            if log_enabled {
                eprintln!(
                    "native(render): warning: maintain did not observe noop submit completion in time"
                );
            }
        } else {
            if log_enabled {
                eprintln!("native(render): device maintain settled after noop submit");
            }
        }
    }
}

#[derive(Resource, Default, Clone)]
struct GlobalFrameCount(u64);

fn main_frame_tick_system(mut frame: ResMut<GlobalFrameCount>) {
    frame.0 = frame.0.saturating_add(1);
}

fn apply_deferred_window_resolution_change_system(
    frame: Res<GlobalFrameCount>,
    mut deferred: ResMut<DeferredWindowResolutionChange>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
) {
    if !deferred.pending {
        return;
    }
    if frame.0 < deferred.apply_frame {
        return;
    }

    let Some(mut window) = windows.iter_mut().next() else {
        return;
    };

    window
        .resolution
        .set_physical_resolution(deferred.target_w, deferred.target_h);
    deferred.pending = false;
    eprintln!(
        "native: applying deferred window geometry -> {}x{} (frame={})",
        deferred.target_w,
        deferred.target_h,
        frame.0
    );
}

fn schedule_window_geometry_change(
    pending_window_geometry: &mut PendingWindowGeometry,
    deferred: &mut DeferredWindowResolutionChange,
    w: u32,
    h: u32,
    now_frame: u64,
    reason: &'static str,
) {
    // Mark pending immediately so camera gating can kick in this frame,
    // but defer the actual window resize until next frame so the render
    // world sees cameras disabled before the surface is reconfigured.
    pending_window_geometry.pending = true;
    pending_window_geometry.target_w = w;
    pending_window_geometry.target_h = h;
    pending_window_geometry.scheduled_frame = Some(now_frame + 1);

    deferred.pending = true;
    deferred.target_w = w;
    deferred.target_h = h;
    deferred.apply_frame = now_frame + 1;

    eprintln!(
        "native: {reason} window geometry -> {}x{} (defer_apply_frame={}, now_frame={})",
        w,
        h,
        deferred.apply_frame,
        now_frame
    );
}

fn auto_cycle_resolution_system(
    frame: Res<GlobalFrameCount>,
    mut auto: ResMut<AutoResolutionCycle>,
    mut capture_state: ResMut<EguiCaptureState>,
    mut pending_window_geometry: ResMut<PendingWindowGeometry>,
    mut deferred: ResMut<DeferredWindowResolutionChange>,
    mut app_exit: MessageWriter<AppExit>,
) {
    if !auto.active {
        return;
    }
    if frame.0 < auto.start_frame {
        return;
    }
    if auto.every_frames == 0 {
        return;
    }
    if (frame.0 - auto.start_frame) % auto.every_frames != 0 {
        return;
    }
    if pending_window_geometry.pending || deferred.pending {
        return;
    }
    if auto.max_cycles > 0 && auto.cycles_done >= auto.max_cycles {
        app_exit.write(AppExit::Success);
        return;
    }

    let resolutions: &[(u32, u32, &str)] = &[
        (256, 144, "144p (256x144)"),
        (426, 240, "240p (426x240)"),
        (640, 360, "360p (640x360)"),
        (854, 480, "480p (854x480)"),
        (1280, 720, "720p (1280x720)"),
        (1920, 1080, "1080p (1920x1080)"),
        (2560, 1440, "1440p (2560x1440)"),
        (3840, 2160, "2160p (3840x2160)"),
        (1080, 1920, "Phone Portrait (1080x1920)"),
        (1170, 2532, "iPhone Pro (1170x2532)"),
        (1080, 2340, "Modern Phone (1080x2340)"),
    ];

    if auto.next_index >= resolutions.len() {
        auto.next_index = 0;
    }
    let (w, h, _label) = resolutions[auto.next_index];
    capture_state.selected_resolution = auto.next_index as i32;
    auto.next_index = (auto.next_index + 1) % resolutions.len();
    auto.cycles_done = auto.cycles_done.saturating_add(1);

    schedule_window_geometry_change(
        &mut pending_window_geometry,
        &mut deferred,
        w,
        h,
        frame.0,
        "auto-cycled",
    );
}

fn render_frame_tick_system(mut frame: ResMut<GlobalFrameCount>) {
    frame.0 = frame.0.saturating_add(1);
}

// Main-thread system: send a simple marker token to the render thread
// requesting it perform a safe teardown check. We do not attempt to move
// Bevy-internal `WindowSurfaces` types here; that would require deeper
// access. Instead, the render thread will perform the necessary device
// drain and then signal pending teardown flags already registered in
// `GLOBAL_RENDER_TEARDOWNS`.
fn capture_window_surfaces_system() {
    // Only send a teardown request when there is an armed handshake entry.
    // This avoids repeatedly triggering expensive render-thread device
    // idle waits while the app is running normally.
    let map_lock = match GLOBAL_RENDER_TEARDOWNS.get() {
        Some(m) => m,
        None => return,
    };
    let map = match map_lock.lock() {
        Ok(m) => m,
        Err(_) => return,
    };
    if map.is_empty() {
        return;
    }
    let sender_lock = match WINDOW_SURFACE_SENDER.get() {
        Some(s) => s,
        None => return,
    };
    let sender = match sender_lock.lock() {
        Ok(s) => s.clone(),
        Err(_) => return,
    };
    let _ = sender.send(Box::new(()));
}

// Render-thread system: drain tokens from the channel and, for each token,
// perform a strong device drain and then set any pending teardown flags
// in `GLOBAL_RENDER_TEARDOWNS`. This ensures the render thread performs
// the fence/drain before the main world proceeds with destructive drops.
fn render_receive_surfaces_system(world: &mut bevy::ecs::world::World) {
    let recv_lock = match WINDOW_SURFACE_RECEIVER.get() {
        Some(r) => r,
        None => return,
    };
    let recv = match recv_lock.lock() {
        Ok(r) => r,
        Err(_) => return,
    };

    // Drain without blocking
    let mut saw = false;
    while let Ok(_tok) = recv.try_recv() {
        saw = true;
    }
    if !saw {
        return;
    }
    eprintln!("native(render): received teardown request token(s) - performing noop submit + strong device idle and signalling");

    let mut drained_flags: Vec<(u64, Arc<AtomicBool>)> = Vec::new();
    if let Some(map_lock) = GLOBAL_RENDER_TEARDOWNS.get() {
        if let Ok(mut map) = map_lock.lock() {
            for (id, flag) in map.drain() {
                drained_flags.push((id, flag));
            }
        }
    }
    if drained_flags.is_empty() {
        return;
    }

    let rt = RENDER_TEARDOWN
        .get_or_init(|| Arc::new(RenderTeardown::default()))
        .clone();
    rt.in_progress.store(true, Ordering::SeqCst);

    if let (Some(render_device), Some(render_queue)) = (
        world.get_resource::<bevy::render::renderer::RenderDevice>(),
        world.get_resource::<bevy::render::renderer::RenderQueue>(),
    ) {
        submit_noop_and_wait(render_device, render_queue);
        wait_for_device_idle_strong(render_device);
    } else if let Some(render_device) = world.get_resource::<bevy::render::renderer::RenderDevice>() {
        wait_for_device_idle_strong(render_device);
    }

    for (_id, flag) in drained_flags {
        flag.store(true, Ordering::SeqCst);
    }
    rt.in_progress.store(false, Ordering::SeqCst);
}

#[derive(Resource, Default)]
struct ReadbackStaging {
    buffer: Option<BevyBuffer>,
    size: u64,
}

#[derive(Resource, Clone)]
struct PendingRenderAssetDrops {
    images: Vec<Handle<Image>>,
    scheduled_frame: Option<u64>,
    // Hint set on the render thread when the main world requested the drop
    // (placeholder Some(0) -> converted to a real hint using the render
    // world's `GlobalFrameCount`). This is a heuristic proxy for the
    // submission index; the consumer will wait for a few frames after
    // this hint and poll the device before performing destructive drops.
    submit_frame_hint: Option<u64>,
}

impl PendingRenderAssetDrops {
    fn clear_scheduled(&mut self) {
        self.scheduled_frame = None;
        self.submit_frame_hint = None;
    }
}

impl Default for PendingRenderAssetDrops {
    fn default() -> Self {
        PendingRenderAssetDrops {
            images: Vec::new(),
            scheduled_frame: None,
            submit_frame_hint: None,
        }
    }
}

// Final-stage delayed drops queued on the render thread. Each entry stores
// the handle and the earliest render-frame when it is allowed to be destroyed.
#[derive(Resource, Default)]
struct DelayedAttachmentDrops {
    // (handle, earliest_allowed_frame, optional_submission_complete_flag)
    entries: Vec<(Handle<Image>, u64, Option<Arc<AtomicBool>>)>,
}

// When we move `WindowSurfaces` ownership into the render thread we stash
// the removed resource here so it can be dropped on the render thread after
// we have performed a strong device idle and are sure no in-flight work
// references surface acquires anymore.
#[derive(Resource, Default)]
struct StashedWindowSurfaces {
    // (teardown_id, surfaces, earliest_allowed_frame, optional_submission_complete_flag)
    entries: Vec<(u64, WindowSurfaces, u64, Option<Arc<AtomicBool>>)>,
}

#[derive(Resource, Default)]
struct StashedPendingRenderAssetDrops {
    // (teardown_id, pending_drops)
    entries: Vec<(u64, PendingRenderAssetDrops)>,
}


// `SurfaceData` is a private Bevy-internal type and cannot be stashed/dropped
// safely from user code. We intentionally avoid referencing it here.

#[derive(Resource, Default)]
struct StashedDelayedAttachmentDrops {
    // (teardown_id, delayed_drops)
    entries: Vec<(u64, DelayedAttachmentDrops)>,
}


#[derive(Resource, Default)]
struct StashedReadbackStaging {
    // (teardown_id, readback_staging)
    entries: Vec<(u64, ReadbackStaging)>,
}


#[derive(Resource, Default)]
struct StashedAssetsImage {
    // (teardown_id, Assets<Image>)
    entries: Vec<(u64, bevy::asset::Assets<Image>)>,
}

#[derive(Resource, Default)]
struct StashedRenderAssetsGpuImage {
    // (teardown_id, RenderAssets<GpuImage>)
    entries: Vec<(u64, bevy::render::render_asset::RenderAssets<GpuImage>)>,
}

#[derive(Resource, Default)]
struct CompletedDropIds(pub Vec<AssetId<Image>>);


// Guard used to coordinate teardown: prevents new submits while a teardown
// is in progress and records a lightweight submit sequence id so callers
// can wait for the exact noop submission to complete via the queue
// callback. Stored in a global OnceLock so helper functions that don't
// otherwise have access to the RenderApp world can still consult it.
#[derive(Default)]
struct RenderTeardown {
    in_progress: AtomicBool,
    last_submit_seq: AtomicU64,
    last_complete_seq: AtomicU64,
}

static RENDER_TEARDOWN: OnceLock<Arc<RenderTeardown>> = OnceLock::new();





impl bevy::render::extract_resource::ExtractResource for PendingRenderAssetDrops {
    type Source = PendingRenderAssetDrops;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

#[derive(Resource, Default, Clone)]
struct PendingWindowGeometry {
    pending: bool,
    target_w: u32,
    target_h: u32,
    scheduled_frame: Option<u64>,
}

// Env-var driven automatic cycling through the same preset resolutions as the
// right-click path. This exists to reproduce/validate resize safety without
// needing interactive input.
#[derive(Resource, Default)]
struct AutoResolutionCycle {
    active: bool,
    every_frames: u64,
    start_frame: u64,
    max_cycles: u32,
    cycles_done: u32,
    next_index: usize,
}

// Right-click resolution cycling is prone to driver/backend races if we apply the
// OS/window size change in the same frame we set our gating flags.
// We defer the actual Window resolution update by one main-world frame so the
// render world sees cameras disabled before the surface reconfiguration occurs.
#[derive(Resource, Default)]
struct DeferredWindowResolutionChange {
    pending: bool,
    target_w: u32,
    target_h: u32,
    apply_frame: u64,
}

impl bevy::render::extract_resource::ExtractResource for PendingWindowGeometry {
    type Source = PendingWindowGeometry;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

// Controls whether window resolution changes should be applied as logical
// (layout/DPI-independent) sizes or as physical framebuffer pixel sizes.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
enum ResolutionSetMode {
    Logical,
    Physical,
}

impl Default for ResolutionSetMode {
    fn default() -> Self {
        ResolutionSetMode::Logical
    }
}

// Phase machine used by the resize automation (mirror of `resize_test.rs`).
#[derive(Copy, Clone)]
enum Phase {
    ContractingY,
    ContractingX,
    ExpandingY,
    ExpandingX,
}

// Resize automation: toggled by the UI resize icon to exercise the
// same window-geometry stepping logic as `resize_test.rs`.
#[derive(Resource)]
struct ResizeAutomation {
    active: bool,
    env_forced: bool,
    width: u16,
    height: u16,
    phase: Phase,
    first_frame: bool,
    next_step_frame: u64,
    step_every_frames: u64,
    start_frame: u64,
    max_steps: u32,
    steps_done: u32,
    exit_requested: bool,
    seeded_from_window: bool,
}

impl Default for ResizeAutomation {
    fn default() -> Self {
        ResizeAutomation {
            active: false,
            env_forced: false,
            width: 401,
            height: 401,
            phase: Phase::ContractingY,
            first_frame: false,
            next_step_frame: 0,
            step_every_frames: 12,
            start_frame: 0,
            max_steps: 0,
            steps_done: 0,
            exit_requested: false,
            seeded_from_window: false,
        }
    }
}

#[derive(Resource, Default)]
struct CameraRestartRequested {
    requested: bool,
    // When a camera restart is requested we create a shared flag and place
    // it into the global teardown map. The render-subapp will run a cleanup
    // task, set the flag to true when it's safe, and the main app will then
    // perform the destructive camera respawn on the next frame.
    pending_flag: Option<Arc<AtomicBool>>,
}

#[derive(Resource, Default)]
struct SceneRestartRequested {
    requested: bool,
    pending_flag: Option<Arc<AtomicBool>>,
}

#[derive(Resource, Default)]
struct TeardownState {
    active: bool,
    frames_left: u8,
}

// Global coordination structures used to handshake between the main App
// world and the RenderApp cleanup stage. We use a small global table of
// Arc<AtomicBool> entries so both worlds can communicate safely without
// attempting to move resources across world boundaries.
static GLOBAL_RENDER_TEARDOWNS: OnceLock<Mutex<HashMap<u64, Arc<AtomicBool>>>> = OnceLock::new();
// Per-teardown countdown state inserted on the render thread. When the
// main world arms a teardown we move the entry here and count down a small
// number of render frames before finally signalling the AtomicBool. This
// provides an additional frame window to ensure drivers release any
// SurfaceTexture/acquire-semaphore resources before we unconfigure or drop
// the underlying surface, which mitigates Vulkan driver races observed
// during fast resize cycles.
static GLOBAL_RENDER_TEARDOWN_COUNTDOWNS: OnceLock<Mutex<HashMap<u64, (Arc<AtomicBool>, u8)>>> = OnceLock::new();
static NEXT_TEARDOWN_ID: AtomicU64 = AtomicU64::new(1);

// Channel used to transfer WindowSurfaces (or their owning wrappers)
// from the main thread into the render thread. The sender is initialized
// during app startup and is accessible from main-world systems. The
// render-world will take ownership of the receiver and drain it on the
// render thread, moving items into a thread-local stash for dropping.
// Use a boxed Any token so we don't need to move Bevy-internal types here.
static WINDOW_SURFACE_SENDER: OnceLock<Mutex<Sender<Box<dyn Any + Send>>>> = OnceLock::new();
static WINDOW_SURFACE_RECEIVER: OnceLock<Mutex<Receiver<Box<dyn Any + Send>>>> = OnceLock::new();

// Registry mapping an acquire_seq -> list of waiters to notify when a
// submit that referenced that acquire completes. We use a simple
// crossbeam channel sender so the notifier (render submit probe) can
// signal waiting threads/tasks without requiring async executor state.
// Shared sender wrapper used for per-acquire notifications.
type AcquireWaitSender = Arc<Mutex<Option<oneshot::Sender<()>>>>;
// Map acquire_seq -> list of shared oneshot senders. The shared wrapper
// allows either the queue callback or the acquire-notifier to fulfill the
// oneshot exactly once.
static ACQUIRE_WAITS: OnceLock<Mutex<HashMap<u64, Vec<AcquireWaitSender>>>> = OnceLock::new();

// Holders for WindowSurfaces moved out of the main world so they can be
// dropped on the render thread. Keyed by the same teardown id used for the
// render cleanup handshake. Accessed from both worlds; synchronization via
// a Mutex ensures safe coordination.
// (experimental surface-holder removed)

#[derive(Resource)]
struct LoadingState {
    stage: LoadingStage,
    frames_left: u8,
}

#[derive(PartialEq, Eq)]
enum LoadingStage {
    LoadingAssets,
    WaitingForGpu,
    Ready,
}

impl Default for LoadingState {
    fn default() -> Self {
        LoadingState {
            stage: LoadingStage::LoadingAssets,
            frames_left: 0,
        }
    }
}

#[derive(Component)]
struct GpuReadbackCaptureCamera {
    index: u8,
}

fn ensure_gpu_capture_camera_exists(
    mut commands: Commands,
    readback: Option<Res<GpuReadbackImage>>,
    existing: Query<(Entity, &GpuReadbackCaptureCamera)>,
    capture_state: Res<EguiCaptureState>,
    multi_view: Res<MultiView>,
    render_device: Res<bevy::render::renderer::RenderDevice>,
) {
    let Some(readback) = readback else { return; };
    if readback.0 == Handle::default() {
        return;
    }

    let desired_count = if capture_state.capture_multi {
        multi_view.count as u8
    } else {
        1
    };

    // Despawn extras. Before removing camera entities that may hold
    // GPU-backed resources, perform a strong device drain to reduce the
    // chance of destroying swapchain/surface resources while still in use.
    if existing.iter().len() > 0 {
        wait_for_device_idle_strong(&*render_device);
    }
    for (entity, cap_cam) in &existing {
        if cap_cam.index >= desired_count {
            commands.entity(entity).despawn();
        }
    }

    // Spawn missing
    let mut present = vec![false; desired_count as usize];
    for (_, cap_cam) in &existing {
        if (cap_cam.index as usize) < present.len() {
            present[cap_cam.index as usize] = true;
        }
    }

    for i in 0..desired_count {
        if present.get(i as usize).copied().unwrap_or(false) {
            continue;
        }

        commands.spawn((
            Camera3d::default(),
            Camera { order: 100, ..default() },
            GpuReadbackCaptureCamera { index: i },
            Name::new(format!("gpu_readback_capture_camera_{}", i)),
        ));
    }
}

fn update_gpu_capture_viewports(
    readback: Option<Res<GpuReadbackImage>>,
    capture_state: Res<EguiCaptureState>,
    multi_view: Res<MultiView>,
    images: Res<Assets<Image>>,
    pending: Res<GpuReadbackPending>,
    mut cap_cams: Query<(&GpuReadbackCaptureCamera, &mut Camera)>,
) {
    // Update viewports while a readback is active. We need per-frame viewport
    // configuration to ensure capture cameras remain aligned during recording.
    if !pending.active { return; }
    let Some(readback) = readback else { return; };
    let Some(img) = images.get(&readback.0) else { return; };
    let size = img.texture_descriptor.size;
    let w = size.width;
    let h = size.height;
    if w == 0 || h == 0 { return; }

    let count = if capture_state.capture_multi {
        multi_view.count.clamp(1, MultiView::MAX_VIEWS) as u32
    } else {
        1
    };
    
    let base_h = (h / count).max(1);
    let mut y = 0u32;

    for idx in 0..count {
        let view_h = if idx + 1 == count {
            h.saturating_sub(y).max(1)
        } else {
            base_h
        };

        for (cap_cam, mut cam) in &mut cap_cams {
            if cap_cam.index as u32 == idx {
                let new_vp = Viewport {
                    physical_position: UVec2::new(0, y),
                    physical_size: UVec2::new(w, view_h),
                    ..default()
                };
                cam.viewport = Some(new_vp);
                break;
            }
        }
        y = y.saturating_add(base_h);
    }
}

fn sync_gpu_capture_cameras_from_views(
    pending: Res<GpuReadbackPending>,
    view_cams: Query<(&ViewCamera, &Projection, &Transform, Option<&DistanceFog>), Without<GpuReadbackCaptureCamera>>,
    mut cap_cams: Query<(&GpuReadbackCaptureCamera, &mut Projection, &mut Transform, &mut Camera), Without<ViewCamera>>,
) {
    // Sync transforms/projections while readback is active so capture cameras
    // follow the corresponding view cameras.
    if !pending.active { return; }
    for (cap_cam, mut proj, mut transform, mut cam) in &mut cap_cams {
        let mut found = false;
        for (view_cam, view_proj, view_transform, _fog) in &view_cams {
            if view_cam.index == cap_cam.index {
                *proj = view_proj.clone();
                *transform = view_transform.clone();
                found = true;
                break;
            }
        }
        // If no matching view camera, disable it (shouldn't happen with ensure system)
        if !found {
            cam.is_active = false;
        }
    }
}

fn sync_gpu_capture_camera_fog(
    pending: Res<GpuReadbackPending>,
    view_cams: Query<(&ViewCamera, Option<&DistanceFog>), Without<GpuReadbackCaptureCamera>>,
    mut cap_cams: Query<(Entity, &GpuReadbackCaptureCamera, Option<&DistanceFog>), Without<ViewCamera>>,
    mut commands: Commands,
) {
    // Fog component insert/remove is mutating; only apply fog sync on the
    // frame the readback is armed to avoid per-frame churn.
    if !pending.active || !pending.just_armed { return; }
    for (entity, cap_cam, existing_fog) in &mut cap_cams {
        for (view_cam, fog) in &view_cams {
            if view_cam.index == cap_cam.index {
                match (fog, existing_fog) {
                    (Some(f), Some(_)) => {
                        // Cloning a component into an insert() always triggers a change.
                        // Since we can't easily PartialEq DistanceFog, we'll just insert 
                        // and trust that it's okay for now, OR we could avoid inserting
                        // if we're not armed. 
                        // Actually, distance fog is small. The main issue was per-frame
                        // insertion when NOT needed.
                        commands.entity(entity).insert((*f).clone());
                    }
                    (Some(f), None) => {
                        commands.entity(entity).insert((*f).clone());
                    }
                    (None, Some(_)) => {
                        commands.entity(entity).remove::<DistanceFog>();
                    }
                    (None, None) => {}
                }
                break;
            }
        }
    }
}

fn configure_gpu_capture_camera_settings(
    readback: Option<Res<GpuReadbackImage>>,
    pending: Res<GpuReadbackPending>,
    images: Res<Assets<Image>>,
    mut cap_cams: Query<(&GpuReadbackCaptureCamera, &mut Camera)>,
) {
    if !pending.active {
        for (_, mut cam) in &mut cap_cams {
            if cam.is_active {
                cam.is_active = false;
            }
        }
        return;
    }
    let Some(readback) = readback else { return; };
    if readback.0 == Handle::default() {
        return;
    }

    // Ensure the target image has a valid non-zero size before attaching it
    if let Some(img) = images.get(&readback.0) {
        let s = img.texture_descriptor.size;
        if s.width == 0 || s.height == 0 {
            // Can't attach a zero-sized image as a render target; wait until resized
            return;
        }
    } else {
        // Image asset not yet available in asset storage; skip for now
        return;
    }
    for (cap_cam, mut cam) in &mut cap_cams {
        cam.target = RenderTarget::Image(readback.0.clone().into());

        let clear_color = if cap_cam.index == 0 {
            ClearColorConfig::Default
        } else {
            ClearColorConfig::None
        };

        // Set per-camera clear color instead of assigning the private CameraOutputMode
        cam.clear_color = clear_color;

        if cam.order != cap_cam.index as isize {
            cam.order = cap_cam.index as isize;
        }
    }
}

// (Old sync function removed, replaced by sync_gpu_capture_cameras_from_views)

fn ensure_readback_image_exists(
    mut readback: ResMut<GpuReadbackImage>,
    mut images: ResMut<Assets<Image>>,
) {
    if readback.0 != Handle::default() {
        return;
    }

    // Start with a tiny image; it will be resized to the window size.
    let mut image = Image::new_fill(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = bevy::render::render_resource::TextureUsages::RENDER_ATTACHMENT
        | bevy::render::render_resource::TextureUsages::COPY_SRC
        | bevy::render::render_resource::TextureUsages::TEXTURE_BINDING;

    // Label the readback image texture so wgpu/core logs include a recognisable
    // label when the underlying GPU texture is created/destroyed.
    #[allow(unused_assignments)]
    {
        // Try to set a descriptive label; different bevy/wgpu versions store
        // the label as either Option<String> or Option<&'static str> on the
        // descriptor. Attempt the common owned-String form first.
        #[allow(unused_mut)]
        let _ = match &mut image.texture_descriptor.label {
            // common case: Option<String>
            Some(_) => {
                image.texture_descriptor.label = Some("mcbaise_readback_image");
                true
            }
            None => {
                // If there's no label field or it's a different type, ignore.
                // We still continue; the staging buffer and command encoder
                // already have labels.
                false
            }
        };
    }

    let handle = images.add(image);
    readback.0 = handle;
}

fn resize_readback_image_to_window(
    windows: Query<&Window, With<PrimaryWindow>>,
    readback: Option<Res<GpuReadbackImage>>,
    mut images: ResMut<Assets<Image>>,
    capture_state: Res<EguiCaptureState>,
    pending: Res<GpuReadbackPending>,
) {
    if pending.active { return; }
    let Some(readback_res) = readback else { return; };
    let Some(primary_window) = windows.iter().next() else { return; };
    
    let target_width = (primary_window.physical_width() as f32 * capture_state.output_scale).round().max(1.0) as u32;
    let target_height = (primary_window.physical_height() as f32 * capture_state.output_scale).round().max(1.0) as u32;
    
    if target_width == 0 || target_height == 0 { return; }

    if let Some(img) = images.get_mut(&readback_res.0) {
        let current_size = img.texture_descriptor.size;
        if current_size.width != target_width || current_size.height != target_height {
            img.resize(Extent3d {
                width: target_width,
                height: target_height,
                depth_or_array_layers: 1,
            });
        }
    }
}

fn arm_cameras_for_gpu_readback(
    mut pending: ResMut<GpuReadbackPending>,
    readback: Option<Res<GpuReadbackImage>>,
    mut cap_cam: Query<&mut Camera, With<GpuReadbackCaptureCamera>>,
) {
    let seq = READBACK_REQUEST_SEQ.load(Ordering::SeqCst);
    if seq == 0 {
        return;
    }

    if pending.active {
        return;
    }

    let Some(readback) = readback else {
        // If this happens, extraction into the render world will also fail.
        let last = READBACK_DEBUG_LAST_SEQ_ARM.load(Ordering::SeqCst);
        if last != seq {
            READBACK_DEBUG_LAST_SEQ_ARM.store(seq, Ordering::SeqCst);
            wasm_dbg_kv(
                "wasm:gpu_readback: cannot arm (missing GpuReadbackImage) seq=",
                &format!("{seq}"),
            );
        }
        return;
    };

    if readback.0 == Handle::default() {
        let last = READBACK_DEBUG_LAST_SEQ_ARM.load(Ordering::SeqCst);
        if last != seq {
            READBACK_DEBUG_LAST_SEQ_ARM.store(seq, Ordering::SeqCst);
            wasm_dbg_kv(
                "wasm:gpu_readback: cannot arm (GpuReadbackImage not initialized) seq=",
                &format!("{seq}"),
            );
        }
        return;
    }

    // Enable ALL capture cameras for a few frames.
    if cap_cam.is_empty() {
        let last = READBACK_DEBUG_LAST_SEQ_ARM.load(Ordering::SeqCst);
        if last != seq {
            READBACK_DEBUG_LAST_SEQ_ARM.store(seq, Ordering::SeqCst);
            wasm_dbg_kv(
                "wasm:gpu_readback: cannot arm (missing capture camera) seq=",
                &format!("{seq}"),
            );
        }
        return;
    }

    for mut cam in &mut cap_cam {
        cam.is_active = true;
    }

    pending.active = true;
    pending.just_armed = true;
    // Keep the capture camera alive long enough for extraction + one full render + copy.
    pending.frames_left = 3;

    // Advance the render-world readback state machine.
    READBACK_GPU_PHASE.store(1, Ordering::SeqCst);

    let last = READBACK_DEBUG_LAST_SEQ_ARM.load(Ordering::SeqCst);
    if last != seq {
        READBACK_DEBUG_LAST_SEQ_ARM.store(seq, Ordering::SeqCst);
        wasm_dbg_kv(
            "wasm:gpu_readback: armed capture cameras seq=",
            &format!("{seq} count={} frames_left={}", cap_cam.iter().count(), pending.frames_left),
        );
    }
}

fn restore_cameras_after_gpu_readback(
    mut pending: ResMut<GpuReadbackPending>,
    mut cap_cam: Query<&mut Camera, With<GpuReadbackCaptureCamera>>,
) {
    if !pending.active {
        return;
    }

    // We arm cameras during `Update` so extraction can see the new targets.
    // If we restore in the same `Update`, the render world never sees the offscreen target.
    // Skip one tick to allow a single frame to render into the readback image.
    if pending.just_armed {
        pending.just_armed = false;
        return;
    }

    if pending.frames_left > 0 {
        pending.frames_left -= 1;
        return;
    }

    for mut cam in &mut cap_cam {
        cam.is_active = false;
    }
    pending.active = false;
}

fn gpu_readback_render_system(
    render_device: Res<bevy::render::renderer::RenderDevice>,
    render_queue: Res<bevy::render::renderer::RenderQueue>,
    gpu_images: Res<bevy::render::render_asset::RenderAssets<GpuImage>>,
    mut staging_res: ResMut<ReadbackStaging>,
    readback: Option<Res<GpuReadbackImage>>,
) {
    let seq = READBACK_REQUEST_SEQ.load(Ordering::SeqCst);
    if seq == 0 {
        return;
    }

    let Some(readback) = readback else {
        let last = READBACK_DEBUG_LAST_SEQ_RENDER.load(Ordering::SeqCst);
        if last != seq {
            READBACK_DEBUG_LAST_SEQ_RENDER.store(seq, Ordering::SeqCst);
            wasm_dbg_kv(
                "wasm:gpu_readback: render world missing GpuReadbackImage seq=",
                &format!("{seq}"),
            );
        }
        return;
    };

    let phase = READBACK_GPU_PHASE.load(Ordering::SeqCst);
    if phase == 0 {
        // Not armed yet (camera hasn't been retargeted).
        let last = READBACK_DEBUG_LAST_SEQ_RENDER.load(Ordering::SeqCst);
        if last != seq {
            READBACK_DEBUG_LAST_SEQ_RENDER.store(seq, Ordering::SeqCst);
            wasm_dbg_kv(
                "wasm:gpu_readback: render sees seq but phase=0 seq=",
                &format!("{seq}"),
            );
        }
        return;
    }
    if phase == 1 {
        // Wait one full render so the camera target size / view extraction settles.
        READBACK_GPU_PHASE.store(2, Ordering::SeqCst);
        return;
    }
    if phase == 2 {
        // Wait an additional frame so we don't copy the freshly-cleared target
        // before the capture camera has actually rendered into it.
        READBACK_GPU_PHASE.store(3, Ordering::SeqCst);
        return;
    }

    let Some(gpu_image) = gpu_images.get(readback.0.id()) else {
        let last = READBACK_DEBUG_LAST_SEQ_RENDER.load(Ordering::SeqCst);
        if last != seq {
            READBACK_DEBUG_LAST_SEQ_RENDER.store(seq, Ordering::SeqCst);
            wasm_dbg_kv(
                "wasm:gpu_readback: render missing GpuImage for handle seq=",
                &format!("{seq}"),
            );
        }
        return;
    };

    wasm_dbg_kv(
        "wasm:gpu_readback: have gpu_image ",
        &format!("(size={}x{})", gpu_image.size.width, gpu_image.size.height),
    );

    // Prevent overlapping mapping operations: if a previous map is still in-flight,
    // skip this readback request. This avoids creating many staging buffers and
    // racing with buffer lifetimes which can trigger wgpu validation errors.
    if READBACK_MAP_IN_FLIGHT.load(Ordering::SeqCst) != 0 {
        wasm_dbg_kv(
            "gpu_readback: skipping because map in flight ",
            &format!("seq={}", seq),
        );
        return;
    }

    // We may be called multiple times per frame; mark handled as soon as we schedule the copy.
    READBACK_REQUEST_SEQ.store(0, Ordering::SeqCst);
    READBACK_GPU_PHASE.store(0, Ordering::SeqCst);

    // Determine texture size.
    let size = gpu_image.size;
    let width = size.width.max(1);
    let height = size.height.max(1);
    let bytes_per_pixel: u32 = 4;
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
    let buffer_size = (padded_bytes_per_row as u64) * (height as u64);

    eprintln!("gpu_readback: need staging {} bytes ({}x{})", buffer_size, width, height);

    // Reuse a persistent staging buffer when possible to reduce allocation churn.
    let buffer = if staging_res.size >= buffer_size && staging_res.buffer.is_some() {
        staging_res.buffer.as_ref().unwrap().clone()
    } else {
        eprintln!("gpu_readback: creating new staging buffer {} bytes", buffer_size);
        let b = render_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcbaise_readback_staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        staging_res.buffer = Some(b.clone());
        staging_res.size = buffer_size;
        b
    };

    let mut encoder = render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mcbaise_readback_encoder"),
    });

    eprintln!("native(render): create encoder label=mcbaise_readback_encoder frame_seq={}", READBACK_REQUEST_SEQ.load(Ordering::SeqCst));

    encoder.copy_texture_to_buffer(
        gpu_image.texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let finished = encoder.finish();
    let my_seq = SUBMIT_SEQ_COUNTER.fetch_add(1, Ordering::SeqCst) + 1;
    render_queue.on_submitted_work_done(move || {
        eprintln!("native(render): readback submit seq={} completed", my_seq);
    });
    let acq_seq = ACQUIRE_SEQ_COUNTER.load(Ordering::SeqCst);
    render_queue.submit(std::iter::once(finished));
    eprintln!("gpu_readback: submitted encoder for readback (size={}x{}) seq={} acquire_seq={}", width, height, my_seq, acq_seq);

    // On wasm/WebGPU, some browsers/drivers appear to need explicit polling to
    // progress mapping callbacks reliably.
    let _ = render_device.poll(wgpu::PollType::Poll);

    // Map async and post to parent when ready.
    let map_buffer = buffer.clone();
    let map_slice = map_buffer.slice(..);
    let buffer2 = buffer.clone();
    READBACK_MAP_IN_FLIGHT.store(1, Ordering::SeqCst);
    eprintln!("gpu_readback: mapping staging buffer (size={})", staging_res.size);
    map_slice.map_async(wgpu::MapMode::Read, move |res| {
        if let Err(e) = res {
            wasm_dbg_kv("gpu_readback: map_async error ", &format!("{:?}", e));
            READBACK_MAP_IN_FLIGHT.store(0, Ordering::SeqCst);
            return;
        }
        // Re-slice after mapping to avoid capturing a non-'static slice in the callback.
        let data = buffer2.slice(..).get_mapped_range();
        let mut out = vec![0u8; (unpadded_bytes_per_row as usize) * (height as usize)];
        for row in 0..(height as usize) {
            let src_start = row * (padded_bytes_per_row as usize);
            let dst_start = row * (unpadded_bytes_per_row as usize);
            out[dst_start..dst_start + (unpadded_bytes_per_row as usize)]
                .copy_from_slice(&data[src_start..src_start + (unpadded_bytes_per_row as usize)]);
        }

        // checksum/sample
        let sample_len = out.len().min(64);
        let mut checksum: u32 = 0;
        for i in 0..sample_len {
            checksum = checksum.wrapping_add(out[i] as u32);
        }

        eprintln!("gpu_readback: mapped {} bytes ({}x{}, checksum_sample={})", out.len(), width, height, checksum);
        wasm_dbg_kv(
            "gpu_readback: mapped ",
            &format!("{} bytes ({}x{}, checksum_sample={})", out.len(), width, height, checksum),
        );

        #[cfg(target_arch = "wasm32")]
        {
            let win = web_sys::window().unwrap();
            let msg = js_sys::Object::new();
            let uint8 = js_sys::Uint8Array::from(out.as_slice());
            let pixels_buf = uint8.buffer();
            let _ = js_sys::Reflect::set(&msg, &wasm_bindgen::JsValue::from_str("type"), &wasm_bindgen::JsValue::from_str("wasm_pixels"));
            // Send the ArrayBuffer itself for maximum interop.
            let _ = js_sys::Reflect::set(&msg, &wasm_bindgen::JsValue::from_str("pixels"), &pixels_buf);
            let _ = js_sys::Reflect::set(&msg, &wasm_bindgen::JsValue::from_str("width"), &wasm_bindgen::JsValue::from_f64(width as f64));
            let _ = js_sys::Reflect::set(&msg, &wasm_bindgen::JsValue::from_str("height"), &wasm_bindgen::JsValue::from_f64(height as f64));
            let _ = js_sys::Reflect::set(&msg, &wasm_bindgen::JsValue::from_str("checksum"), &wasm_bindgen::JsValue::from_f64(checksum as f64));
            let _ = js_sys::Reflect::set(&msg, &wasm_bindgen::JsValue::from_str("gpu"), &wasm_bindgen::JsValue::from_bool(true));
            let sample_arr = js_sys::Array::new();
            for i in 0..out.len().min(16) {
                sample_arr.push(&wasm_bindgen::JsValue::from_f64(out[i] as f64));
            }
            let _ = js_sys::Reflect::set(&msg, &wasm_bindgen::JsValue::from_str("sample"), &sample_arr);

            // If this wasm instance runs inside an iframe, transfer to parent.
            // If it's top-level, avoid transferring to self (can yield 0-byte buffers).
            let in_frame = win.frame_element().ok().flatten().is_some();
            if in_frame {
                if let Ok(Some(parent)) = win.parent() {
                    let transfer = js_sys::Array::new();
                    transfer.push(&pixels_buf);
                    let _ = parent.post_message_with_transfer(&wasm_bindgen::JsValue::from(msg), "*", &transfer);
                    wasm_dbg("wasm:gpu_readback: posted wasm_pixels to parent (transfer)");
                } else {
                    let _ = win.post_message(&wasm_bindgen::JsValue::from(msg), "*");
                    wasm_dbg("wasm:gpu_readback: posted wasm_pixels to self (no parent)");
                }
            } else {
                let _ = win.post_message(&wasm_bindgen::JsValue::from(msg), "*");
                wasm_dbg("wasm:gpu_readback: posted wasm_pixels to self (no transfer)");
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Native: write PNGs to disk if `MCBAISE_READBACK_OUTPUT` is set.
            if let Ok(dir) = std::env::var("MCBAISE_READBACK_OUTPUT") {
                let _ = std::fs::create_dir_all(&dir);
                let t = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis();
                let fname = format!("{}/frame_{}_{}x{}.png", dir, t, width, height);
                // `out` is RGBA packed, width x height. Write atomically via tmp file + rename
                let tmp = format!("{}.tmp", fname);
                match std::fs::File::create(&tmp) {
                    Ok(mut fout) => {
                        use image::codecs::png::PngEncoder;
                        use image::ColorType;
                        use image::ImageEncoder;
                        let encoder = PngEncoder::new(&mut fout);
                        match encoder.write_image(&out, width as u32, height as u32, ColorType::Rgba8) {
                            Ok(()) => {
                                // ensure file is flushed/dropped before rename
                                drop(fout);
                                if let Err(e) = std::fs::rename(&tmp, &fname) {
                                    eprintln!("failed to rename readback tmp file: {} -> {}: {}", tmp, fname, e);
                                } else {
                                    eprintln!("native: wrote PNG {}", fname);
                                }
                            }
                            Err(e) => {
                                eprintln!("failed to encode PNG to tmp file {}: {}", tmp, e);
                                let _ = std::fs::remove_file(&tmp);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("failed to create readback tmp file {}: {}", tmp, e);
                    }
                }

                // If oneshot requested, clear the env var so subsequent frames aren't recorded.
                if std::env::var("MCBAISE_READBACK_ONESHOT").as_deref().ok() == Some("1") {
                    unsafe { let _ = std::env::remove_var("MCBAISE_READBACK_OUTPUT"); }
                    unsafe { let _ = std::env::remove_var("MCBAISE_READBACK_ONESHOT"); }
                }
            }
        }

        drop(data);
        buffer2.unmap();
        READBACK_MAP_IN_FLIGHT.store(0, Ordering::SeqCst);
    });
}

fn gpu_readback_poll_system(render_device: Res<bevy::render::renderer::RenderDevice>) {
    if READBACK_MAP_IN_FLIGHT.load(Ordering::SeqCst) == 0 {
        return;
    }
    let _ = render_device.poll(wgpu::PollType::Poll);
}

#[cfg(target_arch = "wasm32")]
const META_BYTES: &[u8] = include_bytes!("../../../assets/model/fullbody_default.meta.json");

#[cfg(target_arch = "wasm32")]
const TENSOR_LZ4_BYTES: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/fullbody_default.safetensors.lz4"
));

const VIDEO_ID: &str = "v2hcW03gcus";

// ---------------------------- native mpv sync (optional) ----------------------------

// This is intentionally opt-in:
// - compile-time: `--features native-mpv`
// - runtime: set `MCBAISE_NATIVE_MPV=1`
// It spawns `mpv` and makes mpv authoritative for time/play state via mpv JSON IPC.

#[cfg(all(windows, not(target_arch = "wasm32"), feature = "native-mpv"))]
mod native_mpv {
    use std::io::{BufRead, BufReader as StdBufReader};
    use std::process::{Child, Command as ProcessCommand, Stdio};
    use std::sync::mpsc::TryRecvError;
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::thread;
    use std::thread::JoinHandle;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use serde_json::json;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::windows::named_pipe::ClientOptions;

    #[derive(Debug, Clone)]
    #[allow(dead_code)]
    pub enum Command {
        SetPlaying(bool),
        SeekSeconds(f32),
        Shutdown,
    }

    #[derive(Debug, Clone)]
    pub enum Event {
        State { time_sec: f32, playing: bool },
        Error(String),
    }

    fn command_exists(cmd: &str) -> bool {
        ProcessCommand::new(cmd)
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    pub fn spawn(
        url: String,
        mpv_path: Option<String>,
        extra_args: Vec<String>,
    ) -> (Sender<Command>, Receiver<Event>, JoinHandle<()>) {
        let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();
        let (evt_tx, evt_rx) = mpsc::channel::<Event>();

        let join = thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_io()
                .enable_time()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    let _ =
                        evt_tx.send(Event::Error(format!("failed to create tokio runtime: {e}")));
                    return;
                }
            };

            runtime.block_on(async move {
                let mpv_exe = mpv_path.unwrap_or_else(|| "mpv".to_string());

                let looks_like_youtube = url.contains("youtube.com") || url.contains("youtu.be");
                let disable_auto_cookies = std::env::var("MCBAISE_MPV_DISABLE_AUTO_COOKIES")
                    .ok()
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);

                let cookie_file = std::env::var("MCBAISE_MPV_COOKIES_FILE")
                    .ok()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
                    .or_else(|| {
                        if !disable_auto_cookies && std::path::Path::new("cookies.txt").exists() {
                            Some("cookies.txt".to_string())
                        } else {
                            None
                        }
                    });

                let cookie_source = std::env::var("MCBAISE_MPV_COOKIES_FROM_BROWSER")
                    .ok()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
                    .or_else(|| {
                        if looks_like_youtube && !disable_auto_cookies && cookie_file.is_none() {
                            Some("chrome".to_string())
                        } else {
                            None
                        }
                    });

                let has_ytdlp = command_exists("yt-dlp");
                let has_youtubedl = command_exists("youtube-dl");

                if !has_ytdlp {
                    if has_youtubedl {
                        println!("[native-mpv] warning: yt-dlp not found; falling back to youtube-dl (often broken). Install yt-dlp.");
                        let _ = evt_tx.send(Event::Error(
                            "yt-dlp not found; mpv will fall back to youtube-dl which often breaks on modern YouTube. Install yt-dlp and retry."
                                .to_string(),
                        ));
                    } else {
                        println!("[native-mpv] warning: yt-dlp not found (recommended for YouTube URLs). Install yt-dlp.");
                        let _ = evt_tx.send(Event::Error(
                            "yt-dlp not found (recommended for YouTube URLs). Install yt-dlp and retry."
                                .to_string(),
                        ));
                    }
                }

                // Preflight: if we're going to rely on cookies-from-browser, run yt-dlp directly
                // first to get a clear error (Windows Chrome cookie DB lock, bot-check, etc.).
                // This avoids confusing mpv ytdl_hook failures like "NoneType has no attribute decode".
                if looks_like_youtube
                    && has_ytdlp {
                        // NOTE: `yt-dlp` preflight can sometimes be slow. Never block mpv launch
                        // indefinitely; time out and continue (especially when using cookies.txt).
                        let preflight_timeout_sec: u64 = std::env::var("MCBAISE_MPV_YTDLP_PREFLIGHT_TIMEOUT_SEC")
                            .ok()
                            .and_then(|v| v.trim().parse::<u64>().ok())
                            .filter(|&v| v > 0)
                            .unwrap_or(20);

                        let url2 = url.clone();
                        let cookie_file2 = cookie_file.clone();
                        let cookie_source2 = cookie_source.clone();

                        let preflight_label = if let Some(path) = cookie_file2.as_ref() {
                            format!("--cookies {path}")
                        } else if let Some(source) = cookie_source2.as_ref() {
                            format!("cookies-from-browser={source}")
                        } else {
                            "(no cookies)".to_string()
                        };

                        println!(
                            "[native-mpv] yt-dlp preflight ({preflight_label}) timeout={preflight_timeout_sec}s"
                        );

                        let preflight_task = tokio::task::spawn_blocking(move || {
                            let mut cmd = ProcessCommand::new("yt-dlp");
                            cmd.arg("--simulate").arg("--no-playlist");

                            if let Some(path) = cookie_file2.as_ref() {
                                cmd.arg("--cookies").arg(path);
                            } else if let Some(source) = cookie_source2.as_ref() {
                                cmd.arg("--cookies-from-browser").arg(source);
                            }

                            cmd.arg(&url2).stdin(Stdio::null());
                            cmd.output()
                        });

                        let output: Option<std::process::Output> = match tokio::time::timeout(
                            Duration::from_secs(preflight_timeout_sec),
                            preflight_task,
                        )
                        .await
                        {
                            Ok(joined) => match joined {
                                Ok(res) => match res {
                                    Ok(out) => Some(out),
                                    Err(e) => {
                                        let _ = evt_tx.send(Event::Error(format!(
                                            "yt-dlp preflight failed to run: {e}. Continuing to launch mpv."
                                        )));
                                        None
                                    }
                                },
                                Err(e) => {
                                    let _ = evt_tx.send(Event::Error(format!(
                                        "yt-dlp preflight task failed: {e}. Continuing to launch mpv."
                                    )));
                                    None
                                }
                            },
                            Err(_) => {
                                let _ = evt_tx.send(Event::Error(format!(
                                    "yt-dlp preflight timed out after {preflight_timeout_sec}s; continuing to launch mpv."
                                )));
                                None
                            }
                        };

                        if let Some(out) = output {
                            if out.status.success() {
                                // ok
                            } else {
                                let stderr = String::from_utf8_lossy(&out.stderr);
                                let stdout = String::from_utf8_lossy(&out.stdout);
                                let mut msg = String::new();
                                msg.push_str("yt-dlp preflight failed. ");

                                let combined = format!("{}\n{}", stdout, stderr);
                                let combined_lc = combined.to_ascii_lowercase();

                                // Common Windows/YouTube failure modes with actionable guidance.
                                if combined_lc.contains("only images are available")
                                    || combined_lc.contains("n challenge")
                                    || combined_lc.contains("js challenge")
                                    || combined_lc.contains("remote components challenge")
                                {
                                    let has_deno = command_exists("deno");
                                    msg.push_str("YouTube extraction hit a JS challenge (EJS), so yt-dlp only sees storyboards and no real audio/video formats. ");
                                    if !has_deno {
                                        msg.push_str("Fix: install `deno` and ensure it is on PATH, then retry. ");
                                    }
                                    msg.push_str("Also try enabling EJS remote components and forcing the web client. Example yt-dlp command: `yt-dlp --cookies cookies.txt --extractor-args \"youtube:player_client=web\" --remote-components ejs:github -F <url>`. ");
                                    msg.push_str("For this app/mpv, you can pass: `MCBAISE_MPV_YTDL_RAW_OPTIONS=remote-components=ejs:github,extractor-args=youtube:player_client=web`. ");
                                } else if combined_lc.contains("http error 502")
                                    || combined_lc.contains("bad gateway")
                                    || combined_lc.contains("http error 5")
                                {
                                    msg.push_str("YouTube returned HTTP 5xx (temporary upstream/network issue). Retry in a bit; if it persists, check VPN/proxy/firewall. ");
                                } else if combined_lc.contains("request contains an invalid argument")
                                    || combined_lc.contains("http error 400")
                                {
                                    msg.push_str("YouTube returned HTTP 400 (invalid argument) for one of the player clients. Try forcing the web client: `extractor-args=youtube:player_client=web`. ");
                                } else {
                                    msg.push_str("If you see Chrome cookie database copy errors, Windows Chrome is locking the cookie DB while Chrome is running. ");
                                }

                                // Grab a short snippet for UI.
                                let snippet = stderr
                                    .lines()
                                    .chain(stdout.lines())
                                    .filter(|l| !l.trim().is_empty())
                                    .take(3)
                                    .collect::<Vec<_>>()
                                    .join(" | ");

                                if !snippet.is_empty() {
                                    msg.push_str(" Output: ");
                                    msg.push_str(&snippet);
                                }

                                let _ = evt_tx.send(Event::Error(msg));

                                // If we're relying on cookies-from-browser, preflight failures are
                                // usually fatal and mpv will likely fail similarly.
                                // If we have a cookies file, keep going so mpv can try anyway.
                                if cookie_file.is_none() {
                                    return;
                                }
                            }
                        }
                    }

                let uniq = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis();
                let pipe_name = format!("\\\\.\\\\pipe\\\\mcbaise_mpv_{uniq}");

                println!(
                    "[native-mpv] launching: {mpv_exe} --input-ipc-server={pipe_name} --force-window=yes --ontop --ytdl=yes{} ... {url}",
                    if has_ytdlp { " --script-opts=ytdl_hook-ytdl_path=yt-dlp" } else { "" }
                );

                let mut child: Child = match ProcessCommand::new(&mpv_exe)
                    .arg(format!("--input-ipc-server={pipe_name}"))
                    .arg("--force-window=yes")
                    .arg("--ontop")
                    // Prefer yt-dlp if present; mpv will invoke it for YouTube URLs.
                    .arg("--ytdl=yes")
                    .args(has_ytdlp.then_some("--script-opts=ytdl_hook-ytdl_path=yt-dlp"))
                    .args(extra_args.iter())
                    .arg(&url)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::piped())
                    .spawn()
                {
                    Ok(c) => c,
                    Err(e) => {
                        println!("[native-mpv] failed to launch mpv ({mpv_exe}): {e}");
                        let _ = evt_tx.send(Event::Error(format!(
                            "failed to launch mpv ({mpv_exe}): {e}. Is mpv installed and on PATH?"
                        )));
                        return;
                    }
                };

                let pid = child.id();
                println!("[native-mpv] spawned mpv pid={pid}");

                // Drain mpv stderr so we don't lose useful error messages.
                if let Some(stderr) = child.stderr.take() {
                    let evt_tx2 = evt_tx.clone();
                    thread::spawn(move || {
                        let reader = StdBufReader::new(stderr);
                        for line in reader.lines().map_while(Result::ok).take(400) {
                            let line = line.trim().to_string();
                            if line.is_empty() {
                                continue;
                            }
                            println!("[native-mpv][stderr] {line}");

                            if line.contains("Could not copy Chrome cookie database") {
                                let _ = evt_tx2.send(Event::Error(
                                    "yt-dlp couldn't read Chrome cookies because Chrome locks the cookie DB on Windows. This app will NOT close Chrome. Workarounds: fully exit Chrome (including background processes) then retry; or start Chrome with --disable-features=LockProfileCookieDatabase."
                                        .to_string(),
                                ));
                                continue;
                            }

                            let interesting = line.contains("yt-dlp")
                                || line.contains("ytdl")
                                || line.contains("youtube-dl")
                                || line.contains("ERROR")
                                || line.contains("Error")
                                || line.contains("fatal")
                                || line.contains("Failed");
                            if interesting {
                                let _ = evt_tx2.send(Event::Error(format!("mpv: {line}")));
                            }
                        }
                    });
                }

                // Connect to the mpv IPC pipe (mpv creates it).
                println!("[native-mpv] connecting IPC: {pipe_name}");
                let mut ipc = None;
                for _ in 0..60 {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    match ClientOptions::new().open(&pipe_name) {
                        Ok(c) => {
                            ipc = Some(c);
                            break;
                        }
                        Err(_) => {
                            if let Ok(Some(status)) = child.try_wait() {
                                println!("[native-mpv] mpv exited before IPC connected (status={status})");
                                let _ = evt_tx.send(Event::Error(format!(
                                    "mpv exited before IPC connected (status={status})"
                                )));
                                return;
                            }
                        }
                    }
                }

                let Some(ipc) = ipc else {
                    println!("[native-mpv] timed out connecting to mpv IPC: {pipe_name}");
                    let _ = evt_tx.send(Event::Error(
                        "timed out connecting to mpv IPC (named pipe not ready)".to_string(),
                    ));
                    let _ = child.kill();
                    return;
                };

                println!("[native-mpv] IPC connected");

                let (reader, mut writer) = tokio::io::split(ipc);
                let mut reader = BufReader::new(reader);

                async fn send(writer: &mut (impl AsyncWriteExt + Unpin), v: serde_json::Value) -> Result<(), String> {
                    let s = v.to_string();
                    writer
                        .write_all(s.as_bytes())
                        .await
                        .map_err(|e| e.to_string())?;
                    writer
                        .write_all(b"\n")
                        .await
                        .map_err(|e| e.to_string())?;
                    writer.flush().await.map_err(|e| e.to_string())?;
                    Ok(())
                }

                async fn read_response(
                    reader: &mut BufReader<tokio::io::ReadHalf<tokio::net::windows::named_pipe::NamedPipeClient>>,
                    want_id: i64,
                ) -> Result<serde_json::Value, String> {
                    let mut line = String::new();
                    loop {
                        line.clear();
                        let n = reader.read_line(&mut line).await.map_err(|e| e.to_string())?;
                        if n == 0 {
                            return Err("mpv IPC closed".to_string());
                        }
                        let v: serde_json::Value = match serde_json::from_str(&line) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        if v.get("request_id").and_then(|x| x.as_i64()) == Some(want_id) {
                            return Ok(v);
                        }
                    }
                }

                let mut next_id: i64 = 1;
                let poll_interval = Duration::from_millis(250);
                let mut last_sent_time = -1.0f32;
                let mut last_sent_playing = None::<bool>;

                loop {
                    // Commands
                    match cmd_rx.try_recv() {
                        Ok(Command::Shutdown) => {
                            let _ = send(&mut writer, json!({"command": ["quit"]})).await;
                            let _ = child.kill();
                            return;
                        }
                        Ok(Command::SetPlaying(playing)) => {
                            let id = next_id;
                            next_id += 1;
                            let _ = send(
                                &mut writer,
                                json!({"command": ["set_property", "pause", !playing], "request_id": id}),
                            )
                            .await;
                            let _ = read_response(&mut reader, id).await;

                            // Best-effort: if the app requests play, also request unmute.
                            // This avoids cases where mpv is playing but muted.
                            if playing {
                                let id_mute = next_id;
                                next_id += 1;
                                let _ = send(
                                    &mut writer,
                                    json!({"command": ["set_property", "mute", false], "request_id": id_mute}),
                                )
                                .await;
                                let _ = read_response(&mut reader, id_mute).await;
                            }
                        }
                        Ok(Command::SeekSeconds(t)) => {
                            let id = next_id;
                            next_id += 1;
                            let _ = send(
                                &mut writer,
                                json!({"command": ["set_property", "time-pos", t], "request_id": id}),
                            )
                            .await;
                            let _ = read_response(&mut reader, id).await;
                        }
                        Err(TryRecvError::Empty) => {}
                        Err(TryRecvError::Disconnected) => {
                            let _ = send(&mut writer, json!({"command": ["quit"]})).await;
                            let _ = child.kill();
                            return;
                        }
                    }

                    // Poll time-pos
                    let id_time = next_id;
                    next_id += 1;
                    if let Err(e) = send(
                        &mut writer,
                        json!({"command": ["get_property", "time-pos"], "request_id": id_time}),
                    )
                    .await
                    {
                        let _ = evt_tx.send(Event::Error(format!("mpv IPC write failed: {e}")));
                        let _ = child.kill();
                        return;
                    }
                    let v_time = match read_response(&mut reader, id_time).await {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = evt_tx.send(Event::Error(format!("mpv IPC read failed: {e}")));
                            let _ = child.kill();
                            return;
                        }
                    };
                    let time_sec = v_time.get("data").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;

                    // Poll pause
                    let id_pause = next_id;
                    next_id += 1;
                    let _ = send(
                        &mut writer,
                        json!({"command": ["get_property", "pause"], "request_id": id_pause}),
                    )
                    .await;
                    let v_pause = read_response(&mut reader, id_pause).await.unwrap_or(json!({}));
                    let paused = v_pause.get("data").and_then(|x| x.as_bool()).unwrap_or(true);
                    let playing = !paused;

                    if (time_sec - last_sent_time).abs() > 0.02 || last_sent_playing != Some(playing) {
                        last_sent_time = time_sec;
                        last_sent_playing = Some(playing);
                        let _ = evt_tx.send(Event::State { time_sec, playing });
                    }

                    if let Ok(Some(_)) = child.try_wait() {
                        let _ = evt_tx.send(Event::Error("mpv exited".to_string()));
                        return;
                    }

                    tokio::time::sleep(poll_interval).await;
                }
            });
        });

        (cmd_tx, evt_rx, join)
    }
}

// ---------------------------- native YouTube sync (optional) ----------------------------

// This is intentionally opt-in:
// - compile-time: `--features native-youtube`
// - runtime: set `MCBAISE_NATIVE_YOUTUBE=1`
// It requires a running WebDriver (e.g. chromedriver) and will open a real browser window.

#[cfg(all(not(target_arch = "wasm32"), feature = "native-youtube"))]
mod native_youtube {
    use std::collections::{BTreeMap, VecDeque};
    use std::sync::mpsc::TryRecvError;
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::thread;
    use std::thread::JoinHandle;
    use std::time::Duration;

    use std::process::{Child, Command as ProcessCommand};

    use fantoccini::ClientBuilder;
    use serde_json::json;
    use url::Url;

    fn parse_bool(s: &str) -> bool {
        matches!(s.trim(), "1" | "true" | "TRUE" | "yes" | "YES")
    }

    #[derive(Debug, Clone)]
    struct NetscapeCookie {
        domain_host: String,
        path: String,
        secure: bool,
        expires_unix: Option<i64>,
        name: String,
        value: String,
        http_only: bool,
    }

    fn parse_netscape_cookies(content: &str) -> Vec<NetscapeCookie> {
        let mut out = Vec::new();
        for raw_line in content.lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('#') {
                continue;
            }

            // Netscape cookie file: domain \t flag \t path \t secure \t expiration \t name \t value
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 7 {
                continue;
            }

            let mut domain = parts[0].trim().to_string();
            let mut http_only = false;
            if let Some(rest) = domain.strip_prefix("#HttpOnly_") {
                domain = rest.to_string();
                http_only = true;
            }

            let domain_host = domain.trim_start_matches('.').to_string();
            if domain_host.is_empty() {
                continue;
            }

            let path = parts[2].trim().to_string();
            let secure = parts[3].trim().eq_ignore_ascii_case("true");
            let expires_unix = parts[4]
                .trim()
                .parse::<i64>()
                .ok()
                .and_then(|v| if v > 0 { Some(v) } else { None });
            let name = parts[5].to_string();
            let value = parts[6].to_string();
            if name.is_empty() {
                continue;
            }

            out.push(NetscapeCookie {
                domain_host,
                path,
                secure,
                expires_unix,
                name,
                value,
                http_only,
            });
        }
        out
    }

    async fn import_cookies_txt_if_present(client: &fantoccini::Client, evt_tx: &Sender<Event>) {
        let disabled = std::env::var("MCBAISE_YOUTUBE_DISABLE_COOKIE_TXT")
            .ok()
            .map(|v| parse_bool(&v))
            .unwrap_or(false);
        if disabled {
            return;
        }

        let cookie_path = std::env::var("MCBAISE_CHROME_COOKIES_TXT")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .or_else(|| {
                if std::path::Path::new("cookies.txt").exists() {
                    Some("cookies.txt".to_string())
                } else {
                    None
                }
            });

        let Some(cookie_path) = cookie_path else {
            return;
        };

        let content = match std::fs::read_to_string(&cookie_path) {
            Ok(s) => s,
            Err(e) => {
                let _ = evt_tx.send(Event::Error(format!(
                    "failed to read cookies file '{cookie_path}': {e}"
                )));
                return;
            }
        };

        let all = parse_netscape_cookies(&content);
        if all.is_empty() {
            let _ = evt_tx.send(Event::Error(format!(
                "cookies file '{cookie_path}' contained no parseable cookies"
            )));
            return;
        }

        // Group by host so we can navigate to a matching origin before calling AddCookie.
        let mut by_host: BTreeMap<String, Vec<NetscapeCookie>> = BTreeMap::new();
        for c in all {
            by_host.entry(c.domain_host.clone()).or_default().push(c);
        }

        let mut imported = 0usize;
        let mut failed = 0usize;

        println!(
            "[native-youtube] importing cookies from {cookie_path} into WebDriver session ({} host groups)",
            by_host.len()
        );

        for (host, cookies) in by_host {
            // Navigate to a page on the host so WebDriver allows setting cookies.
            let scheme = "https";
            let url = format!("{scheme}://{host}/");
            if let Err(e) = client.goto(&url).await {
                failed += cookies.len();
                println!("[native-youtube] cookie import: failed to navigate to {url}: {e}");
                continue;
            }

            for c in cookies {
                // Skip expired cookies.
                if let Some(exp) = c.expires_unix
                    && exp
                        <= (std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs() as i64)
                {
                    continue;
                }

                let mut cookie = fantoccini::cookies::Cookie::new(c.name, c.value);
                if !c.path.trim().is_empty() {
                    cookie.set_path(c.path);
                }
                // Set a domain cookie so it applies across subdomains when possible.
                // (Chrome generally accepts this when current host is within the domain.)
                cookie.set_domain(host.clone());
                cookie.set_secure(c.secure);
                cookie.set_http_only(c.http_only);
                if let Some(exp) = c.expires_unix {
                    cookie.set_expires(time::OffsetDateTime::from_unix_timestamp(exp).ok());
                }

                match client.add_cookie(cookie).await {
                    Ok(_) => imported += 1,
                    Err(e) => {
                        failed += 1;
                        println!(
                            "[native-youtube] cookie import: add_cookie failed for {host}: {e}"
                        );
                    }
                }
            }
        }

        println!("[native-youtube] cookie import done: imported={imported} failed={failed}");
        if imported == 0 {
            let _ = evt_tx.send(Event::Error(format!(
                "imported 0 cookies from '{cookie_path}'. If you're still seeing bot-check, regenerate the cookies file and retry."
            )));
        }
    }

    #[derive(Debug, Clone)]
    pub enum Command {
        SetPlaying(bool),
        SeekSeconds(f32),
        ReloadAndSeek {
            time_sec: f32,
            playing: bool,
        },
        SetWindowRect {
            x: u32,
            y: u32,
            width: u32,
            height: u32,
        },
        Shutdown,
    }

    #[derive(Debug, Clone)]
    pub enum Event {
        State {
            time_sec: f32,
            playing: bool,
        },
        PlayerErrorOverlay,
        AdState {
            playing_ad: bool,
            label: Option<String>,
        },
        Error(String),
    }

    pub fn spawn(
        video_id: &str,
        webdriver_url: &str,
        launch_webdriver: bool,
        chrome_user_data_dir: Option<String>,
        chrome_profile_dir: Option<String>,
    ) -> (Sender<Command>, Receiver<Event>, JoinHandle<()>) {
        let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();
        let (evt_tx, evt_rx) = mpsc::channel::<Event>();

        let video_id = video_id.to_string();
        let webdriver_url = webdriver_url.to_string();

        let join = thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_io()
                .enable_time()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    let _ =
                        evt_tx.send(Event::Error(format!("failed to create tokio runtime: {e}")));
                    return;
                }
            };

            runtime.block_on(async move {
                let mut chromedriver_child: Option<Child> = None;

                // Prefer a visible, regular Chrome window (not headless).
                let chrome_caps = || {
                    let using_profile = chrome_user_data_dir
                        .as_ref()
                        .map(|s| !s.trim().is_empty())
                        .unwrap_or(false);

                    let mut args = vec![
                        "--new-window".to_string(),
                        "--no-first-run".to_string(),
                        "--no-default-browser-check".to_string(),
                        "--log-level=3".to_string(),
                        "--autoplay-policy=no-user-gesture-required".to_string(),
                        // Avoid Chrome prompting for notification permissions.
                        "--disable-notifications".to_string(),
                        // Reduce startup flakiness (especially with real profiles).
                        "--disable-extensions".to_string(),
                        "--disable-component-extensions-with-background-pages".to_string(),
                    ];

                    // When we're *not* using the user's real profile, try to reduce background noise.
                    // But when we *are* using the real profile, these flags can interfere with
                    // YouTube/ads/network beacons.
                    if !using_profile {
                        args.push("--disable-sync".to_string());
                        args.push("--disable-background-networking".to_string());
                    }

                    if let Some(dir) = chrome_user_data_dir.as_ref().filter(|s| !s.trim().is_empty()) {
                        args.push(format!("--user-data-dir={dir}"));
                    }
                    if let Some(profile) = chrome_profile_dir.as_ref().filter(|s| !s.trim().is_empty()) {
                        args.push(format!("--profile-directory={profile}"));
                    }

                    json!({
                        "browserName": "chrome",
                        "goog:chromeOptions": {
                            "args": args,
                            "prefs": {
                                // 1=allow, 2=block.
                                "profile.default_content_setting_values.notifications": 2
                            },
                            // Avoid Chrome spewing verbose logs into our parent process output.
                            "excludeSwitches": ["enable-logging"]
                        }
                    })
                };

                if let Some(dir) = chrome_user_data_dir.as_ref().filter(|s| !s.trim().is_empty()) {
                    println!(
                        "[native-youtube] using Chrome profile: user-data-dir={dir}{}",
                        chrome_profile_dir
                            .as_deref()
                            .filter(|s| !s.trim().is_empty())
                            .map(|p| format!(", profile-directory={p}"))
                            .unwrap_or_default()
                    );
                    println!(
                        "[native-youtube] note: close all Chrome windows or chromedriver may fail to start (profile lock)"
                    );
                }

                let client = match ClientBuilder::native()
                    .capabilities(chrome_caps().as_object().unwrap().clone())
                    .connect(&webdriver_url)
                    .await
                {
                    Ok(c) => c,
                    Err(e) => {
                        if launch_webdriver {
                            let err_s = e.to_string();
                            // If we reached the WebDriver but session creation failed (common when Chrome crashes
                            // or Chrome/ChromeDriver versions mismatch), do NOT try to spawn another chromedriver.
                            // Spawning would just fail with "port already in use" and hide the real issue.
                            let looks_like_connectivity = err_s.contains("client error (Connect)")
                                || err_s.contains("server did not respond")
                                || err_s.contains("Connection refused")
                                || err_s.contains("connect failed")
                                || err_s.contains("Connect");

                            if !looks_like_connectivity {
                                let msg = format!(
                                    "failed to connect to webdriver at {webdriver_url}: {err_s}.\n\
If chromedriver is already running, this usually means session creation failed (Chrome crashed or version mismatch)."
                                );
                                println!("[native-youtube] {msg}");
                                let _ = evt_tx.send(Event::Error(msg));
                                return;
                            }

                            // Best-effort: if chromedriver is on PATH, start it and retry.
                            // This won't work if you're using a remote WebDriver URL.
                            println!(
                                "[native-youtube] webdriver connect failed ({e}); attempting to launch chromedriver..."
                            );

                            match ProcessCommand::new("chromedriver")
                                .args(["--port=9515"])
                                .spawn()
                            {
                                Ok(child) => {
                                    chromedriver_child = Some(child);
                                }
                                Err(spawn_err) => {
                                    let msg = format!(
                                        "failed to launch chromedriver: {spawn_err}.\n\
Install chromedriver and ensure it's on PATH.\n\
Or start it manually: chromedriver --port=9515"
                                    );
                                    println!("[native-youtube] {msg}");
                                    let _ = evt_tx.send(Event::Error(msg));
                                    return;
                                }
                            }

                            let mut connected = None;
                            let mut last_retry_err: Option<String> = None;
                            for _ in 0..40 {
                                tokio::time::sleep(Duration::from_millis(150)).await;
                                match ClientBuilder::native()
                                    .capabilities(chrome_caps().as_object().unwrap().clone())
                                    .connect(&webdriver_url)
                                    .await
                                {
                                    Ok(c) => {
                                        connected = Some(c);
                                        break;
                                    }
                                    Err(retry_err) => {
                                        last_retry_err = Some(retry_err.to_string());
                                    }
                                }
                            }
                            if let Some(c) = connected {
                                c
                            } else {
                                let detail = last_retry_err.unwrap_or_else(|| e.to_string());
                                let mut msg = format!(
                                    "failed to connect to webdriver at {webdriver_url}: {detail}.\n\
Start chromedriver (example): chromedriver --port=9515"
                                );
                                if detail.contains("DevToolsActivePort") {
                                    msg.push_str(
                                        "\n\
\n\
Troubleshooting:\n\
- Ensure ChromeDriver major version matches your installed Chrome major version.\n\
- If using `MCBAISE_CHROME_USER_DATA_DIR`, fully close Chrome (including background `chrome.exe`) to release the profile lock.\n\
- If it still crashes with your real profile, try `task mcbaise:native:youtube:fresh` (no profile reuse).",
                                    );
                                }
                                println!("[native-youtube] {msg}");
                                let _ = evt_tx.send(Event::Error(msg));
                                return;
                            }
                        } else {
                            let msg = format!(
                                "failed to connect to webdriver at {webdriver_url}: {e}.\n\
Start chromedriver (example): chromedriver --port=9515"
                            );
                            println!("[native-youtube] {msg}");
                            let _ = evt_tx.send(Event::Error(msg));
                            return;
                        }
                    }
                };

                // Immediately switch to a cheap "launch" page so we can apply window geometry
                // before the user sees a default-sized Chrome window sitting on a random page.
                let _ = client.goto("about:blank").await;

                // Give the app a brief moment to send us a desired window rect.
                // Store any other early commands and replay them once the watch page is loaded.
                let mut pending_cmds: VecDeque<Command> = VecDeque::new();
                let mut applied_rect = false;
                for _ in 0..40 {
                    match cmd_rx.try_recv() {
                        Ok(Command::SetWindowRect { x, y, width, height }) => {
                            let _ = client.set_window_rect(x, y, width.max(1), height.max(1)).await;
                            applied_rect = true;
                        }
                        Ok(other) => pending_cmds.push_back(other),
                        Err(TryRecvError::Empty) => {}
                        Err(TryRecvError::Disconnected) => {
                            shutdown_browser(client, &mut chromedriver_child).await;
                            return;
                        }
                    }

                    if applied_rect {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(15)).await;
                }

                async fn shutdown_browser(
                    client: fantoccini::Client,
                    chromedriver_child: &mut Option<Child>,
                ) {
                    println!("[native-youtube] shutdown: closing webdriver session...");

                    // Close all windows (YouTube can spawn additional windows/tabs).
                    // Ignore failures and proceed to end session.
                    if let Ok(handles) = client.windows().await {
                        for handle in handles {
                            let _ = client.switch_to_window(handle).await;
                            let _ = client.close_window().await;
                        }
                    } else {
                        let _ = client.close_window().await;
                    }

                    // End the WebDriver session.
                    let _ = client.close().await;

                    // Best-effort: stop chromedriver we started.
                    if let Some(mut child) = chromedriver_child.take() {
                        let _ = child.kill();
                    }

                    println!("[native-youtube] shutdown: done");
                }

                println!("[native-youtube] webdriver session created; opening YouTube watch page...");

                // If a repo-root cookies.txt exists (or MCBAISE_CHROME_COOKIES_TXT is set),
                // import it into this automation session before navigating to the watch page.
                // This avoids relying on a locked Chrome profile.
                import_cookies_txt_if_present(&client, &evt_tx).await;

                // NOTE: Do NOT open the raw `/embed/...` page directly.
                // YouTube can treat that as an embedded context with missing referrer/identity and refuse playback.
                // The regular watch page works reliably when driven by a real browser.
                let mut watch = Url::parse("https://www.youtube.com/watch").unwrap();
                {
                    let mut qp = watch.query_pairs_mut();
                    qp.append_pair("v", &video_id);
                    qp.append_pair("autoplay", "1");
                    qp.append_pair("playsinline", "1");
                }

                let watch_url = watch.as_str().to_string();

                if let Err(e) = client.goto(&watch_url).await {
                    let msg = format!("failed to open YouTube watch page: {e}");
                    println!("[native-youtube] {msg}");
                    let _ = evt_tx.send(Event::Error(msg));
                    shutdown_browser(client, &mut chromedriver_child).await;
                    return;
                }

                println!("[native-youtube] YouTube page loaded; waiting for <video>...");

                // Wait for a usable <video> (present + metadata loaded).
                // On the watch page, <video> may exist early but be unusable until the player boots.
                let mut ready = false;
                for _ in 0..200 {
                    match client
                        .execute(
                            r#"
                            const v = document.querySelector('video');
                            if (!v) return false;
                            const dur = v.duration;
                            return Number.isFinite(dur) && dur > 0 && v.readyState >= 2;
                            "#,
                            vec![],
                        )
                        .await
                    {
                        Ok(v) if v.as_bool().unwrap_or(false) => {
                            ready = true;
                            break;
                        }
                        _ => {}
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }

                if !ready {
                    let _ = evt_tx.send(Event::Error(
                        "YouTube player did not become ready (<video> not playable)".to_string(),
                    ));
                    shutdown_browser(client, &mut chromedriver_child).await;
                    return;
                }

                // Best-effort: enable autoplay by muting and calling play().
                // User can unmute in the browser window.
                let _ = client
                    .execute(
                        r#"
                        const v = document.querySelector('video');
                        if (!v) return false;
                        v.muted = true;
                        v.play();
                        return true;
                        "#,
                        vec![],
                    )
                    .await;

                println!("[native-youtube] <video> ready; syncing playback.");

                async fn wait_for_video_ready(client: &fantoccini::Client) -> bool {
                    for _ in 0..200 {
                        match client
                            .execute(
                                r#"
                                const v = document.querySelector('video');
                                if (!v) return false;
                                const dur = v.duration;
                                return Number.isFinite(dur) && dur > 0 && v.readyState >= 2;
                                "#,
                                vec![],
                            )
                            .await
                        {
                            Ok(v) if v.as_bool().unwrap_or(false) => return true,
                            _ => {}
                        }
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                    false
                }

                let mut last_sent_time = -1.0f32;
                let mut last_sent_playing = None::<bool>;
                let mut last_sent_error_overlay = false;
                let mut error_overlay_cooldown_ticks: u32 = 0;
                let mut last_sent_ad = false;
                let mut last_sent_ad_label: Option<String> = None;
                let mut skip_ad_cooldown_ticks: u32 = 0;
                let mut consecutive_poll_errors: u32 = 0;
                let mut poll_tick: u64 = 0;
                let poll_interval = Duration::from_millis(300);

                loop {
                    loop {
                        let next_cmd = if let Some(cmd) = pending_cmds.pop_front() {
                            Some(cmd)
                        } else {
                            match cmd_rx.try_recv() {
                                Ok(cmd) => Some(cmd),
                                Err(TryRecvError::Empty) => None,
                                Err(TryRecvError::Disconnected) => {
                                    // If the app exits and drops the sender, shut down cleanly.
                                    shutdown_browser(client, &mut chromedriver_child).await;
                                    return;
                                }
                            }
                        };

                        let Some(cmd) = next_cmd else {
                            break;
                        };

                        match cmd {
                            Command::SetPlaying(playing) => {
                                let _ = client
                                    .execute(
                                        r#"
                                        const v = document.querySelector('video');
                                        if (!v) return false;
                                        if (arguments[0]) {
                                            // Best-effort: unmute when the app requests play.
                                            // Autoplay policies may reject play-with-sound, so fall back to
                                            // muted play, then unmute shortly after.
                                            const ensureUnmuted = () => {
                                                try { v.muted = false; } catch (_) {}
                                                try {
                                                    if (typeof v.volume === 'number' && v.volume === 0) v.volume = 0.2;
                                                } catch (_) {}
                                            };

                                            try { ensureUnmuted(); } catch (_) {}

                                            let p;
                                            try { p = v.play(); } catch (_) {}

                                            try {
                                                if (p && typeof p.catch === 'function') {
                                                    p.catch(() => {
                                                        try { v.muted = true; } catch (_) {}
                                                        try { v.play(); } catch (_) {}
                                                        setTimeout(ensureUnmuted, 250);
                                                    });
                                                }
                                            } catch (_) {}

                                            setTimeout(ensureUnmuted, 250);
                                        } else {
                                            try { v.pause(); } catch (_) {}
                                        }
                                        return true;
                                        "#,
                                        vec![json!(playing)],
                                    )
                                    .await;
                            }
                            Command::SeekSeconds(time_sec) => {
                                let _ = client
                                    .execute(
                                        r#"
                                        const v = document.querySelector('video');
                                        if (!v) return false;
                                        v.currentTime = arguments[0];
                                        return true;
                                        "#,
                                        vec![json!(time_sec)],
                                    )
                                    .await;
                            }
                            Command::ReloadAndSeek { time_sec, playing } => {
                                // Best-effort recover from YouTube's transient player error overlay.
                                // Reload the watch page, wait for <video>, then seek and restore play/pause.
                                let _ = client.goto(&watch_url).await;

                                if wait_for_video_ready(&client).await {
                                    // Mute then attempt play() to satisfy autoplay policies.
                                    let _ = client
                                        .execute(
                                            r#"
                                            const v = document.querySelector('video');
                                            if (!v) return false;
                                            v.muted = true;
                                            v.play();
                                            return true;
                                            "#,
                                            vec![],
                                        )
                                        .await;

                                    let _ = client
                                        .execute(
                                            r#"
                                            const v = document.querySelector('video');
                                            if (!v) return false;
                                            v.currentTime = arguments[0];
                                            return true;
                                            "#,
                                            vec![json!(time_sec.max(0.0))],
                                        )
                                        .await;

                                    let _ = client
                                        .execute(
                                            r#"
                                            const v = document.querySelector('video');
                                            if (!v) return false;
                                            if (arguments[0]) {
                                                const ensureUnmuted = () => {
                                                    try { v.muted = false; } catch (_) {}
                                                    try {
                                                        if (typeof v.volume === 'number' && v.volume === 0) v.volume = 0.2;
                                                    } catch (_) {}
                                                };

                                                let p;
                                                try { p = v.play(); } catch (_) {}
                                                try {
                                                    if (p && typeof p.catch === 'function') {
                                                        p.catch(() => {
                                                            try { v.muted = true; } catch (_) {}
                                                            try { v.play(); } catch (_) {}
                                                            setTimeout(ensureUnmuted, 250);
                                                        });
                                                    }
                                                } catch (_) {}
                                                setTimeout(ensureUnmuted, 250);
                                            } else {
                                                try { v.pause(); } catch (_) {}
                                            }
                                            return true;
                                            "#,
                                            vec![json!(playing)],
                                        )
                                        .await;
                                }
                            }
                            Command::SetWindowRect { x, y, width, height } => {
                                // Best-effort: reposition the browser window.
                                // This requires a non-headless driver and may be ignored by some platforms.
                                let _ = client
                                    .set_window_rect(x, y, width.max(1), height.max(1))
                                    .await;
                            }
                            Command::Shutdown => {
                                shutdown_browser(client, &mut chromedriver_child).await;
                                return;
                            }
                        }
                    }

                                        poll_tick = poll_tick.wrapping_add(1);
                                        let slow_scan = poll_tick.is_multiple_of(25);
                                        let try_skip = skip_ad_cooldown_ticks == 0;

                                        // One JS round-trip for video/ad/error/skip. Keep it cheap: use fast selectors every tick,
                                        // and only do a limited shadow-root scan occasionally.
                                        let poll_val = client
                                                .execute(
                                                        r#"
                                                        const slowScan = !!arguments[0];
                                                        const trySkip = !!arguments[1];

                                                        function qs(sel) {
                                                            try { return document.querySelector(sel); } catch (_) { return null; }
                                                        }

                                                        function findInShadow(sel, maxEls) {
                                                            const stack = [document];
                                                            const max = maxEls || 600;
                                                            while (stack.length) {
                                                                const root = stack.pop();
                                                                try {
                                                                    const found = root.querySelector ? root.querySelector(sel) : null;
                                                                    if (found) return found;
                                                                } catch (_) {}

                                                                try {
                                                                    const els = root.querySelectorAll ? root.querySelectorAll('*') : [];
                                                                    const lim = Math.min(els.length, max);
                                                                    for (let i = 0; i < lim; i++) {
                                                                        const el = els[i];
                                                                        if (el && el.shadowRoot) stack.push(el.shadowRoot);
                                                                    }
                                                                } catch (_) {}
                                                            }
                                                            return null;
                                                        }

                                                        // Video
                                                        let v = qs('video');
                                                        if (!v && slowScan) v = findInShadow('video', 600);
                                                        const hasVideo = !!v;
                                                        const time = hasVideo ? v.currentTime : null;
                                                        const paused = hasVideo ? v.paused : true;

                                                        // Ad detection
                                                        let player = qs('.html5-video-player');
                                                        if (!player && slowScan) player = findInShadow('.html5-video-player', 600);
                                                        let ad = !!(player && player.classList && player.classList.contains('ad-showing'));
                                                        if (!ad) {
                                                            const adBadge = qs('.ytp-ad-badge__text--clean-player') || qs('.ad-simple-attributed-string');
                                                            ad = !!adBadge;
                                                        }

                                                        let adLabel = null;
                                                        if (ad) {
                                                            const n = qs('.ad-simple-attributed-string') || qs('.ytp-ad-badge__text--clean-player');
                                                            if (n) {
                                                                const txt = (n.innerText || '').trim();
                                                                const aria = (n.getAttribute && n.getAttribute('aria-label')) ? n.getAttribute('aria-label') : '';
                                                                adLabel = txt || aria || null;
                                                            }
                                                        }

                                                        // Error overlay
                                                        let e = qs('.ytp-error');
                                                        if (!e && slowScan) e = findInShadow('.ytp-error', 600);
                                                        let overlay = false;
                                                        if (e) {
                                                            try {
                                                                const r = e.getBoundingClientRect();
                                                                const t = (e.innerText || '').toLowerCase();
                                                                overlay = !!r && r.width > 1 && r.height > 1 && t.includes('something went wrong');
                                                            } catch (_) {
                                                                overlay = false;
                                                            }
                                                        }

                                                        // Skip button
                                                        let skipClicked = false;
                                                        let skipPresent = false;
                                                        if (ad && trySkip) {
                                                            let btn = qs('button.ytp-skip-ad-button') || qs('.ytp-skip-ad-button');
                                                            if (!btn && slowScan) btn = findInShadow('button.ytp-skip-ad-button, .ytp-skip-ad-button', 600);
                                                            if (btn) {
                                                                skipPresent = true;
                                                                try {
                                                                    const r = btn.getBoundingClientRect();
                                                                    if (r && r.width > 1 && r.height > 1 && !btn.disabled) {
                                                                        btn.click();
                                                                        skipClicked = true;
                                                                    }
                                                                } catch (_) {}
                                                            }
                                                        }

                                                        return { hasVideo, time, paused, overlay, ad, adLabel, skipPresent, skipClicked };
                                                        "#,
                                                        vec![json!(slow_scan), json!(try_skip)],
                                                )
                                                .await;

                    if let Ok(v) = poll_val.as_ref() {
                        consecutive_poll_errors = 0;

                        let has_video = v
                            .get("hasVideo")
                            .and_then(|x| x.as_bool())
                            .unwrap_or(false);
                        let time_sec = v.get("time").and_then(|x| x.as_f64()).unwrap_or(-1.0) as f32;
                        let paused = v.get("paused").and_then(|x| x.as_bool()).unwrap_or(true);
                        let playing = !paused;

                        // Emit a non-fatal overlay event when it appears.
                        // Throttle to avoid spamming if it stays visible.
                        let overlay_visible = v
                            .get("overlay")
                            .and_then(|x| x.as_bool())
                            .unwrap_or(false);
                        if overlay_visible {
                            if error_overlay_cooldown_ticks == 0 && !last_sent_error_overlay {
                                let _ = evt_tx.send(Event::PlayerErrorOverlay);
                                error_overlay_cooldown_ticks = 10; // ~2s at 200ms polling
                            }
                            last_sent_error_overlay = true;
                        } else {
                            last_sent_error_overlay = false;
                        }
                        error_overlay_cooldown_ticks = error_overlay_cooldown_ticks.saturating_sub(1);

                        // Emit ad state changes.
                        let ad_visible = v.get("ad").and_then(|x| x.as_bool()).unwrap_or(false);
                        let ad_label = v
                            .get("adLabel")
                            .and_then(|x| x.as_str())
                            .map(|s| s.to_string());
                        if ad_visible != last_sent_ad || ad_label != last_sent_ad_label {
                            last_sent_ad = ad_visible;
                            last_sent_ad_label = ad_label.clone();
                            let _ = evt_tx.send(Event::AdState {
                                playing_ad: ad_visible,
                                label: ad_label,
                            });
                        }

                        // Skip click throttle.
                        skip_ad_cooldown_ticks = skip_ad_cooldown_ticks.saturating_sub(1);
                        let skip_clicked = v
                            .get("skipClicked")
                            .and_then(|x| x.as_bool())
                            .unwrap_or(false);
                        let skip_present = v
                            .get("skipPresent")
                            .and_then(|x| x.as_bool())
                            .unwrap_or(false);
                        if skip_clicked {
                            skip_ad_cooldown_ticks = 12;
                        } else if ad_visible && skip_present {
                            skip_ad_cooldown_ticks = 3;
                        }

                        // Only emit State when we actually have a real video/time.
                        // If the video element is missing, don't spam fake State that would
                        // prevent the Bevy side from noticing a stall.
                        if has_video && time_sec.is_finite() && time_sec >= 0.0 {
                            let should_send = (last_sent_time < 0.0
                                || (time_sec - last_sent_time).abs() > 0.05)
                                || last_sent_playing.map(|lp| lp != playing).unwrap_or(true);

                            if should_send {
                                last_sent_time = time_sec;
                                last_sent_playing = Some(playing);
                                let _ = evt_tx.send(Event::State { time_sec, playing });
                            }
                        }
                    } else {
                        consecutive_poll_errors = consecutive_poll_errors.saturating_add(1);
                        let msg = match poll_val {
                            Err(e) => e.to_string(),
                            _ => "unknown poll error".to_string(),
                        };

                        // Common transient: YouTube/Chrome can change window handles during ads / reloads.
                        // Try to recover by switching to an available window.
                        if (msg.contains("no such window")
                            || msg.contains("web view not found")
                            || msg.contains("target window already closed"))
                            && let Ok(handles) = client.windows().await
                            && let Some(handle) = handles.last().cloned()
                        {
                            let _ = client.switch_to_window(handle).await;
                            consecutive_poll_errors = 0;
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            continue;
                        }

                        // If we're accumulating poll failures, attempt a reload before giving up.
                        if consecutive_poll_errors == 10 {
                            let _ = client.goto(&watch_url).await;
                            let _ = wait_for_video_ready(&client).await;
                            consecutive_poll_errors = 0;
                            continue;
                        }

                        if consecutive_poll_errors >= 20 {
                            let _ = evt_tx.send(Event::Error(format!(
                                "lost WebDriver session (poll failed): {msg}"
                            )));
                            shutdown_browser(client, &mut chromedriver_child).await;
                            return;
                        }
                    }

                    tokio::time::sleep(poll_interval).await;
                }
            });
        });

        (cmd_tx, evt_rx, join)
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "native-youtube"))]
#[derive(Resource, Default)]
struct NativeYoutubeWindowLayout {
    applied: bool,
    force_topmost: bool,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "native-youtube"))]
#[derive(Resource)]
struct NativeYoutubeSync {
    enabled: bool,
    tx: std::sync::mpsc::Sender<native_youtube::Command>,
    rx: std::sync::Mutex<std::sync::mpsc::Receiver<native_youtube::Event>>,
    join: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
    last_error: std::sync::Mutex<Option<String>>,

    // Interpolation state: YouTube may only report time at a low/irregular rate.
    // We integrate locally between samples for smooth animation.
    has_remote: bool,
    last_remote_time_sec: f32,
    last_remote_playing: bool,
    sample_age_sec: f32,
    remote_age_sec: f32,
    interp_time_sec: f32,

    // Ad/healing state.
    in_ad: bool,
    ad_label: Option<String>,
    last_good_time_sec: f32,
    pending_seek_after_ad: bool,
    ad_nudge_cooldown_sec: f32,

    // Simple healing: if the remote state stops updating while we're trying to play,
    // re-seek to the last known remote time and press play.
    heal_cooldown_sec: f32,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "native-youtube"))]
#[derive(Resource, Clone)]
struct NativeYoutubeConfig {
    webdriver_url: String,
    launch_webdriver: bool,
    chrome_user_data_dir: Option<String>,
    chrome_profile_dir: Option<String>,
}

// Stub types so `ui_overlay` can take optional resources on all targets/features.
#[cfg(not(all(not(target_arch = "wasm32"), feature = "native-youtube")))]
#[derive(Resource)]
struct NativeYoutubeSync {
    pub enabled: bool,
    pub has_remote: bool,
    pub last_error: std::sync::Mutex<Option<String>>,
}

#[cfg(not(all(not(target_arch = "wasm32"), feature = "native-youtube")))]
#[derive(Resource, Clone)]
struct NativeYoutubeConfig;

#[cfg(all(not(target_arch = "wasm32"), feature = "native-youtube"))]
impl Drop for NativeYoutubeSync {
    fn drop(&mut self) {
        // Best-effort shutdown if the app exits without running our exit system.
        let _ = self.tx.send(native_youtube::Command::Shutdown);

        // Critical: wait for the webdriver thread to actually close the browser
        // before the process exits (otherwise Chrome can be left running).
        if let Ok(mut slot) = self.join.lock()
            && let Some(handle) = slot.take()
        {
            let _ = handle.join();
        }
    }
}

// ---------------------------- native mpv sync (optional) ----------------------------

#[cfg(all(windows, not(target_arch = "wasm32"), feature = "native-mpv"))]
#[derive(Resource)]
struct NativeMpvSync {
    enabled: bool,
    tx: std::sync::mpsc::Sender<native_mpv::Command>,
    rx: std::sync::Mutex<std::sync::mpsc::Receiver<native_mpv::Event>>,
    join: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
    last_error: std::sync::Mutex<Option<String>>,

    has_remote: bool,
    last_remote_time_sec: f32,
    last_remote_playing: bool,
    sample_age_sec: f32,
    interp_time_sec: f32,
}

#[cfg(all(windows, not(target_arch = "wasm32"), feature = "native-mpv"))]
#[derive(Resource, Clone)]
struct NativeMpvConfig {
    url: String,
    mpv_path: Option<String>,
    extra_args: Vec<String>,
}

// Stub types so `ui_overlay` can take optional resources on all targets/features.
#[cfg(not(all(windows, not(target_arch = "wasm32"), feature = "native-mpv")))]
#[derive(Resource)]
struct NativeMpvSync {
    pub enabled: bool,
    pub has_remote: bool,
    pub last_error: std::sync::Mutex<Option<String>>,
}

#[cfg(not(all(windows, not(target_arch = "wasm32"), feature = "native-mpv")))]
#[derive(Resource, Clone)]
struct NativeMpvConfig;

#[cfg(all(windows, not(target_arch = "wasm32"), feature = "native-mpv"))]
impl Drop for NativeMpvSync {
    fn drop(&mut self) {
        let _ = self.tx.send(native_mpv::Command::Shutdown);
        if let Ok(mut slot) = self.join.lock()
            && let Some(handle) = slot.take()
        {
            let _ = handle.join();
        }
    }
}

const TUBE_RADIUS: f32 = 3.4;
const SUBJECT_RADIUS: f32 = 0.78;
const HUMAN_SCALE: f32 = 1.15;
const HUMAN_RADIUS: f32 = SUBJECT_RADIUS * HUMAN_SCALE;
const BALL_RADIUS: f32 = 0.70;
const CONTACT_EPS: f32 = 0.01;
const GRAVITY: f32 = 9.81;

const FRAMES_SAMPLES: usize = 3200;

#[cfg(not(target_arch = "wasm32"))]
const TUBULAR_SEGMENTS: usize = 2600;
#[cfg(not(target_arch = "wasm32"))]
const RADIAL_SEGMENTS: usize = 96;

#[cfg(target_arch = "wasm32")]
const TUBULAR_SEGMENTS: usize = 1800;
#[cfg(target_arch = "wasm32")]
const RADIAL_SEGMENTS: usize = 64;

#[derive(Component)]
struct TubeTag;

#[derive(Component)]
struct SubjectTag;

#[derive(Component)]
struct BallTag;

#[derive(Component)]
struct SubjectLightTag;

#[derive(Resource, Clone, Copy, PartialEq, Eq, Default)]
enum SubjectMode {
    #[default]
    Auto,
    Random,
    Human,
    Doughnut,
    Ball,
}

impl SubjectMode {
    fn label(self) -> &'static str {
        match self {
            SubjectMode::Auto => "Subject: auto",
            SubjectMode::Random => "Subject: random",
            SubjectMode::Human => "Subject: human",
            SubjectMode::Doughnut => "Subject: doughnut",
            SubjectMode::Ball => "Subject: ball",
        }
    }

    fn short_label(self) -> &'static str {
        match self {
            SubjectMode::Auto => "auto",
            SubjectMode::Random => "random",
            SubjectMode::Human => "human",
            SubjectMode::Doughnut => "doughnut",
            SubjectMode::Ball => "ball",
        }
    }
}

#[derive(Resource, Clone, Copy, PartialEq, Eq, Default)]
enum ColorSchemeMode {
    #[default]
    Auto,
    Random,
    OrangeWhite,
    Nin,
    BlackWhite,
    RandomGrey,
    Blue,
    Dynamic,
    Fluid,
    Sun,
    Psychedelic,
    Neon,
    Matrix,
}

impl ColorSchemeMode {
    fn label(self) -> &'static str {
        match self {
            ColorSchemeMode::Auto => "Colors: auto",
            ColorSchemeMode::Random => "Colors: random",
            ColorSchemeMode::OrangeWhite => "Colors: orange and white",
            ColorSchemeMode::Nin => "Colors: NIN",
            ColorSchemeMode::BlackWhite => "Colors: black and white",
            ColorSchemeMode::RandomGrey => "Colors: random grey",
            ColorSchemeMode::Blue => "Colors: blue",
            ColorSchemeMode::Dynamic => "Colors: dynamic",
            ColorSchemeMode::Fluid => "Colors: fluid",
            ColorSchemeMode::Sun => "Colors: sun",
            ColorSchemeMode::Psychedelic => "Colors: psychedelic",
            ColorSchemeMode::Neon => "Colors: neon",
            ColorSchemeMode::Matrix => "Colors: matrix",
        }
    }

    fn short_label_from_value(v: u32) -> &'static str {
        match v {
            0 => "orange/white",
            1 => "NIN",
            2 => "black/white",
            3 => "random grey",
            4 => "blue",
            5 => "dynamic",
            6 => "fluid",
            7 => "sun",
            8 => "psychedelic",
            9 => "neon",
            10 => "matrix",
            _ => "random grey",
        }
    }

    fn from_value(v: u32) -> Self {
        match v {
            0 => ColorSchemeMode::OrangeWhite,
            1 => ColorSchemeMode::Nin,
            2 => ColorSchemeMode::BlackWhite,
            3 => ColorSchemeMode::RandomGrey,
            4 => ColorSchemeMode::Blue,
            5 => ColorSchemeMode::Dynamic,
            6 => ColorSchemeMode::Fluid,
            7 => ColorSchemeMode::Sun,
            8 => ColorSchemeMode::Psychedelic,
            9 => ColorSchemeMode::Neon,
            10 => ColorSchemeMode::Matrix,
            _ => ColorSchemeMode::RandomGrey,
        }
    }
}

#[derive(Resource, Clone, Copy, PartialEq, Eq, Default)]
enum TexturePatternMode {
    #[default]
    Auto,
    Random,
    Stripe,
    Swirl,
    StripeWire,
    SwirlWire,
    Fluid,
    FluidStripe,
    FluidSwirl,
    Wave,
    Fractal,
    Particle,
    Grid,
    HoopWire,
    HoopAlt,
}

#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum AppearancePreset {
    BlueGlass,
    OpaqueWhite,
    Blue,
    Polkadot,
    MatteLightBlue,
    Wireframe,
}

impl AppearancePreset {
    fn label(self) -> &'static str {
        match self {
            AppearancePreset::BlueGlass => "blue glass",
            AppearancePreset::OpaqueWhite => "opaque white",
            AppearancePreset::Blue => "blue",
            AppearancePreset::Polkadot => "polkadot",
            AppearancePreset::MatteLightBlue => "matte light blue",
            AppearancePreset::Wireframe => "wireframe",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum AppearanceMode {
    Auto,
    Random,
    #[default]
    BlueGlass,
    OpaqueWhite,
    Blue,
    Polkadot,
    MatteLightBlue,
    Wireframe,
}

impl AppearanceMode {
    fn label(self) -> &'static str {
        match self {
            AppearanceMode::Auto => "auto",
            AppearanceMode::Random => "random",
            AppearanceMode::BlueGlass => "blue glass",
            AppearanceMode::OpaqueWhite => "opaque white",
            AppearanceMode::Blue => "blue",
            AppearanceMode::Polkadot => "polkadot",
            AppearanceMode::MatteLightBlue => "matte light blue",
            AppearanceMode::Wireframe => "wireframe",
        }
    }

    #[allow(dead_code)]
    fn preset(self) -> Option<AppearancePreset> {
        match self {
            AppearanceMode::BlueGlass => Some(AppearancePreset::BlueGlass),
            AppearanceMode::OpaqueWhite => Some(AppearancePreset::OpaqueWhite),
            AppearanceMode::Blue => Some(AppearancePreset::Blue),
            AppearanceMode::Polkadot => Some(AppearancePreset::Polkadot),
            AppearanceMode::MatteLightBlue => Some(AppearancePreset::MatteLightBlue),
            AppearanceMode::Wireframe => Some(AppearancePreset::Wireframe),
            AppearanceMode::Auto | AppearanceMode::Random => None,
        }
    }
}

#[derive(Resource, Clone, Copy)]
struct HumanAppearanceMode(AppearanceMode);

impl Default for HumanAppearanceMode {
    fn default() -> Self {
        Self(AppearanceMode::Auto)
    }
}

#[derive(Resource, Clone, Copy)]
struct BallAppearanceMode(AppearanceMode);

impl Default for BallAppearanceMode {
    fn default() -> Self {
        Self(AppearanceMode::Auto)
    }
}

#[derive(Clone, Copy)]
struct AutoAppearanceState {
    rng: u32,
    since_switch_sec: f32,
    next_switch_sec: f32,
    current: AppearancePreset,
}

impl AutoAppearanceState {
    fn lcg_next(&mut self) -> u32 {
        self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
        self.rng
    }

    fn rand01(&mut self) -> f32 {
        (self.lcg_next() as f32) / (u32::MAX as f32)
    }

    fn rand_range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.rand01()
    }

    fn schedule_next_switch(&mut self) {
        self.since_switch_sec = 0.0;
        self.next_switch_sec = self.rand_range(3.0, 6.0);
    }

    #[allow(dead_code)]
    fn pick_next_preset(&mut self) -> AppearancePreset {
        let candidates = [
            AppearancePreset::BlueGlass,
            AppearancePreset::OpaqueWhite,
            AppearancePreset::Blue,
            AppearancePreset::Polkadot,
            AppearancePreset::MatteLightBlue,
            AppearancePreset::Wireframe,
        ];

        for _ in 0..6 {
            let idx = (self.rand01() * candidates.len() as f32)
                .floor()
                .clamp(0.0, (candidates.len() - 1) as f32) as usize;
            let p = candidates[idx];
            if p != self.current {
                return p;
            }
        }

        // Fallback cycle.
        match self.current {
            AppearancePreset::BlueGlass => AppearancePreset::OpaqueWhite,
            AppearancePreset::OpaqueWhite => AppearancePreset::Blue,
            AppearancePreset::Blue => AppearancePreset::Polkadot,
            AppearancePreset::Polkadot => AppearancePreset::MatteLightBlue,
            AppearancePreset::MatteLightBlue => AppearancePreset::Wireframe,
            AppearancePreset::Wireframe => AppearancePreset::BlueGlass,
        }
    }
}

#[derive(Resource, Clone, Copy)]
struct AutoHumanAppearanceState(AutoAppearanceState);

impl Default for AutoHumanAppearanceState {
    fn default() -> Self {
        let mut s = AutoAppearanceState {
            rng: 0xA11C_E001,
            since_switch_sec: 0.0,
            next_switch_sec: 0.0,
            current: AppearancePreset::OpaqueWhite,
        };
        s.schedule_next_switch();
        Self(s)
    }
}

#[derive(Resource, Clone, Copy)]
struct AutoBallAppearanceState(AutoAppearanceState);

impl Default for AutoBallAppearanceState {
    fn default() -> Self {
        let mut s = AutoAppearanceState {
            rng: 0xB411_C0DE,
            since_switch_sec: 0.0,
            next_switch_sec: 0.0,
            current: AppearancePreset::BlueGlass,
        };
        s.schedule_next_switch();
        Self(s)
    }
}

#[allow(dead_code)]
#[derive(Resource, Clone)]
struct AppearanceTextures {
    #[allow(dead_code)]
    polkadot: Handle<Image>,
}

impl TexturePatternMode {
    fn label(self) -> &'static str {
        match self {
            TexturePatternMode::Auto => "Texture: auto",
            TexturePatternMode::Random => "Texture: random",
            TexturePatternMode::Stripe => "Texture: stripe",
            TexturePatternMode::Swirl => "Texture: swirl",
            TexturePatternMode::StripeWire => "Texture: stripe (wire)",
            TexturePatternMode::SwirlWire => "Texture: swirl (wire)",
            TexturePatternMode::Fluid => "Texture: fluid",
            TexturePatternMode::FluidStripe => "Texture: fluid stripe",
            TexturePatternMode::FluidSwirl => "Texture: fluid swirl",
            TexturePatternMode::Wave => "Texture: wave",
            TexturePatternMode::Fractal => "Texture: fractal",
            TexturePatternMode::Particle => "Texture: particle",
            TexturePatternMode::Grid => "Texture: grid",
            TexturePatternMode::HoopWire => "Texture: hoop (wire)",
            TexturePatternMode::HoopAlt => "Texture: hoop (alt)",
        }
    }

    fn short_label_from_value(v: u32) -> &'static str {
        match v {
            0 => "stripe",
            1 => "swirl",
            2 => "stripe wire",
            3 => "swirl wire",
            4 => "fluid",
            5 => "fluid stripe",
            6 => "fluid swirl",
            7 => "wave",
            8 => "fractal",
            9 => "particle",
            10 => "grid",
            11 => "hoop",
            12 => "hoop-alt",
            _ => "stripe",
        }
    }

    fn from_value(v: u32) -> Self {
        match v {
            0 => TexturePatternMode::Stripe,
            1 => TexturePatternMode::Swirl,
            2 => TexturePatternMode::StripeWire,
            3 => TexturePatternMode::SwirlWire,
            4 => TexturePatternMode::Fluid,
            5 => TexturePatternMode::FluidStripe,
            6 => TexturePatternMode::FluidSwirl,
            7 => TexturePatternMode::Wave,
            8 => TexturePatternMode::Fractal,
            9 => TexturePatternMode::Particle,
            10 => TexturePatternMode::Grid,
            11 => TexturePatternMode::HoopWire,
            12 => TexturePatternMode::HoopAlt,
            _ => TexturePatternMode::Stripe,
        }
    }
}

#[derive(Resource, Clone, Copy, PartialEq, Eq, Default)]
enum PoseMode {
    #[default]
    Auto,
    Random,
    Standing,
    Belly,
    Back,
    LeftSide,
    RightSide,
}

impl PoseMode {
    fn label(self) -> &'static str {
        match self {
            PoseMode::Auto => "Pose: auto",
            PoseMode::Random => "Pose: random",
            PoseMode::Standing => "Pose: standing",
            PoseMode::Belly => "Pose: belly",
            PoseMode::Back => "Pose: back",
            PoseMode::LeftSide => "Pose: left side",
            PoseMode::RightSide => "Pose: right side",
        }
    }

    fn short_label(self) -> &'static str {
        match self {
            PoseMode::Auto => "auto",
            PoseMode::Random => "random",
            PoseMode::Standing => "standing",
            PoseMode::Belly => "belly",
            PoseMode::Back => "back",
            PoseMode::LeftSide => "left side",
            PoseMode::RightSide => "right side",
        }
    }
}

#[derive(Resource, Clone, Copy)]
struct AutoPoseState {
    rng: u32,
    current: PoseMode,
    since_switch_sec: f32,
    next_switch_sec: f32,
    collision_cooldown_sec: f32,
}

#[derive(Resource, Clone)]
struct AutoCameraState {
    rng: u32,
    since_switch_sec: f32,
    next_switch_sec: f32,
    current: CameraMode,
    subject_distance: f32,
    last_effective_mode: CameraMode,
    mode_since_sec: f32,
    pass_anchor_progress: f32,
    pass_anchor_valid: bool,
    pass_reanchor_since_sec: f32,
}

#[derive(Resource, Clone, Copy)]
struct AutoSubjectState {
    rng: u32,
    since_switch_sec: f32,
    next_switch_sec: f32,
    current: SubjectMode,
}

impl Default for AutoSubjectState {
    fn default() -> Self {
        let mut s = Self {
            rng: 0xBADC_0FFEu32,
            since_switch_sec: 0.0,
            next_switch_sec: 0.0,
            current: if cfg!(feature = "burn_human") {
                SubjectMode::Human
            } else {
                SubjectMode::Doughnut
            },
        };
        s.schedule_next_switch();
        s
    }
}

impl AutoSubjectState {
    fn lcg_next(&mut self) -> u32 {
        self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
        self.rng
    }

    fn rand01(&mut self) -> f32 {
        (self.lcg_next() as f32) / (u32::MAX as f32)
    }

    fn rand_range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.rand01()
    }

    fn schedule_next_switch(&mut self) {
        self.since_switch_sec = 0.0;
        self.next_switch_sec = self.rand_range(3.0, 6.0);
    }

    fn pick_next_subject(&mut self) -> SubjectMode {
        match self.current {
            SubjectMode::Ball => {
                if cfg!(feature = "burn_human") {
                    SubjectMode::Human
                } else {
                    SubjectMode::Doughnut
                }
            }
            _ => SubjectMode::Ball,
        }
    }
}

#[derive(Resource, Clone, Copy)]
struct AutoTubeStyleState {
    rng: u32,
    scheme_current: u32,
    scheme_since_switch_sec: f32,
    scheme_next_switch_sec: f32,
    pattern_current: u32,
    pattern_since_switch_sec: f32,
    pattern_next_switch_sec: f32,
    auto_wire_since_switch_sec: f32,
    auto_wire_next_switch_sec: f32,
}

impl Default for AutoTubeStyleState {
    fn default() -> Self {
        let mut s = Self {
            rng: 0xC0DE_CAFEu32,
            scheme_current: 0,
            scheme_since_switch_sec: 0.0,
            scheme_next_switch_sec: 0.0,
            pattern_current: 0,
            pattern_since_switch_sec: 0.0,
            pattern_next_switch_sec: 0.0,
            auto_wire_since_switch_sec: 0.0,
            auto_wire_next_switch_sec: 0.0,
        };
        s.schedule_next_scheme_switch();
        s.schedule_next_pattern_switch();
        s
    }
}

impl AutoTubeStyleState {
    fn lcg_next(&mut self) -> u32 {
        self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
        self.rng
    }

    fn rand01(&mut self) -> f32 {
        (self.lcg_next() as f32) / (u32::MAX as f32)
    }

    fn rand_range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.rand01()
    }

    fn schedule_next_scheme_switch(&mut self) {
        self.scheme_since_switch_sec = 0.0;
        self.scheme_next_switch_sec = self.rand_range(3.0, 6.0);
    }

    fn schedule_next_pattern_switch(&mut self) {
        self.pattern_since_switch_sec = 0.0;
        self.pattern_next_switch_sec = self.rand_range(3.0, 6.0);
    }

    fn schedule_next_auto_wire_pattern_switch(&mut self) {
        self.auto_wire_since_switch_sec = 0.0;
        self.auto_wire_next_switch_sec = self.rand_range(1.0, 2.0);
    }

    fn pick_next_scheme(&mut self) {
        let prev = self.scheme_current % 4;
        let mut next = prev;
        for _ in 0..8 {
            next = self.lcg_next() % 4;
            if next != prev {
                break;
            }
        }
        self.scheme_current = next;
    }

    fn pick_next_pattern(&mut self) {
        let prev = self.pattern_current;
        let mut next = prev;
        for _ in 0..8 {
            // Use full set of patterns (0..12) so Auto cycles through all shaders
            next = self.lcg_next() % 13;
            if next != prev {
                break;
            }
        }
        self.pattern_current = next;
    }

    fn pick_next_wire_pattern(&mut self) {
        let prev = self.pattern_current;
        let mut next = prev;
        for _ in 0..8 {
            // Expand wire choices to include all wire-like pattern indices (2..=12)
            next = 2 + (self.lcg_next() % 11);
            if next != prev {
                break;
            }
        }
        self.pattern_current = next;
    }
}

impl Default for AutoCameraState {
    fn default() -> Self {
        let mut s = Self {
            rng: 0xC0FFEE_u32,
            since_switch_sec: 0.0,
            next_switch_sec: 0.0,
            current: CameraMode::BallChase,
            subject_distance: 5.0,
            last_effective_mode: CameraMode::BallChase,
            mode_since_sec: 0.0,
            pass_anchor_progress: 0.0,
            pass_anchor_valid: false,
            pass_reanchor_since_sec: 0.0,
        };
        s.apply_mode_params(s.current);
        s
    }
}

impl AutoCameraState {
    fn lcg_next(&mut self) -> u32 {
        // Deterministic, cheap RNG.
        self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
        self.rng
    }

    fn rand01(&mut self) -> f32 {
        let v = self.lcg_next();
        (v as f32) / (u32::MAX as f32)
    }

    fn rand_range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.rand01()
    }

    fn apply_mode_params(&mut self, mode: CameraMode) {
        let (min_sec, max_sec) = mode.suggested_duration_range_sec();
        // Fallback if a mode doesn't specify anything.
        let (min_sec, max_sec) = if max_sec > 0.0 {
            (min_sec, max_sec)
        } else {
            (3.0, 6.0)
        };

        self.since_switch_sec = 0.0;
        self.next_switch_sec = self.rand_range(min_sec, max_sec);

        let (dmin, dmax) = mode.distance_range_to_subject();
        if dmax > 0.0 {
            self.subject_distance = self.rand_range(dmin, dmax);
        }
    }

    fn pick_next_mode(&mut self) -> CameraMode {
        // All camera modes (keep this in sync with enum `CameraMode`).
        // Avoid picking the same mode twice.
        let candidates = [
            CameraMode::First,
            CameraMode::Over,
            CameraMode::Back,
            CameraMode::BallChase,
            CameraMode::Side,
            CameraMode::FocusedChase,
            CameraMode::FocusedSide,
            CameraMode::PassingLeft,
            CameraMode::PassingRight,
            CameraMode::PassingTop,
            CameraMode::PassingBottom,
        ];

        // Try a few times to avoid repeats.
        for _ in 0..6 {
            let idx = (self.rand01() * candidates.len() as f32)
                .floor()
                .clamp(0.0, (candidates.len() - 1) as f32) as usize;
            let m = candidates[idx];
            if m != self.current {
                return m;
            }
        }
        // Fallback: deterministic step.
        match self.current {
            CameraMode::BallChase => CameraMode::Back,
            CameraMode::Back => CameraMode::Side,
            CameraMode::Side => CameraMode::First,
            CameraMode::First => CameraMode::Over,
            CameraMode::Over => CameraMode::BallChase,
            CameraMode::FocusedChase => CameraMode::FocusedSide,
            CameraMode::FocusedSide => CameraMode::PassingLeft,
            CameraMode::PassingLeft => CameraMode::PassingRight,
            CameraMode::PassingRight => CameraMode::PassingTop,
            CameraMode::PassingTop => CameraMode::PassingBottom,
            CameraMode::PassingBottom => CameraMode::FocusedChase,
        }
    }
}

impl Default for AutoPoseState {
    fn default() -> Self {
        Self {
            rng: 0xC0FF_EE11,
            current: PoseMode::Standing,
            since_switch_sec: 0.0,
            next_switch_sec: 2.3,
            collision_cooldown_sec: 0.0,
        }
    }
}

impl AutoPoseState {
    fn lcg_next(&mut self) -> u32 {
        // Deterministic, fast RNG (good enough for pose variation).
        self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
        self.rng
    }

    fn rand_f32_01(&mut self) -> f32 {
        let v = self.lcg_next();
        // Use top 24 bits for a stable mantissa.
        let mant = (v >> 8) & 0x00FF_FFFF;
        mant as f32 / 16_777_215.0
    }

    fn pick_random_pose(&mut self) -> PoseMode {
        let poses = [
            PoseMode::Standing,
            PoseMode::Belly,
            PoseMode::Back,
            PoseMode::LeftSide,
            PoseMode::RightSide,
        ];
        let mut idx = (self.lcg_next() as usize) % poses.len();
        if poses[idx] == self.current {
            idx = (idx + 1) % poses.len();
        }
        poses[idx]
    }

    fn schedule_next_switch(&mut self) {
        // 1.2s .. 3.6s
        self.next_switch_sec = 1.2 + 2.4 * self.rand_f32_01();
        self.since_switch_sec = 0.0;
    }
}

fn pose_targets(pose: PoseMode) -> (f32, f32) {
    match pose {
        PoseMode::Standing => (0.0, 0.0),
        PoseMode::Belly => (0.0, 1.0),
        PoseMode::Back => (0.0, -1.0),
        PoseMode::LeftSide => (std::f32::consts::FRAC_PI_2, 0.0),
        PoseMode::RightSide => (-std::f32::consts::FRAC_PI_2, 0.0),
        PoseMode::Auto | PoseMode::Random => (0.0, 0.0),
    }
}

fn pose_from_local_hit_dir(local_hit_dir: Vec3) -> PoseMode {
    // Choose the dominant axis in the model-local frame:
    // - +Z belly, -Z back
    // - +X left side, -X right side
    // - +Y standing (upright contact)
    let d = local_hit_dir.normalize_or_zero();
    if d.length_squared() < 1e-6 {
        return PoseMode::Standing;
    }

    if d.y > 0.78 {
        return PoseMode::Standing;
    }

    if d.z.abs() >= d.x.abs() {
        if d.z >= 0.0 {
            PoseMode::Belly
        } else {
            PoseMode::Back
        }
    } else if d.x >= 0.0 {
        PoseMode::LeftSide
    } else {
        PoseMode::RightSide
    }
}

#[derive(Component)]
struct MainCamera;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Component)]
struct OverlayUiCamera;

#[derive(Component, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ViewCamera {
    index: u8,
}

#[derive(Resource, Clone, Copy)]
struct MultiView {
    count: u8,
}

impl Default for MultiView {
    fn default() -> Self {
        Self { count: 1 }
    }
}

impl MultiView {
    const MAX_VIEWS: u8 = 4;

    fn increment(&mut self) -> bool {
        let prev = self.count;
        self.count = (self.count + 1).clamp(1, Self::MAX_VIEWS);
        self.count != prev
    }

    fn decrement(&mut self) -> bool {
        let prev = self.count;
        self.count = self.count.saturating_sub(1).clamp(1, Self::MAX_VIEWS);
        self.count != prev
    }
}

#[derive(Resource, Default, Clone)]
struct MultiViewHint {
    text: String,
    until_sec: f64,
}

impl MultiViewHint {
    const SHOW_FOR_SEC: f64 = 1.6;

    fn show(&mut self, now_sec: f64, text: impl Into<String>) {
        self.text = text.into();
        self.until_sec = now_sec + Self::SHOW_FOR_SEC;
    }

    fn active(&self, now_sec: f64) -> bool {
        !self.text.is_empty() && now_sec < self.until_sec
    }
}

#[derive(Resource, Default)]
struct SubjectNormalsComputed(std::collections::HashSet<bevy::asset::AssetId<Mesh>>);

#[derive(Resource, Clone, Copy)]
struct SubjectDynamics {
    initialized: bool,
    theta: f32,
    omega: f32,
    human_r: f32,
    human_vr: f32,
    human_roll: f32,
    human_pitch: f32,
    ball_r: f32,
    ball_vr: f32,
    ball_roll: f32,
}

impl Default for SubjectDynamics {
    fn default() -> Self {
        let human_r_max = TUBE_RADIUS - HUMAN_RADIUS;
        let ball_r_max = TUBE_RADIUS - BALL_RADIUS;
        Self {
            initialized: false,
            theta: 0.0,
            omega: 0.0,
            human_r: human_r_max - CONTACT_EPS,
            human_vr: 0.0,
            human_roll: 0.0,
            human_pitch: 0.0,
            ball_r: ball_r_max - CONTACT_EPS,
            ball_vr: 0.0,
            ball_roll: 0.0,
        }
    }
}

#[derive(Resource, Clone)]
struct Playback {
    time_sec: f32,
    playing: bool,
    speed: f32,
}

#[derive(Resource, Clone)]
struct TubeSettings {
    scheme: u32,
    pattern: u32,
}

#[derive(Resource)]
struct TubeScene {
    curve: CatmullRomCurve,
    frames: Frames,
    tube_material: Handle<TubeMaterial>,
}

#[derive(Resource)]
struct OverlayState {
    last_credit_idx: i32,
    last_caption_idx: i32,
    last_visible: bool,
    last_caption_visible: bool,
}

#[derive(Resource, Clone, Copy)]
struct OverlayVisibility {
    show: bool,
}

#[derive(Resource, Clone, Copy)]
struct CaptionVisibility {
    show: bool,
}

#[cfg(target_arch = "wasm32")]
#[derive(Resource, Clone, Copy)]
struct VideoVisibility {
    show: bool,
}

#[derive(Resource, Clone, Default)]
struct OverlayText {
    credit: String,
    caption: String,
    caption_is_meta: bool,
}

impl Default for OverlayState {
    fn default() -> Self {
        Self {
            last_credit_idx: -1,
            last_caption_idx: -1,
            last_visible: false,
            last_caption_visible: true,
        }
    }
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static JS_INPUT: std::cell::RefCell<JsInput> = const { std::cell::RefCell::new(JsInput::new()) };
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
struct JsInput {
    has_time: bool,
    time_sec: f32,
    playing: bool,
    sample_age_sec: f32,
    toggle_scheme: bool,
    toggle_texture: bool,
    speed_delta: i32,
    toggle_overlay: bool,
}

#[cfg(target_arch = "wasm32")]
impl JsInput {
    const fn new() -> Self {
        Self {
            has_time: false,
            time_sec: 0.0,
            playing: false,
            sample_age_sec: 0.0,
            toggle_scheme: false,
            toggle_texture: false,
            speed_delta: 0,
            toggle_overlay: false,
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_video_time(time_sec: f32, playing: bool) {
    JS_INPUT.with(|s| {
        let mut st = s.borrow_mut();
        st.has_time = true;
        st.time_sec = time_sec.max(0.0);
        st.playing = playing;
    });
}

// Allow JS to set an internal render scale for the wasm app. We store it in an
// AtomicU32 (bits of an f32) so it can be written from JS quickly and read by
// a Bevy system without heavy synchronization.
#[cfg(target_arch = "wasm32")]
mod render_scale_api {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static RENDER_SCALE_BITS: AtomicU32 = AtomicU32::new(f32::to_bits(1.0));

    #[wasm_bindgen]
    pub fn set_render_scale(scale: f32) {
        // sanitize input: must be finite and > 0
        let s = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
        RENDER_SCALE_BITS.store(f32::to_bits(s), Ordering::SeqCst);
    }

    // JS-visible getter (optional; useful for debugging from JS)
    #[wasm_bindgen]
    pub fn get_render_scale() -> f32 {
        f32::from_bits(RENDER_SCALE_BITS.load(Ordering::SeqCst))
    }

    // Internal accessor used by Bevy systems.
    pub fn current_render_scale() -> f32 {
        f32::from_bits(RENDER_SCALE_BITS.load(Ordering::SeqCst))
    }
}

// Resource to expose render scale inside ECS. Systems can read this and adapt
// rendering accordingly (e.g. adjust render targets or camera scale).
#[derive(Resource, Debug, Clone, Copy)]
#[allow(dead_code)]
struct RenderScale(f32);

impl Default for RenderScale {
    fn default() -> Self {
        RenderScale(1.0)
    }
}

#[cfg(target_arch = "wasm32")]
fn update_render_scale_resource(mut rs: ResMut<RenderScale>) {
    rs.0 = render_scale_api::current_render_scale();
}

// Apply the requested render scale by adjusting the primary window's render
// resolution on wasm. This avoids touching the DOM canvas directly; instead
// Bevy will render at the smaller resolution and the browser will upscale it.
#[cfg(target_arch = "wasm32")]
fn apply_render_scale_to_window(
    mut query: Query<&mut Window, With<PrimaryWindow>>,
    rs: Res<RenderScale>,
) {
    // Local imports required only on wasm target
    use wasm_bindgen::JsCast;
    use web_sys::HtmlElement;

    if !rs.is_changed() {
        return;
    }
    // Query the primary window via Bevy's ECS query interface.
    if let Ok(mut primary) = query.single_mut() {
        // Query the canvas client size via web-sys.
        if let Some(win) = web_sys::window() {
            if let Some(doc) = win.document() {
                if let Some(el) = doc.get_element_by_id("bevy-canvas") {
                    // Cast to HtmlElement so we can query clientWidth/clientHeight
                    if let Ok(html) = el.dyn_into::<HtmlElement>() {
                        let clw = html.client_width() as f32;
                        let clh = html.client_height() as f32;
                        // devicePixelRatio gives the CSS->device pixel ratio
                        let ratio = win.device_pixel_ratio() as f32;
                        let target_w = (clw * rs.0 * ratio).max(1.0);
                        let target_h = (clh * rs.0 * ratio).max(1.0);
                        // Use Bevy 0.17 WindowResolution API to request the physical
                        // resolution we want the renderer to target. `set_physical_resolution`
                        // takes integer physical pixel dimensions.
                        primary
                            .resolution
                            .set_physical_resolution(target_w as u32, target_h as u32);
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn toggle_color_scheme() {
    JS_INPUT.with(|s| s.borrow_mut().toggle_scheme = true);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn toggle_texture() {
    JS_INPUT.with(|s| s.borrow_mut().toggle_texture = true);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn speed_up() {
    JS_INPUT.with(|s| s.borrow_mut().speed_delta += 1);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn slow_down() {
    JS_INPUT.with(|s| s.borrow_mut().speed_delta -= 1);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn toggle_overlay() {
    JS_INPUT.with(|s| s.borrow_mut().toggle_overlay = true);
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use web_sys::{HtmlCanvasElement, ImageBitmap};

// Request a readback of the primary `#bevy-canvas` from wasm and post an
// ImageBitmap to the parent window. The returned Promise resolves when the
// ImageBitmap has been transferred to the parent (or rejects on error).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn request_wasm_readback() -> Result<(), JsValue> {
    let win = web_sys::window().ok_or(JsValue::from_str("no window"))?;
    let doc = win.document().ok_or(JsValue::from_str("no document"))?;
    let el = doc.get_element_by_id("bevy-canvas").ok_or(JsValue::from_str("no canvas"))?;
    let canvas = el.dyn_into::<HtmlCanvasElement>().map_err(|_| JsValue::from_str("element not canvas"))?;

    // Use `createImageBitmap(canvas)` which returns a Promise<ImageBitmap>.
    let promise = win.create_image_bitmap_with_html_canvas_element(&canvas)
        .map_err(|e| e)?;
    let ib_val = JsFuture::from(promise).await?;
    let ib = ib_val.dyn_into::<ImageBitmap>()?;

    // Build a message containing the bitmap and transfer it to parent.
    let msg = js_sys::Object::new();
    js_sys::Reflect::set(&msg, &JsValue::from_str("type"), &JsValue::from_str("wasm_imagebitmap"))?;
    js_sys::Reflect::set(&msg, &JsValue::from_str("bitmap"), &ib.clone().into())?;
    let transfer = js_sys::Array::new();
    transfer.push(&ib);
    win.post_message_with_transfer(&JsValue::from(msg), "*", &transfer)
        .map_err(|e| e)?;
    Ok(())
}

// Pixel-buffer fallback: read RGBA pixels from the `#bevy-canvas` and post an
// ArrayBuffer containing raw u8 RGBA pixels to the parent. This is more
// interoperable across UAs when ImageBitmap transfers fail.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn request_wasm_readback_pixels() -> Result<(), JsValue> {
    let win = web_sys::window().ok_or(JsValue::from_str("no window"))?;
    let doc = win.document().ok_or(JsValue::from_str("no document"))?;
    let el = doc.get_element_by_id("bevy-canvas").ok_or(JsValue::from_str("no canvas"))?;
    let canvas = el.dyn_into::<HtmlCanvasElement>().map_err(|_| JsValue::from_str("element not canvas"))?;

    // Create an ImageBitmap from the canvas so we get a stable snapshot.
    let promise = win.create_image_bitmap_with_html_canvas_element(&canvas).map_err(|e| e)?;
    let ib_val = JsFuture::from(promise).await?;
    let ib = ib_val.dyn_into::<ImageBitmap>()?;

    // Create a temporary canvas and draw the ImageBitmap into its 2D context.
    let tmp = doc.create_element("canvas")?.dyn_into::<HtmlCanvasElement>()?;
    let w = ib.width();
    let h = ib.height();
    tmp.set_width(w);
    tmp.set_height(h);
    let ctx = tmp.get_context("2d")?.ok_or(JsValue::from_str("no 2d context"))?;
    let ctx2 = ctx.dyn_into::<web_sys::CanvasRenderingContext2d>()?;
    // drawImage(ImageBitmap, 0, 0)
    ctx2.draw_image_with_image_bitmap(&ib, 0.0, 0.0)?;

    // Extract ImageData (RGBA u8 clamped) and copy into a Uint8Array for transfer.
    let image_data = ctx2.get_image_data(0.0, 0.0, w as f64, h as f64)?;
    let mut buf = image_data.data(); // Clamped<Vec<u8>>
    let uint8 = js_sys::Uint8Array::new_with_length(buf.len() as u32);
    uint8.copy_from(&buf[..]);
    let pixels_buf = uint8.buffer();

    // Build message and transfer the underlying ArrayBuffer.
    let msg = js_sys::Object::new();
    js_sys::Reflect::set(&msg, &JsValue::from_str("type"), &JsValue::from_str("wasm_pixels"))?;
    // Send the ArrayBuffer itself for maximum interop.
    js_sys::Reflect::set(&msg, &JsValue::from_str("pixels"), &pixels_buf)?;
    js_sys::Reflect::set(&msg, &JsValue::from_str("width"), &JsValue::from_f64(w as f64))?;
    js_sys::Reflect::set(&msg, &JsValue::from_str("height"), &JsValue::from_f64(h as f64))?;
    // Diagnostic: compute a small checksum and sample of the first bytes so parent can verify pre-transfer contents.
    let mut sample_len = std::cmp::min(64usize, buf.len());
    let mut sum: u32 = 0;
    for i in 0..sample_len {
        sum = sum.wrapping_add(buf[i] as u32);
    }
    // If the initial ImageBitmap → canvas draw yields an all-zero sample, try
    // drawing the source canvas directly into the temp canvas and re-sample.
    // Some UAs/compositors may produce empty ImageBitmaps — this gives a
    // fallback before we escalate to GPU readback.
    let mut attempted_alt_draw = false;
    if sum == 0 {
        // Draw the original canvas directly
        if ctx2.draw_image_with_html_canvas_element(&canvas, 0.0, 0.0).is_ok() {
            if let Ok(new_image) = ctx2.get_image_data(0.0, 0.0, w as f64, h as f64) {
                buf = new_image.data();
                sample_len = std::cmp::min(64usize, buf.len());
                sum = 0;
                for i in 0..sample_len {
                    sum = sum.wrapping_add(buf[i] as u32);
                }
                attempted_alt_draw = true;
            }
        }
    }
    js_sys::Reflect::set(&msg, &JsValue::from_str("checksum"), &JsValue::from_f64(sum as f64))?;
    // Expose a short sample as a JS Array for quick inspection.
    let sample_arr = js_sys::Array::new();
    for i in 0..std::cmp::min(16usize, buf.len()) {
        sample_arr.push(&JsValue::from_f64(buf[i] as f64));
    }
    js_sys::Reflect::set(&msg, &JsValue::from_str("sample"), &sample_arr)?;
    js_sys::Reflect::set(&msg, &JsValue::from_str("attempted_alt_draw"), &JsValue::from_bool(attempted_alt_draw))?;

    // If we are inside an iframe, transfer the buffer to the parent (fast, zero-copy).
    // If we are top-level, DO NOT transfer to ourselves: some runtimes end up delivering
    // a detached (0-byte) buffer to the receiver when sender==receiver.
    let in_frame = win.frame_element().ok().flatten().is_some();
    if in_frame {
        if let Ok(Some(parent)) = win.parent() {
            let transfer = js_sys::Array::new();
            transfer.push(&pixels_buf);
            parent
                .post_message_with_transfer(&JsValue::from(msg), "*", &transfer)
                .map_err(|e| e)?;
        } else {
            // Fallback: no parent available; post locally without transfer.
            win.post_message(&JsValue::from(msg), "*").map_err(|e| e)?;
        }
    } else {
        win.post_message(&JsValue::from(msg), "*").map_err(|e| e)?;
    }
    Ok(())
}

// JS-visible hook to request a GPU readback. This increments an atomic counter
// that the ECS system observes and will serve as the trigger for a true
// GPU->CPU readback implementation (added iteratively).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn request_wasm_readback_gpu() {
    // Increment the request sequence; the scaffold system will observe this
    // and clear it when handled.
    READBACK_REQUEST_SEQ.fetch_add(1, Ordering::SeqCst);
    web_sys::console::debug_1(&JsValue::from_str("wasm:gpu_readback requested"));
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = globalThis, js_name = mcbaise_request_playing)]
    fn mcbaise_request_playing(playing: bool);

    #[wasm_bindgen(js_namespace = globalThis, js_name = mcbaise_set_credit)]
    fn mcbaise_set_credit(html: &str, show: bool);

    #[wasm_bindgen(js_namespace = globalThis, js_name = mcbaise_set_caption)]
    fn mcbaise_set_caption(text: &str, show: bool, is_meta: bool);

    #[wasm_bindgen(js_namespace = globalThis, js_name = mcbaise_set_video_visible)]
    fn mcbaise_set_video_visible(show: bool);

    #[wasm_bindgen(js_namespace = globalThis, js_name = mcbaise_set_loading)]
    fn mcbaise_set_loading(show: bool, text: &str);

    #[wasm_bindgen(js_namespace = globalThis, js_name = mcbaise_set_wasm_ready)]
    fn mcbaise_set_wasm_ready();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn main() {
    #[cfg(target_arch = "wasm32")]
    wasm_dbg("wasm: main() entered");
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    #[cfg(not(target_arch = "wasm32"))]
    {
        fn truthy_env(name: &str) -> bool {
            std::env::var(name)
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        }

        #[cfg(feature = "burn_human")]
        fn assets_self_test() {
            use std::fs;
            use std::io;
            use std::path::{Path, PathBuf};
            use std::time::{SystemTime, UNIX_EPOCH};

            fn exists_file(p: &Path) -> bool {
                fs::metadata(p).map(|m| m.is_file()).unwrap_or(false)
            }

            fn ensure_parent(p: &Path) -> io::Result<()> {
                if let Some(parent) = p.parent() {
                    fs::create_dir_all(parent)?;
                }
                Ok(())
            }

            let local_tensor = PathBuf::from("assets/model/fullbody_default.safetensors");
            let local_meta = PathBuf::from("assets/model/fullbody_default.meta.json");

            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let backup_dir = PathBuf::from(format!(".tmp/mcbaise_assets_self_test_{ts}"));
            let backup_tensor = backup_dir.join("fullbody_default.safetensors");
            let backup_meta = backup_dir.join("fullbody_default.meta.json");

            let had_tensor = exists_file(&local_tensor);
            let had_meta = exists_file(&local_meta);

            if had_tensor || had_meta {
                let _ = fs::create_dir_all(&backup_dir);
            }

            // Move local assets aside (if present) so we test the download/cache path.
            if had_tensor {
                let _ = ensure_parent(&backup_tensor);
                fs::rename(&local_tensor, &backup_tensor)
                    .unwrap_or_else(|e| panic!("failed to move {}: {e}", local_tensor.display()));
            }
            if had_meta {
                let _ = ensure_parent(&backup_meta);
                fs::rename(&local_meta, &backup_meta)
                    .unwrap_or_else(|e| panic!("failed to move {}: {e}", local_meta.display()));
            }

            struct Restore {
                local_tensor: PathBuf,
                local_meta: PathBuf,
                backup_dir: PathBuf,
                backup_tensor: PathBuf,
                backup_meta: PathBuf,
                had_tensor: bool,
                had_meta: bool,
            }
            impl Drop for Restore {
                fn drop(&mut self) {
                    if self.had_tensor {
                        let _ = fs::create_dir_all(
                            self.local_tensor.parent().unwrap_or(Path::new(".")),
                        );
                        let _ = fs::rename(&self.backup_tensor, &self.local_tensor);
                    }
                    if self.had_meta {
                        let _ =
                            fs::create_dir_all(self.local_meta.parent().unwrap_or(Path::new(".")));
                        let _ = fs::rename(&self.backup_meta, &self.local_meta);
                    }
                    let _ = fs::remove_dir_all(&self.backup_dir);
                }
            }

            let _restore = Restore {
                local_tensor,
                local_meta,
                backup_dir: backup_dir.clone(),
                backup_tensor,
                backup_meta,
                had_tensor,
                had_meta,
            };

            // Force download even if cache already exists.
            // NOTE: Rust 2024 makes env mutation unsafe; this runs before Bevy spawns threads.
            unsafe {
                std::env::set_var("MCBAISE_ASSETS_AUTO_DOWNLOAD", "1");
                std::env::set_var("MCBAISE_ASSETS_FORCE_DOWNLOAD", "1");
            }

            let src = native_assets::resolve_burn_human_source();
            match src {
                BurnHumanSource::Paths { tensor, meta } => {
                    if !exists_file(&tensor) {
                        panic!(
                            "assets self-test failed: tensor missing at {}",
                            tensor.display()
                        );
                    }
                    if !exists_file(&meta) {
                        panic!(
                            "assets self-test failed: meta missing at {}",
                            meta.display()
                        );
                    }
                    eprintln!("[mcbaise] assets self-test ok");
                    eprintln!("[mcbaise] tensor: {}", tensor.display());
                    eprintln!("[mcbaise] meta:   {}", meta.display());
                }
                BurnHumanSource::Bytes { .. } => {
                    eprintln!("[mcbaise] assets self-test ok (embedded bytes)");
                }
                BurnHumanSource::Preloaded(_) => {
                    eprintln!("[mcbaise] assets self-test ok (preloaded)");
                }
                BurnHumanSource::AssetPath(meta_path) => {
                    let raw = std::path::PathBuf::from(&meta_path);
                    let candidate = if exists_file(&raw) {
                        raw
                    } else {
                        std::path::PathBuf::from("assets").join(&meta_path)
                    };

                    if exists_file(&candidate) {
                        eprintln!("[mcbaise] assets self-test ok (asset pipeline)");
                        eprintln!("[mcbaise] meta:   {}", candidate.display());
                    } else {
                        eprintln!(
                            "[mcbaise] assets self-test: meta asset path does not exist on disk: {meta_path}"
                        );
                    }
                }
                BurnHumanSource::Asset(_) => {
                    eprintln!("[mcbaise] assets self-test ok (asset pipeline handle)");
                }
            }
        }

        let args: Vec<String> = std::env::args().collect();
        let self_test = args.iter().any(|a| a == "--assets-self-test");
        if self_test {
            #[cfg(feature = "burn_human")]
            {
                assets_self_test();
            }
            #[cfg(not(feature = "burn_human"))]
            {
                eprintln!("assets self-test suppressed: burn_human feature disabled");
            }
            return;
        }

        // Env-var preflight: download/resolve + print paths, then exit.
        let preflight = truthy_env("MCBAISE_ASSETS_PREFLIGHT");
        if preflight {
            #[cfg(feature = "burn_human")]
            {
                let src = native_assets::resolve_burn_human_source();
                match src {
                    BurnHumanSource::Paths { tensor, meta } => {
                        eprintln!("[mcbaise] assets preflight ok");
                        eprintln!("[mcbaise] tensor: {}", tensor.display());
                        eprintln!("[mcbaise] meta:   {}", meta.display());
                    }
                    BurnHumanSource::Bytes { .. } => {
                        eprintln!("[mcbaise] assets preflight ok (embedded bytes)");
                    }
                    BurnHumanSource::Preloaded(_) => {
                        eprintln!("[mcbaise] assets preflight ok (preloaded)");
                    }
                    BurnHumanSource::AssetPath(meta_path) => {
                        eprintln!("[mcbaise] assets preflight ok (asset pipeline)");
                        eprintln!("[mcbaise] meta:   {meta_path}");
                    }
                    BurnHumanSource::Asset(_) => {
                        eprintln!("[mcbaise] assets preflight ok (asset pipeline handle)");
                    }
                }
            }
            #[cfg(not(feature = "burn_human"))]
            {
                eprintln!("assets preflight skipped: burn_human feature disabled");
            }
            return;
        }
    }

    #[cfg(feature = "burn_human")]
    let burn_plugin = {
        #[cfg(target_arch = "wasm32")]
        {
            wasm_dbg("wasm: decompressing embedded tensor (lz4)...");
            let tensor_vec = lz4_flex::decompress_size_prepended(TENSOR_LZ4_BYTES)
                .expect("decompress embedded burn_human tensor (lz4)");
            wasm_dbg("wasm: tensor decompressed");
            let tensor: &'static [u8] = Box::leak(tensor_vec.into_boxed_slice());
            BurnHumanPlugin {
                source: BurnHumanSource::Bytes {
                    tensor,
                    meta: META_BYTES,
                },
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            BurnHumanPlugin {
                source: native_assets::resolve_burn_human_source(),
            }
        }
    };

    let plugins = DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: format!("{VIDEO_ID} • tube ride"),
            #[cfg(target_arch = "wasm32")]
            canvas: Some("#bevy-canvas".to_string()),
            #[cfg(target_arch = "wasm32")]
            fit_canvas_to_parent: true,
            ..default()
        }),
        ..default()
    });

    // On wasm, Bevy's HTTP asset reader will try to fetch `<asset>.meta` as well.
    // We don't ship meta files for this demo, so disable meta-checking to avoid 404 spam.
    #[cfg(target_arch = "wasm32")]
    let plugins = plugins.set(bevy::asset::AssetPlugin {
        meta_check: bevy::asset::AssetMetaCheck::Never,
        ..default()
    });

    let mut app = App::new();

    #[cfg(target_arch = "wasm32")]
    wasm_dbg("wasm: building Bevy App");

    // Ensure EmbeddedAssetRegistry exists in the World before embedding assets.
    app.init_resource::<EmbeddedAssetRegistry>();

    // Register a small local plugin that embeds the shader into the in-memory registry.
    // We omit the crate path from the embedded URL by passing it as `omit_prefix` so
    // the resulting asset URL becomes: embedded://<crate>/src/mcbaise_tube.wgsl
    #[allow(non_camel_case_types)]
    struct local_embedded_asset_plugin;

    impl Plugin for local_embedded_asset_plugin {
        fn build(&self, app: &mut App) {
            // Register the shader file located alongside this source file.
            embedded_asset!(app, "mcbaise_tube.wgsl");

            // Embedded asset bytes were registered above. The embedded AssetSource
            // is registered later at startup by `register_embedded_asset_source` so
            // the AssetPlugin has been added and `AssetSourceBuilders` exists.
        }
    }

    app.insert_resource(ClearColor(Color::BLACK))
        .insert_resource(Playback {
            time_sec: 0.0,
            playing: cfg!(not(target_arch = "wasm32")),
            speed: 1.0,
        })
        .insert_resource(TubeSettings {
            scheme: 0,
            pattern: 0,
        })
        .insert_resource(OverlayVisibility { show: false })
        .insert_resource(CaptionVisibility { show: true })
        .insert_resource(OverlayText::default());

    // GPU readback wiring: create an offscreen render target image, retarget
    // cameras for one frame when a request arrives, then in RenderApp copy
    // that texture to a mapped buffer and write/post pixels.
    app.init_resource::<GpuReadbackImage>();
    app.init_resource::<GpuReadbackPending>();
    app.init_resource::<PendingWindowGeometry>();
    app.init_resource::<ResizeAutomation>();
    app.init_resource::<CameraRestartRequested>();
    app.init_resource::<SceneRestartRequested>();
    app.init_resource::<TeardownState>();
    app.init_resource::<LoadingState>();
    app.init_resource::<BurnHumanSpawned>();
    app.insert_resource(BurnHumanEnabled::default());
    app.add_systems(Startup, ensure_readback_image_exists);

    // Optional automation: drive the resize stepper (the same behavior as the UI
    // resize icon) without user interaction.
    // - MCBAISE_AUTORESIZE=1
    // - MCBAISE_AUTORESIZE_START_FRAME=120 (default)
    // - MCBAISE_AUTORESIZE_STEP_EVERY_FRAMES=12 (default)
    // - MCBAISE_AUTORESIZE_MAX_STEPS=0 (0 = infinite)
    if std::env::var("MCBAISE_AUTORESIZE").as_deref().ok() == Some("1") {
        let autoresize_start_frame: u64 = std::env::var("MCBAISE_AUTORESIZE_START_FRAME")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(120);
        let autoresize_step_every_frames: u64 = std::env::var("MCBAISE_AUTORESIZE_STEP_EVERY_FRAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(12);
        let autoresize_max_steps: u32 = std::env::var("MCBAISE_AUTORESIZE_MAX_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        let mut auto = app.world_mut().resource_mut::<ResizeAutomation>();
        auto.active = true;
        auto.env_forced = true;
        auto.start_frame = autoresize_start_frame;
        auto.step_every_frames = autoresize_step_every_frames.max(1);
        auto.max_steps = autoresize_max_steps;
        auto.steps_done = 0;
        auto.exit_requested = false;
        auto.seeded_from_window = false;
        auto.first_frame = false;
        auto.next_step_frame = 0;
        eprintln!(
            "native: env autoresize enabled (start_frame={}, step_every_frames={}, max_steps={})",
            auto.start_frame,
            auto.step_every_frames,
            auto.max_steps
        );
    }
    // Spawn the offscreen capture camera(s) once the main camera exists.
    // We moved this to Update to support toggling capture_multi at runtime.
    // Deterministic GPU readback flow:
    // - keep capture camera synced to main camera
    // - size the offscreen target
    // - arm capture camera when a request arrives
    // - disable capture camera after a few frames
    app.add_systems(Update, loading_watch_system);
    // Ordering matters for resize automation:
    // - Apply deferred resize first.
    // - Run the automation step before the settle watcher so the watcher can
    //   clear the previous step without the automation immediately re-arming
    //   `pending` in the same frame.
    app.add_systems(
        Update,
        window_geometry_watch_system
            .after(main_frame_tick_system)
            .after(apply_deferred_window_resolution_change_system)
            .after(resize_automation_step),
    );
    app.add_systems(
        Update,
        resize_automation_step
            .after(main_frame_tick_system)
            // Ensure camera gating observes the previous frame's pending state,
            // but do not let scheduling a new step disable cameras in the same frame.
            .after(gate_main_cameras_during_geometry_system)
            .before(window_geometry_watch_system),
    );
    app.add_systems(Update, sync_resize_toggle_system);
    // Critical ordering: apply the requested window resolution change before any
    // camera/view viewport systems run this frame. Otherwise, Bevy may reconfigure
    // the surface to the new size while viewports are still computed from the old
    // size, producing a one-frame mismatch (e.g. surface=256x144, viewport=1280x240).
    // Also run after the main frame tick so we can stamp a scheduled frame.
    app.add_systems(
        Update,
        cycle_resolution_on_resize_icon_request_system
            .after(main_frame_tick_system)
            .before(sync_view_cameras),
    );
    app.add_systems(Update, teardown_countdown_system);
    app.add_systems(Update, restart_cameras_system);
    app.add_systems(Update, restart_scene_system);
    // (capture_window_surfaces_exclusive removed)
    app.add_systems(Update, ensure_gpu_capture_camera_exists);
    app.add_systems(Update, sync_gpu_capture_cameras_from_views);
    app.add_systems(Update, sync_gpu_capture_camera_fog);
    app.add_systems(Update, update_gpu_capture_viewports);
    app.add_systems(Update, configure_gpu_capture_camera_settings);
    app.add_systems(Update, resize_readback_image_to_window);
    app.add_systems(Update, arm_cameras_for_gpu_readback);
    app.add_systems(Update, restore_cameras_after_gpu_readback);
        // Native: we create the primary egui context ourselves (on a dedicated overlay camera)
        // so it isn't constrained by the first 3D camera's viewport.
        // WASM: keep the default auto-created primary context.
    app.insert_resource(bevy_egui::EguiGlobalSettings {
            auto_create_primary_context: cfg!(target_arch = "wasm32"),
            ..default()
        })
        .init_resource::<SubjectMode>()
        .init_resource::<PoseMode>()
        .init_resource::<CameraPreset>()
        .init_resource::<AutoPoseState>()
        .init_resource::<AutoCameraState>()
        .init_resource::<AutoSubjectState>()
        .init_resource::<ColorSchemeMode>()
        .init_resource::<TexturePatternMode>()
        .init_resource::<AutoTubeStyleState>()
        .init_resource::<HumanAppearanceMode>()
        .init_resource::<BallAppearanceMode>()
        .init_resource::<AutoHumanAppearanceState>()
        .init_resource::<AutoBallAppearanceState>()
        .init_resource::<SubjectDynamics>()
        .init_resource::<SubjectNormalsComputed>()
        .init_resource::<FluidSimulation>()
        .init_resource::<MultiView>()
        .init_resource::<MultiViewHint>()
        .init_resource::<EguiCaptureState>()
        .init_resource::<RenderScale>()
        .add_plugins(plugins);

        // Diagnostic: optionally skip PBR-related plugins for isolation testing.
        // If `MCBAISE_DISABLE_PBR=1` is set, do not register the wireframe/material/burn plugins.
        if std::env::var("MCBAISE_DISABLE_PBR").as_deref().ok() == Some("1") {
            eprintln!("diagnostic: MCBAISE_DISABLE_PBR=1 - skipping wireframe/material/burn plugins");
        } else {
            app.add_plugins(bevy::pbr::wireframe::WireframePlugin::default());
        }

    #[cfg(target_arch = "wasm32")]
    app.insert_resource(VideoVisibility { show: true });

    app.add_plugins(EguiPlugin::default())
        // Ensure embedded assets are registered before materials request shaders.
        .add_plugins(local_embedded_asset_plugin);

    // Always register material and burn plugins; skipping these breaks startup systems.
    app.add_plugins(MaterialPlugin::<TubeMaterial>::default());

    #[cfg(feature = "burn_human")]
    app.add_plugins(burn_plugin);

    app.add_systems(Startup, setup_scene)
        // Diagnostic: print embedded registry contents at startup to verify registered paths
        .add_systems(
            Startup,
            (register_embedded_asset_source, print_embedded_registry),
        )
        .add_systems(Update, sync_view_cameras)
        .add_systems(Update, update_multiview_viewports.after(sync_view_cameras))
        .add_systems(Update, ensure_subject_normals)
        .add_systems(Update, update_fluid_simulation)
        .add_systems(Update, update_tube_and_subject)
            .add_systems(Update, apply_subject_mode)
            .add_systems(Update, enforce_burn_human_subject_mode.before(apply_subject_mode))
        .add_systems(Update, update_overlays)
        .add_systems(EguiPrimaryContextPass, ui_overlay);
    #[cfg(feature = "burn_human")]
    app.add_systems(Update, spawn_burn_human_when_ready);
    
    #[cfg(all(not(target_arch = "wasm32"), feature = "capture_ui"))]
    {
        // No extra registration needed for egui polling/state handled in ui_overlay
    }

    #[cfg(target_arch = "wasm32")]
    {
        app.init_resource::<WasmDebugFrameOnce>();
        app.add_systems(Update, wasm_debug_first_update_tick);
    }

    // IMPORTANT: register extraction only after DefaultPlugins/RenderPlugin has created the RenderApp.
    // If we add this plugin earlier, the RenderApp doesn't exist yet and the readback resource won't
    // be extracted into the render world, causing "render world missing GpuReadbackImage".
    app.add_plugins(bevy::render::extract_resource::ExtractResourcePlugin::<GpuReadbackImage>::default());
    app.add_plugins(bevy::render::extract_resource::ExtractResourcePlugin::<PendingWindowGeometry>::default());

    // Initialize a persistent staging buffer resource used by the render-side
    // readback system so we don't allocate/destroy many wgpu buffers per capture.
    app.init_resource::<ReadbackStaging>();

    use bevy::ecs::schedule::IntoScheduleConfigs;
    use bevy::render::{RenderApp, RenderSystems};
    app.init_resource::<PendingRenderAssetDrops>();
    // Global frame counter used to implement a small safety delay (frames)
    // before destructive render-thread drops are consumed. Incremented
    // separately in the main and render sub-apps to provide a simple
    // frame-based coordination point.
    app.init_resource::<GlobalFrameCount>();
    app.init_resource::<ResolutionSetMode>();
    app.add_systems(Update, main_frame_tick_system);
    // Optional automation: cycle the same preset resolutions without user input.
    // Enable via env var to reproduce resize timing issues:
    // - MCBAISE_AUTOCYCLE_RES=1
    // - MCBAISE_AUTOCYCLE_EVERY_FRAMES=60 (default)
    // - MCBAISE_AUTOCYCLE_START_FRAME=120 (default)
    // - MCBAISE_AUTOCYCLE_MAX_CYCLES=0 (0 = infinite)
    // If you need to ensure autocycle is off (e.g. for autoresize-only tests), set:
    // - MCBAISE_DISABLE_AUTOCYCLE=1
    let autocycle_disabled = std::env::var("MCBAISE_DISABLE_AUTOCYCLE").as_deref().ok() == Some("1");
    let autocycle_active = !autocycle_disabled
        && std::env::var("MCBAISE_AUTOCYCLE_RES").as_deref().ok() == Some("1");
    let autocycle_every_frames: u64 = std::env::var("MCBAISE_AUTOCYCLE_EVERY_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    let autocycle_start_frame: u64 = std::env::var("MCBAISE_AUTOCYCLE_START_FRAME")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);
    let autocycle_max_cycles: u32 = std::env::var("MCBAISE_AUTOCYCLE_MAX_CYCLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    app.insert_resource(AutoResolutionCycle {
        active: autocycle_active,
        every_frames: autocycle_every_frames,
        start_frame: autocycle_start_frame,
        max_cycles: autocycle_max_cycles,
        cycles_done: 0,
        next_index: 0,
    });
    app.init_resource::<DeferredWindowResolutionChange>();
    app.add_systems(
        Update,
        apply_deferred_window_resolution_change_system
            .after(main_frame_tick_system)
            .before(sync_view_cameras),
    );
    app.add_systems(
        Update,
        auto_cycle_resolution_system
            .after(main_frame_tick_system)
            .before(sync_view_cameras),
    );
    // Register main-world capture system to request render-thread teardown.
    app.add_systems(Update, capture_window_surfaces_system);

    // Initialize the window-surface token channel and store sender/receiver
    // into the global OnceLock so systems can use them. Do this before
    // borrowing `app` mutably for the render sub-app to avoid double-borrows.
    let (s, r) = unbounded::<Box<dyn Any + Send>>();
    let _ = WINDOW_SURFACE_SENDER.get_or_init(|| Mutex::new(s));
    let _ = WINDOW_SURFACE_RECEIVER.get_or_init(|| Mutex::new(r));

    let render_app = app.sub_app_mut(RenderApp);
    // Make the staging resource available to the render world where the
    // `gpu_readback_render_system` runs. The main `app.init_resource` call
    // created it in the main world; the render sub-app needs its own instance.
    render_app.init_resource::<ReadbackStaging>();
    // Make the pending-render-drops resource available in both worlds so the
    // main app can queue handles and the render app can consume/remove them
    // from its `Assets<Image>` on the render thread.
    render_app.init_resource::<PendingRenderAssetDrops>();
    render_app.init_resource::<GlobalFrameCount>();
    render_app.init_resource::<PendingWindowGeometry>();
    render_app.init_resource::<ResolutionSetMode>();
    render_app.init_resource::<DelayedAttachmentDrops>();
    render_app.init_resource::<StashedWindowSurfaces>();
    render_app.init_resource::<StashedPendingRenderAssetDrops>();
    render_app.init_resource::<StashedDelayedAttachmentDrops>();
    render_app.init_resource::<StashedReadbackStaging>();
    render_app.init_resource::<StashedAssetsImage>();
    render_app.init_resource::<StashedRenderAssetsGpuImage>();
    render_app.init_resource::<CompletedDropIds>();
    #[derive(Resource, Default)]
    struct RenderGraphDumpDone(pub bool);
    render_app.add_systems(
        bevy::render::Render,
        render_frame_tick_system.in_set(RenderSystems::Cleanup),
    );

    // Debug/probe logging for render scheduling and attachments can be very
    // noisy. Enable explicitly when needed.
    let render_probes_enabled = std::env::var("MCBAISE_RENDER_PROBES").as_deref().ok() == Some("1");

    // Debug helper: print the `RenderGraph` debug dump once on the render
    // thread so we can discover the canonical node names to instrument.
    fn render_graph_dump_system(
        rg: Option<Res<bevy::render::render_graph::RenderGraph>>,
        mut done: Local<bool>,
    ) {
        if *done {
            return;
        }
        if let Some(rg) = rg {
            eprintln!("native(render): RenderGraph dump:\n{:?}", &*rg);
            *done = true;
        }
    }
    if render_probes_enabled {
        render_app.init_resource::<RenderGraphDumpDone>();
        render_app.add_systems(
            bevy::render::Render,
            render_graph_dump_system.in_set(RenderSystems::Prepare),
        );
    }

    // Instrumentation nodes: insert lightweight SystemNodes into the
    // RenderGraph around the main opaque 3D pass so we can log entry/exit
    // timings and resource snapshots for that specific node.
    fn render_wrapper_before_system(frame: Res<GlobalFrameCount>) {
        eprintln!("native(render): wrapper-before main_opaque_pass_3d_node frame={}", frame.0);
    }
    fn render_wrapper_after_system(frame: Res<GlobalFrameCount>) {
        eprintln!("native(render): wrapper-after main_opaque_pass_3d_node frame={}", frame.0);
    }
    // Probe: approximate a swapchain/surface acquire point. We increment the
    // ACQUIRE_SEQ_COUNTER here so downstream submits (including Bevy's
    // internal submit and our explicit submits) can be correlated with the
    // acquire that started the frame. This is best-effort instrumentation
    // — Bevy performs actual swapchain acquires inside its renderer.
    fn render_acquire_probe_system() {
        let acq = ACQUIRE_SEQ_COUNTER.fetch_add(1, Ordering::SeqCst) + 1;
        eprintln!("native(render): acquire probe incremented acquire_seq={}", acq);
    }
    // The RenderGraph internals and label types vary between Bevy versions;
    // attempting to create `SystemNode`s with string labels is not portable
    // across versions and caused build failures. Instead, register lightweight
    // wrapper logging systems in the render schedule so they execute on the
    // render thread and still provide timing information for investigation.
    if render_probes_enabled {
        render_app.add_systems(
            bevy::render::Render,
            (
                render_wrapper_before_system.in_set(RenderSystems::Prepare),
                render_acquire_probe_system.in_set(RenderSystems::Prepare),
                render_wrapper_after_system.in_set(RenderSystems::Cleanup),
            ),
        );
    }

    // After the render graph runs and submits, register a probe callback
    // that will log when the submitted work completes. This captures the
    // current global submit seq so we can correlate encoder/submit timing
    // with surface teardown logs.
    fn render_graph_submit_probe_system(render_queue: Res<bevy::render::renderer::RenderQueue>) {
        // Assign a submit sequence id for this probe so it appears in the
        // same global timeline as our explicit submits.
        let seq = SUBMIT_SEQ_COUNTER.fetch_add(1, Ordering::SeqCst) + 1;
        let acq = ACQUIRE_SEQ_COUNTER.load(Ordering::SeqCst);
        eprintln!("native(render): post-graph submit probe registered seq={} acquire_seq={}", seq, acq);
        render_queue.on_submitted_work_done(move || {
            eprintln!("native(render): post-graph submit probe seq={} acquire_seq={} on_submitted_work_done fired", seq, acq);
            // Notify any waiters that registered interest in this acquire
            if let Some(w) = ACQUIRE_WAITS.get() {
                if let Ok(mut map) = w.lock() {
                    if let Some(waiters) = map.remove(&acq) {
                        for shared in waiters {
                            if let Some(tx) = shared.lock().unwrap().take() {
                                let _ = tx.send(());
                            }
                        }
                    }
                }
            }
        });
    }
    // list of completed drop ids so finalizers can consult them without
    // relying on atomics or cross-thread signalling.
    fn render_async_completion_pump_system(
        mut task_pool: TaskPool<'_, bevy::asset::AssetId<Image>>,
        mut completed: ResMut<CompletedDropIds>,
    ) {
        use std::task::Poll;
        for status in task_pool.iter_poll() {
            if let Poll::Ready(id) = status {
                eprintln!("native(render): async completion observed for id={:?}", id);
                completed.0.push(id);
            }
        }
    }

    // Register render-world receiver to run in Cleanup so it executes on the
    // render thread prior to resource drops.
    if render_probes_enabled {
        render_app.add_systems(
            bevy::render::Render,
            (
                render_graph_submit_probe_system.in_set(RenderSystems::Cleanup),
                render_receive_surfaces_system.in_set(RenderSystems::Cleanup),
            ),
        );
    } else {
        render_app.add_systems(
            bevy::render::Render,
            render_receive_surfaces_system.in_set(RenderSystems::Cleanup),
        );
    }

    // Render-thread: apply any pending geometry change on the render thread
    // instead of the main thread. This performs the `Window` resolution set
    // from the render world so the surface reconfigure/unconfigure happens
    // on the same thread that owns GPU resources and avoids cross-thread
    // races observed on some Vulkan drivers.
    fn render_apply_pending_geometry_system(
        mut pending_geom: ResMut<PendingWindowGeometry>,
        mut windows: Query<&mut Window, With<PrimaryWindow>>,
        // If available, attempt to update the render-extracted windows map.
        mut extracted_windows: Option<ResMut<bevy::render::view::window::ExtractedWindows>>,
        render_device: Res<bevy::render::renderer::RenderDevice>,
        frame: Res<GlobalFrameCount>,
        mode: Res<ResolutionSetMode>,
    ) {
        // This was an experimental mitigation to force surface reconfigure on
        // the render thread. In practice it can desynchronize the render
        // world's swapchain/surface size from the main world's window/camera
        // state (viewport sizes, multiview splits), which can trigger wgpu
        // validation errors like "Encoder is invalid".
        //
        // Keep it available for experiments, but default it off.
        let enable = std::env::var("MCBAISE_RENDER_THREAD_GEOM_APPLY")
            .as_deref()
            .unwrap_or("0")
            == "1";
        if !enable {
            return;
        }

        // Only act when a pending geometry change has been scheduled by
        // the main world (we set `pending = true` there to request the
        // render thread to actually apply the change).
        if !pending_geom.pending {
            return;
        }

        // Apply immediately during Render `Prepare`.
        // This runs before the render graph records command buffers, so we
        // should not be "mid-recording" here. Delaying by one frame can
        // temporarily desync the configured surface size from extracted view
        // sizes, which has been observed to trigger wgpu validation errors.
        pending_geom.scheduled_frame = Some(frame.0);

        // Conservative drain before mutating surface-backed state.
        wait_for_device_idle_strong(&*render_device);

        // Apply on the render thread if a Window is available in the
        // render world. Bevy extracts `Window` into the render app, so
        // this usually succeeds.
            if let Ok(mut window) = windows.single_mut() {
            let curr_w = window.physical_width();
            let curr_h = window.physical_height();
            let mut w = pending_geom.target_w;
            let mut h = pending_geom.target_h;
            if w == 0 { w = curr_w; }
            if h == 0 { h = curr_h; }
            eprintln!("native(render): render-thread applying pending geometry -> {}x{} (was {}x{})", w, h, curr_w, curr_h);
            if *mode == ResolutionSetMode::Physical {
                window.resolution.set_physical_resolution(w, h);
            } else {
                window.resolution.set(w as f32, h as f32);
            }
            eprintln!("native(render): render-thread post-set window physical size = {}x{} (requested {}x{})", window.physical_width(), window.physical_height(), w, h);
            pending_geom.pending = false;
            pending_geom.scheduled_frame = None;
            // Clear any handshake ids (already cleared on main) to be safe.
            // Note: `request_id` and `pending_flag` may not exist in this
            // build's `PendingWindowGeometry` shape; only clear if present.
            // (No-op if fields absent.)
        } else if let Some(mut extracted) = extracted_windows {
            // Attempt to apply the requested geometry directly into the
            // render-extracted windows map. This ensures the surface
            // reconfigure occurs on the render thread even when `Window`
            // is not available as a resource in this world.
            let curr = extracted.iter().next().map(|(_, w)| (w.physical_width, w.physical_height));
            let mut w = pending_geom.target_w;
            let mut h = pending_geom.target_h;
            if w == 0 {
                if let Some((cw, _ch)) = curr { w = cw; }
            }
            if h == 0 {
                if let Some((_cw, ch)) = curr { h = ch; }
            }
            let mut applied_any = false;
            for (_ent, ex_win) in extracted.iter_mut() {
                // Update the extracted window physical dimensions where available.
                ex_win.physical_width = w;
                ex_win.physical_height = h;
                applied_any = true;
            }
            if applied_any {
                eprintln!("native(render): applied pending geometry into ExtractedWindows -> {}x{}", w, h);
                pending_geom.pending = false;
                pending_geom.scheduled_frame = None;
            } else {
                eprintln!("native(render): ExtractedWindows present but contained no entries to apply geometry");
            }
        } else {
            eprintln!("native(render): no Window or ExtractedWindows resource in render world to apply pending geometry");
        }
    }

    // Run the render-thread apply during the Render Prepare stage so that
    // the render world has extracted `Window` and other view state available
    // before the render graph executes. Place it after the attachment
    // snapshot so we have instrumentation prior to applying geometry.
    render_app.add_systems(
        bevy::render::Render,
        render_apply_pending_geometry_system
            .in_set(RenderSystems::Prepare)
            .before(render_graph_attachment_snapshot_system),
    );

    // Diagnostic probe: log which window/surface-related resources exist
    // in the render world during Prepare. This helps determine whether the
    // render world exposes `Window`, `WindowSurfaces`, or `ExtractedWindows`.
    fn render_prepare_resource_probe_system(
        ws_res: Option<Res<WindowSurfaces>>,
        // Use a fully-qualified path to the ExtractedWindows type. If the
        // type isn't available in this Bevy build the `Option<Res<...>>`
        // will simply be `None` and logging will reflect that.
        extracted: Option<Res<bevy::render::view::window::ExtractedWindows>>,
    ) {
        let mut parts: Vec<&'static str> = Vec::new();
        if ws_res.is_some() { parts.push("WindowSurfaces"); }
        if extracted.is_some() { parts.push("ExtractedWindows"); }
        if parts.is_empty() {
            eprintln!("native(render): Prepare probe - no window/surface resources present");
        } else {
            eprintln!("native(render): Prepare probe - found: {}", parts.join(", "));
        }
    }

    if render_probes_enabled {
        render_app.add_systems(
            bevy::render::Render,
            render_prepare_resource_probe_system
                .in_set(RenderSystems::Prepare)
                .after(render_apply_pending_geometry_system),
        );
    }

    // Instrumentation: capture a snapshot of attachments and pending drops
    // at the start of each render frame. This helps correlate which images
    // are present in the render world immediately before the render graph
    // executes with downstream wgpu destroy logs.
    fn render_graph_attachment_snapshot_system(
        images: Option<Res<Assets<Image>>>,
        gpu_images: Option<Res<bevy::render::render_asset::RenderAssets<GpuImage>>>,
        pending: Option<Res<PendingRenderAssetDrops>>,
        frame: Res<GlobalFrameCount>,
        render_device: Option<Res<bevy::render::renderer::RenderDevice>>,
        render_queue: Option<Res<bevy::render::renderer::RenderQueue>>,
    ) {
        eprintln!("native(render): render attachment snapshot frame={}", frame.0);
        if let Some(imgs) = images.as_ref() {
            eprintln!("native(render): Assets<Image> count={}", imgs.len());
            for (h, img) in imgs.iter().take(48) {
                eprintln!(
                    "native(render): Image handle={:?} size={:?} label={:?}",
                    h, img.texture_descriptor.size, img.texture_descriptor.label
                );
            }
        }
        if let Some(gimgs) = gpu_images {
            let mut seen = 0usize;
            for (id, g) in gimgs.iter() {
                if seen < 48 {
                    let name = images.as_ref().and_then(|imgs| {
                        imgs.get(id.clone()).and_then(|img| img.texture_descriptor.label.as_ref().map(|s| s.to_string()))
                    });
                    if let Some(n) = name {
                        eprintln!(
                            "native(render): GpuImage id={:?} name={} size={:?}",
                            id, n, g.size
                        );
                    } else {
                        eprintln!(
                            "native(render): GpuImage id={:?} size={:?}",
                            id, g.size
                        );
                    }
                }
                seen += 1;
            }
            eprintln!("native(render): RenderAssets<GpuImage> sample_count={}", seen.min(48));
        }
        if let Some(p) = pending {
            eprintln!(
                "native(render): PendingRenderAssetDrops scheduled_frame={:?} count={}",
                p.scheduled_frame,
                p.images.len()
            );
            for h in p.images.iter().take(48) {
                eprintln!("native(render): pending image handle id={}", h.id());
            }

            // Submit a no-op and wait immediately after logging pending drops.
            // This creates a clear GPU submission boundary right where we
            // observe the pending drops so we can better ensure the driver
            // has completed prior work before any destructive operations.
            if let (Some(dev), Some(queue)) = (render_device, render_queue) {
                submit_noop_and_wait(&*dev, &*queue);
            }
        }
    }
    // Conditionally register GPU readback systems; set `MCBAISE_DISABLE_READBACK=1` to skip.
    if std::env::var("MCBAISE_DISABLE_READBACK").as_deref().ok() != Some("1") {
        render_app.add_systems(
            bevy::render::Render,
            gpu_readback_render_system.in_set(RenderSystems::Cleanup),
        );
        render_app.add_systems(
            bevy::render::Render,
            gpu_readback_poll_system.in_set(RenderSystems::Cleanup),
        );
    } else {
        eprintln!("diagnostic: MCBAISE_DISABLE_READBACK=1 - skipping gpu readback systems");
    }

    // Render-side monitor: when the main app requests a teardown/restart we
    // arm a shared flag. The render-subapp executes this monitor in its
    // Cleanup stage and sets the flag after performing a blocking device
    // poll, ensuring the render thread has drained in-flight work before
    // the main world destructively drops GPU-backed resources.
    // Register the instrumentation snapshot early in the render pipeline so
    // we observe the attachment set just prior to graph execution.
    if render_probes_enabled {
        render_app.add_systems(
            bevy::render::Render,
            render_graph_attachment_snapshot_system.in_set(RenderSystems::Prepare),
        );
    }

    // Post-graph instrumentation: capture a snapshot immediately after
    // the render graph has executed so we can correlate which attachments
    // remain live when the graph finishes (helps identify late-held
    // SurfaceTexture references or lingering encoder state).
    fn render_graph_post_frame_snapshot_system(
        images: Option<Res<Assets<Image>>>,
        gpu_images: Option<Res<bevy::render::render_asset::RenderAssets<GpuImage>>>,
        frame: Res<GlobalFrameCount>,
        render_queue: Option<Res<bevy::render::renderer::RenderQueue>>,
    ) {
        eprintln!("native(render): post-graph attachment snapshot frame={}", frame.0);
        if let Some(imgs) = images.as_ref() {
            eprintln!("native(render): [post] Assets<Image> count={}", imgs.len());
            for (h, img) in imgs.iter().take(48) {
                eprintln!(
                    "native(render): [post] Image handle={:?} size={:?} label={:?}",
                    h, img.texture_descriptor.size, img.texture_descriptor.label
                );
            }
        }
        if let Some(gimgs) = gpu_images {
            let mut seen = 0usize;
            for (id, g) in gimgs.iter() {
                if seen < 48 {
                    eprintln!(
                        "native(render): [post] GpuImage id={:?} size={:?}",
                        id, g.size
                    );
                }
                seen += 1;
            }
            eprintln!("native(render): [post] RenderAssets<GpuImage> sample_count={}", seen.min(48));
        }
        if render_queue.is_some() {
            eprintln!("native(render): [post] RenderQueue present");
        }
    }

    if render_probes_enabled {
        render_app.add_systems(
            bevy::render::Render,
            render_graph_post_frame_snapshot_system
                .in_set(RenderSystems::Cleanup)
                .after(render_receive_surfaces_system),
        );
    }

    // Render-side: resize our size-dependent Images (readback / preview / capture)
    // when the main world has requested a window geometry change. Running this
    // on the render thread ensures texture recreation happens where GPU
    // allocations are owned and avoids cross-thread destruction races.
    fn render_resize_images_system(
        pending_geom: Option<Res<PendingWindowGeometry>>,
        readback: Option<Res<GpuReadbackImage>>,
        mut images: Option<ResMut<Assets<Image>>>,
        render_device: Res<bevy::render::renderer::RenderDevice>,
        mut last: Local<Option<(u32, u32)>>,
    ) {
        let Some(pending) = pending_geom else { return; };
        // Only act when the UI has settled and produced a target size.
        if pending.target_w == 0 || pending.target_h == 0 { return; }

        let target_w = pending.target_w;
        let target_h = pending.target_h;

        // Avoid repeated resizes when size hasn't changed.
        if let Some((lw, lh)) = *last {
            if lw == target_w && lh == target_h {
                return;
            }
        }

        // Give the device a conservative drain so drivers have time to finish
        // any work that may reference older attachments before we recreate.
        wait_for_device_idle_strong(&*render_device);

        if let Some(mut imgs) = images {
            // Resize the explicit readback image if present.
            if let Some(rb) = readback.as_ref() {
                if let Some(img) = imgs.get_mut(&rb.0) {
                    let cur = img.texture_descriptor.size;
                    if cur.width != target_w || cur.height != target_h {
                        eprintln!("native(render): resizing readback image {}x{} -> {}x{}", cur.width, cur.height, target_w, target_h);
                        img.resize(Extent3d { width: target_w, height: target_h, depth_or_array_layers: 1 });
                    }
                }
            }

            // Heuristic: resize any image whose label indicates it's a capture/preview/readback.
            for (h, img) in imgs.iter_mut() {
                if let Some(lbl) = img.texture_descriptor.label {
                    let lname = lbl.to_ascii_lowercase();
                    if lname.contains("readback") || lname.contains("preview") || lname.contains("capture") {
                        let cur = img.texture_descriptor.size;
                        if cur.width != target_w || cur.height != target_h {
                            eprintln!("native(render): resizing image label={:?} handle={:?} {}x{} -> {}x{}", lbl, h, cur.width, cur.height, target_w, target_h);
                            img.resize(Extent3d { width: target_w, height: target_h, depth_or_array_layers: 1 });
                        }
                    }
                }
            }
        }

        *last = Some((target_w, target_h));
    }

    render_app.add_systems(
        bevy::render::Render,
        render_resize_images_system.in_set(RenderSystems::Prepare),
    );

    render_app.add_systems(
        bevy::render::Render,
        // First consume any pending asset-drops on the render thread, then
        // run the existing teardown monitor which polls and signals readiness.
        (render_asset_drop_consume_system.before(render_teardown_monitor_system)).in_set(RenderSystems::Cleanup),
    );

    // Finalizer: actually remove queued attachments from `Assets<Image>` once
    // the conservative scheduled frame has been reached and the device has
    // been polled/drained. This performs the destructive removal on the
    // render thread where GPU allocations are owned.
    fn render_attachment_final_drop_system(
        mut images: Option<ResMut<Assets<Image>>>,
        mut pending_opt: Option<ResMut<PendingRenderAssetDrops>>,
        mut delayed_opt: Option<ResMut<DelayedAttachmentDrops>>,
        render_device: Res<bevy::render::renderer::RenderDevice>,
        frame: Res<GlobalFrameCount>,
        completed: Option<Res<CompletedDropIds>>,
    ) {
        // If there's no pending drops table, nothing to finalize.
        let mut pending = match pending_opt {
            Some(p) => p,
            None => return,
        };
        // Only act when a scheduled_frame is present.
        let sched = match pending.scheduled_frame {
            Some(s) => s,
            None => return,
        };
        if frame.0 < sched {
            return;
        }

        // Strong poll before destructive operations.
        let _ = render_device.poll(wgpu::PollType::Wait);
        sleep(Duration::from_millis(8));
        let _ = render_device.poll(wgpu::PollType::Wait);

        let mut images = match images {
            Some(i) => i,
            None => return,
        };

        let delayed_count = delayed_opt.as_ref().map(|d| d.entries.len()).unwrap_or(0);
        eprintln!(
            "native(render): finalizing {} queued attachment drops at frame={}",
            delayed_count,
            frame.0
        );

        // Drain and re-queue entries that are not yet ready. For entries
        // that are allowed by frame and whose submission-flag (if any) is
        // set, perform the destructive removal.
        let mut remaining: Vec<(Handle<Image>, u64, Option<Arc<AtomicBool>>)> = Vec::new();
        if let Some(mut delayed) = delayed_opt {
            for (handle, allowed_frame, flag_opt) in delayed.entries.drain(..) {
                if frame.0 < allowed_frame {
                    remaining.push((handle, allowed_frame, flag_opt));
                    continue;
                }
                // If a completion flag was registered, prefer that. Otherwise,
                // consult the `CompletedDropIds` list populated by the async
                // pump system. If neither indicates completion, keep entry.
                let completed_present = completed.as_ref().map_or(false, |c| c.0.contains(&handle.id()));
                if let Some(flag) = flag_opt {
                    if !flag.load(Ordering::SeqCst) && !completed_present {
                        // submission not yet completed; keep for later
                        remaining.push((handle, allowed_frame, Some(flag)));
                        continue;
                    }
                } else if !completed_present {
                    // No atomic flag recorded; wait for async completion marker.
                    remaining.push((handle, allowed_frame, None));
                    continue;
                }
                let present = images.get(&handle).is_some();
                eprintln!(
                    "native(render): final drop id={} present_in_render_assets={}",
                    handle.id(),
                    present
                );
                // If the image is still present in the render `Assets<Image>`, it
                // may be backed by swapchain/surface resources. Perform an extra
                // conservative device drain here to reduce the chance of destroying
                // surface-acquire semaphores while still referenced by in-flight
                // GPU work (wgpu-hal/vulkan drivers have been observed to panic
                // in that case). This is targeted and only runs for present
                // attachments to avoid unnecessary stalls.
                if present {
                    eprintln!("native(render): present attachment - performing extra device idle drain before removal");
                    wait_for_device_idle_strong(&*render_device);
                }
                images.remove(&handle);
            }
            delayed.entries = remaining;
        }

        pending.scheduled_frame = None;
        pending.submit_frame_hint = None;
    }

    render_app.add_systems(
        bevy::render::Render,
        render_async_completion_pump_system.in_set(RenderSystems::Cleanup).before(render_attachment_final_drop_system),
    );

    render_app.add_systems(
        bevy::render::Render,
        render_attachment_final_drop_system.in_set(RenderSystems::Cleanup).after(render_asset_drop_consume_system),
    );

    // Finalizer for stashed WindowSurfaces: only drop stashed surfaces once
    // both the scheduled frame has been reached and the on_submitted_work_done
    // callback (if registered) has fired. This provides a strong multi-factor
    // gate to avoid destroying semaphore-backed surface resources while the
    // driver still references them.
    fn render_stashed_window_surfaces_finalizer(
        mut stashed_res: Option<ResMut<StashedWindowSurfaces>>,
        frame: Res<GlobalFrameCount>,
        render_device: Option<Res<bevy::render::renderer::RenderDevice>>,
        render_queue: Option<Res<bevy::render::renderer::RenderQueue>>,
        completed: Option<Res<CompletedDropIds>>,
    ) {
        let mut stashed = match stashed_res {
            Some(s) => s,
            None => return,
        };

        if stashed.entries.is_empty() { return; }

        eprintln!("native(render): stashed WindowSurfaces finalizer running frame={} entries={}", frame.0, stashed.entries.len());

        let mut remaining: Vec<(u64, WindowSurfaces, u64, Option<Arc<AtomicBool>>)> = Vec::new();
        for (id, surfaces, allowed_frame, flag_opt) in stashed.entries.drain(..) {
            if frame.0 < allowed_frame {
                remaining.push((id, surfaces, allowed_frame, flag_opt));
                continue;
            }
            if let Some(flag) = flag_opt {
                if !flag.load(Ordering::SeqCst) {
                    // on_submitted_work_done hasn't fired yet; keep it.
                    remaining.push((id, surfaces, allowed_frame, Some(flag)));
                    continue;
                }
            }

            eprintln!("native(render): finalizing stashed WindowSurfaces id={} at frame={}", id, frame.0);
            // As a last defense, perform a strong device idle before drop if
            // we have a RenderDevice available.
            if let Some(dev) = render_device.as_ref() {
                wait_for_device_idle_strong(&**dev);
            }
            // Conservative per-stash noop-before-drop: if we have access to
            // the render queue, emit an explicit noop submit+wait for this
            // particular stash entry immediately before dropping it. This
            // creates a clear submission/fence boundary that reduces the
            // chance of driver-held acquire-semaphores still referencing the
            // surface when we unconfigure/drop it.
            if let (Some(dev), Some(queue)) = (render_device.as_ref(), render_queue.as_ref()) {
                eprintln!("native(render): issuing per-stash noop-before-drop id={} acquire_snapshot={}", id, ACQUIRE_SEQ_COUNTER.load(Ordering::SeqCst));
                submit_noop_and_wait(&**dev, &**queue);
            }
            // dropping `surfaces` here can unconfigure/destroy backing
            // surface resources (wgpu Surface / swapchain) which some
            // Vulkan drivers will panic on if acquire semaphores are
            // still referenced. As a conservative development-time
            // workaround we intentionally leak the stashed `WindowSurfaces`
            // instead of dropping them immediately. This prevents the
            // panic while preserving the ability to observe and debug
            // which submits remain live. If you prefer the original
            // behavior, replace `mem::forget` with `drop`.
            eprintln!(
                "native(render): skipping drop (leak) of stashed WindowSurfaces id={} to avoid semaphore race",
                id
            );
            std::mem::forget(surfaces);
        }

        stashed.entries = remaining;
    }

    render_app.add_systems(
        bevy::render::Render,
        render_stashed_window_surfaces_finalizer.in_set(RenderSystems::Cleanup).after(render_attachment_final_drop_system),
    );

    // Note: SurfaceData is private to Bevy; we avoid trying to stash or
    // manually drop it from application code. See comments where we
    // attempted this change earlier; instead we rely on increased gating
    // and noop-submits to reduce race conditions on drivers.

    // Aggressive pre-cleanup wait: run at the start of Cleanup to give the
    // device an extra-long drain before any cleanup or surface/resource
    // destruction occurs. This is a conservative mitigation for drivers
    // that report `SurfaceAcquireSemaphores` still in use when surfaces are
    // reconfigured/dropped while GPU work is outstanding.
    fn render_cleanup_blocking_wait_system(
        render_device: Res<bevy::render::renderer::RenderDevice>,
        render_queue: Option<Res<bevy::render::renderer::RenderQueue>>,
        frame: Res<GlobalFrameCount>,
        pending: Option<Res<PendingRenderAssetDrops>>,
        delayed: Option<Res<DelayedAttachmentDrops>>,
        stashed: Option<Res<StashedWindowSurfaces>>,
    ) {
        // Only perform the aggressive blocking drain when we have queued
        // delayed drops or pending render-asset drops. Unconditional strong
        // waits here cause long stalls and make the native app unusable.
        let has_pending = match &pending {
            Some(p) => !p.images.is_empty() || p.scheduled_frame.is_some(),
            None => false,
        };
        let has_delayed = match &delayed {
            Some(d) => !d.entries.is_empty(),
            None => false,
        };
        let has_stashed = match &stashed {
            Some(s) => !s.entries.is_empty(),
            None => false,
        };
        // Also trigger when there are stashed `WindowSurfaces` waiting to be
        // dropped on the render thread. Those are sensitive to acquire
        // semaphore lifetimes and benefit from an extra explicit noop+wait
        // submission boundary in addition to the device poll.
        if !has_pending && !has_delayed && !has_stashed {
            return;
        }
        eprintln!("native(render): pre-cleanup device idle wait frame={}", frame.0);
        wait_for_device_idle_strong(&*render_device);
        // Also emit an explicit noop submission and wait if we have access
        // to the render queue. This creates a clear fence boundary at the
        // driver level which some Vulkan drivers require before surface
        // reconfigure/unconfigure.
        if let Some(queue) = render_queue {
            submit_noop_and_wait(&*render_device, &*queue);
        }
    }

    render_app.add_systems(
        bevy::render::Render,
        render_cleanup_blocking_wait_system.in_set(RenderSystems::Cleanup).before(render_asset_drop_consume_system),
    );

    // Submit a no-op command buffer and wait for its completion on the
    // render thread. This produces an explicit submission/fence boundary
    // so drivers that otherwise keep acquire semaphores alive until a
    // later submit will be forced to complete prior work before any
    // subsequent surface unconfigure/drop occurs.
    fn render_submit_noop_and_wait_system(
        render_device: Res<bevy::render::renderer::RenderDevice>,
        render_queue: Res<bevy::render::renderer::RenderQueue>,
    ) {
        submit_noop_and_wait(&*render_device, &*render_queue);
    }

    render_app.add_systems(
        bevy::render::Render,
        render_submit_noop_and_wait_system.in_set(RenderSystems::Cleanup).after(render_attachment_final_drop_system),
    );


    #[cfg(not(target_arch = "wasm32"))]
    app.add_systems(Update, update_overlay_ui_camera);
    #[cfg(not(target_arch = "wasm32"))]
    app.add_systems(Update, gate_overlay_ui_camera_system.after(update_overlay_ui_camera));
    #[cfg(not(target_arch = "wasm32"))]
    app.add_systems(
        Update,
        gate_main_cameras_during_geometry_system.after(gate_overlay_ui_camera_system),
    );

    // Execution-phase systems: perform actual camera/scene despawn+recreate
    // once the render cleanup monitor has signalled readiness.
    app.add_systems(Update, restart_cameras_execute_system);
    app.add_systems(Update, restart_scene_execute_system);

    #[cfg(target_arch = "wasm32")]
    app.add_systems(Update, apply_js_input);

    #[cfg(target_arch = "wasm32")]
    app.add_systems(Update, update_render_scale_resource);
    #[cfg(target_arch = "wasm32")]
    app.add_systems(
        Update,
        apply_render_scale_to_window.after(update_render_scale_resource),
    );

    #[cfg(not(target_arch = "wasm32"))]
    app.add_systems(Update, (advance_time_native, native_controls, gif_auto_capture_system));

    #[cfg(all(not(target_arch = "wasm32"), feature = "native-youtube"))]
    {
        app.add_systems(
            Update,
            (native_youtube_shutdown_on_exit, native_youtube_sync),
        );
        app.add_systems(
            Update,
            native_youtube_align_browser_window.after(native_youtube_sync),
        );
        init_native_youtube(&mut app);
    }

    #[cfg(all(windows, not(target_arch = "wasm32"), feature = "native-mpv"))]
    {
        app.add_systems(Update, (native_mpv_shutdown_on_exit, native_mpv_sync));
        init_native_mpv(&mut app);
    }

    app.run();
}

#[cfg(all(not(target_arch = "wasm32"), feature = "native-youtube"))]
fn init_native_youtube(app: &mut App) {
    let enabled = std::env::var("MCBAISE_NATIVE_YOUTUBE")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if !enabled {
        return;
    }

    let webdriver_url = std::env::var("MCBAISE_WEBDRIVER_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "http://localhost:9515".to_string());

    let launch_webdriver = std::env::var("MCBAISE_LAUNCH_WEBDRIVER")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let chrome_user_data_dir = std::env::var("MCBAISE_CHROME_USER_DATA_DIR")
        .ok()
        .filter(|s| !s.trim().is_empty());

    let chrome_profile_dir = std::env::var("MCBAISE_CHROME_PROFILE_DIR")
        .ok()
        .filter(|s| !s.trim().is_empty());

    eprintln!(
        "[native-youtube] enabled. webdriver={webdriver_url} (auto-launch={launch_webdriver})"
    );

    app.insert_resource(NativeYoutubeConfig {
        webdriver_url: webdriver_url.clone(),
        launch_webdriver,
        chrome_user_data_dir: chrome_user_data_dir.clone(),
        chrome_profile_dir: chrome_profile_dir.clone(),
    });

    app.init_resource::<NativeYoutubeWindowLayout>();

    if let Some(mut playback) = app.world_mut().get_resource_mut::<Playback>() {
        // YouTube is authoritative.
        playback.speed = 1.0;
    }

    let (tx, rx, join) = native_youtube::spawn(
        VIDEO_ID,
        &webdriver_url,
        launch_webdriver,
        chrome_user_data_dir,
        chrome_profile_dir,
    );
    app.insert_resource(NativeYoutubeSync {
        enabled: true,
        tx,
        rx: std::sync::Mutex::new(rx),
        join: std::sync::Mutex::new(Some(join)),
        last_error: std::sync::Mutex::new(None),

        has_remote: false,
        last_remote_time_sec: 0.0,
        last_remote_playing: false,
        sample_age_sec: 0.0,
        remote_age_sec: 0.0,
        interp_time_sec: 0.0,

        in_ad: false,
        ad_label: None,
        last_good_time_sec: 0.0,
        pending_seek_after_ad: false,
        ad_nudge_cooldown_sec: 0.0,

        heal_cooldown_sec: 0.0,
    });
}

#[cfg(all(not(target_arch = "wasm32"), feature = "native-youtube"))]
fn native_youtube_align_browser_window(
    sync: Option<Res<NativeYoutubeSync>>,
    layout: Option<ResMut<NativeYoutubeWindowLayout>>,
    mut primary: Query<(Entity, &mut Window), With<PrimaryWindow>>,
    winit_windows: Option<NonSend<bevy::winit::WinitWindows>>,
) {
    let Some(sync) = sync else {
        return;
    };
    if !sync.enabled {
        return;
    }

    let Some(mut layout) = layout else {
        return;
    };

    let Some((primary_entity, mut bevy_window)) = primary.iter_mut().next() else {
        return;
    };

    // Best-effort: keep Bevy on top for a short window after we move Chrome.
    #[cfg(windows)]
    {
        use bevy::window::WindowLevel;
        use raw_window_handle::{HasWindowHandle, RawWindowHandle};
        use windows_sys::Win32::Foundation::HWND;
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            HWND_TOPMOST, SW_RESTORE, SWP_NOMOVE, SWP_NOSIZE, SWP_SHOWWINDOW, SetWindowPos,
            ShowWindow,
        };

        // Once we've positioned Chrome, keep Bevy as topmost so it doesn't fall behind.
        if layout.force_topmost {
            bevy_window.window_level = WindowLevel::AlwaysOnTop;
        }

        // Manual HWND manipulation while rendering can trigger validation errors
        // in some drivers. Stick to Bevy's WindowLevel for stability.
    }

    if layout.applied {
        return;
    }

    // Prefer true OS window geometry via winit (includes current monitor coords).
    if let Some(winit_windows) = winit_windows {
        let Some(w) = winit_windows.get_window(primary_entity) else {
            return;
        };

        let Ok(pos) = w.outer_position() else {
            return;
        };
        let size = w.outer_size();

        let x_i32 = pos.x.saturating_add(size.width as i32);
        let y_i32 = pos.y;
        let width_u32 = size.width.max(1);
        let height_u32 = size.height.max(1);

        let x = x_i32.max(0) as u32;
        let y = y_i32.max(0) as u32;

        let _ = sync.tx.send(native_youtube::Command::SetWindowRect {
            x,
            y,
            width: width_u32,
            height: height_u32,
        });
        layout.applied = true;
        layout.force_topmost = true;
        return;
    }

    // Fallback: we can't get the OS position, but we can at least match the size.
    // Place at (width, 0) so it ends up "to the right" on the primary monitor.
    let width_u32 = bevy_window.resolution.physical_width().max(1);
    let height_u32 = bevy_window.resolution.physical_height().max(1);
    let _ = sync.tx.send(native_youtube::Command::SetWindowRect {
        x: width_u32,
        y: 0,
        width: width_u32,
        height: height_u32,
    });
    layout.applied = true;
    layout.force_topmost = true;
}

#[cfg(all(windows, not(target_arch = "wasm32"), feature = "native-mpv"))]
fn init_native_mpv(app: &mut App) {
    let enabled = std::env::var("MCBAISE_NATIVE_MPV")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if !enabled {
        return;
    }

    let url = std::env::var("MCBAISE_MPV_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| format!("https://www.youtube.com/watch?v={VIDEO_ID}"));

    let mpv_path = std::env::var("MCBAISE_MPV_PATH")
        .ok()
        .filter(|s| !s.trim().is_empty());

    let mut extra_args = std::env::var("MCBAISE_MPV_EXTRA_ARGS")
        .ok()
        .unwrap_or_default()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let looks_like_youtube = url.contains("youtube.com") || url.contains("youtu.be");

    // Help with YouTube's "sign in to confirm you're not a bot" gate by letting mpv/yt-dlp
    // reuse your browser cookies.
    // Examples:
    // - MCBAISE_MPV_COOKIES_FROM_BROWSER=chrome
    // - MCBAISE_MPV_YTDL_RAW_OPTIONS=cookies-from-browser=chrome
    let mut ytdl_raw_options: Vec<String> = Vec::new();

    let disable_auto_cookies = std::env::var("MCBAISE_MPV_DISABLE_AUTO_COOKIES")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let cookie_file = std::env::var("MCBAISE_MPV_COOKIES_FILE")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .or_else(|| {
            if !disable_auto_cookies && std::path::Path::new("cookies.txt").exists() {
                Some("cookies.txt".to_string())
            } else {
                None
            }
        });

    if let Some(path) = cookie_file.as_ref() {
        ytdl_raw_options.push(format!("cookies={path}"));
    }

    if cookie_file.is_none()
        && let Ok(v) = std::env::var("MCBAISE_MPV_COOKIES_FROM_BROWSER")
    {
        let v = v.trim();
        if !v.is_empty() {
            ytdl_raw_options.push(format!("cookies-from-browser={v}"));
        }
    }
    if let Ok(v) = std::env::var("MCBAISE_MPV_YTDL_RAW_OPTIONS") {
        let v = v.trim();
        if !v.is_empty() {
            // Allow comma-separated key=value pairs.
            ytdl_raw_options.push(v.to_string());
        }
    }

    // Default YouTube extractor behavior: prefer the web client.
    // This helps avoid some iOS/tv client failures and can improve format availability.
    // Opt out with: `MCBAISE_MPV_DISABLE_DEFAULT_YOUTUBE_EXTRACTOR_ARGS=1`
    if looks_like_youtube {
        let disable_default_extractor_args =
            std::env::var("MCBAISE_MPV_DISABLE_DEFAULT_YOUTUBE_EXTRACTOR_ARGS")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);

        if !disable_default_extractor_args {
            let has_extractor_args = ytdl_raw_options
                .iter()
                .any(|s| s.to_ascii_lowercase().contains("extractor-args="));
            if !has_extractor_args {
                ytdl_raw_options.push("extractor-args=youtube:player_client=web".to_string());
            }
        }
    }

    // Fully automated default: if this looks like YouTube and the user didn't specify cookies,
    // try Chrome cookies by default.
    if !disable_auto_cookies {
        let has_any_cookie_opt = ytdl_raw_options.iter().any(|s| {
            let s = s.to_ascii_lowercase();
            s.contains("cookies-from-browser=") || s.contains("cookies=")
        });
        if looks_like_youtube && !has_any_cookie_opt {
            ytdl_raw_options.push("cookies-from-browser=chrome".to_string());
        }
    }

    // If the user didn't specify any format selection, default to something permissive.
    // This avoids ytdl_hook errors like: "Requested format is not available".
    if looks_like_youtube {
        let has_mpv_ytdl_format = extra_args
            .iter()
            .any(|a| a == "--ytdl-format" || a.to_ascii_lowercase().starts_with("--ytdl-format="));
        let has_ytdlp_format = ytdl_raw_options
            .iter()
            .any(|s| s.to_ascii_lowercase().contains("format="));
        if !has_mpv_ytdl_format && !has_ytdlp_format {
            extra_args.push("--ytdl-format=bestvideo+bestaudio/best".to_string());
            println!("[native-mpv] defaulting to --ytdl-format=bestvideo+bestaudio/best");
        }
    }
    if !ytdl_raw_options.is_empty() {
        let joined = ytdl_raw_options
            .into_iter()
            .flat_map(|s| {
                s.split(',')
                    .map(|x| x.trim().to_string())
                    .collect::<Vec<_>>()
            })
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(",");

        if !joined.is_empty() {
            extra_args.push(format!("--ytdl-raw-options={joined}"));
            println!("[native-mpv] ytdl raw options: {joined}");
        }
    }

    println!("[native-mpv] enabled. url={url}");

    app.insert_resource(NativeMpvConfig {
        url: url.clone(),
        mpv_path: mpv_path.clone(),
        extra_args: extra_args.clone(),
    });

    if let Some(mut playback) = app.world_mut().get_resource_mut::<Playback>() {
        // mpv is authoritative.
        playback.speed = 1.0;
    }

    let (tx, rx, join) = native_mpv::spawn(url, mpv_path, extra_args);
    app.insert_resource(NativeMpvSync {
        enabled: true,
        tx,
        rx: std::sync::Mutex::new(rx),
        join: std::sync::Mutex::new(Some(join)),
        last_error: std::sync::Mutex::new(None),

        has_remote: false,
        last_remote_time_sec: 0.0,
        last_remote_playing: false,
        sample_age_sec: 0.0,
        interp_time_sec: 0.0,
    });
}

#[cfg(all(windows, not(target_arch = "wasm32"), feature = "native-mpv"))]
fn native_mpv_sync(
    time: Res<Time>,
    mut playback: ResMut<Playback>,
    mpv: Option<ResMut<NativeMpvSync>>,
) {
    let Some(mut mpv) = mpv else {
        return;
    };
    if !mpv.enabled {
        return;
    }

    let dt = time.delta_secs();
    mpv.sample_age_sec += dt;

    let events = if let Ok(rx) = mpv.rx.lock() {
        rx.try_iter().collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    for evt in events {
        match evt {
            native_mpv::Event::State { time_sec, playing } => {
                mpv.has_remote = true;
                mpv.last_remote_time_sec = time_sec;
                mpv.last_remote_playing = playing;
                mpv.sample_age_sec = 0.0;
            }
            native_mpv::Event::Error(e) => {
                if let Ok(mut slot) = mpv.last_error.lock() {
                    *slot = Some(e);
                }
            }
        }
    }

    if !mpv.has_remote {
        return;
    }

    mpv.interp_time_sec = if mpv.last_remote_playing {
        mpv.last_remote_time_sec + mpv.sample_age_sec
    } else {
        mpv.last_remote_time_sec
    };

    playback.time_sec = mpv.interp_time_sec;
    playback.playing = mpv.last_remote_playing;
}

#[cfg(all(windows, not(target_arch = "wasm32"), feature = "native-mpv"))]
fn native_mpv_shutdown_on_exit(
    app_exit: MessageReader<bevy::app::AppExit>,
    window_close: MessageReader<bevy::window::WindowCloseRequested>,
    mpv: Option<Res<NativeMpvSync>>,
) {
    if app_exit.is_empty() && window_close.is_empty() {
        return;
    }

    if let Some(mpv) = mpv.as_ref() {
        let _ = mpv.tx.send(native_mpv::Command::Shutdown);
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "native-youtube"))]
fn native_youtube_sync(
    time: Res<Time>,
    mut playback: ResMut<Playback>,
    sync: Option<ResMut<NativeYoutubeSync>>,
) {
    let Some(mut sync) = sync else {
        return;
    };
    if !sync.enabled {
        return;
    }

    let dt = time.delta_secs();
    sync.sample_age_sec += dt;
    sync.heal_cooldown_sec = (sync.heal_cooldown_sec - dt).max(0.0);
    sync.ad_nudge_cooldown_sec = (sync.ad_nudge_cooldown_sec - dt).max(0.0);

    let mut disable = false;
    let mut disable_reason: Option<String> = None;
    let mut force_heal = false;
    let mut ad_update: Option<(bool, Option<String>)> = None;

    // Drain events while holding the receiver lock, then update interpolation state.
    let mut last_state: Option<(f32, bool)> = None;
    {
        let Ok(rx) = sync.rx.lock() else {
            return;
        };

        loop {
            match rx.try_recv() {
                Ok(native_youtube::Event::State { time_sec, playing }) => {
                    last_state = Some((time_sec, playing));
                }
                Ok(native_youtube::Event::PlayerErrorOverlay) => {
                    force_heal = true;
                }
                Ok(native_youtube::Event::AdState { playing_ad, label }) => {
                    ad_update = Some((playing_ad, label));
                }
                Ok(native_youtube::Event::Error(msg)) => {
                    println!("[native-youtube] Error: {msg}");
                    // disable = true;
                    // disable_reason = Some(msg);
                    // break;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    disable = true;
                    disable_reason = Some("native youtube sync disconnected".to_string());
                    break;
                }
            }
        }
    }

    if let Some((time_sec, playing)) = last_state {
        let time_sec = time_sec.max(0.0);
        sync.last_remote_time_sec = time_sec;
        sync.last_remote_playing = playing;
        sync.sample_age_sec = 0.0;
        sync.remote_age_sec = 0.0;

        // Track last-known-good time only when not in an ad.
        if !sync.in_ad {
            // Never regress last_good to ~0 due to transient player resets.
            if time_sec > 0.01 {
                sync.last_good_time_sec = time_sec;
            }
        }

        // If we were showing a transient healing message, clear it once we have a valid sample again.
        if let Ok(mut slot) = sync.last_error.lock()
            && slot.as_deref().is_some_and(|s| s.starts_with("healing:"))
        {
            *slot = None;
        }

        if !sync.has_remote {
            sync.has_remote = true;
            sync.interp_time_sec = time_sec;
        }
    }

    if let Some((playing_ad, label)) = ad_update {
        let was_in_ad = sync.in_ad;
        sync.in_ad = playing_ad;
        sync.ad_label = label;

        if playing_ad {
            // While an ad is visible, ensure we keep playing and defer seeking.
            // If we don't have a good time yet, seed it from our current predicted time.
            if sync.last_good_time_sec <= 0.01 {
                sync.last_good_time_sec = playback.time_sec.max(0.0);
            }

            // Only defer seek if we're actually trying to reach content.
            if (playback.playing || force_heal) && sync.last_good_time_sec > 0.01 {
                sync.pending_seek_after_ad = true;
            }

            // Freeze our local animation time at the last known good content time.
            sync.remote_age_sec = 0.0;
            sync.interp_time_sec = sync.last_good_time_sec.max(0.0);
        } else if was_in_ad && !playing_ad {
            // Ad just ended; we'll seek back to last_good_time_sec below.
        }
    }

    if disable {
        if let Some(msg) = disable_reason {
            println!("[native-youtube] {msg}");
            if let Ok(mut slot) = sync.last_error.lock() {
                *slot = Some(msg);
            }
        }
        sync.enabled = false;
        return;
    }

    // Smooth playback between samples by extrapolating from the last remote sample.
    // Also allow the heal path to run even if we haven't received a valid sample yet.
    if sync.has_remote || force_heal {
        // If an ad is showing, keep the browser playing; don't attempt seeks until it ends.
        if sync.in_ad {
            // If we're trying to get through an ad (pending seek / healing), force play.
            if (sync.pending_seek_after_ad || force_heal) && sync.ad_nudge_cooldown_sec <= 0.0 {
                let _ = sync.tx.send(native_youtube::Command::SetPlaying(true));
                sync.ad_nudge_cooldown_sec = 1.0;
            }

            // Do not animate while ads are playing.
            playback.time_sec = sync.last_good_time_sec.max(0.0);
            playback.speed = 1.0;

            if let Ok(mut slot) = sync.last_error.lock() {
                let label = sync.ad_label.as_deref().unwrap_or("ad playing");
                *slot = Some(format!(
                    "healing: ad detected ({label}) — waiting; t≈{:.2}",
                    playback.time_sec
                ));
            }
        }

        // If remote is fresh, treat it as authoritative for play/pause.
        // If remote is stale, keep user's desired state so they can heal.
        if sync.has_remote && sync.sample_age_sec < 0.6 {
            playback.playing = sync.last_remote_playing;
        }

        // Predict time forward when we're trying to play.
        if sync.has_remote && playback.playing {
            sync.remote_age_sec += dt;
        }

        if sync.has_remote && !sync.in_ad {
            let expected_remote_now = sync.last_remote_time_sec
                + if playback.playing {
                    sync.remote_age_sec
                } else {
                    0.0
                };

            if playback.playing {
                sync.interp_time_sec += dt;
            }

            let err = expected_remote_now - sync.interp_time_sec;
            if err.abs() > 0.75 {
                // Big jump => seek/reload; snap.
                sync.interp_time_sec = expected_remote_now;
            } else {
                // Small drift => correct smoothly.
                sync.interp_time_sec += err * 0.35;
            }

            sync.interp_time_sec = sync.interp_time_sec.max(0.0);
            playback.time_sec = sync.interp_time_sec;
            playback.speed = 1.0;
        }

        // Healing: if we're trying to play but remote samples stall, seek to the last known
        // remote position and press play again.
        if force_heal {
            if let Ok(mut slot) = sync.last_error.lock() {
                *slot = Some("healing: YouTube player error overlay (reloading)".to_string());
            }
        } else if playback.playing
            && sync.sample_age_sec > 1.25
            && let Ok(mut slot) = sync.last_error.lock()
        {
            *slot = Some("healing: YouTube state stalled (seeking)".to_string());
        }

        if sync.heal_cooldown_sec <= 0.0
            && (force_heal || (playback.playing && sync.sample_age_sec > 1.25))
        {
            // Prefer last known good content time so we don't jump back to 0 after reload/ads.
            let seek_to = if sync.last_good_time_sec > 0.01 {
                sync.last_good_time_sec
            } else if sync.has_remote {
                sync.last_remote_time_sec.max(0.0)
            } else {
                playback.time_sec.max(0.0)
            };

            // If the state has been stalled for a while, escalate to a full reload.
            let should_reload = force_heal || sync.sample_age_sec > 2.5;

            if should_reload {
                let _ = sync.tx.send(native_youtube::Command::ReloadAndSeek {
                    time_sec: seek_to,
                    playing: true,
                });
            } else {
                let _ = sync.tx.send(native_youtube::Command::SeekSeconds(seek_to));
                let _ = sync.tx.send(native_youtube::Command::SetPlaying(true));
            }

            // Snap our local time back to the last known remote time so we don't keep drifting.
            sync.remote_age_sec = 0.0;
            sync.interp_time_sec = seek_to;
            playback.time_sec = seek_to;

            // Throttle heal attempts.
            sync.heal_cooldown_sec = 2.0;
            sync.sample_age_sec = 0.0;
        }

        // If we were waiting for an ad to finish, seek back to last known good time once it's gone.
        if !sync.in_ad
            && sync.pending_seek_after_ad
            && sync.has_remote
            && sync.last_good_time_sec > 0.01
            && sync.heal_cooldown_sec <= 0.0
        {
            let seek_to = sync.last_good_time_sec;
            let _ = sync.tx.send(native_youtube::Command::SeekSeconds(seek_to));
            let _ = sync.tx.send(native_youtube::Command::SetPlaying(true));

            sync.interp_time_sec = seek_to;
            sync.remote_age_sec = 0.0;
            playback.time_sec = seek_to;
            playback.playing = true;
            sync.pending_seek_after_ad = false;
            sync.heal_cooldown_sec = 1.0;

            if let Ok(mut slot) = sync.last_error.lock() {
                *slot = Some(format!(
                    "healing: ad ended — seeking back to {:.2}",
                    seek_to
                ));
            }
        }
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "native-youtube"))]
fn native_youtube_shutdown_on_exit(
    mut app_exit: MessageReader<bevy::app::AppExit>,
    mut window_close: MessageReader<bevy::window::WindowCloseRequested>,
    sync: Option<Res<NativeYoutubeSync>>,
) {
    if app_exit.is_empty() && window_close.is_empty() {
        return;
    }
    // Drain so we don't repeatedly send.
    app_exit.clear();
    window_close.clear();

    let Some(sync) = sync else { return };
    if !sync.enabled {
        return;
    }

    let _ = sync.tx.send(native_youtube::Command::Shutdown);
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut tube_materials: ResMut<Assets<TubeMaterial>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    #[cfg(target_arch = "wasm32")]
    wasm_dbg("wasm: setup_scene() entered");
    let polkadot = images.add(make_polkadot_texture());
    commands.insert_resource(AppearanceTextures { polkadot });

    // commands.spawn((
    //     Camera3d::default(),
    //     Camera { order: 1, ..default() },
    //     Transform::from_xyz(10.0, 18.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    // ));

    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(-12.0, 10.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(-6.0, 14.0, -16.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let curve = make_random_loop_curve(1337);
    let frames = build_frames(&curve, FRAMES_SAMPLES);

    let tube_mesh = meshes.add(build_tube_mesh(
        &curve,
        &frames,
        TUBULAR_SEGMENTS,
        RADIAL_SEGMENTS,
        TUBE_RADIUS,
    ));

    let tube_mat = tube_materials.add(TubeMaterial::default());

    // Create initial fluid textures (will be updated by fluid simulation)
    let fluid_size = 64;
    let mut velocity_data = Vec::with_capacity(fluid_size * fluid_size * 4);
    let mut density_data = Vec::with_capacity(fluid_size * fluid_size * 4);

    for _ in 0..(fluid_size * fluid_size) {
        // Initial neutral values
        velocity_data.extend_from_slice(&[128, 128, 0, 255]); // RG = (0.5, 0.5) for neutral velocity
        density_data.extend_from_slice(&[128, 128, 128, 255]); // Gray for neutral density
    }

    let velocity_texture = Image::new(
        Extent3d {
            width: fluid_size as u32,
            height: fluid_size as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        velocity_data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );

    let density_texture = Image::new(
        Extent3d {
            width: fluid_size as u32,
            height: fluid_size as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        density_data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );

    let velocity_handle = images.add(velocity_texture);
    let density_handle = images.add(density_texture);

    // Update tube material with initial fluid textures
    if let Some(material) = tube_materials.get_mut(&tube_mat) {
        material.fluid_velocity = Some(velocity_handle);
        material.fluid_density = Some(density_handle);
    }

    commands.spawn((
        Mesh3d(tube_mesh),
        MeshMaterial3d(tube_mat.clone()),
        Transform::default(),
        TubeTag,
        Name::new("tube"),
    ));

    // Defer heavy BurnHuman model spawn until the lightweight scene is up.
    // A deferred system `spawn_burn_human_when_ready` will insert the subject
    // once `BurnHumanAssets` are available.

    // Optional alternate subject: a glossy, light-blue "glass" ball.
    // (We keep it around and just toggle visibility.)
    let ball_mesh = meshes.add(Mesh::from(bevy::math::primitives::Sphere::new(BALL_RADIUS)));
    let ball_mat = std_materials.add(StandardMaterial {
        base_color: Color::srgb(0.45, 0.82, 1.0).with_alpha(0.35),
        metallic: 0.0,
        reflectance: 0.95,
        perceptual_roughness: 0.04,
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        ..default()
    });

    commands.spawn((
        Mesh3d(ball_mesh),
        MeshMaterial3d(ball_mat),
        Transform::default(),
        Visibility::Hidden,
        BallTag,
        Name::new("subject_ball"),
    ));

    // Local light that follows the subject; helps avoid "flat" shading when the tube
    // occludes directional light contributions.
    commands.spawn((
        PointLight {
            intensity: 2200.0,
            range: 20.0,
            radius: 0.2,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0),
        SubjectLightTag,
        Name::new("subject_fill_light"),
    ));

    // If the `burn_human` feature is not enabled, spawn a lightweight
    // glass doughnut (torus) as a placeholder subject so the scene isn't empty.
    #[cfg(not(feature = "burn_human"))]
    {
        // Torus parameters (kept small for performance).
        let major = HUMAN_RADIUS * 0.9;
        let minor = HUMAN_RADIUS * 0.28;
        let tubular_segments: usize = 128;
        let radial_segments: usize = 32;

        // Build a smooth parametric torus to avoid seam/loop artifacts.
        let torus_mesh = meshes.add(build_torus_mesh(
            major,
            minor,
            tubular_segments,
            radial_segments,
        ));
        let torus_mat = std_materials.add(StandardMaterial {
            base_color: Color::srgb(0.45, 0.82, 1.0).with_alpha(0.35),
            metallic: 0.0,
            reflectance: 0.95,
            perceptual_roughness: 0.04,
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            ..default()
        });

        // Spawn as `SubjectTag` so existing subject transforms/logic apply.
        commands.spawn((
            Mesh3d(torus_mesh),
            MeshMaterial3d(torus_mat),
            Transform::default(),
            SubjectTag,
            Name::new("placeholder_doughnut"),
        ));

        // Mark spawned so other systems don't try to spawn again.
        commands.insert_resource(BurnHumanSpawned(true));
    }

    // Native: a dedicated full-window overlay camera for egui captions/credits/UI.
    // This avoids anchoring egui to the first 3D camera's viewport when multiview is enabled.
    #[cfg(not(target_arch = "wasm32"))]
    commands.spawn((
        Camera2d,
        Camera {
            viewport: None,
            order: 10_000,
            clear_color: ClearColorConfig::None,
            ..default()
        },
        OverlayUiCamera,
        bevy_egui::PrimaryEguiContext,
        Name::new("overlay_ui_camera"),
    ));

    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            near: 0.02,
            far: 3000.0,
            ..default()
        }),
        Transform::from_xyz(0.0, 0.0, -8.0).looking_at(Vec3::ZERO, Vec3::Y),
        MainCamera,
        ViewCamera { index: 0 },
        DistanceFog {
            color: Color::srgb_u8(0x12, 0x00, 0x00),
            falloff: FogFalloff::Linear {
                start: 40.0,
                end: 300.0,
            },
            ..default()
        },
    ));

    commands.insert_resource(TubeScene {
        curve,
        frames,
        tube_material: tube_mat,
    });

    commands.insert_resource(OverlayState::default());

    #[cfg(target_arch = "wasm32")]
    wasm_dbg("wasm: setup_scene() done; calling mcbaise_set_wasm_ready()");
    #[cfg(target_arch = "wasm32")]
    mcbaise_set_wasm_ready();
}

#[allow(clippy::type_complexity)]
fn sync_view_cameras(
    mut commands: Commands,
    multi_view: Res<MultiView>,
    main_cam: Query<(&Projection, &Transform, Option<&DistanceFog>), With<MainCamera>>,
    extras: Query<(Entity, &ViewCamera), (With<ViewCamera>, Without<MainCamera>)>,
) {
    if !multi_view.is_changed() {
        return;
    }

    let desired = multi_view.count.clamp(1, MultiView::MAX_VIEWS);
    let Some((projection, main_tr, fog)) = main_cam.iter().next() else {
        return;
    };

    // Remove any extra cameras beyond the desired count.
    for (entity, vc) in &extras {
        if vc.index >= desired {
            commands.entity(entity).despawn();
        }
    }

    // Spawn any missing cameras in 1..desired.
    let mut present = vec![false; desired as usize];
    for (_, vc) in &extras {
        if (vc.index as usize) < present.len() {
            present[vc.index as usize] = true;
        }
    }

    for idx in 1..desired {
        if present.get(idx as usize).copied().unwrap_or(false) {
            continue;
        }

        let mut e = commands.spawn((
            Camera3d::default(),
            Camera { order: idx as isize, ..default() },
            (*projection).clone(),
            *main_tr,
            ViewCamera { index: idx },
        ));

        if let Some(fog) = fog {
            e.insert((*fog).clone());
        }
    }
}

fn update_multiview_viewports(
    multi_view: Res<MultiView>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    cam_ids: Query<(Entity, &ViewCamera), With<ViewCamera>>,
    mut cameras: Query<&mut Camera>,
) {
    let Some(window) = windows.iter().next() else {
        return;
    };

    let w = window.physical_width();
    let h = window.physical_height();
    if w == 0 || h == 0 {
        return;
    }

    let count = (multi_view.count.clamp(1, MultiView::MAX_VIEWS)) as u32;
    let base_h = (h / count).max(1);

    let mut ordered: Vec<(u8, Entity)> = cam_ids.iter().map(|(e, vc)| (vc.index, e)).collect();
    ordered.sort_by_key(|(idx, _)| *idx);

    let mut y = 0u32;
    for (idx_u8, entity) in ordered {
        let idx = idx_u8 as u32;
        if idx >= count {
            continue;
        }

        let view_h = if idx + 1 == count {
            h.saturating_sub(y).max(1)
        } else {
            base_h
        };

        let Ok(mut cam) = cameras.get_mut(entity) else {
            continue;
        };

        cam.viewport = Some(bevy::camera::Viewport {
            physical_position: UVec2::new(0, y),
            physical_size: UVec2::new(w, view_h),
            ..default()
        });

        cam.order = idx as isize;
        cam.clear_color = if idx == 0 {
            ClearColorConfig::Default
        } else {
            ClearColorConfig::None
        };

        y = y.saturating_add(base_h);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn update_overlay_ui_camera(
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    mut q: Query<&mut Camera, With<OverlayUiCamera>>,
) {
    let Some(window) = windows.iter().next() else {
        return;
    };
    let w = window.physical_width();
    let h = window.physical_height();
    if w == 0 || h == 0 {
        return;
    }

    for mut cam in &mut q {
        cam.viewport = None;
        cam.order = 10_000;
        cam.clear_color = ClearColorConfig::None;
    }
}

// During teardown / resize / exit, disable the overlay 2D camera.
// This prevents the Core2D main opaque pass from running while the window
// surface/swapchain is being reconfigured or destroyed, which otherwise can
// trigger wgpu validation errors like "In a pass parameter: Encoder is invalid".
#[cfg(not(target_arch = "wasm32"))]
fn gate_overlay_ui_camera_system(
    teardown: Res<TeardownState>,
    pending_geom: Res<PendingWindowGeometry>,
    mut cams: Query<&mut Camera, With<OverlayUiCamera>>,
    mut app_exit: MessageReader<bevy::app::AppExit>,
    mut window_close: MessageReader<bevy::window::WindowCloseRequested>,
) {
    let exit_requested = !app_exit.is_empty() || !window_close.is_empty();
    if exit_requested {
        app_exit.clear();
        window_close.clear();
    }

    let disable = exit_requested || teardown.active || pending_geom.pending;
    for mut cam in &mut cams {
        cam.is_active = !disable;
        if disable {
            cam.clear_color = ClearColorConfig::None;
        }
    }
}

// Disable non-overlay cameras while a resize/teardown is in progress.
// Important: only forces cameras OFF; it does not force-enable them again,
// so other systems remain in control of camera activation.
#[cfg(not(target_arch = "wasm32"))]
fn gate_main_cameras_during_geometry_system(
    teardown: Res<TeardownState>,
    pending_geom: Res<PendingWindowGeometry>,
    mut cams: Query<&mut Camera, (With<ViewCamera>, Without<OverlayUiCamera>)>,
) {
    let disable = teardown.active || pending_geom.pending;
    for mut cam in &mut cams {
        cam.is_active = !disable;
        if disable {
            cam.clear_color = ClearColorConfig::None;
        }
    }
}

// Watch for a UI-requested window geometry change to settle, then request a camera restart.
fn window_geometry_watch_system(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut pending_geom: ResMut<PendingWindowGeometry>,
    mut restart_req: ResMut<CameraRestartRequested>,
    mut teardown: ResMut<TeardownState>,
    loading: Res<LoadingState>,
    frame: Res<GlobalFrameCount>,
) {
    if !pending_geom.pending {
        return;
    }

    // The `Window` component updates immediately when we call
    // `set_physical_resolution`, but the render surface reconfiguration happens
    // later, and Bevy's pipelined rendering means render-world state can lag.
    // Keep `pending` asserted for a few frames after the deferred apply frame
    // to avoid transient mismatches (e.g. depth vs color attachment size).
    const WINDOW_GEOM_STABILIZE_FRAMES: u64 = 3;
    if let Some(applied_frame) = pending_geom.scheduled_frame {
        let hold_until = applied_frame.saturating_add(WINDOW_GEOM_STABILIZE_FRAMES);
        if frame.0 <= hold_until {
            return;
        }
    }

    let Some(window) = windows.iter().next() else { return; };
    let ww = window.physical_width();
    let wh = window.physical_height();
    if ww == 0 || wh == 0 { return; }
    if ww == pending_geom.target_w && wh == pending_geom.target_h {
        pending_geom.pending = false;
        pending_geom.scheduled_frame = None;
        // Only activate teardown if the app has finished initial loading and
        // GPU-ready handshake — avoid tearing down resources during startup.
        if loading.stage == LoadingStage::Ready {
            // By default, avoid performing a destructive teardown on simple window
            // resizes. Destroying swapchain/surface-backed resources during a resize
            // can race with in-flight SurfaceTexture semaphores on some drivers.
            // If the user explicitly wants the old behavior, set
            // MCBAISE_FORCE_GEOM_RESTART=1 in the environment.
            let force = std::env::var("MCBAISE_FORCE_GEOM_RESTART").as_deref().unwrap_or("0") == "1";
            if force {
                teardown.active = true;
                teardown.frames_left = 4;
                eprintln!("native: window geometry settled -> starting teardown countdown (forced)");
            } else {
                // We no longer do destructive teardown for simple resizes, but we
                // still need to ensure view cameras come back and their state is
                // refreshed. Without this, cameras can remain inactive from the
                // pending-resize gating frame and views appear black.
                restart_req.requested = true;
                eprintln!("native: window geometry settled -> resizing in-place (no teardown)");
            }
        } else {
            eprintln!("native: window geometry settled but loading not ready; skipping teardown");
        }
    }
}

// Automated resize stepper (mirrors `resize_test.rs` stepping behavior).
fn resize_automation_step(
    mut auto: ResMut<ResizeAutomation>,
    mut pending_geom: ResMut<PendingWindowGeometry>,
    mut deferred: ResMut<DeferredWindowResolutionChange>,
    frame: Res<GlobalFrameCount>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut app_exit: MessageWriter<AppExit>,
) {
    if !auto.active {
        return;
    }

    if auto.env_forced && frame.0 < auto.start_frame {
        return;
    }

    if auto.exit_requested {
        if !pending_geom.pending && !deferred.pending {
            app_exit.write(AppExit::Success);
        }
        return;
    }

    // Defer one frame to avoid interfering with immediate UI layout.
    if !auto.first_frame {
        auto.first_frame = true;
        return;
    }

    // Throttle how quickly we step the window size. If we resize too often,
    // camera gating will blank most frames and the output looks "mostly black".
    let step_every_frames = auto.step_every_frames.max(1);
    if auto.next_step_frame != 0 && frame.0 < auto.next_step_frame {
        return;
    }

    // Only issue the next step when the previous resize has fully settled.
    // Otherwise `pending` stays asserted continuously and view cameras remain
    // gated off (black views).
    if pending_geom.pending || deferred.pending {
        return;
    }

    if !auto.seeded_from_window {
        if let Some(win) = windows.iter().next() {
            let w = win.physical_width().max(1).min(u32::from(u16::MAX)) as u16;
            let h = win.physical_height().max(1).min(u32::from(u16::MAX)) as u16;
            auto.width = w;
            auto.height = h;
        }
        auto.seeded_from_window = true;
    }

    const MAX_WIDTH: u16 = 401;
    const MAX_HEIGHT: u16 = 401;
    const MIN_WIDTH: u16 = 1;
    const MIN_HEIGHT: u16 = 1;
    const RESIZE_STEP: u16 = 4;

    let mut w = auto.width;
    let mut h = auto.height;

    match auto.phase {
        Phase::ContractingY => {
            if h <= MIN_HEIGHT {
                auto.phase = Phase::ContractingX;
            } else {
                h = h.saturating_sub(RESIZE_STEP);
            }
        }
        Phase::ContractingX => {
            if w <= MIN_WIDTH {
                auto.phase = Phase::ExpandingY;
            } else {
                w = w.saturating_sub(RESIZE_STEP);
            }
        }
        Phase::ExpandingY => {
            if h >= MAX_HEIGHT {
                auto.phase = Phase::ExpandingX;
            } else {
                h = h.saturating_add(RESIZE_STEP);
            }
        }
        Phase::ExpandingX => {
            if w >= MAX_WIDTH {
                auto.phase = Phase::ContractingY;
            } else {
                w = w.saturating_add(RESIZE_STEP);
            }
        }
    }

    // If changed, schedule a deferred window resize and arm the resize handshake.
    if w != auto.width || h != auto.height {
        auto.width = w;
        auto.height = h;

        // Defer the actual resize to next frame to keep the render-world surface
        // reconfigure away from the same frame as UI layout and to give camera
        // gating time to kick in.
        pending_geom.pending = true;
        pending_geom.target_w = w as u32;
        pending_geom.target_h = h as u32;
        // `scheduled_frame` tracks the deferred apply frame (not the schedule frame).
        // The settle watcher will keep `pending` asserted for a few frames after.
        pending_geom.scheduled_frame = Some(frame.0 + 1);

        deferred.pending = true;
        deferred.target_w = w as u32;
        deferred.target_h = h as u32;
        deferred.apply_frame = frame.0 + 1;

        // Leave some frames between steps so the scene is visible.
        auto.next_step_frame = frame.0 + step_every_frames;

        auto.steps_done = auto.steps_done.saturating_add(1);
        if auto.max_steps > 0 && auto.steps_done >= auto.max_steps {
            auto.exit_requested = true;
        }

        eprintln!(
            "native: resize-automation scheduled geometry -> {}x{} (apply_frame={}, frame={})",
            w,
            h,
            deferred.apply_frame,
            frame.0
        );
    }
}

// Sync UI toggle into the ResizeAutomation resource and initialize dimensions
fn sync_resize_toggle_system(
    capture_state: Res<EguiCaptureState>,
    mut auto: ResMut<ResizeAutomation>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    if auto.env_forced {
        return;
    }

    if capture_state.resize_automation_active && !auto.active {
        // Activate: seed from current window physical size
        if let Some(win) = windows.iter().next() {
            let w = win.physical_width().max(1).min(65535) as u16;
            let h = win.physical_height().max(1).min(65535) as u16;
            auto.width = w;
            auto.height = h;
        }
        auto.first_frame = false;
        auto.next_step_frame = 0;
        auto.active = true;
        auto.seeded_from_window = true;
        auto.steps_done = 0;
        auto.exit_requested = false;
        eprintln!("native: resize automation enabled");
    } else if !capture_state.resize_automation_active && auto.active {
        auto.active = false;
        auto.next_step_frame = 0;
        eprintln!("native: resize automation disabled");
    }
}

// Restart main and overlay cameras to ensure GPU state is consistent after big geometry changes.
fn restart_cameras_system(
    mut commands: Commands,
    mut restart_req: ResMut<CameraRestartRequested>,
    mut params: ParamSet<(
        Query<&mut Camera, With<OverlayUiCamera>>,
        Query<(&mut Camera, &mut Transform), With<MainCamera>>,
    )>,
    render_device: Res<bevy::render::renderer::RenderDevice>,
    mut pending_drops: ResMut<PendingRenderAssetDrops>,
    loading: Res<LoadingState>,
) {
    if loading.stage != LoadingStage::Ready {
        return;
    }

    // Perform a non-destructive, in-place camera restart when requested.
    // This avoids despawning/spawning camera entities which can free GPU
    // resources and trigger validation errors during render.
    if !restart_req.requested {
        return;
    }
    restart_req.requested = false;
    eprintln!("native: performing in-place camera restart (no destructive drops)");

    // Give the device an extra conservative drain to reduce races.
    let _ = render_device.poll(wgpu::PollType::Wait);
    sleep(Duration::from_millis(4));

    for mut cam in params.p0().iter_mut() {
        cam.is_active = true;
        cam.order = 10_000;
        cam.clear_color = ClearColorConfig::None;
    }

    for (mut cam, _transform) in params.p1().iter_mut() {
        // Critical: do not reset the camera transform here.
        // The tube-ride camera is animated; resetting the transform makes the
        // scene look like it's replaying previously-rendered frames.
        cam.is_active = true;
        cam.order = 0;
        cam.clear_color = ClearColorConfig::Default;
    }
    // Keep pending_drops resource present for future destructive operations.
    let _ = &mut *pending_drops;
}

// Execution-phase: once the render cleanup has signalled readiness we perform
// the actual camera despawn and respawn in the main world. This ensures the
// render thread has had a chance to drain before we drop swapchain-backed
// resources.
fn restart_cameras_execute_system(
    mut commands: Commands,
    mut restart_req: ResMut<CameraRestartRequested>,
    overlay_q: Query<Entity, With<OverlayUiCamera>>,
    main_q: Query<Entity, With<MainCamera>>,
    render_device: Res<bevy::render::renderer::RenderDevice>,
) {
    if let Some(flag) = restart_req.pending_flag.as_ref() {
        if !flag.load(Ordering::SeqCst) {
            return;
        }
        // Clear the pending flag so this only runs once.
        restart_req.pending_flag = None;

        eprintln!("native: render cleanup acknowledged -> performing camera restart (non-destructive)");
        // Give the device an extra conservative drain and a short pause before
        // touching any swapchain-backed resources.
        wait_for_device_idle_strong(&*render_device);

        let has_overlay = overlay_q.iter().next().is_some();
        let has_main = main_q.iter().next().is_some();

        if !has_overlay {
            #[cfg(not(target_arch = "wasm32"))]
            commands.spawn((
                Camera2d,
                Camera {
                    viewport: None,
                    order: 10_000,
                    clear_color: ClearColorConfig::None,
                    ..default()
                },
                OverlayUiCamera,
                bevy_egui::PrimaryEguiContext,
                Name::new("overlay_ui_camera"),
            ));
        }

        if !has_main {
            commands.spawn((
                Camera3d::default(),
                Camera { order: 0, ..default() },
                Projection::Perspective(PerspectiveProjection {
                    near: 0.02,
                    far: 3000.0,
                    ..default()
                }),
                Transform::from_xyz(0.0, 0.0, -8.0).looking_at(Vec3::ZERO, Vec3::Y),
                MainCamera,
                ViewCamera { index: 0 },
                DistanceFog {
                    color: Color::srgb_u8(0x12, 0x00, 0x00),
                    falloff: FogFalloff::Linear {
                        start: 40.0,
                        end: 300.0,
                    },
                    ..default()
                },
            ));
        } else {
            eprintln!("native: non-destructive camera restart - keeping existing camera entities");
        }
    }
}

// Cycle through preset output/window resolutions when the user right-clicks
// the resize icon in the egui overlay.
fn cycle_resolution_on_resize_icon_request_system(
    mut capture_state: ResMut<EguiCaptureState>,
    mut pending_window_geometry: ResMut<PendingWindowGeometry>,
    mut deferred: ResMut<DeferredWindowResolutionChange>,
    frame: Res<GlobalFrameCount>,
) {
    if !capture_state.cycle_resolution_requested {
        return;
    }
    // Consume the request even if we're busy to avoid repeated cycling.
    capture_state.cycle_resolution_requested = false;

    if pending_window_geometry.pending || deferred.pending {
        return;
    }

    let resolutions: &[(u32, u32, &str)] = &[
        (256, 144, "144p (256x144)"),
        (426, 240, "240p (426x240)"),
        (640, 360, "360p (640x360)"),
        (854, 480, "480p (854x480)"),
        (1280, 720, "720p (1280x720)"),
        (1920, 1080, "1080p (1920x1080)"),
        (2560, 1440, "1440p (2560x1440)"),
        (3840, 2160, "2160p (3840x2160)"),
        (1080, 1920, "Phone Portrait (1080x1920)"),
        (1170, 2532, "iPhone Pro (1170x2532)"),
        (1080, 2340, "Modern Phone (1080x2340)"),
    ];

    let next_index = if capture_state.selected_resolution >= 0
        && (capture_state.selected_resolution as usize) < resolutions.len()
    {
        let idx = capture_state.selected_resolution as usize;
        (idx + 1) % resolutions.len()
    } else {
        0usize
    };

    let (w, h, _label) = resolutions[next_index];
    capture_state.selected_resolution = next_index as i32;

    schedule_window_geometry_change(
        &mut pending_window_geometry,
        &mut deferred,
        w,
        h,
        frame.0,
        "resize icon right-click cycled",
    );
}

// Countdown system: while teardown is active, decrement frames; when it reaches
// zero, trigger camera and scene restart requests.
fn teardown_countdown_system(
    mut teardown: ResMut<TeardownState>,
    mut restart_req: ResMut<CameraRestartRequested>,
    mut scene_restart: ResMut<SceneRestartRequested>,
    render_device: Res<bevy::render::renderer::RenderDevice>,
) {
    if !teardown.active {
        return;
    }
    // Poll GPU each frame to help drive completion of in-flight submissions.
    let _ = render_device.poll(wgpu::PollType::Poll);

    if teardown.frames_left > 0 {
        // Decrement and wait; keep this conservative to avoid races during swapchain teardown.
        teardown.frames_left = teardown.frames_left.saturating_sub(1);
        return;
    }
    // Countdown complete: request restart and clear teardown flag.
    teardown.active = false;
    restart_req.requested = true;
    scene_restart.requested = true;
    eprintln!("native: teardown countdown complete -> requesting camera + scene restart");
}

// If requested, despawn the main scene entities and recreate them at full resolution.
#[derive(SystemParam)]
struct RestartSceneParams<'w, 's> {
    commands: Commands<'w, 's>,
    scene_restart: ResMut<'w, SceneRestartRequested>,
    meshes: ResMut<'w, Assets<Mesh>>,
    tube_materials: ResMut<'w, Assets<TubeMaterial>>,
    std_materials: ResMut<'w, Assets<StandardMaterial>>,
    images: ResMut<'w, Assets<Image>>,
    assets: Option<Res<'w, BurnHumanAssets>>,
    tube_q: Query<'w, 's, Entity, With<TubeTag>>,
    subj_q: Query<'w, 's, Entity, With<SubjectTag>>,
    ball_q: Query<'w, 's, Entity, With<BallTag>>,
    light_q: Query<'w, 's, Entity, With<SubjectLightTag>>,
    render_device: Res<'w, bevy::render::renderer::RenderDevice>,
    pending_drops: ResMut<'w, PendingRenderAssetDrops>,
    loading: Res<'w, LoadingState>,
    maybe_readback: Option<Res<'w, GpuReadbackImage>>,
    maybe_appearance: Option<Res<'w, AppearanceTextures>>,
    maybe_capture: Option<Res<'w, EguiCaptureState>>,
}

fn restart_scene_system(mut p: RestartSceneParams, frame: Res<GlobalFrameCount>) {
    // Avoid arming a full scene teardown while the app is still in its
    // startup/loading phase. This prevents accidental early drops that
    // would tear down GPU resources before the scene is first created.
    if p.loading.stage != LoadingStage::Ready {
        return;
    }

    // Arm a render-world handshake to safely perform destructive scene teardown
    // after the render-subapp acknowledges it's idle. We don't destroy GPU-
    // backed resources here; instead we create a shared flag and wait for the
    // render-app cleanup system to set it.
    if !p.scene_restart.requested {
        return;
    }
    p.scene_restart.requested = false;

    let id = NEXT_TEARDOWN_ID.fetch_add(1, Ordering::SeqCst);
    let flag = Arc::new(AtomicBool::new(false));
    let map = GLOBAL_RENDER_TEARDOWNS.get_or_init(|| Mutex::new(HashMap::new()));
    map.lock().unwrap().insert(id, flag.clone());
    p.scene_restart.pending_flag = Some(flag);
    eprintln!("native: armed scene restart handshake (id={}) - render cleanup will signal readiness", id);

    let mut drops = PendingRenderAssetDrops::default();
    for (_h, mat) in p.tube_materials.iter() {
        if let Some(h) = &mat.fluid_velocity {
            drops.images.push(h.clone());
        }
        if let Some(h) = &mat.fluid_density {
            drops.images.push(h.clone());
        }
    }
    for (_h, mat) in p.std_materials.iter() {
        if let Some(h) = &mat.base_color_texture {
            drops.images.push(h.clone());
        }
        if let Some(h) = &mat.normal_map_texture {
            drops.images.push(h.clone());
        }
        if let Some(h) = &mat.occlusion_texture {
            drops.images.push(h.clone());
        }
    }
    if let Some(rb) = p.maybe_readback.as_ref() {
        drops.images.push(rb.0.clone());
    }
    if let Some(app) = p.maybe_appearance.as_ref() {
        drops.images.push(app.polkadot.clone());
    }
    if let Some(capture) = p.maybe_capture.as_ref() {
        for entry in capture.loaded.values() {
            for h in &entry.handles {
                drops.images.push(h.clone());
            }
        }
    }

    drops.images.sort_by_key(|h| h.id());
    drops.images.dedup_by_key(|h| h.id());
    p.pending_drops.images = drops.images;
    // Mark pending drops for the render side. We can't reliably assign
    // a render-frame-based delay from the main world (frame counters
    // differ between main and render sub-apps), so use `Some(0)` as a
    // placeholder. The render-side consumer will convert this to a
    // real scheduled frame using the render world's `GlobalFrameCount`.
    p.pending_drops.scheduled_frame = Some(0);
}

// Execution-phase: after render cleanup has signalled it's safe, perform
// the scene despawn and re-setup in the main world.
fn restart_scene_execute_system(
    mut commands: Commands,
    mut scene_restart: ResMut<SceneRestartRequested>,
    // resources required by setup_scene
    mut meshes: ResMut<Assets<Mesh>>,
    mut tube_materials: ResMut<Assets<TubeMaterial>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    assets: Option<Res<BurnHumanAssets>>,
    // queries to find and despawn existing scene entities
    tube_q: Query<Entity, With<TubeTag>>,
    subj_q: Query<Entity, With<SubjectTag>>,
    ball_q: Query<Entity, With<BallTag>>,
    light_q: Query<Entity, With<SubjectLightTag>>,
    render_device: Res<bevy::render::renderer::RenderDevice>,
    mut pending_drops: ResMut<PendingRenderAssetDrops>,
) {
    if let Some(flag) = scene_restart.pending_flag.as_ref() {
        if !flag.load(Ordering::SeqCst) {
            return;
        }
        // consume the pending flag so we only run once
        scene_restart.pending_flag = None;
        eprintln!("native: render cleanup acknowledged -> performing scene restart (non-destructive)");
        // Extra strong drain before any work that could touch swapchain-backed
        // resources to reduce races on some drivers.
        wait_for_device_idle_strong(&*render_device);
        // Clear the main-world pending-drops list now that the render thread
        // should have consumed and removed the GPU-backed images.
        pending_drops.images.clear();
        pending_drops.scheduled_frame = None;

        // Instead of destructively despawning and re-creating many GPU-backed
        // entities (which can cause swapchain/resource lifecycle races), prefer
        // a non-destructive restart: if the expected scene entities are
        // already present, keep them and only call `setup_scene` when the
        // scene is missing (first-run or fully torn-down). This avoids
        // freeing swapchain-backed textures on the main thread while the
        // render thread may still reference them.
        let has_tube = tube_q.iter().next().is_some();
        let has_subject = subj_q.iter().next().is_some();
        let has_ball = ball_q.iter().next().is_some();
        let has_light = light_q.iter().next().is_some();

        if !(has_tube || has_subject || has_ball || has_light) {
            // Nothing present: perform the full setup.
            setup_scene(commands, meshes, tube_materials, std_materials, images);
        } else {
            // Otherwise, keep existing entities; we may still want to update
            // materials/transforms (left to existing Update systems such as
            // `apply_subject_mode` and `update_fluid_simulation`).
            eprintln!("native: non-destructive restart - keeping existing scene entities");
        }
    }
}

// (capture_window_surfaces_exclusive removed)

// Watch for initial scene setup completing and wait for GPU to settle before
// clearing the loading overlay. This prevents showing the model before the
// GPU/driver has fully initialized and helps avoid swapchain/surface races.
fn loading_watch_system(
    render_device: Res<bevy::render::renderer::RenderDevice>,
    main_cam_q: Query<Entity, With<MainCamera>>,
    mut loading: ResMut<LoadingState>,
) {
    // If already ready, nothing to do.
    if loading.stage == LoadingStage::Ready {
        return;
    }

    match loading.stage {
        LoadingStage::LoadingAssets => {
            // Consider the main cameras as a signal that setup_scene has run.
            if main_cam_q.iter().next().is_some() {
                loading.stage = LoadingStage::WaitingForGpu;
                loading.frames_left = 6; // conservative wait to let submissions complete
                eprintln!("native: setup_scene done -> waiting for GPU ({} frames)", loading.frames_left);
                // Give the device a blocking poll to flush and submit pending work.
                let _ = render_device.poll(wgpu::PollType::Wait);
            }
        }
        LoadingStage::WaitingForGpu => {
            // Progress any in-flight work; do a light poll each frame.
            let _ = render_device.poll(wgpu::PollType::Poll);
            if loading.frames_left > 0 {
                loading.frames_left = loading.frames_left.saturating_sub(1);
                return;
            }
            loading.stage = LoadingStage::Ready;
            eprintln!("native: GPU ready -> clearing loading overlay");
        }
        LoadingStage::Ready => {}
    }
}

fn ensure_subject_normals(
    mut meshes: ResMut<Assets<Mesh>>,
    mut done: ResMut<SubjectNormalsComputed>,
    q: Query<&Mesh3d, With<SubjectTag>>,
) {
    for mesh3d in &q {
        let id = mesh3d.0.id();
        if done.0.contains(&id) {
            continue;
        }

        let Some(mesh) = meshes.get_mut(&mesh3d.0) else {
            continue;
        };

        if subject_mesh_has_usable_normals(mesh) {
            done.0.insert(id);
            continue;
        }

        if compute_smooth_normals(mesh) {
            done.0.insert(id);
        }
    }
}

fn subject_mesh_has_usable_normals(mesh: &Mesh) -> bool {
    let Some(values) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL) else {
        return false;
    };

    match values {
        // Guard against the "present but zero" case.
        bevy::mesh::VertexAttributeValues::Float32x3(ns) => ns.iter().any(|n| {
            let v = Vec3::from(*n);
            v.length_squared() > 1.0e-10
        }),
        _ => true,
    }
}

fn update_fluid_simulation(
    time: Res<Time>,
    mut fluid_sim: ResMut<FluidSimulation>,
    mut images: ResMut<Assets<Image>>,
    tube_scene: Res<TubeScene>,
    mut tube_materials: ResMut<Assets<TubeMaterial>>,
    playback: Res<Playback>,
) {
    // Update fluid simulation
    fluid_sim.update(time.delta_secs(), playback.time_sec);

    // If fluid textures are not enabled, do not allocate GPU textures yet.
    // The simulation still advances (so enabling later will be smooth), but
    // we avoid creating `Image` assets until the feature is enabled.
    if !fluid_sim.enabled {
        return;
    }

    // Create or update velocity texture (create-once, then update data)
    let velocity_image = create_fluid_texture(
        &fluid_sim.velocity_field,
        fluid_sim.width,
        fluid_sim.height,
        true,
    );
    let velocity_handle = if let Some(handle) = &fluid_sim.velocity_handle {
        if let Some(img) = images.get_mut(handle) {
            img.data = velocity_image.data;
            Some(handle.clone())
        } else {
            let h = images.add(velocity_image);
            fluid_sim.velocity_handle = Some(h.clone());
            Some(h)
        }
    } else {
        let h = images.add(velocity_image);
        fluid_sim.velocity_handle = Some(h.clone());
        Some(h)
    };

    // Create or update density texture (create-once, then update data)
    let density_image = create_fluid_texture_density(&fluid_sim.density_field, fluid_sim.width, fluid_sim.height);
    let density_handle = if let Some(handle) = &fluid_sim.density_handle {
        if let Some(img) = images.get_mut(handle) {
            img.data = density_image.data;
            Some(handle.clone())
        } else {
            let h = images.add(density_image);
            fluid_sim.density_handle = Some(h.clone());
            Some(h)
        }
    } else {
        let h = images.add(density_image);
        fluid_sim.density_handle = Some(h.clone());
        Some(h)
    };

    // Update tube material with fluid textures
    if let Some(material) = tube_materials.get_mut(&tube_scene.tube_material) {
        material.fluid_velocity = velocity_handle.clone();
        material.fluid_density = density_handle.clone();
    }
}

fn create_fluid_texture(
    velocity_field: &[Vec2],
    width: usize,
    height: usize,
    is_velocity: bool,
) -> Image {
    let mut data = Vec::with_capacity(width * height * 4);

    for &vel in velocity_field {
        if is_velocity {
            // Store velocity as RG channels, BA as zero
            let r = ((vel.x * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            let g = ((vel.y * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            data.extend_from_slice(&[r, g, 0, 255]);
        } else {
            // For density, store as grayscale
            let gray = ((vel.x * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            data.extend_from_slice(&[gray, gray, gray, 255]);
        }
    }

    let mut img = Image::new(
        Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    // Label the fluid texture for clearer wgpu logs.
    let _ = match &mut img.texture_descriptor.label {
        Some(_) => {
            img.texture_descriptor.label = Some(if is_velocity { "fluid_velocity" } else { "fluid_generic" });
            true
        }
        None => false,
    };
    img
}

fn create_fluid_texture_density(density_field: &[f32], width: usize, height: usize) -> Image {
    let mut data = Vec::with_capacity(width * height * 4);

    for &density in density_field {
        let gray = (density.clamp(0.0, 1.0) * 255.0) as u8;
        data.extend_from_slice(&[gray, gray, gray, 255]);
    }

    let mut img = Image::new(
        Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    let _ = match &mut img.texture_descriptor.label {
        Some(_) => {
            img.texture_descriptor.label = Some("fluid_density");
            true
        }
        None => false,
    };
    img
}

fn compute_smooth_normals(mesh: &mut Mesh) -> bool {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return false;
    }

    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(bevy::mesh::VertexAttributeValues::Float32x3(ps)) => {
            ps.iter().map(|p| Vec3::from(*p)).collect::<Vec<Vec3>>()
        }
        _ => return false,
    };

    if positions.len() < 3 {
        return false;
    }

    let indices: Vec<u32> = match mesh.indices() {
        Some(Indices::U16(is)) => is.iter().map(|&i| i as u32).collect(),
        Some(Indices::U32(is)) => is.clone(),
        None => (0..positions.len() as u32).collect(),
    };

    if indices.len() < 3 {
        return false;
    }

    let mut normals = vec![Vec3::ZERO; positions.len()];

    for tri in indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }

        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];

        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let n = e1.cross(e2);

        if n.length_squared() <= 1.0e-20 {
            continue;
        }

        normals[i0] += n;
        normals[i1] += n;
        normals[i2] += n;
    }

    let normals: Vec<[f32; 3]> = normals
        .into_iter()
        .map(|n| {
            let n = n.normalize_or_zero();
            [n.x, n.y, n.z]
        })
        .collect();

    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    true
}

#[cfg(not(target_arch = "wasm32"))]
fn advance_time_native(
    time: Res<Time>,
    mut playback: ResMut<Playback>,
    _native_youtube: Option<Res<NativeYoutubeSync>>,
    _native_mpv: Option<Res<NativeMpvSync>>,
) {
    #[cfg(feature = "native-youtube")]
    {
        if _native_youtube
            .as_ref()
            .map(|yt| yt.enabled)
            .unwrap_or(false)
        {
            return;
        }
    }

    #[cfg(all(windows, feature = "native-mpv"))]
    {
        if let Some(mpv) = _native_mpv.as_ref() {
            // Only suppress local time advance once mpv is actually providing samples.
            if mpv.enabled && mpv.has_remote {
                return;
            }
        }
    }
    if playback.playing {
        playback.time_sec += time.delta_secs() * playback.speed;
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn gif_auto_capture_system(
    mut capture_state: ResMut<EguiCaptureState>,
    time: Res<Time>,
) {
    use std::sync::atomic::Ordering;
    // If GIF recording is enabled, request readbacks when idle.
    if std::env::var("MCBAISE_GIF_RECORD").as_deref().ok() == Some("1") {
        let now = time.elapsed().as_secs_f64();
        if capture_state.capture_fps > 0 {
            if now - capture_state.last_capture_time_secs < 1.0 / capture_state.capture_fps as f64 {
                return;
            }
        }

        capture_state.capture_skip_counter += 1;
        if capture_state.capture_skip_counter < capture_state.capture_skip {
            return;
        }

        // Only request if no request is pending and no map is in flight.
        if READBACK_REQUEST_SEQ.load(Ordering::SeqCst) == 0
            && READBACK_MAP_IN_FLIGHT.load(Ordering::SeqCst) == 0
        {
            READBACK_REQUEST_SEQ.fetch_add(1, Ordering::SeqCst);
            capture_state.last_capture_time_secs = now;
            capture_state.capture_skip_counter = 0;
        }
    }
}

// --- Capture UI + preview (native only) ---------------------------------

// Preview entry stores one or more frames (for GIFs) with their bevy handles and egui texture ids.
#[derive(Default)]
struct PreviewEntry {
    handles: Vec<Handle<Image>>,
    tex_ids: Vec<egui::TextureId>,
    // per-frame durations in seconds (if empty, use 0.1s)
    durations: Vec<f32>,
    current_idx: usize,
    last_advance_secs: f64,
}

// Egui capture state: holds loaded previews and last poll time.
#[derive(Resource)]
struct EguiCaptureState {
    last_dir: Option<String>,
    // map from filename -> PreviewEntry
    loaded: std::collections::HashMap<String, PreviewEntry>,
    // order of filenames (most-recent first)
    order: std::collections::VecDeque<String>,
    last_polled_secs: f64,
    // whether the capture UI is visible; default: hidden until user triggers capture
    visible: bool,
    // whether the keyboard shortcuts info overlay is visible
    show_info: bool,
    // whether the resize automation (UI button) is active
    resize_automation_active: bool,
    // whether to show options instead of previews in the capture UI
    show_options: bool,
    // whether to include captions in the recording
    capture_captions: bool,
    // whether to capture the whole multi-view layout or just the main camera
    capture_multi: bool,
    // target capture FPS (0 = as fast as possible)
    capture_fps: u32,
    // how many frames to skip between captures (1 = every frame, 2 = every other frame)
    capture_skip: u32,
    // internal counter for frame skipping
    capture_skip_counter: u32,
    // last time a frame was captured
    last_capture_time_secs: f64,
    // output scale factor (e.g. 0.5 for half-size, 2.0 for double-size)
    output_scale: f32,
    // selected window resolution index (-1 for custom/none)
    selected_resolution: i32,

    // UI request: cycle preset resolution (set by resize icon right-click)
    cycle_resolution_requested: bool,
}

impl Default for EguiCaptureState {
    fn default() -> Self {
        Self {
            last_dir: None,
            loaded: Default::default(),
            order: Default::default(),
            last_polled_secs: 0.0,
            visible: false,
            show_info: false,
            resize_automation_active: false,
            show_options: false,
            capture_captions: true,
            capture_multi: false,
            capture_fps: 30,
            capture_skip: 1,
            capture_skip_counter: 0,
            last_capture_time_secs: 0.0,
            output_scale: 1.0,
            selected_resolution: -1,

            cycle_resolution_requested: false,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_controls(
    keys: Res<ButtonInput<KeyCode>>,
    mut playback: ResMut<Playback>,
    mut settings: ResMut<TubeSettings>,
    mut scheme_mode: ResMut<ColorSchemeMode>,
    mut pattern_mode: ResMut<TexturePatternMode>,
    _native_youtube: Option<Res<NativeYoutubeSync>>,
    _native_mpv: Option<Res<NativeMpvSync>>,
    mut readback: ResMut<GpuReadbackImage>,
    mut images: ResMut<Assets<Image>>,
    #[cfg(feature = "capture_ui")]
    mut capture_state_opt: Option<ResMut<EguiCaptureState>>,
) {
    if keys.just_pressed(KeyCode::Space) {
        playback.playing = !playback.playing;

        #[cfg(feature = "native-youtube")]
        {
            if let Some(yt) = _native_youtube.as_ref()
                && yt.enabled
            {
                let _ = yt
                    .tx
                    .send(crate::native_youtube::Command::SetPlaying(playback.playing));

                if playback.playing && yt.sample_age_sec > 2.0 {
                    let seek_to = if yt.last_good_time_sec > 0.01 {
                        yt.last_good_time_sec
                    } else {
                        playback.time_sec
                    };
                    let _ = yt.tx.send(crate::native_youtube::Command::ReloadAndSeek {
                        time_sec: seek_to,
                        playing: true,
                    });
                }
            }
        }

        #[cfg(all(windows, feature = "native-mpv"))]
        {
            if let Some(mpv) = _native_mpv.as_ref()
                && mpv.enabled
            {
                let _ = mpv
                    .tx
                    .send(crate::native_mpv::Command::SetPlaying(playback.playing));
            }
        }
    }
    if keys.just_pressed(KeyCode::Digit1) {
        settings.scheme = (settings.scheme + 1) % 4;
        *scheme_mode = ColorSchemeMode::from_value(settings.scheme);
    }
    if keys.just_pressed(KeyCode::Digit2) {
        settings.pattern = (settings.pattern + 1) % 6;
        *pattern_mode = TexturePatternMode::from_value(settings.pattern);
    }
    if keys.just_pressed(KeyCode::ArrowUp) {
        #[cfg(feature = "native-youtube")]
        {
            if _native_youtube
                .as_ref()
                .map(|yt| yt.enabled)
                .unwrap_or(false)
            {
                return;
            }
        }
        playback.speed = (playback.speed + 0.25).clamp(0.25, 3.0);
    }
    if keys.just_pressed(KeyCode::ArrowDown) {
        #[cfg(feature = "native-youtube")]
        {
            if _native_youtube
                .as_ref()
                .map(|yt| yt.enabled)
                .unwrap_or(false)
            {
                return;
            }
        }
        playback.speed = (playback.speed - 0.25).clamp(0.25, 3.0);
    }

    // One-shot PNG capture: press `P` to write a single frame PNG to .tmp/readback_<ts>/
    if keys.just_pressed(KeyCode::F5) {
        // Ensure readback image exists immediately so the render-side system can copy from it.
        if readback.0 == Handle::default() {
            let mut image = Image::new_fill(
                Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                &[0, 0, 0, 0],
                TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
            );
            image.texture_descriptor.usage = bevy::render::render_resource::TextureUsages::RENDER_ATTACHMENT
                | bevy::render::render_resource::TextureUsages::COPY_SRC
                | bevy::render::render_resource::TextureUsages::TEXTURE_BINDING;
            let handle = images.add(image);
            readback.0 = handle;
        }
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let dir = format!(".tmp/readback_{}", ts);
        let _ = std::fs::create_dir_all(&dir);
        unsafe { std::env::set_var("MCBAISE_READBACK_OUTPUT", &dir); }
        unsafe { std::env::set_var("MCBAISE_READBACK_ONESHOT", "1"); }
        READBACK_REQUEST_SEQ.fetch_add(1, Ordering::SeqCst);
        eprintln!("native: requested one-shot readback -> {}", dir);
            #[cfg(feature = "capture_ui")]
            if let Some(cs) = capture_state_opt.as_deref_mut() {
                cs.visible = true;
                // Ensure the UI polls the newly-created output dir immediately and
                // don't treat files from this dir as already loaded (in case we
                // recorded here previously). This forces fresh registration.
                cs.last_dir = Some(dir.clone());
                cs.loaded.retain(|k, _| !k.starts_with(&dir));
                cs.order.retain(|k| !k.starts_with(&dir));
            }

            // Also clear Egui's loaded set/order so they refresh
            #[cfg(feature = "capture_ui")]
            if let Some(cs) = capture_state_opt.as_deref_mut() {
                cs.last_dir = Some(dir.clone());
                cs.loaded.retain(|k, _| !k.starts_with(&dir));
                cs.order.retain(|k| !k.starts_with(&dir));
            }
    }

    // GIF recording toggle: press `G` to start/stop recording. Frames are written
    // to `.tmp/readback_<ts>/frame_*.png`. On stop, frames are assembled into
    // `.tmp/readback_<ts>/animation.gif` in a background thread.
    if keys.just_pressed(KeyCode::F6) {
        match std::env::var("MCBAISE_GIF_RECORD").as_deref().ok() {
            Some("1") => {
                // stop recording
                unsafe { std::env::remove_var("MCBAISE_GIF_RECORD"); }
                if let Ok(dir) = std::env::var("MCBAISE_READBACK_OUTPUT") {
                    // clear output env var so mapping won't keep writing
                    unsafe { std::env::remove_var("MCBAISE_READBACK_OUTPUT"); }
                    eprintln!("native: stopping GIF recording, assembling {}", dir);
                    // assemble GIF in background
                    std::thread::spawn(move || {
                        if let Err(e) = assemble_gif_from_dir(&dir) {
                            eprintln!("failed to assemble gif: {}", e);
                        }
                    });
                }
            }
            _ => {
                // start recording
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let dir = format!(".tmp/readback_{}", ts);
                let _ = std::fs::create_dir_all(&dir);
                unsafe { std::env::set_var("MCBAISE_READBACK_OUTPUT", &dir); }
                unsafe { std::env::set_var("MCBAISE_GIF_RECORD", "1"); }
                eprintln!("native: started GIF recording -> {}", dir);
                #[cfg(feature = "capture_ui")]
                if let Some(cs) = capture_state_opt.as_deref_mut() {
                    cs.visible = true;
                }
            }
        }
    }
}

fn assemble_gif_from_dir(dir: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::fs::{read_dir, File};
    use std::path::Path;

    let mut paths = read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("png")).unwrap_or(false))
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err("no PNG frames found".into());
    }

    let first = image::open(&paths[0])?.to_rgba8();
    let width = first.width();
    let height = first.height();

    let out_path = Path::new(dir).join("animation.gif");
    let tmp_path = Path::new(dir).join("animation.gif.tmp");

    let progress_path = Path::new(dir).join(".progress");
    // Write to a temp file first, then rename into place to avoid other readers
    // seeing a partially-written GIF.
    let result = (|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let fout = File::create(&tmp_path)?;
        let mut encoder = image::codecs::gif::GifEncoder::new(fout);
        // infinite loop
        let _ = encoder.set_repeat(image::codecs::gif::Repeat::Infinite);

        let total = paths.len();
        for (idx, p) in paths.iter().enumerate() {
            let img = image::open(p)?.to_rgba8();
            let raw = img.into_raw();
            let rgba = image::RgbaImage::from_raw(width, height, raw)
                .ok_or("frame size mismatch")?;
            // Provide a modest default delay (100ms) to make the resulting GIF playable
            // even if frame timing isn't available. Use `from_parts` to provide the
            // delay since the `delay` field is private.
            let delay = image::Delay::from_numer_denom_ms(100, 1);
            let frame = image::Frame::from_parts(rgba, 0, 0, delay);
            encoder.encode_frame(frame)?;
            
            // Write progress and check for cancellation marker
            let progress = ((idx + 1) * 100) / total;
            let _ = std::fs::write(&progress_path, progress.to_string());

            // If a .cancel marker appears in the dir, abort assembly and
            // clean up the temp files (best-effort).
            if Path::new(dir).join(".cancel").exists() {
                let _ = std::fs::remove_file(&tmp_path);
                let _ = std::fs::remove_file(&progress_path);
                return Err("assembly cancelled".into());
            }
        }
        Ok(())
    })();

    let _ = std::fs::remove_file(&progress_path);

    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }

    std::fs::rename(&tmp_path, &out_path)?;
    eprintln!("native: wrote GIF {}", out_path.display());
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn apply_js_input(
    time: Res<Time>,
    mut playback: ResMut<Playback>,
    mut settings: ResMut<TubeSettings>,
    mut scheme_mode: ResMut<ColorSchemeMode>,
    mut pattern_mode: ResMut<TexturePatternMode>,
    mut overlay_vis: ResMut<OverlayVisibility>,
) {
    JS_INPUT.with(|s| {
        let mut st = s.borrow_mut();

        st.sample_age_sec += time.delta_secs().min(0.1);

        if st.has_time {
            playback.time_sec = st.time_sec;
            playback.playing = st.playing;
            st.has_time = false;
            st.sample_age_sec = 0.0;
        } else if playback.playing && st.sample_age_sec > 0.35 {
            // If the YouTube sampling loop stalls (rAF throttling / transient JS hiccups),
            // keep the simulation timeline moving until the next sample arrives.
            playback.time_sec += time.delta_secs();
        }
        if st.toggle_scheme {
            settings.scheme = (settings.scheme + 1) % 4;
            *scheme_mode = ColorSchemeMode::from_value(settings.scheme);
            st.toggle_scheme = false;
        }
        if st.toggle_texture {
            settings.pattern = (settings.pattern + 1) % 6;
            *pattern_mode = TexturePatternMode::from_value(settings.pattern);
            st.toggle_texture = false;
        }
        if st.speed_delta != 0 {
            // +/- 0.25 per click, clamped.
            playback.speed = (playback.speed + st.speed_delta as f32 * 0.25).clamp(0.25, 3.0);
            st.speed_delta = 0;
        }

        if st.toggle_overlay {
            overlay_vis.show = !overlay_vis.show;
            st.toggle_overlay = false;
        }
    });
}

#[derive(SystemParam)]
struct TubeUpdateParams<'w> {
    playback: Res<'w, Playback>,
    time: Res<'w, Time>,
    settings: ResMut<'w, TubeSettings>,
    scheme_mode: Res<'w, ColorSchemeMode>,
    pattern_mode: Res<'w, TexturePatternMode>,
    auto_style: ResMut<'w, AutoTubeStyleState>,
    tube_scene: Res<'w, TubeScene>,
    tube_materials: ResMut<'w, Assets<TubeMaterial>>,
    dyns: ResMut<'w, SubjectDynamics>,
    subject_mode: Res<'w, SubjectMode>,
    pose_mode: Res<'w, PoseMode>,
    camera_preset: Res<'w, CameraPreset>,
    auto_pose: ResMut<'w, AutoPoseState>,
    auto_cam: ResMut<'w, AutoCameraState>,
    auto_subject: ResMut<'w, AutoSubjectState>,
}

#[allow(clippy::type_complexity)]
fn update_tube_and_subject(
    TubeUpdateParams {
        playback,
        time,
        mut settings,
        scheme_mode,
        pattern_mode,
        mut auto_style,
        tube_scene,
        mut tube_materials,
        mut dyns,
        subject_mode,
        pose_mode,
        camera_preset,
        mut auto_pose,
        mut auto_cam,
        mut auto_subject,
    }: TubeUpdateParams,
    mut subject: Query<
        &mut Transform,
        (
            With<SubjectTag>,
            Without<MainCamera>,
            Without<SubjectLightTag>,
            Without<BallTag>,
        ),
    >,
    mut subject_light: Query<
        &mut Transform,
        (
            With<SubjectLightTag>,
            Without<SubjectTag>,
            Without<MainCamera>,
            Without<BallTag>,
        ),
    >,
    mut ball: Query<
        &mut Transform,
        (
            With<BallTag>,
            Without<SubjectTag>,
            Without<MainCamera>,
            Without<SubjectLightTag>,
        ),
    >,
    mut cam: Query<
        &mut Transform,
        (
            With<ViewCamera>,
            Without<SubjectTag>,
            Without<SubjectLightTag>,
            Without<BallTag>,
        ),
    >,
) {
    let t = playback.time_sec;

    // Keep smoothing behavior consistent across frame rates.
    let dt = time.delta_secs().min(0.1);
    let pos_alpha = 1.0 - (-12.0 * dt).exp();
    let rot_alpha = 1.0 - (-12.0 * dt).exp();

    // Resolve tube style (colors/texture) from explicit / auto timeline / random.
    let scheme = match *scheme_mode {
        ColorSchemeMode::Auto => timeline_color_scheme(t),
        ColorSchemeMode::Random => {
            auto_style.scheme_since_switch_sec += dt;
            if auto_style.scheme_since_switch_sec >= auto_style.scheme_next_switch_sec {
                auto_style.pick_next_scheme();
                auto_style.schedule_next_scheme_switch();
            }
            auto_style.scheme_current % 4
        }
        ColorSchemeMode::OrangeWhite => 0,
        ColorSchemeMode::Nin => 1,
        ColorSchemeMode::BlackWhite => 2,
        ColorSchemeMode::RandomGrey => 3,
        ColorSchemeMode::Blue => 4,
        ColorSchemeMode::Dynamic => 5,
        ColorSchemeMode::Fluid => 6,
        ColorSchemeMode::Sun => 7,
        ColorSchemeMode::Psychedelic => 8,
        ColorSchemeMode::Neon => 9,
        ColorSchemeMode::Matrix => 10,
    };

    let pattern = match *pattern_mode {
        TexturePatternMode::Auto => {
            // For the first 3 minutes, only use wireframe textures and switch quickly.
            if t < 180.0 {
                if auto_style.auto_wire_next_switch_sec == 0.0 {
                    auto_style.schedule_next_auto_wire_pattern_switch();
                }
                if auto_style.pattern_current % 4 < 2 {
                    auto_style.pick_next_wire_pattern();
                    auto_style.schedule_next_auto_wire_pattern_switch();
                }

                auto_style.auto_wire_since_switch_sec += dt;
                if auto_style.auto_wire_since_switch_sec >= auto_style.auto_wire_next_switch_sec {
                    auto_style.pick_next_wire_pattern();
                    auto_style.schedule_next_auto_wire_pattern_switch();
                }

                auto_style.pattern_current % 6
            } else {
                timeline_texture_pattern(t)
            }
        }
        TexturePatternMode::Random => {
            auto_style.pattern_since_switch_sec += dt;
            if auto_style.pattern_since_switch_sec >= auto_style.pattern_next_switch_sec {
                auto_style.pick_next_pattern();
                auto_style.schedule_next_pattern_switch();
            }
            auto_style.pattern_current % 6
        }
        TexturePatternMode::Stripe => 0,
        TexturePatternMode::Swirl => 1,
        TexturePatternMode::StripeWire => 2,
        TexturePatternMode::SwirlWire => 3,
        TexturePatternMode::Fluid => 4,
        TexturePatternMode::FluidStripe => 5,
        TexturePatternMode::FluidSwirl => 6,
        TexturePatternMode::Wave => 7,
        TexturePatternMode::Fractal => 8,
        TexturePatternMode::Particle => 9,
        TexturePatternMode::Grid => 10,
        TexturePatternMode::HoopWire => 11,
        TexturePatternMode::HoopAlt => 12,
    };

    // If color scheme is fluid, force fluid pattern regardless of pattern mode selection
    let final_pattern = if scheme == 5 { 4 } else { pattern };

    auto_style.scheme_current = scheme;
    auto_style.pattern_current = final_pattern;
    settings.scheme = scheme;
    settings.pattern = final_pattern;

    // Resolve subject from explicit / auto timeline / random.
    match *subject_mode {
        SubjectMode::Auto => {
            auto_subject.current = timeline_subject_mode(t);
        }
        SubjectMode::Random => {
            auto_subject.since_switch_sec += dt;
            if auto_subject.since_switch_sec >= auto_subject.next_switch_sec {
                auto_subject.current = auto_subject.pick_next_subject();
                auto_subject.schedule_next_switch();
            }
        }
        SubjectMode::Human | SubjectMode::Doughnut | SubjectMode::Ball => {
            auto_subject.current = *subject_mode;
        }
    }

    let active_subject_mode = match *subject_mode {
        SubjectMode::Auto | SubjectMode::Random => auto_subject.current,
        _ => *subject_mode,
    };

    // Update tube shader params.
    if let Some(mat) = tube_materials.get_mut(&tube_scene.tube_material) {
        mat.set_time(t);
        mat.set_scheme(settings.scheme, t);
        mat.set_pattern(settings.pattern);
    }

    // Drive along curve.
    let progress = progress_from_video_time(t);

    let cam_center = tube_scene.curve.point_at(progress);
    let f = tube_scene.frames.frame_at(progress);

    let cam_tangent = f.tan;
    let cam_n = f.nor;
    let cam_b = f.bin;

    // Wrap instead of clamping to avoid a discontinuity at the loop boundary.
    let look_ahead = tube_scene
        .curve
        .point_at((progress + 0.003).rem_euclid(1.0));

    // Resolve the effective camera mode up-front so we can use it for both look-ahead and camera.
    let selected_camera_mode = match *camera_preset {
        CameraPreset::Auto => {
            auto_cam.current = timeline_camera_mode(t);
            auto_cam.current
        }
        CameraPreset::Random => {
            auto_cam.since_switch_sec += dt;
            if auto_cam.since_switch_sec >= auto_cam.next_switch_sec {
                let next_mode = auto_cam.pick_next_mode();
                auto_cam.current = next_mode;
                auto_cam.apply_mode_params(next_mode);
            }
            auto_cam.current
        }
        CameraPreset::FollowActiveFirst => CameraMode::First,
        CameraPreset::FollowActiveOver | CameraPreset::TubeOver => CameraMode::Over,
        CameraPreset::FollowActiveBack => CameraMode::Back,
        CameraPreset::FollowActiveSide => CameraMode::Side,
        CameraPreset::FollowActiveChase
        | CameraPreset::FollowHumanChase
        | CameraPreset::FollowBallChase => CameraMode::BallChase,
        CameraPreset::FollowActiveFocusedChase => CameraMode::FocusedChase,
        CameraPreset::FollowActiveFocusedSide => CameraMode::FocusedSide,
        CameraPreset::PassingLeft => CameraMode::PassingLeft,
        CameraPreset::PassingRight => CameraMode::PassingRight,
        CameraPreset::PassingTop => CameraMode::PassingTop,
        CameraPreset::PassingBottom => CameraMode::PassingBottom,
    };

    let mode_changed = selected_camera_mode != auto_cam.last_effective_mode;
    if mode_changed {
        auto_cam.mode_since_sec = 0.0;
    } else {
        auto_cam.mode_since_sec += dt;
    }

    // Passing modes need a stable camera anchor so the subject can pass through frame.
    // Anchor when entering a passing mode and periodically while it remains active,
    // so the subject repeatedly rolls into view instead of leaving us staring at a wall.
    if mode_changed {
        if selected_camera_mode.is_passing() {
            // Anchor very close to current progress so the shot reads immediately.
            let lead_u = 0.010;
            auto_cam.pass_anchor_progress = (progress + lead_u).rem_euclid(1.0);
            auto_cam.pass_anchor_valid = true;
            auto_cam.pass_reanchor_since_sec = 0.0;
        } else {
            auto_cam.pass_anchor_valid = false;
        }
        auto_cam.last_effective_mode = selected_camera_mode;
    }

    if selected_camera_mode.is_passing() {
        auto_cam.pass_reanchor_since_sec += dt;
        // Roughly once every few seconds, re-anchor to the current progress.
        if auto_cam.pass_reanchor_since_sec >= 3.0 {
            let lead_u = 0.010;
            auto_cam.pass_anchor_progress = (progress + lead_u).rem_euclid(1.0);
            auto_cam.pass_anchor_valid = true;
            auto_cam.pass_reanchor_since_sec = 0.0;
        }
    }

    // Subject position on wall.
    let ball_ahead = if selected_camera_mode == CameraMode::BallChase {
        0.020
    } else {
        0.010
    };
    // Wrap instead of clamping to avoid a discontinuity at the loop boundary.
    let s = (progress + ball_ahead).rem_euclid(1.0);
    let center = tube_scene.curve.point_at(s);
    let bf = tube_scene.frames.frame_at(s);

    // Tube-surface dynamics for motion around the tube wall.
    // Treat the subject like a bead sliding on a ring (the tube cross-section), driven by
    // an effective acceleration = gravity - (curve inertial acceleration).
    let curve_param_speed = 0.0028; // must match progress_from_video_time
    let ds = (curve_param_speed * dt).clamp(0.0, 0.01);
    let center_prev = tube_scene.curve.point_at((s - ds).rem_euclid(1.0));
    let center_next = tube_scene.curve.point_at((s + ds).rem_euclid(1.0));
    let v_center = (center_next - center_prev) / (2.0 * dt.max(1e-4));
    let a_center = (center_next - center * 2.0 + center_prev) / (dt.max(1e-4) * dt.max(1e-4));

    // Time-varying gravity: blend real world-down with a rotating pull direction in the
    // tube cross-section frame so the rider/ball doesn't feel "stuck at the top".
    // This is intentionally artistic (not physically correct), but it creates the desired
    // swirling + pose/collision variety.
    let world_down = Vec3::new(0.0, -1.0, 0.0);
    let swirl_rate = 2.2; // rad/s
    let swirl_blend = 0.90; // 0..1 (higher = more swirl)
    let phase = t * swirl_rate;
    // Pull direction rotates around the tube axis (lies in the normal/binormal plane).
    let swirl_pull_dir = -(bf.nor * phase.cos() + bf.bin * phase.sin()).normalize_or_zero();

    let mut g_dir = world_down * (1.0 - swirl_blend) + swirl_pull_dir * swirl_blend;
    if g_dir.length_squared() < 1e-6 {
        g_dir = world_down;
    }
    let gravity_acc = g_dir.normalize_or_zero() * GRAVITY;
    let eff_acc = gravity_acc - a_center;

    let g_n = eff_acc.dot(bf.nor) / GRAVITY;
    let g_b = eff_acc.dot(bf.bin) / GRAVITY;
    let g_mag = (g_n * g_n + g_b * g_b).sqrt();
    let theta_eq = g_b.atan2(g_n);

    if !dyns.initialized {
        dyns.theta = theta_eq;
        dyns.omega = 0.0;
        dyns.initialized = true;
    }

    let r_ring = match active_subject_mode {
        SubjectMode::Human => dyns.human_r.max(0.25),
        SubjectMode::Doughnut => dyns.human_r.max(0.25),
        SubjectMode::Ball => dyns.ball_r.max(0.25),
        SubjectMode::Auto | SubjectMode::Random => dyns.human_r.max(0.25),
    };
    let theta_rel = (dyns.theta - theta_eq + std::f32::consts::PI)
        .rem_euclid(std::f32::consts::TAU)
        - std::f32::consts::PI;

    // theta¨ = -(g/r) * sin(theta - theta_eq) - damping * theta˙
    // Damping acts like friction along the tube wall.
    let damping = match active_subject_mode {
        SubjectMode::Human => 0.55,
        SubjectMode::Doughnut => 0.55,
        SubjectMode::Ball => 0.35,
        SubjectMode::Auto | SubjectMode::Random => 0.55,
    };
    // When the tangent is close to vertical, g_mag is small; add a small floor so it still
    // responds during fast twists/loops (visual intent > strict physics).
    let g_mag_eff = (g_mag + 0.12).min(4.0);
    let theta_dd = -(GRAVITY * g_mag_eff / r_ring) * theta_rel.sin() - damping * dyns.omega;
    dyns.omega += theta_dd * dt;
    dyns.theta = (dyns.theta + dyns.omega * dt).rem_euclid(std::f32::consts::TAU);

    let theta = dyns.theta;
    let offset = bf.nor * theta.cos() + bf.bin * theta.sin();
    // Tangent direction around the tube wall (d/dθ of offset).
    let tube_wall_tangent = (-bf.nor * theta.sin() + bf.bin * theta.cos()).normalize_or_zero();

    // Tube-wall contact + collision (radial) for both human and ball.
    // Allow slight penetration (via spring), then bounce on collision.
    // Use effective acceleration (gravity - inertial) so loops push/pull on the contact.
    let a_out = eff_acc.dot(offset);

    let human_r_max = (TUBE_RADIUS - HUMAN_RADIUS).max(0.25);
    let human_r_rest = (human_r_max - CONTACT_EPS).max(0.25);
    let k_h = 90.0;
    let c_h = 14.0;

    let human_r_prev = dyns.human_r;
    let human_vr_prev = dyns.human_vr;
    dyns.human_vr += (a_out - k_h * (dyns.human_r - human_r_rest) - c_h * dyns.human_vr) * dt;
    dyns.human_r += dyns.human_vr * dt;
    let mut human_hit_wall = false;
    if dyns.human_r > human_r_max {
        dyns.human_r = human_r_max;
        if dyns.human_vr > 0.0 {
            dyns.human_vr = -dyns.human_vr * 0.25;
            human_hit_wall = true;
        }
    }
    dyns.human_r = dyns.human_r.max(human_r_rest - 0.35);
    // Also treat "slamming" into the limit as a hit even if the spring integration already
    // flipped velocity before the clamp.
    if !human_hit_wall {
        let near_wall = dyns.human_r >= human_r_max - 0.0005;
        let vr_flip = human_vr_prev > 0.6 && dyns.human_vr < -0.1;
        let r_advanced = dyns.human_r > human_r_prev;
        human_hit_wall = near_wall && vr_flip && r_advanced;
    }

    let ball_r_max = (TUBE_RADIUS - BALL_RADIUS).max(0.25);
    let ball_r_rest = (ball_r_max - CONTACT_EPS).max(0.25);
    let k_b = 60.0;
    let c_b = 9.0;
    dyns.ball_vr += (a_out - k_b * (dyns.ball_r - ball_r_rest) - c_b * dyns.ball_vr) * dt;
    dyns.ball_r += dyns.ball_vr * dt;
    if dyns.ball_r > ball_r_max {
        dyns.ball_r = ball_r_max;
        if dyns.ball_vr > 0.0 {
            dyns.ball_vr = -dyns.ball_vr * 0.55;
        }
    }
    dyns.ball_r = dyns.ball_r.max(ball_r_rest - 0.35);

    let subject_pos_human = center + offset * dyns.human_r;
    let subject_pos_ball = center + offset * dyns.ball_r;

    // The burn_human mesh is authored in a different basis than Bevy's default.
    // We apply the same corrective rotation used at spawn-time so the subject
    // doesn't end up edge-on ("flat") depending on where we are along the tube.
    let model_basis = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);

    let subject_up = offset.normalize_or_zero();
    // Orientation should follow motion on the tube wall.
    // Use the velocity computed above (curve advance) plus wall sliding.

    let active_r = match active_subject_mode {
        SubjectMode::Human => dyns.human_r,
        SubjectMode::Doughnut => dyns.human_r,
        SubjectMode::Ball => dyns.ball_r,
        SubjectMode::Auto | SubjectMode::Random => dyns.human_r,
    };
    let v_wall = v_center + tube_wall_tangent * (dyns.omega * active_r);

    // Use a tube-frame forward as the primary orientation anchor.
    // Basing forward purely on instantaneous velocity makes the model/ball appear to
    // "swim"/spin relative to the tube and camera when omega/velocity changes sign.
    let tube_forward = (bf.tan - subject_up * bf.tan.dot(subject_up)).normalize_or_zero();
    let vel_forward = (v_wall - subject_up * v_wall.dot(subject_up)).normalize_or_zero();

    let mut subject_forward = tube_forward;
    if subject_forward.length_squared() < 1e-6 {
        subject_forward = tube_wall_tangent;
    }
    // Allow a small bias toward the actual motion direction when it's well-defined.
    if vel_forward.length_squared() > 1e-6 {
        subject_forward = tube_forward.lerp(vel_forward, 0.25).normalize_or_zero();
        if subject_forward.length_squared() < 1e-6 {
            subject_forward = vel_forward;
        }
    }

    // Right/around direction on the surface.
    let subject_around = subject_up.cross(subject_forward).normalize_or_zero();
    // Re-orthonormalize.
    subject_forward = subject_around.cross(subject_up).normalize_or_zero();

    // Shared target orientation basis (surface frame).
    let base_rot = Quat::from_mat3(&Mat3::from_cols(
        subject_around,
        subject_up,
        subject_forward,
    )) * model_basis;

    // Bank into the sideways motion around the tube (visual "surfing" cue).
    let bank = (dyns.omega * 0.22).clamp(-0.55, 0.55);
    let desired_rot = Quat::from_axis_angle(subject_forward, bank) * base_rot;

    // Human posture dynamics: allow the body to roll/pitch relative to the surface.
    // This avoids the "always orthographic/upright" look (enables belly/back/sideways).
    // Use effective acceleration components to drive posture, then smooth with critical damping.
    match *pose_mode {
        // Timeline pose: deterministic (not random).
        PoseMode::Auto => {
            auto_pose.current = timeline_pose_mode(t);
        }
        // Random pose controller:
        // - randomly changes pose over time
        // - on tube-wall impact, pick a pose based on which side hit (in model-local frame)
        PoseMode::Random => {
            auto_pose.since_switch_sec += dt;
            auto_pose.collision_cooldown_sec = (auto_pose.collision_cooldown_sec - dt).max(0.0);

            if human_hit_wall && auto_pose.collision_cooldown_sec <= 0.0 {
                // "Hit direction" is outward from the tube center.
                let world_hit_dir = subject_up;
                // Convert to model-local using the current desired rotation (includes model basis).
                let local_hit_dir = desired_rot.inverse() * world_hit_dir;
                auto_pose.current = pose_from_local_hit_dir(local_hit_dir);
                auto_pose.schedule_next_switch();
                auto_pose.collision_cooldown_sec = 0.45;
            } else if auto_pose.since_switch_sec >= auto_pose.next_switch_sec {
                auto_pose.current = auto_pose.pick_random_pose();
                auto_pose.schedule_next_switch();
            }
        }
        _ => {}
    }

    let (posture_roll_target, posture_pitch_target) = match *pose_mode {
        PoseMode::Auto | PoseMode::Random => pose_targets(auto_pose.current),
        other => pose_targets(other),
    };

    let posture_rate = if matches!(*pose_mode, PoseMode::Auto | PoseMode::Random) {
        16.0
    } else {
        14.0
    };
    let posture_alpha = 1.0 - (-posture_rate * dt).exp();
    dyns.human_roll = dyns
        .human_roll
        .lerp(posture_roll_target, posture_alpha)
        .clamp(-2.2, 2.2);
    dyns.human_pitch = dyns
        .human_pitch
        .lerp(posture_pitch_target, posture_alpha)
        .clamp(-1.2, 1.2);

    // Roll about the direction of travel, pitch about the across-surface axis.
    let human_posture = Quat::from_axis_angle(subject_forward, dyns.human_roll)
        * Quat::from_axis_angle(subject_around, dyns.human_pitch);
    // Mesh authored facing doesn't match our computed surface-forward; flip it so we
    // don't end up watching the character's back the whole ride.
    let human_facing_fix = Quat::from_axis_angle(subject_up, std::f32::consts::PI);
    let desired_human_rot = human_facing_fix * (human_posture * desired_rot);

    // Ball rolling: integrate roll based on surface speed.
    let v_ball = v_center + tube_wall_tangent * (dyns.omega * dyns.ball_r);
    let v_ball_plane = v_ball - subject_up * v_ball.dot(subject_up);
    let mut ball_roll_axis = subject_around;
    if v_ball_plane.length_squared() > 1e-6 {
        let tangent_dir = v_ball_plane.normalize_or_zero();
        let axis = subject_up.cross(tangent_dir).normalize_or_zero();
        let sign = if axis.dot(subject_around) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let roll_rate = (v_ball_plane.length() / BALL_RADIUS) * sign;
        dyns.ball_roll = (dyns.ball_roll + roll_rate * dt).rem_euclid(std::f32::consts::TAU);
        if axis.length_squared() > 1e-6 {
            ball_roll_axis = axis;
        }
    }
    let desired_ball_rot = Quat::from_axis_angle(ball_roll_axis, dyns.ball_roll) * desired_rot;

    if let Ok(mut tr) = subject.single_mut() {
        let mut desired = desired_human_rot;
        if tr.rotation.dot(desired) < 0.0 {
            desired = -desired;
        }

        tr.translation = tr.translation.lerp(subject_pos_human, pos_alpha);
        tr.rotation = tr.rotation.slerp(desired, rot_alpha);
    }

    if let Ok(mut tr) = ball.single_mut() {
        let mut desired = desired_ball_rot;
        if tr.rotation.dot(desired) < 0.0 {
            desired = -desired;
        }
        tr.translation = tr.translation.lerp(subject_pos_ball, pos_alpha);
        tr.rotation = tr.rotation.slerp(desired, rot_alpha);
    }

    if let Ok(mut light_tr) = subject_light.single_mut() {
        let active_pos = match active_subject_mode {
            SubjectMode::Human => subject_pos_human,
            SubjectMode::Doughnut => subject_pos_human,
            SubjectMode::Ball => subject_pos_ball,
            SubjectMode::Auto | SubjectMode::Random => subject_pos_human,
        };
        let target = active_pos + subject_up * 0.9 - subject_forward * 0.6;
        light_tr.translation = light_tr.translation.lerp(target, pos_alpha);
    }

    let random_subject_distance = if *camera_preset == CameraPreset::Random {
        Some(auto_cam.subject_distance)
    } else {
        None
    };

    let pass_anchor = if selected_camera_mode.is_passing() && auto_cam.pass_anchor_valid {
        let u = auto_cam.pass_anchor_progress;
        let c = tube_scene.curve.point_at(u);
        let fr = tube_scene.frames.frame_at(u);
        Some((c, fr.tan, fr.nor, fr.bin))
    } else {
        None
    };

    let (pos, look, up) = camera_pose(
        t,
        *camera_preset,
        selected_camera_mode,
        active_subject_mode,
        auto_cam.mode_since_sec,
        random_subject_distance,
        pass_anchor,
        cam_center,
        look_ahead,
        cam_tangent,
        cam_n,
        cam_b,
        center,
        subject_pos_human,
        subject_pos_ball,
        subject_forward,
        subject_up,
    );

    let desired = Transform::from_translation(pos).looking_at(look, up);

    for mut cam_tr in &mut cam {
        let mut desired_rot = desired.rotation;
        if cam_tr.rotation.dot(desired_rot) < 0.0 {
            desired_rot = -desired_rot;
        }

        cam_tr.translation = cam_tr.translation.lerp(desired.translation, pos_alpha);
        cam_tr.rotation = cam_tr.rotation.slerp(desired_rot, rot_alpha);
    }
}

fn apply_subject_mode(
    subject_mode: Res<SubjectMode>,
    auto_subject: Res<AutoSubjectState>,
    mut human_vis: Query<&mut Visibility, (With<SubjectTag>, Without<BallTag>)>,
    mut ball_vis: Query<&mut Visibility, (With<BallTag>, Without<SubjectTag>)>,
) {
    if !subject_mode.is_changed() && !auto_subject.is_changed() {
        return;
    }

    let active = match *subject_mode {
        SubjectMode::Auto | SubjectMode::Random => auto_subject.current,
        _ => *subject_mode,
    };

    let (show_human, show_ball) = match active {
        SubjectMode::Human => (true, false),
        SubjectMode::Doughnut => (true, false),
        SubjectMode::Ball => (false, true),
        SubjectMode::Auto | SubjectMode::Random => (true, false),
    };

    for mut v in &mut human_vis {
        *v = if show_human {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }

    for mut v in &mut ball_vis {
        *v = if show_ball {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

fn make_polkadot_texture() -> Image {
    let size: u32 = 256;
    let mut data = vec![0u8; (size * size * 4) as usize];

    let spacing = 64.0_f32;
    let radius = 18.0_f32;
    let r2 = radius * radius;

    for y in 0..size {
        for x in 0..size {
            let fx = x as f32 + 0.5;
            let fy = y as f32 + 0.5;

            // Nearest grid center.
            let gx = (fx / spacing).round();
            let gy = (fy / spacing).round();
            let cx = gx * spacing;
            let cy = gy * spacing;
            let dx = fx - cx;
            let dy = fy - cy;

            let is_dot = dx * dx + dy * dy <= r2;
            let (r, g, b) = if is_dot {
                (40u8, 140u8, 255u8)
            } else {
                (255u8, 255u8, 255u8)
            };

            let i = ((y * size + x) * 4) as usize;
            data[i] = r;
            data[i + 1] = g;
            data[i + 2] = b;
            data[i + 3] = 255;
        }
    }

    let mut img = Image::new_fill(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    let _ = match &mut img.texture_descriptor.label {
        Some(_) => {
            img.texture_descriptor.label = Some("polkadot_texture");
            true
        }
        None => false,
    };
    img
}

#[allow(dead_code)]
fn apply_appearance_preset(
    mat: &mut StandardMaterial,
    preset: AppearancePreset,
    polkadot: &Handle<Image>,
) {
    mat.metallic = 0.0;
    mat.cull_mode = None;
    mat.base_color_texture = None;

    match preset {
        AppearancePreset::BlueGlass => {
            mat.base_color = Color::srgb(0.45, 0.82, 1.0).with_alpha(0.35);
            mat.reflectance = 0.95;
            mat.perceptual_roughness = 0.04;
            mat.alpha_mode = AlphaMode::Blend;
        }
        AppearancePreset::OpaqueWhite => {
            mat.base_color = Color::srgb(0.95, 0.95, 0.95);
            mat.reflectance = 0.45;
            mat.perceptual_roughness = 0.65;
            mat.alpha_mode = AlphaMode::Opaque;
        }
        AppearancePreset::Blue => {
            mat.base_color = Color::srgb(0.12, 0.38, 1.0);
            mat.reflectance = 0.60;
            mat.perceptual_roughness = 0.35;
            mat.alpha_mode = AlphaMode::Opaque;
        }
        AppearancePreset::Polkadot => {
            mat.base_color = Color::srgb(1.0, 1.0, 1.0);
            mat.base_color_texture = Some(polkadot.clone());
            mat.reflectance = 0.40;
            mat.perceptual_roughness = 0.60;
            mat.alpha_mode = AlphaMode::Opaque;
        }
        AppearancePreset::MatteLightBlue => {
            mat.base_color = Color::srgb(0.58, 0.80, 0.98);
            mat.reflectance = 0.35;
            mat.perceptual_roughness = 0.85;
            mat.alpha_mode = AlphaMode::Opaque;
        }
        AppearancePreset::Wireframe => {
            // Actual wireframe rendering is controlled via `Wireframe` component;
            // keep a stable base color here.
            mat.base_color = Color::srgb(0.95, 0.95, 0.95);
            mat.reflectance = 0.40;
            mat.perceptual_roughness = 0.65;
            mat.alpha_mode = AlphaMode::Opaque;
        }
    }
}

#[allow(dead_code)]
fn timeline_human_appearance(video_time_sec: f32) -> AppearancePreset {
    // Deterministic and simple.
    let cycle = 14.0;
    let u = video_time_sec.rem_euclid(cycle);
    if u < 3.5 {
        AppearancePreset::OpaqueWhite
    } else if u < 7.0 {
        AppearancePreset::MatteLightBlue
    } else if u < 10.5 {
        AppearancePreset::Polkadot
    } else if u < 11.0 {
        AppearancePreset::Blue
    } else {
        AppearancePreset::BlueGlass
    }
}

#[allow(dead_code)]
fn timeline_ball_appearance(video_time_sec: f32) -> AppearancePreset {
    // Keep the ball mostly "blue glass".
    let cycle = 14.0;
    let u = video_time_sec.rem_euclid(cycle);
    if u > 11.0 {
        AppearancePreset::BlueGlass
    } else if u > 7.0 {
        AppearancePreset::Polkadot
    } else {
        AppearancePreset::BlueGlass
    }
}

#[allow(dead_code, clippy::too_many_arguments, clippy::type_complexity)]
fn update_subject_appearance(
    mut commands: Commands,
    playback: Res<Playback>,
    time: Res<Time>,
    textures: Res<AppearanceTextures>,
    human_mode: Res<HumanAppearanceMode>,
    ball_mode: Res<BallAppearanceMode>,
    mut auto_human: ResMut<AutoHumanAppearanceState>,
    mut auto_ball: ResMut<AutoBallAppearanceState>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    human_mat: Query<
        (Entity, &MeshMaterial3d<StandardMaterial>),
        (With<SubjectTag>, Without<BallTag>),
    >,
    ball_mat: Query<(Entity, &MeshMaterial3d<StandardMaterial>), With<BallTag>>,
) {
    let t = playback.time_sec;
    let dt = time.delta_secs().min(0.1);

    if human_mode.is_changed() && matches!(human_mode.0, AppearanceMode::Random) {
        auto_human.0.schedule_next_switch();
    }
    if ball_mode.is_changed() && matches!(ball_mode.0, AppearanceMode::Random) {
        auto_ball.0.schedule_next_switch();
    }

    let human_preset = match human_mode.0 {
        AppearanceMode::Auto => {
            auto_human.0.current = timeline_human_appearance(t);
            auto_human.0.current
        }
        AppearanceMode::Random => {
            auto_human.0.since_switch_sec += dt;
            if auto_human.0.since_switch_sec >= auto_human.0.next_switch_sec {
                auto_human.0.current = auto_human.0.pick_next_preset();
                auto_human.0.schedule_next_switch();
            }
            auto_human.0.current
        }
        other => {
            let p = other.preset().unwrap_or(AppearancePreset::OpaqueWhite);
            auto_human.0.current = p;
            p
        }
    };

    let ball_preset = match ball_mode.0 {
        AppearanceMode::Auto => {
            auto_ball.0.current = timeline_ball_appearance(t);
            auto_ball.0.current
        }
        AppearanceMode::Random => {
            auto_ball.0.since_switch_sec += dt;
            if auto_ball.0.since_switch_sec >= auto_ball.0.next_switch_sec {
                auto_ball.0.current = auto_ball.0.pick_next_preset();
                auto_ball.0.schedule_next_switch();
            }
            auto_ball.0.current
        }
        other => {
            let p = other.preset().unwrap_or(AppearancePreset::BlueGlass);
            auto_ball.0.current = p;
            p
        }
    };

    if let Some((e, h)) = human_mat.iter().next() {
        if let Some(mat) = materials.get_mut(&h.0) {
            apply_appearance_preset(mat, human_preset, &textures.polkadot);
        }

        if human_preset == AppearancePreset::Wireframe {
            commands.entity(e).insert((
                bevy::pbr::wireframe::Wireframe,
                bevy::pbr::wireframe::WireframeColor {
                    color: Color::srgb(0.95, 0.95, 0.95),
                },
            ));
        } else {
            commands.entity(e).remove::<(
                bevy::pbr::wireframe::Wireframe,
                bevy::pbr::wireframe::WireframeColor,
            )>();
        }
    }

    if let Some((e, h)) = ball_mat.iter().next() {
        if let Some(mat) = materials.get_mut(&h.0) {
            apply_appearance_preset(mat, ball_preset, &textures.polkadot);
        }

        if ball_preset == AppearancePreset::Wireframe {
            commands.entity(e).insert((
                bevy::pbr::wireframe::Wireframe,
                bevy::pbr::wireframe::WireframeColor {
                    color: Color::srgb(0.20, 0.60, 1.0),
                },
            ));
        } else {
            commands.entity(e).remove::<(
                bevy::pbr::wireframe::Wireframe,
                bevy::pbr::wireframe::WireframeColor,
            )>();
        }
    }
}

fn update_overlays(
    playback: Res<Playback>,
    mut state: ResMut<OverlayState>,
    mut overlay_text: ResMut<OverlayText>,
    overlay_vis: Res<OverlayVisibility>,
    caption_vis: Res<CaptionVisibility>,
    #[cfg(all(not(target_arch = "wasm32"), feature = "native-youtube"))] native_youtube: Option<
        Res<NativeYoutubeSync>,
    >,
    #[cfg(not(target_arch = "wasm32"))] mut window: Query<&mut Window, With<PrimaryWindow>>,
) {
    let t = playback.time_sec;

    let visible_changed = state.last_visible != overlay_vis.show;
    if visible_changed {
        state.last_visible = overlay_vis.show;
    }

    let caption_visible_changed = state.last_caption_visible != caption_vis.show;
    if caption_visible_changed {
        state.last_caption_visible = caption_vis.show;
    }

    let c_idx = find_opening_credit(t);
    let credit_changed = c_idx != state.last_credit_idx;
    if credit_changed {
        state.last_credit_idx = c_idx;
        let credit_overlay = if c_idx < 0 {
            ""
        } else {
            opening_credit_overlay(c_idx as usize)
        };

        overlay_text.credit = credit_overlay.to_string();

        #[cfg(not(target_arch = "wasm32"))]
        {
            let credit_plain = if c_idx < 0 {
                ""
            } else {
                opening_credit_plain(c_idx as usize)
            };

            if !credit_plain.is_empty() {
                println!("[credit] {credit_plain}");
            }

            if let Ok(mut w) = window.single_mut() {
                if credit_plain.is_empty() {
                    w.title = format!("{VIDEO_ID} • tube ride");
                } else {
                    w.title = format!("{VIDEO_ID} • tube ride — {credit_plain}");
                }
            }
        }
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "native-youtube"))]
    let waiting_for_play = native_youtube.as_ref().is_some_and(|s| {
        s.enabled && (!s.has_remote || (!playback.playing && playback.time_sec < 0.05))
    });

    #[cfg(target_arch = "wasm32")]
    let waiting_for_play = !playback.playing && playback.time_sec < 0.05;

    #[cfg(all(not(target_arch = "wasm32"), not(feature = "native-youtube")))]
    let waiting_for_play = false;

    let cues = lyric_cues();
    let l_idx = if waiting_for_play {
        -2
    } else {
        find_cue_index(cues, t)
    };
    let caption_changed = l_idx != state.last_caption_idx;
    if caption_changed {
        state.last_caption_idx = l_idx;
        if l_idx == -2 {
            overlay_text.caption = "Click Play on the YouTube video to start.".to_string();
            overlay_text.caption_is_meta = true;
        } else if l_idx < 0 {
            overlay_text.caption.clear();
            overlay_text.caption_is_meta = false;
        } else {
            let cue = &cues[l_idx as usize];
            overlay_text.caption = cue.text.clone();
            overlay_text.caption_is_meta = cue.is_meta;

            #[cfg(not(target_arch = "wasm32"))]
            {
                if cue.is_meta {
                    println!("[caption] ({})", cue.text);
                } else {
                    println!("[caption] {}", cue.text);
                }
            }
        }
    }

    // On wasm we render overlays in the DOM (see `www/index.html`).
    #[cfg(target_arch = "wasm32")]
    {
        if visible_changed || caption_visible_changed || credit_changed || caption_changed {
            // Credits are independent from the egui overlay UI.
            let credit_show = true;
            // Captions are independent from the egui overlay UI.
            let caption_show = caption_vis.show;

            if c_idx < 0 {
                mcbaise_set_credit("", false);
            } else {
                mcbaise_set_credit(opening_credit_html(c_idx as usize), credit_show);
            }

            if l_idx == -2 {
                mcbaise_set_caption(&overlay_text.caption, caption_show, true);
            } else if l_idx < 0 {
                mcbaise_set_caption("", false, false);
            } else {
                let cue = &cues[l_idx as usize];
                mcbaise_set_caption(&cue.text, caption_show, cue.is_meta);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[derive(SystemParam)]
struct UiOverlayParams<'w, 's> {
    commands: Commands<'w, 's>,
    egui_contexts: EguiContexts<'w, 's>,
    images_assets: ResMut<'w, Assets<Image>>,
    capture_state: ResMut<'w, EguiCaptureState>,
    pending_window_geometry: ResMut<'w, PendingWindowGeometry>,
    teardown: ResMut<'w, TeardownState>,
    overlay_text: Res<'w, OverlayText>,
    overlay_vis: ResMut<'w, OverlayVisibility>,
    caption_vis: ResMut<'w, CaptionVisibility>,
    #[cfg(target_arch = "wasm32")]
    video_vis: ResMut<'w, VideoVisibility>,
    multi_view: ResMut<'w, MultiView>,
    multi_view_hint: ResMut<'w, MultiViewHint>,
    playback: ResMut<'w, Playback>,
    time: Res<'w, Time>,
    settings: ResMut<'w, TubeSettings>,
    scheme_mode: ResMut<'w, ColorSchemeMode>,
    pattern_mode: ResMut<'w, TexturePatternMode>,
    subject_mode: ResMut<'w, SubjectMode>,
    human_appearance: ResMut<'w, HumanAppearanceMode>,
    ball_appearance: ResMut<'w, BallAppearanceMode>,
    pose_mode: ResMut<'w, PoseMode>,
    camera_preset: ResMut<'w, CameraPreset>,
    auto_pose: Res<'w, AutoPoseState>,
    auto_cam: Res<'w, AutoCameraState>,
    auto_subject: Res<'w, AutoSubjectState>,
    auto_style: Res<'w, AutoTubeStyleState>,
    auto_human_appearance: Res<'w, AutoHumanAppearanceState>,
    auto_ball_appearance: Res<'w, AutoBallAppearanceState>,
    overlay_state: Res<'w, OverlayState>,
    _native_youtube: Option<Res<'w, NativeYoutubeSync>>,
    _native_youtube_cfg: Option<Res<'w, NativeYoutubeConfig>>,
    _native_mpv: Option<Res<'w, NativeMpvSync>>,
    _native_mpv_cfg: Option<Res<'w, NativeMpvConfig>>,
    keys: Res<'w, ButtonInput<KeyCode>>,
    windows: Query<'w, 's, &'static mut Window, With<PrimaryWindow>>,
    pending: ResMut<'w, GpuReadbackPending>,
    loading_state: Res<'w, LoadingState>,
}

fn ui_overlay(
    UiOverlayParams {
        mut commands,
        mut egui_contexts,
        overlay_text,
        mut overlay_vis,
        mut caption_vis,
        #[cfg(target_arch = "wasm32")]
        mut video_vis,
        mut multi_view,
        mut multi_view_hint,
        mut playback,
        time,
        mut settings,
        mut scheme_mode,
        mut pattern_mode,
        mut subject_mode,
        mut human_appearance,
        mut ball_appearance,
        mut pose_mode,
        mut camera_preset,
        auto_pose,
        auto_cam,
        auto_subject,
        auto_style,
        auto_human_appearance,
        auto_ball_appearance,
        overlay_state,
        _native_youtube,
        _native_youtube_cfg,
        _native_mpv,
        _native_mpv_cfg,
    mut images_assets,
    mut capture_state,
    keys,
    mut windows,
    mut pending_window_geometry,
    mut teardown,
    pending,
    loading_state,
     }: UiOverlayParams) {
    // Handle ESC priority
    if keys.just_pressed(KeyCode::Escape) {
        if capture_state.show_info {
            capture_state.show_info = false;
        } else if capture_state.visible {
            capture_state.visible = false;
        } else if overlay_vis.show {
            overlay_vis.show = false;
        }
    }

    // Minimal resolution overlay: always-visible small widget that shows
    // the primary window physical resolution. Useful for debugging resize
    // behavior on both native and wasm without adding interactive controls.
    if let Ok(ctx) = egui_contexts.ctx_mut() {
        if let Some(mut win_iter) = windows.iter_mut().next() {
            let w = win_iter.resolution.physical_width();
            let h = win_iter.resolution.physical_height();
            egui::Window::new("resolution_overlay")
                .title_bar(false)
                .resizable(false)
                .collapsible(false)
                .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-8.0, 8.0))
                .show(ctx, |ui| {
                    ui.label(format!("Resolution: {} x {}", w, h));
                });
        }
    }

    // Scan for new captures in any .tmp/readback_* directory
    let mut scan_dirs = Vec::new();
    if let Ok(read_tmp) = std::fs::read_dir(".tmp") {
        for entry in read_tmp.filter_map(|e| e.ok()) {
            let p = entry.path();
            if p.is_dir() && p.file_name().and_then(|n| n.to_str()).map(|n| n.starts_with("readback_")).unwrap_or(false) {
                scan_dirs.push(p);
            }
        }
    }
     // Register any new PNGs with bevy assets + egui before borrowing the egui ctx
     {
         // Only poll the filesystem for previews when the capture UI is visible
         // or when GIF recording is active. This prevents loading user previews
         // while the capture UI feature is compiled out or hidden.
         let recording = std::env::var("MCBAISE_GIF_RECORD").as_deref().ok() == Some("1");
         if !capture_state.visible && !recording {
             // keep the last_polled timestamp moving so we don't accumulate a large
             // backlog when the UI is re-enabled later.
             capture_state.last_polled_secs = time.elapsed().as_secs_f64();
         } else {
             let now = time.elapsed().as_secs_f64();
             if now - capture_state.last_polled_secs > 0.3 {
                 capture_state.last_polled_secs = now;

                 for dir_path in scan_dirs {
                     let dir = match dir_path.to_str() { Some(s) => s, None => continue };
                     if let Ok(mut entries) = std::fs::read_dir(&dir).map(|r| r.filter_map(|e| e.ok()).collect::<Vec<_>>()) {
                         entries.sort_by_key(|e| e.file_name());

                         for ent in entries {
                             if let Some(s) = ent.path().to_str() {
                                 let fname = s.to_string();
                                 let lower = fname.to_ascii_lowercase();
                                 let is_png = lower.ends_with(".png");
                                 let is_gif = lower.ends_with(".gif");
                                 if !(is_png || is_gif) {
                                     continue;
                                 }
                                 if capture_state.loaded.contains_key(&fname) {
                                     continue;
                                 }

                                 // Skip files that are likely mid-write: zero-length or recently
                                 // modified (within 200ms). This avoids attempting to decode
                                 // partially-written files.
                                 if let Ok(meta) = std::fs::metadata(&fname) {
                                     if meta.len() == 0 {
                                         continue;
                                     }
                                     if let Ok(modified) = meta.modified() {
                                         if let Ok(elapsed) = std::time::SystemTime::now().duration_since(modified) {
                                             if elapsed.as_millis() < 200 {
                                                 continue;
                                             }
                                         }
                                     }
                                 }

                                 match std::fs::read(&fname) {
                                     Ok(bytes) => {
                                         if is_gif {
                                             use std::io::Cursor;
                                             use image::codecs::gif::GifDecoder;
                                             use image::AnimationDecoder;

                                             if let Ok(decoder) = GifDecoder::new(Cursor::new(&bytes)) {
                                                 if let Ok(frames_vec) = decoder.into_frames().collect_frames() {
                                                     let mut handles = Vec::with_capacity(frames_vec.len());
                                                     let mut tex_ids = Vec::with_capacity(frames_vec.len());
                                                     let mut durations = Vec::with_capacity(frames_vec.len());
                                                     for frame in frames_vec.into_iter() {
                                                         let buf: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = frame.buffer().clone();
                                                         let w = buf.width();
                                                         let h = buf.height();
                                                         let raw = buf.into_raw();
                                                         if let Some(rgba_img) = image::RgbaImage::from_raw(w, h, raw) {
                                                             let size = bevy::render::render_resource::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
                                                             let bevy_image = Image::new(
                                                                 size,
                                                                 bevy::render::render_resource::TextureDimension::D2,
                                                                 rgba_img.into_raw(),
                                                                 bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
                                                                 RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
                                                             );
                                                             let handle = images_assets.add(bevy_image);
                                                             let tex_id = egui_contexts.add_image(bevy_egui::EguiTextureHandle::Strong(handle.clone()));
                                                             handles.push(handle);
                                                             tex_ids.push(tex_id);
                                                             durations.push(0.1);
                                                         }
                                                     }
                                                     if !handles.is_empty() {
                                                         let entry = PreviewEntry { handles, tex_ids, durations, current_idx: 0, last_advance_secs: time.elapsed().as_secs_f64() };
                                                         capture_state.loaded.insert(fname.clone(), entry);
                                                         capture_state.order.push_front(fname.clone());
                                                     }
                                                 }
                                             }
                                         } else {
                                             match image::load_from_memory(&bytes) {
                                                 Ok(img) => {
                                                     let rgba = img.to_rgba8();
                                                     let (w, h) = (rgba.width(), rgba.height());
                                                     let size = bevy::render::render_resource::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
                                                     let mut bevy_image = Image::new(
                                                         size,
                                                         bevy::render::render_resource::TextureDimension::D2,
                                                         rgba.into_raw(),
                                                         bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
                                                         RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
                                                     );
                                                     // Label preview images for clearer logs.
                                                     let _ = match &mut bevy_image.texture_descriptor.label {
                                                         Some(_) => {
                                                             bevy_image.texture_descriptor.label = Some("preview_image");
                                                             true
                                                         }
                                                         None => false,
                                                     };
                                                     let handle = images_assets.add(bevy_image);
                                                     let tex_id = egui_contexts.add_image(bevy_egui::EguiTextureHandle::Strong(handle.clone()));
                                                     let entry = PreviewEntry { handles: vec![handle], tex_ids: vec![tex_id], durations: vec![0.1], current_idx: 0, last_advance_secs: time.elapsed().as_secs_f64() };
                                                     capture_state.loaded.insert(fname.clone(), entry);
                                                     capture_state.order.push_front(fname.clone());
                                                 }
                                                 Err(_) => {}
                                             }
                                         }
                                     }
                                     Err(_) => {}
                                 }
                             }
                         }
                     }
                 }
             }
         }
     }

    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };

    // If still loading, show a blocking startup overlay with project name and spinner.
    if loading_state.stage != LoadingStage::Ready {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(120.0);
                ui.spinner();
                ui.add_space(10.0);
                ui.heading(egui::RichText::new(format!("{}", VIDEO_ID)).size(20.0));
                ui.add_space(6.0);
                ui.label("Loading assets and initializing GPU...");
                if loading_state.stage == LoadingStage::WaitingForGpu {
                    ui.add_space(6.0);
                    ui.label(format!("Waiting for GPU to settle — frames remaining: {}", loading_state.frames_left));
                }
            });
        });
        return;
    }

    // If we're in teardown mode, show a full-screen blocking overlay so the user
    // clearly sees the reinitialization step instead of stale/broken content.
    if teardown.active {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(120.0);
                ui.spinner();
                ui.add_space(8.0);
                ui.heading(egui::RichText::new("Reinitializing display...").size(20.0));
                ui.add_space(6.0);
                ui.label(format!("Applying new resolution — frames remaining: {}", teardown.frames_left));
            });
        });
        // Still allow the capture side panel to be visible so user can cancel actions,
        // but the central overlay will hide the underlying scene content.
    }

    // Draw side panel with controls and thumbnails (only when visible)
    if capture_state.visible {
        egui::SidePanel::right("mcbaise_capture_panel").min_width(220.0).show(ctx, |ui| {
        ui.heading("Capture");
        ui.horizontal(|ui| {
            if ui.button("Capture PNG").clicked() {
                if let Ok(ts) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                    let sec = ts.as_secs();
                    let dir = format!(".tmp/readback_{}", sec);
                    let _ = std::fs::create_dir_all(&dir);
                    unsafe { std::env::set_var("MCBAISE_READBACK_OUTPUT", &dir); }
                    unsafe { std::env::set_var("MCBAISE_READBACK_ONESHOT", "1"); }
                    READBACK_REQUEST_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    eprintln!("native: UI requested one-shot readback -> {}", dir);
                }
            }

            let recording = std::env::var("MCBAISE_GIF_RECORD").as_deref().ok() == Some("1");
            let mut assembling = false;
            if let Ok(read_tmp) = std::fs::read_dir(".tmp") {
                for ent in read_tmp.filter_map(|e| e.ok()) {
                    let p = ent.path();
                    if p.is_dir() && p.join("animation.gif.tmp").exists() {
                        assembling = true;
                        break;
                    }
                }
            }

            let btn_enabled = recording || !assembling;
            let btn_label = if recording { "Stop GIF" } else { "Start GIF" };

            if ui.add_enabled(btn_enabled, egui::Button::new(btn_label)).clicked() {
                match std::env::var("MCBAISE_GIF_RECORD").as_deref().ok() {
                    Some("1") => {
                        unsafe { std::env::remove_var("MCBAISE_GIF_RECORD"); }
                        if let Ok(dir) = std::env::var("MCBAISE_READBACK_OUTPUT") {
                            unsafe { std::env::remove_var("MCBAISE_READBACK_OUTPUT"); }
                            eprintln!("native: UI stopping GIF recording, assembling {}", dir);
                            std::thread::spawn(move || {
                                if let Err(e) = assemble_gif_from_dir(&dir) {
                                    eprintln!("failed to assemble gif: {}", e);
                                }
                            });
                        }
                    }
                    _ => {
                        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
                        let dir = format!(".tmp/readback_{}", ts);
                        let _ = std::fs::create_dir_all(&dir);
                        unsafe { std::env::set_var("MCBAISE_READBACK_OUTPUT", &dir); }
                        unsafe { std::env::set_var("MCBAISE_GIF_RECORD", "1"); }
                        eprintln!("native: UI started GIF recording -> {}", dir);
                    }
                }
            }
        });

        ui.horizontal(|ui| {
            if ui.button("Hide").clicked() {
                capture_state.visible = false;
            }
            if ui.button(if capture_state.show_options { "Previews" } else { "Options" }).clicked() {
                capture_state.show_options = !capture_state.show_options;
            }
            if ui.button("Remove All").clicked() {
                // remove bevy image assets + egui texture ids
                for entry in capture_state.loaded.values() {
                    for handle in &entry.handles {
                        images_assets.remove(handle);
                    }
                }
                capture_state.loaded.clear();
                capture_state.order.clear();
                let _ = std::fs::remove_dir_all(".tmp");
                let _ = std::fs::create_dir_all(".tmp");
            }
            if ui.button("📂").on_hover_text("Open captures folder").clicked() {
                let _ = open::that(".tmp");
            }
        });

        if capture_state.show_options {
            ui.separator();
            ui.label("Recording Options");
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.checkbox(&mut capture_state.capture_captions, "Capture Captions (if visible)");
                ui.checkbox(&mut capture_state.capture_multi, "Capture Multi-view (experimental)");
                
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label("Target FPS:");
                    ui.add(egui::DragValue::new(&mut capture_state.capture_fps).range(0..=120).suffix(" fps"));
                });
                ui.horizontal(|ui| {
                    ui.label("Frame Skip:");
                    ui.add(egui::DragValue::new(&mut capture_state.capture_skip).range(1..=10));
                });
                ui.label(egui::RichText::new("0 = max render speed").size(10.0).italics().color(egui::Color32::GRAY));

                ui.add_space(8.0);
                let recording = std::env::var("MCBAISE_GIF_RECORD").as_deref().ok() == Some("1");
                let mut assembling = false;
                if let Ok(read_tmp) = std::fs::read_dir(".tmp") {
                    for ent in read_tmp.filter_map(|e| e.ok()) {
                        let p = ent.path();
                        if p.is_dir() && p.join("animation.gif.tmp").exists() {
                            assembling = true;
                            break;
                        }
                    }
                }
                // Lock UI while recording or while an assembly is in progress.
                // Avoid tying UI lock to `pending.active` which flips briefly and
                // causes visible control flicker when not recording.
                let lock_ui = recording || assembling;

                ui.add_enabled_ui(!lock_ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Output Scale:");
                        ui.add(egui::Slider::new(&mut capture_state.output_scale, 0.1..=4.0).step_by(0.1));
                    });

                    ui.add_space(8.0);
                    let resolutions = [
                        // YouTube standard (Landscape)
                        (256, 144, "144p (256x144)"),
                        (426, 240, "240p (426x240)"),
                        (640, 360, "360p (640x360)"),
                        (854, 480, "480p (854x480)"),
                        (1280, 720, "720p (1280x720)"),
                        (1920, 1080, "1080p (1920x1080)"),
                        (2560, 1440, "1440p (2560x1440)"),
                        (3840, 2160, "2160p (3840x2160)"),
                        // Phone aspect ratios (Portrait)
                        (1080, 1920, "Phone Portrait (1080x1920)"),
                        (1170, 2532, "iPhone Pro (1170x2532)"),
                        (1080, 2340, "Modern Phone (1080x2340)"),
                    ];

                    // Check if current window size matches any preset to update selected_resolution index
                    if let Some(window) = windows.iter().next() {
                        let ww = window.physical_width();
                        let wh = window.physical_height();
                        let mut found = false;
                        for (i, (w, h, _)) in resolutions.iter().enumerate() {
                            if *w == ww && *h == wh {
                                capture_state.selected_resolution = i as i32;
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            capture_state.selected_resolution = -1;
                        }
                    }

                    egui::ComboBox::from_label("Window Resolution")
                        .selected_text(if capture_state.selected_resolution >= 0 && (capture_state.selected_resolution as usize) < resolutions.len() {
                            resolutions[capture_state.selected_resolution as usize].2
                        } else {
                            "Custom"
                        })
                        .show_ui(ui, |ui| {
                            for (i, (w, h, label)) in resolutions.iter().enumerate() {
                                if ui.selectable_value(&mut capture_state.selected_resolution, i as i32, *label).clicked() {
                                    if let Some(mut window) = windows.iter_mut().next() {
                                        window.resolution.set_physical_resolution(*w, *h);
                                        pending_window_geometry.pending = true;
                                        pending_window_geometry.target_w = *w;
                                        pending_window_geometry.target_h = *h;
                                        eprintln!("native: UI requested window geometry change -> {}x{}", *w, *h);
                                    }
                                }
                            }
                        });
                });
                
                ui.add_space(16.0);
                ui.label("Frame output directory:");
                ui.label(egui::RichText::new(".tmp/readback_<ts>").small().color(egui::Color32::LIGHT_BLUE));
                
                ui.add_space(20.0);
                ui.label("Tip: Lower FPS saves disk space and CPU during recording.");
            });
        } else {
            ui.separator();
            ui.label("Previews");

                // Recording indicator (show frame count if possible)
                if std::env::var("MCBAISE_GIF_RECORD").as_deref().ok() == Some("1") {
                    ui.horizontal(|ui| {
                        ui.visuals_mut().override_text_color = Some(egui::Color32::from_rgb(255, 100, 100));
                        let mut label = String::from("🔴 Recording GIF...");
                        if let Ok(dir) = std::env::var("MCBAISE_READBACK_OUTPUT") {
                            if let Ok(entries) = std::fs::read_dir(&dir) {
                                let count = entries.filter_map(|e| e.ok()).filter(|e| {
                                    if let Some(n) = e.path().extension() {
                                        n == "png"
                                    } else { false }
                                }).count();
                                label.push_str(&format!("  (frames: {})", count));
                            }
                        }
                        ui.label(label);
                    });
                }

            egui::ScrollArea::vertical().show(ui, |ui| {
                // Check for rendering GIFs
                if let Ok(read_tmp) = std::fs::read_dir(".tmp") {
                    for ent in read_tmp.filter_map(|e| e.ok()) {
                        let p = ent.path();
                        if p.is_dir() && p.join("animation.gif.tmp").exists() {
                            let progress_text = std::fs::read_to_string(p.join(".progress")).ok();
                            ui.horizontal(|ui| {
                                ui.spinner();
                                let label = if let Some(pct) = progress_text {
                                    format!("Assembling GIF ({}%)...", pct.trim())
                                } else {
                                    "Assembling GIF...".to_string()
                                };
                                ui.label(label);
                                // Push a small flexible space so the cancel button sits to the right
                                ui.add_space(8.0);
                                // Add a right-aligned cancel button (small 'x')
                                let cancel_btn = ui.add(egui::Button::new("✖").small());
                                if cancel_btn.clicked() {
                                    // Signal cancellation by writing a .cancel marker file.
                                    // The assembly worker polls for this marker and will
                                    // abort cleanly. Avoid removing the directory here to
                                    // prevent races with the assembler.
                                    if let Err(e) = std::fs::write(p.join(".cancel"), "1") {
                                        eprintln!("Failed to signal cancel for {}: {}", p.display(), e);
                                    }
                                }
                            });
                        }
                    }
                }

                let now = time.elapsed().as_secs_f64();
                // Snapshot the order to avoid borrowing `capture_state` immutably
                // while also mutably borrowing entries from `capture_state.loaded`.
                let order_snapshot: Vec<String> = capture_state.order.iter().cloned().collect();
                for fname in order_snapshot {
                    if let Some(entry) = capture_state.loaded.get_mut(&fname) {
                        // advance frame if needed
                        if entry.tex_ids.len() > 1 {
                            let dur = entry.durations.get(entry.current_idx).cloned().unwrap_or(0.1) as f64;
                            if now - entry.last_advance_secs >= dur {
                                entry.current_idx = (entry.current_idx + 1) % entry.tex_ids.len();
                                entry.last_advance_secs = now;
                            }
                        }
                        let tex_id = entry.tex_ids.get(entry.current_idx).copied().unwrap_or(egui::TextureId::default());
                        let size = egui::vec2(ui.available_width(), 120.0);
                        let resp = ui.add(egui::Image::new((tex_id, size)).sense(egui::Sense::click()));
                        if resp.clicked() {
                            let _ = open::that(&fname);
                        }
                        if resp.secondary_clicked() {
                            #[cfg(windows)]
                            {
                                let _ = std::process::Command::new("explorer")
                                    .arg("/select,")
                                    .arg(fname.replace("/", "\\"))
                                    .spawn();
                            }
                            #[cfg(not(windows))]
                            {
                                if let Some(parent) = std::path::Path::new(&fname).parent() {
                                    let _ = open::that(parent);
                                }
                            }
                        }
                        ui.label(fname);
                    }
                }
            });
        }
        });
    }

    #[cfg(target_arch = "wasm32")]
    let _ = &overlay_text;

    #[cfg(target_arch = "wasm32")]
    let _ = &overlay_state;

    #[cfg(not(any(feature = "native-youtube", feature = "native-mpv")))]
    let _ = &mut commands;

    // Top-right: add an additional synced view.
    egui::Area::new(egui::Id::new("mcbaise_add_view_pie"))
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-10.0, 10.0))
        .show(ctx, |ui| {
            let desired = egui::vec2(28.0, 28.0);
            let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click());

            let painter = ui.painter();
            let center = rect.center();
            let radius = rect.width().min(rect.height()) * 0.36;
            let red = egui::Color32::from_rgba_unmultiplied(235, 40, 40, 220);

            painter.circle_stroke(center, radius, egui::Stroke::new(2.2, red));

            // Fill the "pie" proportionally to the number of active views.
            // 1 view -> 1/MAX filled, MAX views -> full circle.
            let frac = (multi_view.count.clamp(1, MultiView::MAX_VIEWS) as f32)
                / (MultiView::MAX_VIEWS as f32);
            let fill = egui::Color32::from_rgba_unmultiplied(235, 40, 40, 110);
            if frac >= 0.999 {
                // A full-sweep wedge degenerates; draw a proper filled circle at max.
                painter.circle_filled(center, radius, fill);
            } else {
                let a0 = -std::f32::consts::FRAC_PI_2;
                let a1 = a0 + std::f32::consts::TAU * frac;
                let steps = (12.0 * frac.max(0.25)).ceil() as usize;
                let mut pts = Vec::with_capacity(steps + 2);
                pts.push(center);
                for i in 0..=steps {
                    let t = i as f32 / steps as f32;
                    let a = a0 + (a1 - a0) * t;
                    pts.push(egui::pos2(
                        center.x + radius * a.cos(),
                        center.y + radius * a.sin(),
                    ));
                }
                painter.add(egui::Shape::convex_polygon(pts, fill, egui::Stroke::NONE));
            }

            if resp.hovered() {
                painter.rect_stroke(
                    rect,
                    egui::CornerRadius::same(6),
                    egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 40),
                    ),
                    egui::StrokeKind::Inside,
                );
            }

            // Left click adds a view; right click removes a view.
            // On mobile: tap-and-hold removes a view.
            let now_sec = time.elapsed().as_secs_f64();
            if resp.clicked_by(egui::PointerButton::Primary) {
                if !multi_view.increment() {
                    multi_view_hint.show(now_sec, "right click (or tap & hold) to remove");
                }
            } else if resp.clicked_by(egui::PointerButton::Secondary) || resp.long_touched() {
                if !multi_view.decrement() {
                    // Already at 1 view; explain how to add splits back.
                    multi_view_hint.show(now_sec, "left click/tap to add more");
                }
            }
        });

    // Small transient hint beside the view pie button.
    {
        let now_sec = time.elapsed().as_secs_f64();
        if multi_view_hint.active(now_sec) {
            egui::Area::new(egui::Id::new("mcbaise_add_view_pie_hint"))
                .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-46.0, 12.0))
                .show(ctx, |ui| {
                    egui::Frame::NONE
                        .fill(egui::Color32::WHITE)
                        .stroke(egui::Stroke::new(1.2, egui::Color32::BLACK))
                        .corner_radius(egui::CornerRadius::same(6))
                        .inner_margin(egui::Margin::symmetric(8, 6))
                        .show(ui, |ui| {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(&multi_view_hint.text)
                                        .color(egui::Color32::BLACK)
                                        .size(12.0),
                                )
                                .extend(),
                            );
                        });
                });
        }
    }

    if !overlay_vis.show {
        egui::Area::new(egui::Id::new("mcbaise_overlay_restore_pi"))
            .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 10.0))
            .show(ctx, |ui| {
                // Draw a simple pi glyph ourselves so we don't depend on font glyph availability.
                let desired = egui::vec2(28.0, 28.0);
                let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click());
                let stroke = egui::Stroke::new(
                    2.2,
                    egui::Color32::from_rgba_unmultiplied(255, 255, 255, 220),
                );
                let pad = 6.0;
                let x0 = rect.left() + pad;
                let x1 = rect.right() - pad;
                let y0 = rect.top() + pad;
                let y1 = rect.bottom() - pad;
                let y_top = y0 + 3.0;
                let y_bot = y1;
                let leg_w = 3.5;
                let left_leg_x = x0 + leg_w;
                let right_leg_x = x1 - leg_w;
                let painter = ui.painter();
                painter.line_segment([egui::pos2(x0, y_top), egui::pos2(x1, y_top)], stroke);
                painter.line_segment(
                    [egui::pos2(left_leg_x, y_top), egui::pos2(left_leg_x, y_bot)],
                    stroke,
                );
                painter.line_segment(
                    [
                        egui::pos2(right_leg_x, y_top),
                        egui::pos2(right_leg_x, y_bot),
                    ],
                    stroke,
                );

                if resp.hovered() {
                    painter.rect_stroke(
                        rect,
                        egui::CornerRadius::same(6),
                        egui::Stroke::new(
                            1.0,
                            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 40),
                        ),
                        egui::StrokeKind::Inside,
                    );
                }
                if resp.clicked() {
                    overlay_vis.show = true;

                    #[cfg(target_arch = "wasm32")]
                    {
                        // Keep JS in sync immediately.
                        let t = playback.time_sec;
                        let cues = lyric_cues();
                        let l_idx = if !playback.playing && playback.time_sec < 0.05 {
                            -2
                        } else {
                            find_cue_index(cues, t)
                        };
                        let caption_show = caption_vis.show;
                        if l_idx == -2 {
                            mcbaise_set_caption(
                                "Click Play on the YouTube video to start.",
                                caption_show,
                                true,
                            );
                        } else if l_idx < 0 {
                            mcbaise_set_caption("", false, false);
                        } else {
                            let cue = &cues[l_idx as usize];
                            mcbaise_set_caption(&cue.text, caption_show, cue.is_meta);
                        }
                    }
                }
            });
    }

    if overlay_vis.show {
        egui::Window::new("mcbaise_overlay")
            .title_bar(false)
            .resizable(false)
            .collapsible(false)
            .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 10.0))
            .show(ctx, |ui| {
                ui.label(format!("{VIDEO_ID} • tube ride"));

                ui.horizontal(|ui| {
                    if ui.button("Hide overlay").clicked() {
                        overlay_vis.show = false;

                        #[cfg(target_arch = "wasm32")]
                        {
                            // Keep captions in sync immediately (credits are independent).
                            let t = playback.time_sec;
                            let cues = lyric_cues();
                            let l_idx = if !playback.playing && playback.time_sec < 0.05 {
                                -2
                            } else {
                                find_cue_index(cues, t)
                            };
                            if l_idx == -2 {
                                mcbaise_set_caption(
                                    "Click Play on the YouTube video to start.",
                                    caption_vis.show,
                                    true,
                                );
                            } else if l_idx < 0 {
                                mcbaise_set_caption("", false, false);
                            } else {
                                let cue = &cues[l_idx as usize];
                                mcbaise_set_caption(&cue.text, caption_vis.show, cue.is_meta);
                            }
                        }
                    }

                    let caption_toggle_label = if caption_vis.show {
                        "Hide captions"
                    } else {
                        "Show captions"
                    };

                    if ui.button(caption_toggle_label).clicked() {
                        caption_vis.show = !caption_vis.show;

                        #[cfg(target_arch = "wasm32")]
                        {
                            // Keep JS in sync immediately.
                            let t = playback.time_sec;
                            let cues = lyric_cues();
                            let l_idx = if !playback.playing && playback.time_sec < 0.05 {
                                -2
                            } else {
                                find_cue_index(cues, t)
                            };
                            if l_idx == -2 {
                                mcbaise_set_caption(
                                    "Click Play on the YouTube video to start.",
                                    caption_vis.show,
                                    true,
                                );
                            } else if l_idx < 0 {
                                mcbaise_set_caption("", false, false);
                            } else {
                                let cue = &cues[l_idx as usize];
                                mcbaise_set_caption(&cue.text, caption_vis.show, cue.is_meta);
                            }
                        }
                    }

                    #[cfg(target_arch = "wasm32")]
                    {
                        let video_toggle_label = if video_vis.show {
                            "Hide video"
                        } else {
                            "Show video"
                        };

                        if ui.button(video_toggle_label).clicked() {
                            video_vis.show = !video_vis.show;
                            mcbaise_set_video_visible(video_vis.show);
                        }
                    }
                });

                ui.horizontal(|ui| {
                    egui::ComboBox::from_label("Subject")
                        .selected_text(match *subject_mode {
                            SubjectMode::Auto => {
                                format!("Subject: auto ({})", auto_subject.current.short_label())
                            }
                            SubjectMode::Random => {
                                format!("Subject: random ({})", auto_subject.current.short_label())
                            }
                            _ => subject_mode.label().to_string(),
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut *subject_mode,
                                SubjectMode::Auto,
                                SubjectMode::Auto.label(),
                            );
                            ui.selectable_value(
                                &mut *subject_mode,
                                SubjectMode::Random,
                                SubjectMode::Random.label(),
                            );
                            ui.selectable_value(
                                &mut *subject_mode,
                                SubjectMode::Human,
                                SubjectMode::Human.label(),
                            );
                            #[cfg(not(feature = "burn_human"))]
                            ui.selectable_value(
                                &mut *subject_mode,
                                SubjectMode::Doughnut,
                                SubjectMode::Doughnut.label(),
                            );
                            ui.selectable_value(
                                &mut *subject_mode,
                                SubjectMode::Ball,
                                SubjectMode::Ball.label(),
                            );
                        });

                    egui::ComboBox::from_label("Pose")
                        .selected_text(match *pose_mode {
                            PoseMode::Auto => {
                                format!("Pose: auto ({})", auto_pose.current.short_label())
                            }
                            PoseMode::Random => {
                                format!("Pose: random ({})", auto_pose.current.short_label())
                            }
                            _ => pose_mode.label().to_string(),
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut *pose_mode,
                                PoseMode::Auto,
                                PoseMode::Auto.label(),
                            );
                            ui.selectable_value(
                                &mut *pose_mode,
                                PoseMode::Random,
                                PoseMode::Random.label(),
                            );
                            ui.selectable_value(
                                &mut *pose_mode,
                                PoseMode::Standing,
                                PoseMode::Standing.label(),
                            );
                            ui.selectable_value(
                                &mut *pose_mode,
                                PoseMode::Belly,
                                PoseMode::Belly.label(),
                            );
                            ui.selectable_value(
                                &mut *pose_mode,
                                PoseMode::Back,
                                PoseMode::Back.label(),
                            );
                            ui.selectable_value(
                                &mut *pose_mode,
                                PoseMode::LeftSide,
                                PoseMode::LeftSide.label(),
                            );
                            ui.selectable_value(
                                &mut *pose_mode,
                                PoseMode::RightSide,
                                PoseMode::RightSide.label(),
                            );
                        });

                    egui::ComboBox::from_label("Camera")
                        .selected_text(match *camera_preset {
                            CameraPreset::Auto => {
                                format!("Camera: auto ({})", auto_cam.current.label())
                            }
                            CameraPreset::Random => {
                                format!("Camera: random ({})", auto_cam.current.label())
                            }
                            _ => camera_preset.label().to_string(),
                        })
                        .show_ui(ui, |ui| {
                            let before = *camera_preset;
                            for (value, label) in CameraPreset::choices() {
                                ui.selectable_value(&mut *camera_preset, value, label);
                            }

                            // Only these presets imply switching the active subject.
                            if *camera_preset != before {
                                match *camera_preset {
                                    CameraPreset::FollowHumanChase => {
                                        if cfg!(feature = "burn_human") {
                                            *subject_mode = SubjectMode::Human
                                        } else {
                                            *subject_mode = SubjectMode::Doughnut
                                        }
                                    }
                                    CameraPreset::FollowBallChase => {
                                        *subject_mode = SubjectMode::Ball
                                    }
                                    _ => {}
                                }
                            }
                        });

                    let effective_mode = match *camera_preset {
                        CameraPreset::Auto | CameraPreset::Random => auto_cam.current,
                        CameraPreset::FollowActiveFirst => CameraMode::First,
                        CameraPreset::FollowActiveOver | CameraPreset::TubeOver => CameraMode::Over,
                        CameraPreset::FollowActiveBack => CameraMode::Back,
                        CameraPreset::FollowActiveSide => CameraMode::Side,
                        CameraPreset::FollowActiveChase
                        | CameraPreset::FollowHumanChase
                        | CameraPreset::FollowBallChase => CameraMode::BallChase,
                        CameraPreset::FollowActiveFocusedChase => CameraMode::FocusedChase,
                        CameraPreset::FollowActiveFocusedSide => CameraMode::FocusedSide,
                        CameraPreset::PassingLeft => CameraMode::PassingLeft,
                        CameraPreset::PassingRight => CameraMode::PassingRight,
                        CameraPreset::PassingTop => CameraMode::PassingTop,
                        CameraPreset::PassingBottom => CameraMode::PassingBottom,
                    };
                    let (tmin, tmax) = effective_mode.suggested_duration_range_sec();
                    if tmax > 0.0 {
                        ui.label(format!("Suggested: {:.0}–{:.0}s", tmin, tmax));
                    }
                });

                ui.horizontal(|ui| {
                    egui::ComboBox::from_label("Human")
                        .selected_text(match human_appearance.0 {
                            AppearanceMode::Auto => {
                                format!("Human: auto ({})", auto_human_appearance.0.current.label())
                            }
                            AppearanceMode::Random => format!(
                                "Human: random ({})",
                                auto_human_appearance.0.current.label()
                            ),
                            m => format!("Human: {}", m.label()),
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut human_appearance.0,
                                AppearanceMode::Auto,
                                "auto",
                            );
                            ui.selectable_value(
                                &mut human_appearance.0,
                                AppearanceMode::Random,
                                "random",
                            );
                            ui.selectable_value(
                                &mut human_appearance.0,
                                AppearanceMode::BlueGlass,
                                "blue glass",
                            );
                            ui.selectable_value(
                                &mut human_appearance.0,
                                AppearanceMode::OpaqueWhite,
                                "opaque white",
                            );
                            ui.selectable_value(
                                &mut human_appearance.0,
                                AppearanceMode::Blue,
                                "blue",
                            );
                            ui.selectable_value(
                                &mut human_appearance.0,
                                AppearanceMode::Polkadot,
                                "polkadot",
                            );
                            ui.selectable_value(
                                &mut human_appearance.0,
                                AppearanceMode::MatteLightBlue,
                                "matte light blue",
                            );
                            ui.selectable_value(
                                &mut human_appearance.0,
                                AppearanceMode::Wireframe,
                                "wireframe",
                            );
                        });

                    egui::ComboBox::from_label("Ball")
                        .selected_text(match ball_appearance.0 {
                            AppearanceMode::Auto => {
                                format!("Ball: auto ({})", auto_ball_appearance.0.current.label())
                            }
                            AppearanceMode::Random => {
                                format!("Ball: random ({})", auto_ball_appearance.0.current.label())
                            }
                            m => format!("Ball: {}", m.label()),
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut ball_appearance.0,
                                AppearanceMode::Auto,
                                "auto",
                            );
                            ui.selectable_value(
                                &mut ball_appearance.0,
                                AppearanceMode::Random,
                                "random",
                            );
                            ui.selectable_value(
                                &mut ball_appearance.0,
                                AppearanceMode::BlueGlass,
                                "blue glass",
                            );
                            ui.selectable_value(
                                &mut ball_appearance.0,
                                AppearanceMode::OpaqueWhite,
                                "opaque white",
                            );
                            ui.selectable_value(
                                &mut ball_appearance.0,
                                AppearanceMode::Blue,
                                "blue",
                            );
                            ui.selectable_value(
                                &mut ball_appearance.0,
                                AppearanceMode::Polkadot,
                                "polkadot",
                            );
                            ui.selectable_value(
                                &mut ball_appearance.0,
                                AppearanceMode::MatteLightBlue,
                                "matte light blue",
                            );
                            ui.selectable_value(
                                &mut ball_appearance.0,
                                AppearanceMode::Wireframe,
                                "wireframe",
                            );
                        });
                });

                ui.separator();

                #[cfg(all(not(target_arch = "wasm32"), feature = "native-youtube"))]
                {
                    if let Some(cfg) = _native_youtube_cfg.as_ref() {
                        let connected = _native_youtube
                            .as_ref()
                            .map(|yt| yt.enabled)
                            .unwrap_or(false);

                        if let Some(yt) = _native_youtube.as_ref() {
                            if let Ok(slot) = yt.last_error.lock()
                                && let Some(err) = slot.as_ref()
                            {
                                ui.label(format!("YouTube: {err}"));
                            }

                            ui.label(format!("t≈{:.2}s", playback.time_sec));
                            if yt.in_ad {
                                if let Some(label) = yt.ad_label.as_deref() {
                                    ui.label(format!("Ad: {label}"));
                                } else {
                                    ui.label("Ad: playing");
                                }
                            }
                        }

                        if !connected {
                            let label = if _native_youtube.is_some() {
                                "Restart Browser"
                            } else {
                                "Start Browser"
                            };

                            if ui.button(label).clicked() {
                                if let Some(yt) = _native_youtube.as_ref() {
                                    let _ = yt.tx.send(crate::native_youtube::Command::Shutdown);
                                }

                                let (tx, rx, join) = crate::native_youtube::spawn(
                                    VIDEO_ID,
                                    &cfg.webdriver_url,
                                    cfg.launch_webdriver,
                                    cfg.chrome_user_data_dir.clone(),
                                    cfg.chrome_profile_dir.clone(),
                                );

                                commands.insert_resource(NativeYoutubeSync {
                                    enabled: true,
                                    tx,
                                    rx: std::sync::Mutex::new(rx),
                                    join: std::sync::Mutex::new(Some(join)),
                                    last_error: std::sync::Mutex::new(None),

                                    has_remote: false,
                                    last_remote_time_sec: 0.0,
                                    last_remote_playing: false,
                                    sample_age_sec: 0.0,
                                    remote_age_sec: 0.0,
                                    interp_time_sec: 0.0,

                                    in_ad: false,
                                    ad_label: None,
                                    last_good_time_sec: 0.0,
                                    pending_seek_after_ad: false,
                                    ad_nudge_cooldown_sec: 0.0,

                                    heal_cooldown_sec: 0.0,
                                });
                            }
                        }
                    }
                }

                #[cfg(all(windows, not(target_arch = "wasm32"), feature = "native-mpv"))]
                {
                    if let Some(cfg) = _native_mpv_cfg.as_ref() {
                        let connected = _native_mpv.as_ref().map(|m| m.has_remote).unwrap_or(false);

                        if let Some(mpv) = _native_mpv.as_ref() {
                            if let Ok(slot) = mpv.last_error.lock()
                                && let Some(err) = slot.as_ref()
                            {
                                ui.label(format!("mpv: {err}"));
                            }
                            ui.label(format!("t≈{:.2}s", playback.time_sec));
                        }

                        let label = if connected {
                            "Restart mpv"
                        } else {
                            "Start/Restart mpv"
                        };
                        if ui.button(label).clicked() {
                            if let Some(mpv) = _native_mpv.as_ref() {
                                let _ = mpv.tx.send(crate::native_mpv::Command::Shutdown);
                            }

                            let (tx, rx, join) = crate::native_mpv::spawn(
                                cfg.url.clone(),
                                cfg.mpv_path.clone(),
                                cfg.extra_args.clone(),
                            );
                            commands.insert_resource(NativeMpvSync {
                                enabled: true,
                                tx,
                                rx: std::sync::Mutex::new(rx),
                                join: std::sync::Mutex::new(Some(join)),
                                last_error: std::sync::Mutex::new(None),

                                has_remote: false,
                                last_remote_time_sec: 0.0,
                                last_remote_playing: false,
                                sample_age_sec: 0.0,
                                interp_time_sec: 0.0,
                            });
                        }
                    }
                }

                ui.horizontal(|ui| {
                    let yt_authoritative = {
                        #[cfg(all(not(target_arch = "wasm32"), feature = "native-youtube"))]
                        {
                            _native_youtube
                                .as_ref()
                                .map(|yt| yt.enabled)
                                .unwrap_or(false)
                        }
                        #[cfg(not(all(not(target_arch = "wasm32"), feature = "native-youtube")))]
                        {
                            false
                        }
                    };

                    let mpv_authoritative = {
                        #[cfg(all(windows, not(target_arch = "wasm32"), feature = "native-mpv"))]
                        {
                            _native_mpv
                                .as_ref()
                                .map(|m| m.enabled && m.has_remote)
                                .unwrap_or(false)
                        }
                        #[cfg(not(all(
                            windows,
                            not(target_arch = "wasm32"),
                            feature = "native-mpv"
                        )))]
                        {
                            false
                        }
                    };

                    let authoritative = yt_authoritative || mpv_authoritative;

                    let desired_playing = !playback.playing;
                    let label = if playback.playing { "Pause" } else { "Play" };
                    if ui.button(label).clicked() {
                        playback.playing = desired_playing;

                        #[cfg(all(not(target_arch = "wasm32"), feature = "native-youtube"))]
                        {
                            if let Some(yt) = _native_youtube.as_ref()
                                && yt.enabled
                            {
                                let _ = yt.tx.send(crate::native_youtube::Command::SetPlaying(
                                    desired_playing,
                                ));

                                // If the page is stuck (no fresh samples), pressing Play should attempt recovery.
                                if desired_playing && yt.sample_age_sec > 2.0 {
                                    let seek_to = if yt.last_good_time_sec > 0.01 {
                                        yt.last_good_time_sec
                                    } else {
                                        playback.time_sec
                                    };
                                    let _ =
                                        yt.tx.send(crate::native_youtube::Command::ReloadAndSeek {
                                            time_sec: seek_to,
                                            playing: true,
                                        });
                                }
                            }
                        }

                        #[cfg(all(windows, not(target_arch = "wasm32"), feature = "native-mpv"))]
                        {
                            if let Some(mpv) = _native_mpv.as_ref()
                                && mpv.enabled
                            {
                                let _ = mpv
                                    .tx
                                    .send(crate::native_mpv::Command::SetPlaying(desired_playing));
                            }
                        }

                        #[cfg(target_arch = "wasm32")]
                        {
                            mcbaise_request_playing(desired_playing);
                        }
                    }

                    if authoritative {
                        ui.label(if yt_authoritative {
                            "Speed: YouTube (authoritative)"
                        } else {
                            "Speed: mpv (authoritative)"
                        });
                        playback.speed = 1.0;
                    } else {
                        ui.label(format!("Speed: {:.2}x", playback.speed));
                        if ui.button("-").clicked() {
                            playback.speed = (playback.speed - 0.25).clamp(0.25, 3.0);
                        }
                        if ui.button("+").clicked() {
                            playback.speed = (playback.speed + 0.25).clamp(0.25, 3.0);
                        }
                    }
                });

                ui.horizontal(|ui| {
                    egui::ComboBox::from_label("Colors")
                        .selected_text(match *scheme_mode {
                            ColorSchemeMode::Auto => format!(
                                "Colors: auto ({})",
                                ColorSchemeMode::short_label_from_value(auto_style.scheme_current)
                            ),
                            ColorSchemeMode::Random => format!(
                                "Colors: random ({})",
                                ColorSchemeMode::short_label_from_value(auto_style.scheme_current)
                            ),
                            _ => scheme_mode.label().to_string(),
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut *scheme_mode,
                                ColorSchemeMode::Auto,
                                ColorSchemeMode::Auto.label(),
                            );
                            ui.selectable_value(
                                &mut *scheme_mode,
                                ColorSchemeMode::Random,
                                ColorSchemeMode::Random.label(),
                            );

                            if ui
                                .selectable_value(
                                    &mut *scheme_mode,
                                    ColorSchemeMode::OrangeWhite,
                                    ColorSchemeMode::OrangeWhite.label(),
                                )
                                .clicked()
                            {
                                settings.scheme = 0;
                            }
                            if ui
                                .selectable_value(
                                    &mut *scheme_mode,
                                    ColorSchemeMode::Nin,
                                    ColorSchemeMode::Nin.label(),
                                )
                                .clicked()
                            {
                                settings.scheme = 1;
                            }

                            if ui
                                .selectable_value(
                                    &mut *scheme_mode,
                                    ColorSchemeMode::BlackWhite,
                                    ColorSchemeMode::BlackWhite.label(),
                                )
                                .clicked()
                            {
                                settings.scheme = 2;
                            }

                            if ui
                                .selectable_value(
                                    &mut *scheme_mode,
                                    ColorSchemeMode::RandomGrey,
                                    ColorSchemeMode::RandomGrey.label(),
                                )
                                .clicked()
                            {
                                settings.scheme = 3;
                            }

                            if ui
                                .selectable_value(
                                    &mut *scheme_mode,
                                    ColorSchemeMode::Blue,
                                    ColorSchemeMode::Blue.label(),
                                )
                                .clicked()
                            {
                                settings.scheme = 4;
                            }

                            if ui
                                .selectable_value(
                                    &mut *scheme_mode,
                                    ColorSchemeMode::Dynamic,
                                    ColorSchemeMode::Dynamic.label(),
                                )
                                .clicked()
                            {
                                settings.scheme = 5;
                            }

                            if ui
                                .selectable_value(
                                    &mut *scheme_mode,
                                    ColorSchemeMode::Fluid,
                                    ColorSchemeMode::Fluid.label(),
                                )
                                .clicked()
                            {
                                settings.scheme = 6;
                                settings.pattern = 4; // Automatically switch to fluid pattern
                                *pattern_mode = TexturePatternMode::Fluid;
                            }

                            if ui
                                .selectable_value(
                                    &mut *scheme_mode,
                                    ColorSchemeMode::Sun,
                                    ColorSchemeMode::Sun.label(),
                                )
                                .clicked()
                            {
                                settings.scheme = 7;
                            }
                            if ui
                                .selectable_value(
                                    &mut *scheme_mode,
                                    ColorSchemeMode::Psychedelic,
                                    ColorSchemeMode::Psychedelic.label(),
                                )
                                .clicked()
                            {
                                settings.scheme = 8;
                            }

                            if ui
                                .selectable_value(
                                    &mut *scheme_mode,
                                    ColorSchemeMode::Neon,
                                    ColorSchemeMode::Neon.label(),
                                )
                                .clicked()
                            {
                                settings.scheme = 9;
                            }

                            if ui
                                .selectable_value(
                                    &mut *scheme_mode,
                                    ColorSchemeMode::Matrix,
                                    ColorSchemeMode::Matrix.label(),
                                )
                                .clicked()
                            {
                                settings.scheme = 10;
                            }
                        });

                    egui::ComboBox::from_label("Texture")
                        .selected_text(match *pattern_mode {
                            TexturePatternMode::Auto => format!(
                                "Texture: auto ({})",
                                TexturePatternMode::short_label_from_value(
                                    auto_style.pattern_current
                                )
                            ),
                            TexturePatternMode::Random => format!(
                                "Texture: random ({})",
                                TexturePatternMode::short_label_from_value(
                                    auto_style.pattern_current
                                )
                            ),
                            _ => pattern_mode.label().to_string(),
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut *pattern_mode,
                                TexturePatternMode::Auto,
                                TexturePatternMode::Auto.label(),
                            );
                            ui.selectable_value(
                                &mut *pattern_mode,
                                TexturePatternMode::Random,
                                TexturePatternMode::Random.label(),
                            );

                            if ui
                                .selectable_value(
                                    &mut *pattern_mode,
                                    TexturePatternMode::Stripe,
                                    TexturePatternMode::Stripe.label(),
                                )
                                .clicked()
                            {
                                settings.pattern = 0;
                            }
                            if ui
                                .selectable_value(
                                    &mut *pattern_mode,
                                    TexturePatternMode::Swirl,
                                    TexturePatternMode::Swirl.label(),
                                )
                                .clicked()
                            {
                                settings.pattern = 1;
                            }

                            if ui
                                .selectable_value(
                                    &mut *pattern_mode,
                                    TexturePatternMode::StripeWire,
                                    TexturePatternMode::StripeWire.label(),
                                )
                                .clicked()
                            {
                                settings.pattern = 2;
                            }

                            if ui
                                .selectable_value(
                                    &mut *pattern_mode,
                                    TexturePatternMode::SwirlWire,
                                    TexturePatternMode::SwirlWire.label(),
                                )
                                .clicked()
                            {
                                settings.pattern = 3;
                            }

                            if ui
                                .selectable_value(
                                    &mut *pattern_mode,
                                    TexturePatternMode::Fluid,
                                    TexturePatternMode::Fluid.label(),
                                )
                                .clicked()
                            {
                                settings.pattern = 4;
                            }

                            if ui
                                .selectable_value(
                                    &mut *pattern_mode,
                                    TexturePatternMode::FluidStripe,
                                    TexturePatternMode::FluidStripe.label(),
                                )
                                .clicked()
                            {
                                settings.pattern = 5;
                            }

                            if ui
                                .selectable_value(
                                    &mut *pattern_mode,
                                    TexturePatternMode::FluidSwirl,
                                    TexturePatternMode::FluidSwirl.label(),
                                )
                                .clicked()
                            {
                                settings.pattern = 6;
                            }
                            if ui
                                .selectable_value(
                                    &mut *pattern_mode,
                                    TexturePatternMode::Wave,
                                    TexturePatternMode::Wave.label(),
                                )
                                .clicked()
                            {
                                settings.pattern = 7;
                            }

                            if ui
                                .selectable_value(
                                    &mut *pattern_mode,
                                    TexturePatternMode::Fractal,
                                    TexturePatternMode::Fractal.label(),
                                )
                                .clicked()
                            {
                                settings.pattern = 8;
                            }

                            if ui
                                .selectable_value(
                                    &mut *pattern_mode,
                                    TexturePatternMode::Particle,
                                    TexturePatternMode::Particle.label(),
                                )
                                .clicked()
                            {
                                settings.pattern = 9;
                            }

                            if ui
                                .selectable_value(
                                    &mut *pattern_mode,
                                    TexturePatternMode::Grid,
                                    TexturePatternMode::Grid.label(),
                                )
                                .clicked()
                            {
                                settings.pattern = 10;
                            }
                            if ui
                                .selectable_value(
                                    &mut *pattern_mode,
                                    TexturePatternMode::HoopWire,
                                    TexturePatternMode::HoopWire.label(),
                                )
                                .clicked()
                            {
                                settings.pattern = 11;
                            }
                            if ui
                                .selectable_value(
                                    &mut *pattern_mode,
                                    TexturePatternMode::HoopAlt,
                                    TexturePatternMode::HoopAlt.label(),
                                )
                                .clicked()
                            {
                                settings.pattern = 12;
                            }
                        });
                });
            });
    }

    // Non-wasm: draw the captions/credits as an egui overlay over the Bevy view.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let t = ctx.input(|i| i.time) as f32;
        let wobble_x = (t * 4.0).sin() * 2.0;
        let wobble_y = (t * 3.1).cos() * 1.0;

        let credit_color = egui::Color32::from_rgb(0xF2, 0xB1, 0x00);
        let caption_color = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 240);

        if overlay_state.last_credit_idx >= 0 {
            egui::Area::new(egui::Id::new("mcbaise_credit_area"))
                .anchor(
                    egui::Align2::CENTER_TOP,
                    egui::vec2(wobble_x, 26.0 + wobble_y),
                )
                .show(ctx, |ui| {
                    // Give the credit text plenty of width so it doesn't wrap.
                    ui.set_min_width(1200.0);
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                    ui.vertical_centered(|ui| {
                        // Approximate the original styled HTML credits.
                        match overlay_state.last_credit_idx {
                            0 => {
                                ui.label(
                                    egui::RichText::new("DIRTY")
                                        .strong()
                                        .size(52.0)
                                        .color(credit_color),
                                );
                                ui.label(
                                    egui::RichText::new("MELODY")
                                        .strong()
                                        .size(52.0)
                                        .color(credit_color),
                                );
                                ui.label(
                                    egui::RichText::new("RECORDS")
                                        .strong()
                                        .size(52.0)
                                        .color(credit_color),
                                );
                                ui.add_space(2.0);
                                ui.label(
                                    egui::RichText::new("Owns All Rights")
                                        .strong()
                                        .size(34.0)
                                        .color(credit_color),
                                );
                            }
                            1 => {
                                ui.label(
                                    egui::RichText::new("MCBAISE")
                                        .strong()
                                        .size(52.0)
                                        .color(credit_color),
                                );
                                ui.label(
                                    egui::RichText::new("PALE REGARD")
                                        .strong()
                                        .size(36.0)
                                        .color(credit_color),
                                );
                            }
                            2 => {
                                ui.label(
                                    egui::RichText::new("ABSURDIA")
                                        .strong()
                                        .size(70.0)
                                        .color(credit_color),
                                );
                                ui.label(
                                    egui::RichText::new("FANTASMAGORIA")
                                        .strong()
                                        .size(64.0)
                                        .color(credit_color),
                                );
                            }
                            _ => {
                                // Fallback.
                                if !overlay_text.credit.is_empty() {
                                    ui.label(
                                        egui::RichText::new(&overlay_text.credit)
                                            .strong()
                                            .size(34.0)
                                            .color(credit_color),
                                    );
                                }
                            }
                        }
                    });
                });
        }

        if caption_vis.show && !overlay_text.caption.is_empty() {
            egui::Area::new(egui::Id::new("mcbaise_caption_area"))
                .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -26.0))
                .show(ctx, |ui| {
                    let frame = egui::Frame::NONE
                        .fill(egui::Color32::from_rgba_unmultiplied(0, 0, 0, 155))
                        .stroke(egui::Stroke::new(
                            1.0,
                            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 24),
                        ))
                        .corner_radius(egui::CornerRadius::same(14))
                        .inner_margin(egui::Margin::symmetric(14, 10));

                    frame.show(ui, |ui| {
                        ui.set_max_width(900.0);
                        ui.vertical_centered(|ui| {
                            let mut text = egui::RichText::new(&overlay_text.caption)
                                .size(22.0)
                                .color(caption_color);
                            if overlay_text.caption_is_meta {
                                text = text.italics().color(egui::Color32::from_rgba_unmultiplied(
                                    255, 255, 255, 150,
                                ));
                            }
                            ui.label(text);
                        });
                    });
                });
        }
    }

    // Info button Bottom-Right
    egui::Area::new(egui::Id::new("mcbaise_info_button"))
        .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-10.0, -10.0))
        .show(ctx, |ui| {
            let desired = egui::vec2(28.0, 28.0);
            let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click());

            let painter = ui.painter();
            let center = rect.center();
            let radius = rect.width().min(rect.height()) * 0.36;

            // Match the pie indicator style but white-ish
            let white = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 200);
            painter.circle_stroke(center, radius, egui::Stroke::new(2.2, white));

            // Draw an "i"
            painter.text(center, egui::Align2::CENTER_CENTER, "i", egui::FontId::proportional(14.0), white);

            if resp.hovered() {
                painter.rect_stroke(
                    rect,
                    egui::CornerRadius::same(6),
                    egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 40),
                    ),
                    egui::StrokeKind::Inside,
                );
            }

            if resp.clicked() {
                capture_state.show_info = !capture_state.show_info;
            }
        });

    // Resize button above the info button
    egui::Area::new(egui::Id::new("mcbaise_resize_button"))
        .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-10.0, -50.0))
        .show(ctx, |ui| {
            let desired = egui::vec2(28.0, 28.0);
            let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click());

            let painter = ui.painter();
            let center = rect.center();
            let radius = rect.width().min(rect.height()) * 0.36;

            // Color the icon green when automation is active.
            let icon_color = if capture_state.resize_automation_active {
                egui::Color32::from_rgb(120, 255, 120)
            } else {
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 200)
            };

            painter.circle_stroke(center, radius, egui::Stroke::new(2.2, icon_color));

            // Draw a diagonal resize glyph
            painter.text(center, egui::Align2::CENTER_CENTER, "⤢", egui::FontId::proportional(14.0), icon_color);

            if resp.hovered() {
                painter.rect_stroke(
                    rect,
                    egui::CornerRadius::same(6),
                    egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 40),
                    ),
                    egui::StrokeKind::Inside,
                );
            }

            if resp.clicked() {
                // Toggle automation: flip the capture_state flag that mirrors
                // the ResizeAutomation resource. The UI state is used here for
                // immediate feedback; the render/world automation resource will
                // be initialized and driven on the main App.
                capture_state.resize_automation_active = !capture_state.resize_automation_active;
            }

            if resp.secondary_clicked() {
                // Request cycling preset window resolutions. This is consumed
                // on the next main-world Update to ensure the pending geometry
                // handshake is armed before the deferred resize is applied.
                capture_state.cycle_resolution_requested = true;
            }
        });

    if capture_state.show_info {
        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(egui::Color32::from_black_alpha(220)))
            .show(ctx, |ui| {
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.heading(egui::RichText::new("Keyboard Shortcuts").size(32.0).color(egui::Color32::WHITE));
                        ui.add_space(32.0);

                        egui::Grid::new("shortcuts_grid")
                            .num_columns(2)
                            .spacing([60.0, 16.0])
                            .show(ui, |ui| {
                                let label_style = |s: &str| egui::RichText::new(s).size(20.0).color(egui::Color32::LIGHT_BLUE);
                                let desc_style = |s: &str| egui::RichText::new(s).size(20.0).color(egui::Color32::WHITE);

                                ui.label(label_style("Space")); ui.label(desc_style("Toggle Playback")); ui.end_row();
                                ui.label(label_style("1")); ui.label(desc_style("Cycle Color Scheme")); ui.end_row();
                                ui.label(label_style("2")); ui.label(desc_style("Cycle Texture Pattern")); ui.end_row();
                                ui.label(label_style("Arrow Up")); ui.label(desc_style("Increase Playback Speed")); ui.end_row();
                                ui.label(label_style("Arrow Down")); ui.label(desc_style("Decrease Playback Speed")); ui.end_row();
                                ui.label(label_style("F5")); ui.label(desc_style("Capture PNG (One-shot)")); ui.end_row();
                                ui.label(label_style("F6")); ui.label(desc_style("Toggle GIF Recording")); ui.end_row();
                                ui.label(label_style("ESC")); ui.label(desc_style("Close this overlay")); ui.end_row();
                            });

                        ui.add_space(48.0);
                        if ui.add(egui::Button::new(egui::RichText::new("Close").size(24.0)).min_size(egui::vec2(120.0, 40.0))).clicked() {
                            capture_state.show_info = false;
                        }
                    });
                });
            });

        // Also allow ESC to close
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            capture_state.show_info = false;
        }
    }

    // YouTube/MPV initialization splash
    if let Some(sync) = _native_youtube.as_ref() {
        if sync.enabled && !sync.has_remote {
            egui::CentralPanel::default()
                .frame(egui::Frame::default().fill(egui::Color32::from_black_alpha(180)))
                .show(ctx, |ui| {
                    ui.centered_and_justified(|ui| {
                        ui.vertical_centered(|ui| {
                            ui.add_space(20.0);
                            ui.label(egui::RichText::new("SYNCING WITH YOUTUBE").strong().size(32.0).color(egui::Color32::WHITE));
                            ui.add_space(20.0);
                            ui.add(egui::Spinner::new().size(40.0));
                            ui.add_space(20.0);
                            ui.label(egui::RichText::new("Launching browser and establishing WebDriver session...").size(18.0).color(egui::Color32::GRAY));
                            
                            if let Ok(slot) = sync.last_error.lock() {
                                if let Some(err) = slot.as_ref().map(|s| s.as_str()) {
                                    ui.add_space(32.0);
                                    ui.label(egui::RichText::new("Initialization Error:").strong().color(egui::Color32::RED));
                                    ui.label(egui::RichText::new(err).color(egui::Color32::from_rgb(255, 100, 100)));
                                }
                            }
                        });
                    });
                });
        }
    } else if let Some(sync) = _native_mpv.as_ref() {
        if sync.enabled && !sync.has_remote {
            egui::CentralPanel::default()
                .frame(egui::Frame::default().fill(egui::Color32::from_black_alpha(240)))
                .show(ctx, |ui| {
                    ui.centered_and_justified(|ui| {
                        ui.vertical_centered(|ui| {
                            ui.add_space(20.0);
                            ui.label(egui::RichText::new("SYNCING WITH MPV").strong().size(32.0).color(egui::Color32::WHITE));
                            ui.add_space(20.0);
                            ui.add(egui::Spinner::new().size(40.0));
                            ui.add_space(20.0);
                            ui.label(egui::RichText::new("Starting MPV instance and connecting to IPC...").size(18.0).color(egui::Color32::GRAY));
                            
                            if let Ok(slot) = sync.last_error.lock() {
                                if let Some(err) = slot.as_ref().map(|s| s.as_str()) {
                                    ui.add_space(32.0);
                                    ui.label(egui::RichText::new("Initialization Error:").strong().color(egui::Color32::RED));
                                    ui.label(egui::RichText::new(err).color(egui::Color32::from_rgb(255, 100, 100)));
                                }
                            }
                        });
                    });
                });
        }
    }
}

// ---------------------------- time → curve progress ----------------------------

fn progress_from_video_time(video_time_sec: f32) -> f32 {
    let speed = 0.0028;
    // Loop around the closed curve instead of clamping to the end.
    // The curve sampling is not stable at 1.0, so keep a small margin.
    let cycle = 0.985;
    (video_time_sec * speed).rem_euclid(cycle)
}

// ---------------------------- camera ----------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum CameraMode {
    First,
    Over,
    Back,
    BallChase,
    Side,
    FocusedChase,
    FocusedSide,
    PassingLeft,
    PassingRight,
    PassingTop,
    PassingBottom,
}

impl CameraMode {
    fn label(self) -> &'static str {
        match self {
            CameraMode::First => "first",
            CameraMode::Over => "over",
            CameraMode::Back => "back",
            CameraMode::BallChase => "chase",
            CameraMode::Side => "side",
            CameraMode::FocusedChase => "focused chase",
            CameraMode::FocusedSide => "focused side",
            CameraMode::PassingLeft => "passing left",
            CameraMode::PassingRight => "passing right",
            CameraMode::PassingTop => "passing top",
            CameraMode::PassingBottom => "passing bottom",
        }
    }

    fn suggested_duration_range_sec(self) -> (f32, f32) {
        match self {
            // Existing behavior.
            CameraMode::First | CameraMode::Back | CameraMode::BallChase | CameraMode::Side => {
                (3.0, 6.0)
            }
            // Over: hold 1s then move in; prefer a longer hold after the move.
            CameraMode::Over => (6.5, 6.5),

            // Focused modes benefit from a bit more time to feel intentional.
            CameraMode::FocusedChase | CameraMode::FocusedSide => (4.0, 7.0),

            // Passing modes need enough time for the subject to traverse the view.
            CameraMode::PassingLeft
            | CameraMode::PassingRight
            | CameraMode::PassingTop
            | CameraMode::PassingBottom => (3.0, 3.0),
        }
    }

    fn distance_range_to_subject(self) -> (f32, f32) {
        match self {
            // These don't use a target-centered distance meaningfully.
            CameraMode::First | CameraMode::Over => (0.0, 0.0),

            // Keep close-ish for readability.
            CameraMode::Back | CameraMode::BallChase | CameraMode::Side => (4.5, 6.5),

            // Give focused modes room to breathe.
            CameraMode::FocusedChase | CameraMode::FocusedSide => (5.0, 10.0),

            // Passing shots tend to read better from farther out.
            CameraMode::PassingLeft
            | CameraMode::PassingRight
            | CameraMode::PassingTop
            | CameraMode::PassingBottom => (9.0, 16.0),
        }
    }

    fn is_passing(self) -> bool {
        matches!(
            self,
            CameraMode::PassingLeft
                | CameraMode::PassingRight
                | CameraMode::PassingTop
                | CameraMode::PassingBottom
        )
    }
}

fn timeline_camera_mode(video_time_sec: f32) -> CameraMode {
    // Deterministic timeline camera behavior.

    // Auto should respect the suggested times, but remain deterministic.
    // We derive a per-mode "preferred" duration from the suggested range:
    // - fixed range: use that value
    // - otherwise: midpoint
    let preferred = |mode: CameraMode| {
        let (min_sec, max_sec) = mode.suggested_duration_range_sec();
        if (min_sec - max_sec).abs() < 1e-4 {
            min_sec
        } else {
            0.5 * (min_sec + max_sec)
        }
    };

    // Build a deterministic cycle.
    let first_dur = preferred(CameraMode::First);
    let over_dur = preferred(CameraMode::Over);
    let back_dur = preferred(CameraMode::Back);
    let chase_a_dur = preferred(CameraMode::BallChase);
    let side_dur = preferred(CameraMode::Side);

    // Alternate focused chase/side each cycle for variety.
    let focused_mode = CameraMode::FocusedChase;
    let focused_dur = preferred(focused_mode);

    // Keep passing at ~1s per direction (and within the suggested 0.9..1.2 range).
    let pass_each_dur = preferred(CameraMode::PassingLeft).clamp(0.9, 1.2);

    let chase_b_dur = preferred(CameraMode::BallChase);

    let cycle = first_dur
        + over_dur
        + back_dur
        + chase_a_dur
        + side_dur
        + focused_dur
        + pass_each_dur * 4.0
        + chase_b_dur;

    let u = video_time_sec.rem_euclid(cycle);
    let cycle_idx = (video_time_sec / cycle).floor() as i32;

    let focused_mode = if (cycle_idx & 1) == 0 {
        CameraMode::FocusedChase
    } else {
        CameraMode::FocusedSide
    };

    let mut t = u;
    if t < first_dur {
        return CameraMode::First;
    }
    t -= first_dur;
    if t < over_dur {
        return CameraMode::Over;
    }
    t -= over_dur;
    if t < back_dur {
        return CameraMode::Back;
    }
    t -= back_dur;
    if t < chase_a_dur {
        return CameraMode::BallChase;
    }
    t -= chase_a_dur;
    if t < side_dur {
        return CameraMode::Side;
    }
    t -= side_dur;
    if t < focused_dur {
        return focused_mode;
    }
    t -= focused_dur;

    // Passing: 4 directions, each for pass_each_dur.
    let idx = (t / pass_each_dur).floor().clamp(0.0, 3.0) as i32;
    match idx {
        0 => CameraMode::PassingLeft,
        1 => CameraMode::PassingRight,
        2 => CameraMode::PassingTop,
        3 => CameraMode::PassingBottom,
        _ => CameraMode::BallChase,
    }
}

fn timeline_pose_mode(video_time_sec: f32) -> PoseMode {
    // Deterministic pose cycle for timeline mode.
    // Placeholder until the full scripted timeline is ported; guarantees Auto isn't random.
    let cycle = 14.0;
    let u = video_time_sec.rem_euclid(cycle);
    let cycle_idx = (video_time_sec / cycle).floor() as i32;

    if u < 3.5 {
        PoseMode::Standing
    } else if u < 7.0 {
        PoseMode::Belly
    } else if u < 10.5 {
        PoseMode::Back
    } else if (cycle_idx & 1) == 0 {
        PoseMode::LeftSide
    } else {
        PoseMode::RightSide
    }
}

fn timeline_subject_mode(video_time_sec: f32) -> SubjectMode {
    // Deterministic subject timeline.
    // Keep the rider as human for most of the cycle; switch to ball during the chase segment.
    let cycle = 14.0;
    let u = video_time_sec.rem_euclid(cycle);
    if u > 11.0 {
        SubjectMode::Ball
    } else {
        SubjectMode::Human
    }
}

fn timeline_color_scheme(video_time_sec: f32) -> u32 {
    // Deterministic color timeline (four schemes).
    let cycle = 14.0;
    let u = video_time_sec.rem_euclid(cycle);
    if u < 3.5 {
        0
    } else if u < 7.0 {
        1
    } else if u < 10.5 {
        2
    } else {
        3
    }
}

fn timeline_texture_pattern(video_time_sec: f32) -> u32 {
    // Deterministic texture timeline.
    // pattern 0: stripe
    // pattern 1: swirl
    // Flip every 3.5s inside the 14s cycle.
    let step = 3.5;
    // Previously the shader treated 0 as swirl; we now treat 0 as stripe.
    // So invert the old alternating sequence.
    1 - ((((video_time_sec / step).floor() as i32) & 1) as u32)
}

#[derive(Resource, Clone, Copy, PartialEq, Eq, Default)]
enum CameraPreset {
    #[default]
    Auto,
    Random,
    FollowActiveChase,
    FollowActiveBack,
    FollowActiveFirst,
    FollowActiveOver,
    FollowActiveSide,
    FollowActiveFocusedChase,
    FollowActiveFocusedSide,
    PassingLeft,
    PassingRight,
    PassingTop,
    PassingBottom,
    FollowHumanChase,
    FollowBallChase,
    TubeOver,
}

impl CameraPreset {
    fn label(self) -> &'static str {
        match self {
            CameraPreset::Auto => "Camera: auto",
            CameraPreset::Random => "Camera: random",
            CameraPreset::FollowActiveChase => "Camera: follow active (chase)",
            CameraPreset::FollowActiveBack => "Camera: follow active (back)",
            CameraPreset::FollowActiveFirst => "Camera: follow active (first)",
            CameraPreset::FollowActiveOver => "Camera: follow active (over)",
            CameraPreset::FollowActiveSide => "Camera: follow active (side)",
            CameraPreset::FollowActiveFocusedChase => "Camera: follow active (focused chase)",
            CameraPreset::FollowActiveFocusedSide => "Camera: follow active (focused side)",
            CameraPreset::PassingLeft => "Camera: passing (left)",
            CameraPreset::PassingRight => "Camera: passing (right)",
            CameraPreset::PassingTop => "Camera: passing (top)",
            CameraPreset::PassingBottom => "Camera: passing (bottom)",
            CameraPreset::FollowHumanChase => "Camera: follow human (chase)",
            CameraPreset::FollowBallChase => "Camera: follow ball (chase)",
            CameraPreset::TubeOver => "Camera: tube overview",
        }
    }

    fn choices() -> [(CameraPreset, &'static str); 16] {
        [
            (CameraPreset::Auto, CameraPreset::Auto.label()),
            (CameraPreset::Random, CameraPreset::Random.label()),
            (
                CameraPreset::FollowActiveChase,
                CameraPreset::FollowActiveChase.label(),
            ),
            (
                CameraPreset::FollowActiveBack,
                CameraPreset::FollowActiveBack.label(),
            ),
            (
                CameraPreset::FollowActiveFirst,
                CameraPreset::FollowActiveFirst.label(),
            ),
            (
                CameraPreset::FollowActiveOver,
                CameraPreset::FollowActiveOver.label(),
            ),
            (
                CameraPreset::FollowActiveSide,
                CameraPreset::FollowActiveSide.label(),
            ),
            (
                CameraPreset::FollowActiveFocusedChase,
                CameraPreset::FollowActiveFocusedChase.label(),
            ),
            (
                CameraPreset::FollowActiveFocusedSide,
                CameraPreset::FollowActiveFocusedSide.label(),
            ),
            (CameraPreset::PassingLeft, CameraPreset::PassingLeft.label()),
            (
                CameraPreset::PassingRight,
                CameraPreset::PassingRight.label(),
            ),
            (CameraPreset::PassingTop, CameraPreset::PassingTop.label()),
            (
                CameraPreset::PassingBottom,
                CameraPreset::PassingBottom.label(),
            ),
            (
                CameraPreset::FollowHumanChase,
                CameraPreset::FollowHumanChase.label(),
            ),
            (
                CameraPreset::FollowBallChase,
                CameraPreset::FollowBallChase.label(),
            ),
            (CameraPreset::TubeOver, CameraPreset::TubeOver.label()),
        ]
    }
}

#[allow(clippy::too_many_arguments)]
fn camera_pose(
    video_time_sec: f32,
    camera_preset: CameraPreset,
    selected_mode: CameraMode,
    subject_mode: SubjectMode,
    mode_age_sec: f32,
    random_subject_distance: Option<f32>,
    pass_anchor: Option<(Vec3, Vec3, Vec3, Vec3)>,
    cam_center: Vec3,
    look_ahead: Vec3,
    cam_tangent: Vec3,
    cam_n: Vec3,
    cam_b: Vec3,
    ball_center: Vec3,
    subject_pos_human: Vec3,
    subject_pos_ball: Vec3,
    subject_tangent: Vec3,
    _subject_up: Vec3,
) -> (Vec3, Vec3, Vec3) {
    let (intro_show_tube, intro_dive) = if cfg!(target_arch = "wasm32") {
        (2.2, 1.6)
    } else {
        // Native has no embedded YouTube UI; start inside immediately.
        (0.0, 0.0)
    };
    let in_intro = video_time_sec < (intro_show_tube + intro_dive);

    let active_pos = match subject_mode {
        SubjectMode::Human => subject_pos_human,
        SubjectMode::Doughnut => subject_pos_human,
        SubjectMode::Ball => subject_pos_ball,
        SubjectMode::Auto | SubjectMode::Random => subject_pos_human,
    };
    let target_pos = match camera_preset {
        CameraPreset::FollowHumanChase => subject_pos_human,
        CameraPreset::FollowBallChase => subject_pos_ball,
        _ => active_pos,
    };

    let cam_inner_pos = cam_center + cam_b * 0.10;

    let first_pos = cam_inner_pos;
    let first_look = look_ahead;
    let first_up = cam_n;

    let over_pos = cam_center + cam_n * 18.0 + cam_b * 14.0;
    let over_look = cam_center;
    let over_up = cam_tangent.cross(cam_n).normalize_or_zero();

    let back_pos = cam_center + cam_tangent * -12.0 + cam_n * 1.2;
    let back_look = cam_center + cam_tangent * 3.0;
    let back_up = cam_n;

    // Keep the chase camera stable in the tube frame.
    // If we offset using subject_up, the camera orbits with the rider/ball and it can feel like
    // the subject never changes its orientation relative to the camera/tube.
    let chase_pos = target_pos + subject_tangent * -4.8 + cam_n * 0.9;
    let chase_look = target_pos;
    // Keep camera "upright" relative to the tube, not the subject.
    // If we use subject_up here, the camera rolls with the rider/ball and it looks like
    // we're always on the "top" of the tube.
    let chase_up = cam_n;

    let side_pos = target_pos + cam_b * 4.2 + cam_n * 1.0 + subject_tangent * -0.6;
    let side_look = target_pos;
    let side_up = cam_n;

    let focused_base_dist = random_subject_distance.unwrap_or(6.8);
    // Focused modes should visibly move toward the subject when activated.
    // Ramp a zoom-in over ~1s.
    let a = (mode_age_sec / 0.9).clamp(0.0, 1.0);
    let a = a * a * (3.0 - 2.0 * a); // smoothstep
    let focused_dist = focused_base_dist * (1.25 - 0.55 * a);
    let focused_chase_dir =
        (-subject_tangent * 0.90 + cam_n * 0.28 + cam_b * 0.22).normalize_or_zero();
    let focused_chase_pos = target_pos + focused_chase_dir * focused_dist;
    let focused_chase_look = target_pos + subject_tangent * 1.8;
    let focused_chase_up = cam_n;

    let focused_side_dir =
        (cam_b * 0.95 + cam_n * 0.20 - subject_tangent * 0.18).normalize_or_zero();
    let focused_side_pos = target_pos + focused_side_dir * focused_dist;
    let focused_side_look = target_pos + subject_tangent * 1.6;
    let focused_side_up = cam_n;

    // Passing shot: camera is placed at the tube center and looks forward down the opening.
    // Direction (left/right/top/bottom) is expressed by a small look-target bias while keeping
    // the camera centered.
    let (pass_c, pass_tan, pass_n, pass_b) =
        pass_anchor.unwrap_or((cam_center, cam_tangent, cam_n, cam_b));

    let pass_up = pass_n;
    let pass_pos = pass_c + pass_tan * (TUBE_RADIUS * 0.10);

    // Look far down the tube so we read the opening, not the wall.
    let pass_far = 25.0;
    let pass_look_center = pass_c + pass_tan * pass_far;

    // Bias the look target slightly to create an on-screen edge preference.
    let aim = (TUBE_RADIUS * 0.25).clamp(0.6, 1.4);
    // With forward ~= +tan and up ~= +n, camera-right ~= +b.
    // To make the subject appear on the LEFT, yaw right (look toward +b).
    let pass_look_left = pass_look_center + pass_b * aim;
    let pass_look_right = pass_look_center - pass_b * aim;
    let pass_look_top = pass_look_center + pass_n * aim;
    let pass_look_bottom = pass_look_center - pass_n * aim;

    let pass_pos_left = pass_pos;
    let pass_pos_right = pass_pos;
    let pass_pos_top = pass_pos;
    let pass_pos_bottom = pass_pos;

    let mut pos;
    let mut look;
    let mut up;

    match selected_mode {
        CameraMode::First => {
            pos = first_pos;
            look = first_look;
            up = first_up;
        }
        CameraMode::Over => {
            pos = over_pos;
            let base_look = if camera_preset == CameraPreset::TubeOver {
                cam_center
            } else {
                over_look
            };
            look = base_look;
            up = over_up;

            // "Over" should behave like a focused shot: hold the normal framing briefly,
            // then slowly pan/tilt and move toward the subject.
            // Keep TubeOver unchanged (it's meant to present the tube itself).
            if !in_intro && camera_preset != CameraPreset::TubeOver {
                let hold_sec = 1.0;
                let move_sec = 2.0;
                let t = ((mode_age_sec - hold_sec) / move_sec).clamp(0.0, 1.0);
                let t = t * t * (3.0 - 2.0 * t); // smoothstep

                // End pose: closer and slightly behind, still in the tube frame.
                let focus_pos = target_pos + cam_n * 12.0 + cam_b * 9.0 + subject_tangent * -1.3;
                let focus_look = target_pos + subject_tangent * 0.9;

                pos = pos.lerp(focus_pos, t);
                look = look.lerp(focus_look, t);
            }
        }
        CameraMode::Back => {
            pos = back_pos;
            look = back_look;
            up = back_up;
        }
        CameraMode::BallChase => {
            pos = chase_pos;
            look = chase_look;
            up = chase_up;
        }
        CameraMode::Side => {
            pos = side_pos;
            look = side_look;
            up = side_up;
        }
        CameraMode::FocusedChase => {
            pos = focused_chase_pos;
            look = focused_chase_look;
            up = focused_chase_up;
        }
        CameraMode::FocusedSide => {
            pos = focused_side_pos;
            look = focused_side_look;
            up = focused_side_up;
        }
        CameraMode::PassingLeft => {
            pos = pass_pos_left;
            look = pass_look_left;
            up = pass_up;
        }
        CameraMode::PassingRight => {
            pos = pass_pos_right;
            look = pass_look_right;
            up = pass_up;
        }
        CameraMode::PassingTop => {
            pos = pass_pos_top;
            look = pass_look_top;
            up = pass_up;
        }
        CameraMode::PassingBottom => {
            pos = pass_pos_bottom;
            look = pass_look_bottom;
            up = pass_up;
        }
    }

    // If we're in a passing shot and the anchor is provided, gently track the
    // subject after it has moved passed the anchor point so the camera reads
    // the subject as it exits the frame.
    if selected_mode.is_passing() {
        if let Some((pass_c, pass_tan, pass_n, _pass_b)) = pass_anchor {
            let active_pos = match subject_mode {
                SubjectMode::Human => subject_pos_human,
                SubjectMode::Doughnut => subject_pos_human,
                SubjectMode::Ball => subject_pos_ball,
                SubjectMode::Auto | SubjectMode::Random => subject_pos_human,
            };
            let rel = active_pos - pass_c;
            let along = rel.dot(pass_tan);
            // Start following once the subject moves slightly past the anchor.
            if along > 0.05 {
                // Normalize amount and clamp; smaller divisor => stronger follow sooner.
                let follow_strength = (along / 1.2).clamp(0.0, 1.0);
                // Blend the look target toward the subject so the pass reads as tracking.
                let look_alpha = 0.45 + 0.55 * follow_strength; // [0.45..1.0]
                look = look.lerp(active_pos, look_alpha);
                // Nudge camera position toward the subject for a more noticeable follow.
                let pos_target = active_pos + pass_tan * -2.0 + pass_n * 0.2;
                let pos_alpha = 0.12 + 0.28 * follow_strength;
                pos = pos.lerp(pos_target, pos_alpha);
            }
        }
    }

    if in_intro {
        let _mid_s = 0.18_f32.min(0.99);
        // Approx: use current frame as mid; visually close.
        let mid = ball_center;
        let far_pos = mid + cam_n * 70.0 + cam_b * 55.0 + cam_tangent * -40.0;
        let far_look = mid + cam_tangent * 40.0;
        let far_up = cam_n;

        if intro_dive <= 0.0 || video_time_sec < intro_show_tube {
            pos = far_pos;
            look = far_look;
            up = far_up;
        } else {
            let a = (video_time_sec - intro_show_tube) / intro_dive;
            let blend = a.clamp(0.0, 1.0);
            pos = far_pos.lerp(pos, blend);
            look = far_look.lerp(look, blend);
            up = far_up.lerp(up, blend).normalize_or_zero();
        }
    }

    // Keep a steadier subject distance so it doesn't feel like it "flies away" as it rotates.
    // Only apply to the modes intended to feature the subject.
    if !in_intro {
        match selected_mode {
            CameraMode::BallChase
            | CameraMode::Back
            | CameraMode::FocusedChase
            | CameraMode::FocusedSide => {
                let desired_dist = match selected_mode {
                    CameraMode::FocusedChase | CameraMode::FocusedSide => focused_dist,
                    _ => random_subject_distance.unwrap_or(5.0),
                };
                let dir = (pos - target_pos).normalize_or_zero();
                if dir.length_squared() > 0.0 {
                    pos = target_pos + dir * desired_dist;
                }
                // Bias look toward the subject for a consistent 3D read.
                look = look.lerp(target_pos, 0.65);
                // Keep a stable up aligned to the tube frame.
                // Blending toward subject_up reintroduces camera roll and makes the subject
                // appear locked to the "top" of the tube.
                up = up.lerp(cam_n, 0.85).normalize_or_zero();
            }
            _ => {}
        }
    }

    (pos, look, up)
}

// ---------------------------- curve + frames ----------------------------

#[derive(Clone)]
struct CatmullRomCurve {
    pts: Vec<Vec3>,
    tension: f32,
}

impl CatmullRomCurve {
    fn point_at(&self, u: f32) -> Vec3 {
        let u = u.clamp(0.0, 1.0);
        let segs = (self.pts.len() - 1) as f32;
        let scaled = u * segs;
        let i = scaled.floor() as isize;
        let t = scaled - i as f32;

        let i1 = i.clamp(0, (self.pts.len() - 2) as isize) as usize;
        let i0 = i1.saturating_sub(1);
        let i2 = (i1 + 1).min(self.pts.len() - 1);
        let i3 = (i1 + 2).min(self.pts.len() - 1);

        catmull_rom(
            self.pts[i0],
            self.pts[i1],
            self.pts[i2],
            self.pts[i3],
            t,
            self.tension,
        )
    }

    fn tangent_at(&self, u: f32) -> Vec3 {
        let eps = 0.0005;
        let a = self.point_at((u - eps).max(0.0));
        let b = self.point_at((u + eps).min(1.0));
        (b - a).normalize_or_zero()
    }
}

fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32, tension: f32) -> Vec3 {
    // Cubic Hermite form.
    let v0 = (p2 - p0) * tension;
    let v1 = (p3 - p1) * tension;

    let t2 = t * t;
    let t3 = t2 * t;

    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    p1 * h00 + v0 * h10 + p2 * h01 + v1 * h11
}

#[derive(Clone)]
struct Frames {
    tangents: Vec<Vec3>,
    normals: Vec<Vec3>,
    binormals: Vec<Vec3>,
    samples: usize,
}

impl Frames {
    fn frame_at(&self, u: f32) -> Frame {
        let u = u.clamp(0.0, 1.0);
        let i_f = u * (self.samples as f32 - 1.0);
        let i = i_f.floor() as usize;
        let i = i.min(self.samples - 2);
        let t = i_f - i as f32;

        let tan = self.tangents[i]
            .lerp(self.tangents[i + 1], t)
            .normalize_or_zero();
        let nor = self.normals[i]
            .lerp(self.normals[i + 1], t)
            .normalize_or_zero();
        let bin = self.binormals[i]
            .lerp(self.binormals[i + 1], t)
            .normalize_or_zero();
        Frame { tan, nor, bin }
    }
}

#[derive(Clone, Copy)]
struct Frame {
    tan: Vec3,
    nor: Vec3,
    bin: Vec3,
}

fn build_frames(curve: &CatmullRomCurve, samples: usize) -> Frames {
    let mut tangents = Vec::with_capacity(samples);
    for i in 0..samples {
        let u = i as f32 / (samples as f32 - 1.0);
        tangents.push(curve.tangent_at(u));
    }

    let mut normals = Vec::with_capacity(samples);
    let mut binormals = Vec::with_capacity(samples);

    let mut n0 = Vec3::Y;
    if n0.dot(tangents[0]).abs() > 0.9 {
        n0 = Vec3::X;
    }
    n0 = (n0 - tangents[0] * n0.dot(tangents[0])).normalize_or_zero();

    normals.push(n0);
    binormals.push(tangents[0].cross(normals[0]).normalize_or_zero());

    for i in 1..samples {
        let t_prev = tangents[i - 1];
        let t_cur = tangents[i];

        let axis = t_prev.cross(t_cur);
        let axis_len = axis.length();

        let mut n_prev = normals[i - 1];
        if axis_len > 1e-8 {
            let axis_n = axis / axis_len;
            let angle = t_prev.dot(t_cur).clamp(-1.0, 1.0).acos();
            let q = Quat::from_axis_angle(axis_n, angle);
            n_prev = q * n_prev;
        }

        let mut n_cur = (n_prev - t_cur * n_prev.dot(t_cur)).normalize_or_zero();
        if n_cur.length_squared() < 1e-10 {
            // Extremely rare degeneracy; keep continuity by re-projecting the previous normal.
            let n_fallback = normals[i - 1];
            n_cur = (n_fallback - t_cur * n_fallback.dot(t_cur)).normalize_or_zero();
        }

        // Enforce sign continuity between adjacent frames.
        // Without this, `lerp()` in `frame_at()` can go through ~zero which looks like
        // the subject "flips" instead of smoothly rotating.
        if n_cur.dot(normals[i - 1]) < 0.0 {
            n_cur = -n_cur;
        }

        let b_cur = t_cur.cross(n_cur).normalize_or_zero();
        normals.push(n_cur);
        binormals.push(b_cur);
    }

    Frames {
        tangents,
        normals,
        binormals,
        samples,
    }
}

fn make_random_loop_curve(seed: u64) -> CatmullRomCurve {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut pts = Vec::with_capacity(1800);

    let total = 1800;
    let step_z = 1.2;

    let mut x = 0.0;
    let mut y = 0.0;
    let mut z = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;

    let mut loop_countdown: i32 = 0;
    let mut loop_phase: f32 = 0.0;
    let mut loop_radius: f32 = 0.0;
    let mut loop_freq: f32 = 0.0;

    let rand = |rng: &mut fastrand::Rng, a: f32, b: f32| a + (b - a) * rng.f32();
    let rand_sign = |rng: &mut fastrand::Rng| if rng.f32() < 0.5 { -1.0 } else { 1.0 };

    for i in 0..total {
        if loop_countdown <= 0 && rng.f32() < 0.02 && i > 60 {
            loop_countdown = rand(&mut rng, 70.0, 160.0).floor() as i32;
            loop_phase = rand(&mut rng, 0.0, std::f32::consts::TAU);
            loop_radius = rand(&mut rng, 3.5, 7.5);
            loop_freq = rand(&mut rng, 0.20, 0.55) * rand_sign(&mut rng);
        }

        if loop_countdown > 0 {
            loop_phase += loop_freq;
            x += loop_phase.cos() * 0.35 * loop_radius;
            y += loop_phase.sin() * 0.35 * loop_radius;
            loop_countdown -= 1;
        } else {
            vx += rand(&mut rng, -0.08, 0.08);
            vy += rand(&mut rng, -0.08, 0.08);
            vx *= 0.96;
            vy *= 0.96;
            x += vx * 2.2;
            y += vy * 2.2;
            x *= 0.995;
            y *= 0.995;
        }

        z += step_z;
        pts.push(Vec3::new(x, y, z));
    }

    CatmullRomCurve { pts, tension: 0.35 }
}

fn build_tube_mesh(
    curve: &CatmullRomCurve,
    frames: &Frames,
    tubular_segments: usize,
    radial_segments: usize,
    radius: f32,
) -> Mesh {
    let rings = tubular_segments + 1;
    let ring_verts = radial_segments + 1;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(rings * ring_verts);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(rings * ring_verts);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(rings * ring_verts);

    for j in 0..rings {
        let u = j as f32 / tubular_segments as f32;
        let center = curve.point_at(u);
        let f = frames.frame_at(u);

        for i in 0..ring_verts {
            let v = i as f32 / radial_segments as f32;
            let ang = v * std::f32::consts::TAU;
            let dir = f.nor * ang.cos() + f.bin * ang.sin();
            let p = center + dir * radius;
            positions.push([p.x, p.y, p.z]);
            normals.push([dir.x, dir.y, dir.z]);
            uvs.push([u, v]);
        }
    }

    let mut indices: Vec<u32> = Vec::with_capacity(tubular_segments * radial_segments * 6);
    for j in 0..tubular_segments {
        let ring0 = j * ring_verts;
        let ring1 = (j + 1) * ring_verts;
        for i in 0..radial_segments {
            let a = (ring0 + i) as u32;
            let b = (ring1 + i) as u32;
            let c = (ring1 + i + 1) as u32;
            let d = (ring0 + i + 1) as u32;
            indices.extend_from_slice(&[a, b, d, b, c, d]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_indices(Indices::U32(indices));
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh
}

// Parametric torus mesh builder: closed, with consistent normals that wrap.
fn build_torus_mesh(
    major: f32,
    minor: f32,
    tubular_segments: usize,
    radial_segments: usize,
) -> Mesh {
    let rings = tubular_segments;
    let ring_verts = radial_segments;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(rings * ring_verts);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(rings * ring_verts);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(rings * ring_verts);

    for i in 0..rings {
        let u = (i as f32) / (rings as f32);
        let theta = u * std::f32::consts::TAU;
        for j in 0..ring_verts {
            let v = (j as f32) / (ring_verts as f32);
            let phi = v * std::f32::consts::TAU;

            let cos_theta = theta.cos();
            let sin_theta = theta.sin();
            let cos_phi = phi.cos();
            let sin_phi = phi.sin();

            let x = (major + minor * cos_phi) * cos_theta;
            let y = minor * sin_phi;
            let z = (major + minor * cos_phi) * sin_theta;

            let nx = cos_theta * cos_phi;
            let ny = sin_phi;
            let nz = sin_theta * cos_phi;

            positions.push([x, y, z]);
            let n = Vec3::new(nx, ny, nz).normalize_or_zero();
            normals.push([n.x, n.y, n.z]);
            uvs.push([u, v]);
        }
    }

    let mut indices: Vec<u32> = Vec::with_capacity(rings * ring_verts * 6);
    for i in 0..rings {
        let next_i = (i + 1) % rings;
        for j in 0..ring_verts {
            let next_j = (j + 1) % ring_verts;
            let a = (i * ring_verts + j) as u32;
            let b = (next_i * ring_verts + j) as u32;
            let c = (next_i * ring_verts + next_j) as u32;
            let d = (i * ring_verts + next_j) as u32;
            indices.extend_from_slice(&[a, b, d, b, c, d]);
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_indices(Indices::U32(indices));
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh
}

// ---------------------------- fluid simulation ----------------------------

#[derive(Resource)]
struct FluidSimulation {
    wave_field: Vec<f32>,      // Current wave amplitude
    wave_prev: Vec<f32>,       // Previous wave amplitude
    wave2_field: Vec<f32>,     // Second coupled wave
    wave2_prev: Vec<f32>,      // Previous second wave
    velocity_field: Vec<Vec2>, // Derived velocity from waves
    density_field: Vec<f32>,   // Density derived from waves
    width: usize,
    height: usize,
    time: f32,
    frame: u32,
    // Whether the fluid feature is enabled (controls lazy creation of GPU textures)
    enabled: bool,
    velocity_handle: Option<bevy::prelude::Handle<Image>>,
    density_handle: Option<bevy::prelude::Handle<Image>>,
}

impl Default for FluidSimulation {
    fn default() -> Self {
        let width = 64;
        let height = 64;
        let size = width * height;

        let mut wave_field = vec![0.0; size];
        let mut wave_prev = vec![0.0; size];
        let mut wave2_field = vec![0.0; size];
        let mut wave2_prev = vec![0.0; size];
        let mut velocity_field = Vec::new();
        let mut density_field = Vec::new();

        // Initialize wave packets like in the shader
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let coord = Vec2::new(x as f32, y as f32);
                let center = Vec2::new(width as f32 / 2.0, height as f32 / 2.0);
                let center2 = Vec2::new(width as f32 / 3.0, height as f32 / 3.0);

                // Wave packet initialization
                let dist1 = (coord - center).length();
                let dist2 = (coord - center2).length();

                let k1 = (dist1 * 0.1).cos() * (-dist1 * dist1 * 0.01).exp();
                let k2 = (dist2 * 0.1).cos() * (-dist2 * dist2 * 0.01).exp();

                wave_field[idx] = k1;
                wave_prev[idx] = k1;
                wave2_field[idx] = k2;
                wave2_prev[idx] = k2;

                // Initialize derived fields
                velocity_field.push(Vec2::ZERO);
                density_field.push(0.5);
            }
        }

        Self {
            wave_field,
            wave_prev,
            wave2_field,
            wave2_prev,
            velocity_field,
            density_field,
            width,
            height,
            time: 0.0,
            frame: 0,
            enabled: false,
            velocity_handle: None,
            density_handle: None,
        }
    }
}

impl FluidSimulation {
    fn update(&mut self, dt: f32, playback_time: f32) {
        self.time = playback_time;
        self.frame += 1;

        let c = 0.25; // Courant number

        // Create new wave fields
        let mut new_wave = vec![0.0; self.wave_field.len()];
        let mut new_wave2 = vec![0.0; self.wave2_field.len()];

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;

                // Boundary check
                let border = 1;
                let in_border = x < border
                    || y < border
                    || x >= self.width - border
                    || y >= self.height - border;

                if in_border {
                    new_wave[idx] = 0.0;
                    new_wave2[idx] = 0.0;
                    continue;
                }

                // Wave 1 computation
                {
                    let center = self.wave_field[idx];
                    let prev = self.wave_prev[idx];

                    // Get neighbors
                    let up = if y < self.height - 1 {
                        self.wave_field[(y + 1) * self.width + x]
                    } else {
                        0.0
                    };
                    let down = if y > 0 {
                        self.wave_field[(y - 1) * self.width + x]
                    } else {
                        0.0
                    };
                    let right = if x < self.width - 1 {
                        self.wave_field[y * self.width + (x + 1)]
                    } else {
                        0.0
                    };
                    let left = if x > 0 {
                        self.wave_field[y * self.width + (x - 1)]
                    } else {
                        0.0
                    };

                    // Laplacian
                    let ddy = up - 2.0 * center + down;
                    let ddx = right - 2.0 * center + left;

                    let next: f32 = if self.frame <= 3 {
                        // Initial step
                        center - 0.5 * c * (ddy + ddx)
                    } else {
                        // Wave equation with coupling
                        let m2 = 1.0;
                        let coord = Vec2::new(x as f32, y as f32);
                        let resolution = Vec2::new(self.width as f32, self.height as f32);
                        let mut uv = (coord / resolution) * 2.0 - 1.0;
                        uv.x *= resolution.x / resolution.y;
                        let potential = uv.dot(uv);

                        let del = ddy + ddx;
                        let other_wave = self.wave2_field[idx];
                        let coupling = other_wave * other_wave * 50.0;

                        let update = del - (m2 + potential + coupling) * center;
                        -prev + 2.0 * center + 0.5 * c * update
                    };

                    new_wave[idx] = next;
                }

                // Wave 2 computation (coupled)
                {
                    let center = self.wave2_field[idx];
                    let prev = self.wave2_prev[idx];

                    // Get neighbors
                    let up = if y < self.height - 1 {
                        self.wave2_field[(y + 1) * self.width + x]
                    } else {
                        0.0
                    };
                    let down = if y > 0 {
                        self.wave2_field[(y - 1) * self.width + x]
                    } else {
                        0.0
                    };
                    let right = if x < self.width - 1 {
                        self.wave2_field[y * self.width + (x + 1)]
                    } else {
                        0.0
                    };
                    let left = if x > 0 {
                        self.wave2_field[y * self.width + (x - 1)]
                    } else {
                        0.0
                    };

                    // Laplacian
                    let ddy = up - 2.0 * center + down;
                    let ddx = right - 2.0 * center + left;

                    let next: f32 = if self.frame <= 3 {
                        // Initial step
                        center - 0.5 * c * (ddy + ddx)
                    } else {
                        // Wave equation with coupling
                        let m2 = 1.0;
                        let coord = Vec2::new(x as f32, y as f32);
                        let resolution = Vec2::new(self.width as f32, self.height as f32);
                        let mut uv = (coord / resolution) * 2.0 - 1.0;
                        uv.x *= resolution.x / resolution.y;
                        let potential = uv.dot(uv);

                        let del = ddy + ddx;
                        let other_wave = self.wave_field[idx];
                        let coupling = other_wave * other_wave * 50.0;

                        let update = del - (m2 + potential + coupling) * center;
                        -prev + 2.0 * center + 0.5 * c * update
                    };

                    new_wave2[idx] = next;
                }
            }
        }

        // Update wave fields
        self.wave_prev.copy_from_slice(&self.wave_field);
        self.wave_field.copy_from_slice(&new_wave);
        self.wave2_prev.copy_from_slice(&self.wave2_field);
        self.wave2_field.copy_from_slice(&new_wave2);

        // Compute derived velocity and density fields
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;

                // Compute velocity from wave gradients
                let eps = 1.0;
                let wave_center = self.wave_field[idx];
                let wave_right = if x < self.width - 1 {
                    self.wave_field[y * self.width + (x + 1)]
                } else {
                    wave_center
                };
                let wave_up = if y < self.height - 1 {
                    self.wave_field[(y + 1) * self.width + x]
                } else {
                    wave_center
                };

                let grad_x = (wave_right - wave_center) / eps;
                let grad_y = (wave_up - wave_center) / eps;

                self.velocity_field[idx] = Vec2::new(grad_x, grad_y) * 0.1;

                // Compute density from wave amplitudes
                let wave1_amp = wave_center.abs();
                let wave2_amp = self.wave2_field[idx].abs();
                self.density_field[idx] = (wave1_amp + wave2_amp * 0.5).clamp(0.0, 1.0);
            }
        }

        // Update velocity field with some time-varying patterns
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                let x_norm = x as f32 / self.width as f32 - 0.5;
                let y_norm = y as f32 / self.height as f32 - 0.5;

                // Create time-varying swirling motion
                let angle = self.time * 0.5 + (x_norm * y_norm * 4.0);
                let radius = (x_norm * x_norm + y_norm * y_norm).sqrt();

                let vel_x = angle.cos() * radius * 0.2 + (self.time * 0.3).sin() * 0.05;
                let vel_y = angle.sin() * radius * 0.2 + (self.time * 0.3).cos() * 0.05;

                self.velocity_field[idx] = Vec2::new(vel_x, vel_y);

                // Apply damping to prevent runaway velocities
                self.velocity_field[idx] *= 0.98;
            }
        }

        // Add programmatic perturbations for dynamic fluid motion
        self.apply_programmatic_forces(dt);
    }

    fn apply_programmatic_forces(&mut self, dt: f32) {
        // Create moving force points that create interesting fluid patterns
        let time = self.time;

        // Force point 1: orbiting around center
        let force1_u = 0.5 + (time * 0.8).cos() * 0.3;
        let force1_v = 0.5 + (time * 0.8).sin() * 0.3;
        self.apply_force_at_uv(force1_u, force1_v, 0.3, dt);

        // Force point 2: figure-8 pattern
        let force2_u = 0.5 + (time * 1.2).sin() * 0.25;
        let force2_v = 0.5 + (time * 0.6).sin() * 2.0 * (time * 1.2).cos() * 0.15;
        self.apply_force_at_uv(force2_u, force2_v, 0.2, dt);

        // Force point 3: bouncing around edges
        let force3_u = ((time * 1.5).sin() * 0.5 + 0.5).clamp(0.1, 0.9);
        let force3_v = ((time * 1.1).cos() * 0.5 + 0.5).clamp(0.1, 0.9);
        self.apply_force_at_uv(force3_u, force3_v, 0.25, dt);
    }

    fn apply_force_at_uv(&mut self, u: f32, v: f32, force_strength: f32, dt: f32) {
        let x = (u * self.width as f32) as usize;
        let y = (v * self.height as f32) as usize;

        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);

        // Apply force in a small radius around the position
        let radius: i32 = 4;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let nx = (x as i32 + dx).clamp(0, self.width as i32 - 1) as usize;
                let ny = (y as i32 + dy).clamp(0, self.height as i32 - 1) as usize;

                let dist_sq = (dx * dx + dy * dy) as f32;
                if dist_sq <= (radius * radius) as f32 {
                    let idx = ny * self.width + nx;
                    let falloff = 1.0 - dist_sq / (radius * radius) as f32;

                    // Apply velocity impulse outward from center
                    let impulse_x =
                        (nx as f32 - x as f32) * 0.05 * force_strength * falloff * dt * 50.0;
                    let impulse_y =
                        (ny as f32 - y as f32) * 0.05 * force_strength * falloff * dt * 50.0;

                    self.velocity_field[idx] += Vec2::new(impulse_x, impulse_y);

                    // Add some density
                    self.density_field[idx] += force_strength * falloff * dt * 5.0;
                    self.density_field[idx] = self.density_field[idx].clamp(0.0, 1.0);
                }
            }
        }
    }

    #[allow(dead_code)]
    fn get_velocity_at(&self, u: f32, v: f32) -> Vec2 {
        let x = (u * self.width as f32) as usize;
        let y = (v * self.height as f32) as usize;

        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);

        self.velocity_field[y * self.width + x]
    }

    #[allow(dead_code)]
    fn apply_force(&mut self, u: f32, v: f32, force_strength: f32, dt: f32) {
        let x = (u * self.width as f32) as usize;
        let y = (v * self.height as f32) as usize;

        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);

        // Apply force in a small radius around the mouse position
        let radius: i32 = 3;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let nx = (x as i32 + dx).clamp(0, self.width as i32 - 1) as usize;
                let ny = (y as i32 + dy).clamp(0, self.height as i32 - 1) as usize;

                let dist_sq = (dx * dx + dy * dy) as f32;
                if dist_sq <= (radius * radius) as f32 {
                    let idx = ny * self.width + nx;
                    let falloff = 1.0 - dist_sq / (radius * radius) as f32;

                    // Apply velocity impulse
                    let impulse = Vec2::new(
                        (nx as f32 - x as f32) * 0.1 * force_strength * falloff * dt * 100.0,
                        (ny as f32 - y as f32) * 0.1 * force_strength * falloff * dt * 100.0,
                    );

                    self.velocity_field[idx] += impulse;

                    // Add some density
                    self.density_field[idx] += force_strength * falloff * dt * 10.0;
                    self.density_field[idx] = self.density_field[idx].clamp(0.0, 1.0);
                }
            }
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct TubeMaterial {
    #[uniform(0)]
    // Pack everything into a single uniform buffer: WebGPU has a low per-stage uniform-buffer limit.
    // Layout: [params0, params1, orange, white, dark_inside, dark_outside]
    u: [Vec4; 6],

    #[texture(1)]
    #[sampler(2)]
    fluid_velocity: Option<Handle<Image>>,

    #[texture(3)]
    #[sampler(4)]
    fluid_density: Option<Handle<Image>>,
}

impl Default for TubeMaterial {
    fn default() -> Self {
        Self {
            u: [
                Vec4::new(0.0, 5.0, 240.0, -1.35),
                Vec4::new(0.08, 1.70, 0.22, 0.0),
                Color::srgb_u8(0xD2, 0x6B, 0x11).to_linear().to_vec4(),
                Color::srgb_u8(0xFF, 0xF8, 0xF0).to_linear().to_vec4(),
                Color::srgb_u8(0x0A, 0x00, 0x00).to_linear().to_vec4(),
                Color::srgb_u8(0x05, 0x00, 0x00).to_linear().to_vec4(),
            ],
            fluid_velocity: None,
            fluid_density: None,
        }
    }
}

impl TubeMaterial {
    fn set_time(&mut self, t: f32) {
        self.u[0].x = t;
    }

    fn set_scheme(&mut self, scheme: u32, time_sec: f32) {
        match scheme {
            0 => {
                // Orange/white.
                self.u[2] = Color::srgb_u8(0xD2, 0x6B, 0x11).to_linear().to_vec4();
                self.u[3] = Color::srgb_u8(0xFF, 0xF8, 0xF0).to_linear().to_vec4();
            }
            1 => {
                // NIN (legacy alt palette): cyan/pink.
                self.u[2] = Color::srgb_u8(0x18, 0xC5, 0xC5).to_linear().to_vec4();
                self.u[3] = Color::srgb_u8(0xFF, 0xC2, 0xF0).to_linear().to_vec4();
            }
            2 => {
                // Pure black/white.
                self.u[2] = Color::srgb_u8(0x06, 0x06, 0x06).to_linear().to_vec4();
                self.u[3] = Color::srgb_u8(0xFD, 0xFD, 0xFD).to_linear().to_vec4();
            }
            3 => {
                // "Random grey": deterministic grey pairs that change every ~1.5s.
                let tick = (time_sec * 0.666_666_7).floor().max(0.0) as u32;
                let mut x = tick ^ 0xA5A5_5A5A;
                x ^= x >> 16;
                x = x.wrapping_mul(0x7FEB_352D);
                x ^= x >> 15;
                x = x.wrapping_mul(0x846C_A68B);
                x ^= x >> 16;

                let g1 = 0.15 + 0.70 * ((x & 0xFFFF) as f32 / 65535.0);
                let g2 = (g1 * 0.55 + 0.25).clamp(0.05, 0.95);
                self.u[2] = Color::srgb(g1, g1, g1).to_linear().to_vec4();
                self.u[3] = Color::srgb(g2, g2, g2).to_linear().to_vec4();
            }
            4 => {
                // Blue tones
                self.u[2] = Color::srgb_u8(0x1A, 0x4F, 0x8B).to_linear().to_vec4(); // Dark blue
                self.u[3] = Color::srgb_u8(0xA0, 0xD2, 0xFF).to_linear().to_vec4(); // Light blue
            }
            5 => {
                // Dynamic: changing colors based on time
                let hue = (time_sec * 0.1) % 1.0;
                let sat = 0.8;
                let val = 0.8;

                // Simple HSV to RGB for dynamic colors
                let c = val * sat;
                let x = c * (1.0 - ((hue * 6.0) % 2.0 - 1.0).abs());
                let m = val - c;

                let h_sector = (hue * 6.0).floor();
                let rgb = if h_sector == 0.0 {
                    vec3(c, x, 0.0)
                } else if h_sector == 1.0 {
                    vec3(x, c, 0.0)
                } else if h_sector == 2.0 {
                    vec3(0.0, c, x)
                } else if h_sector == 3.0 {
                    vec3(0.0, x, c)
                } else if h_sector == 4.0 {
                    vec3(x, 0.0, c)
                } else {
                    vec3(c, 0.0, x)
                } + vec3(m, m, m);

                self.u[2] = Color::srgb(rgb.x, rgb.y, rgb.z).to_linear().to_vec4();
                self.u[3] = Color::srgb(1.0, 1.0, 1.0).to_linear().to_vec4(); // White
            }
            6 => {
                // Fluid: actual fluid-like colors (will be implemented)
                self.u[2] = Color::srgb_u8(0xD2, 0x6B, 0x11).to_linear().to_vec4(); // Orange
                self.u[3] = Color::srgb_u8(0xFF, 0xF8, 0xF0).to_linear().to_vec4(); // White
            }
            7 => {
                // Sun: plasma-like colors (moved from fluid)
                self.u[2] = Color::srgb_u8(0xFF, 0x45, 0x00).to_linear().to_vec4(); // Orange red
                self.u[3] = Color::srgb_u8(0xFF, 0xFF, 0x00).to_linear().to_vec4(); // Yellow
            }
            8 => {
                // Psychedelic: purple and magenta
                self.u[2] = Color::srgb_u8(0x8A, 0x2B, 0xE2).to_linear().to_vec4(); // Blue violet
                self.u[3] = Color::srgb_u8(0xFF, 0x00, 0xFF).to_linear().to_vec4(); // Magenta
            }
            9 => {
                // Neon: cyan and pink
                self.u[2] = Color::srgb_u8(0x00, 0xFF, 0xFF).to_linear().to_vec4(); // Cyan
                self.u[3] = Color::srgb_u8(0xFF, 0x14, 0x93).to_linear().to_vec4(); // Deep pink
            }
            10 => {
                // Matrix: green and black
                self.u[2] = Color::srgb_u8(0x00, 0x20, 0x00).to_linear().to_vec4(); // Dark green
                self.u[3] = Color::srgb_u8(0x00, 0xFF, 0x00).to_linear().to_vec4(); // Bright green
            }
            _ => {
                // Default to orange/white
                self.u[2] = Color::srgb_u8(0xD2, 0x6B, 0x11).to_linear().to_vec4();
                self.u[3] = Color::srgb_u8(0xFF, 0xF8, 0xF0).to_linear().to_vec4();
            }
        }
    }

    fn set_pattern(&mut self, pattern: u32) {
        self.u[1].w = pattern as f32;
    }
}

impl Material for TubeMaterial {
    fn fragment_shader() -> ShaderRef {
        // Construct the embedded path at compile-time so it exactly matches
        // what `embedded_asset!` registered. `embedded_path!` returns a PathBuf
        // like: `<crate>/.../mcbaise_tube.wgsl` so prefix with `embedded://`.
        let p = bevy::asset::embedded_path!("mcbaise_tube.wgsl");
        // Create an owned AssetPath from the PathBuf and mark it to load from the `embedded` source.
        let ap = bevy::asset::AssetPath::from_path_buf(p).with_source("embedded");
        ShaderRef::from(ap)
    }
}

// ---------------------------- opening credits + captions ----------------------------

struct Credit {
    start: f32,
    end: f32,
    #[allow(dead_code)]
    html: &'static str,
    #[allow(dead_code)]
    overlay: &'static str,
    #[allow(dead_code)]
    plain: &'static str,
}

fn opening_credits() -> &'static [Credit] {
    &[
        Credit {
            start: 0.00,
            end: 2.70,
            html: r#"<span style="font-size:3em; letter-spacing:.06em;">DIRTY<br>MELODY</span><br><span style="font-size:3em; letter-spacing:.06em;">RECORDS</span><br><span style="font-size:2em; opacity:.95; letter-spacing:.08em;">Owns All Rights</span>"#,
            overlay: "DIRTY\nMELODY\nRECORDS\nOwns All Rights",
            plain: "DIRTY MELODY RECORDS — Owns All Rights",
        },
        Credit {
            start: 2.70,
            end: 4.80,
            html: r#"<span style="font-size:3em; letter-spacing:.06em;">MCBAISE<br><span style="font-size:.70em;">PALE REGARD</span>"#,
            overlay: "MCBAISE\nPALE REGARD",
            plain: "MCBAISE — PALE REGARD",
        },
        Credit {
            start: 4.80,
            end: 8.40,
            html: r#"<span style="font-size:5.25em;">ABSURDIA</span><br><span style="font-size:5.05em;">FANTASMAGORIA</span>"#,
            overlay: "ABSURDIA\nFANTASMAGORIA",
            plain: "ABSURDIA FANTASMAGORIA",
        },
    ]
}

fn print_embedded_registry(_registry: Res<EmbeddedAssetRegistry>) {
    // Print the compile-time embedded path for the shader file so we can match the AssetServer lookup.
    // let p = bevy::asset::embedded_path!("mcbaise_tube.wgsl");
    // info!("embedded_path for mcbaise_tube.wgsl = {}", p.display());

    // // We can't rely on Debug for the registry; just confirm the resource exists.
    // let _ = registry;
    // info!("embedded registry resource present");
}

/// Register the embedded AssetSource with the AssetServer's builders so loads
/// from `embedded://...` succeed. Runs at startup.
fn register_embedded_asset_source(
    registry: Res<EmbeddedAssetRegistry>,
    mut builders: ResMut<bevy::asset::io::AssetSourceBuilders>,
) {
    registry.register_source(&mut builders);
}
fn find_opening_credit(t: f32) -> i32 {
    for (i, c) in opening_credits().iter().enumerate() {
        if t >= c.start && t <= c.end {
            return i as i32;
        }
    }
    -1
}
#[allow(dead_code)]
fn opening_credit_html(idx: usize) -> &'static str {
    opening_credits().get(idx).map(|c| c.html).unwrap_or("")
}

#[allow(dead_code)]
fn opening_credit_plain(idx: usize) -> &'static str {
    opening_credits().get(idx).map(|c| c.plain).unwrap_or("")
}

fn opening_credit_overlay(idx: usize) -> &'static str {
    opening_credits().get(idx).map(|c| c.overlay).unwrap_or("")
}
#[derive(Clone)]
struct Cue {
    start: f32,
    end: f32,
    text: String,
    is_meta: bool,
}
fn lyric_srt_text() -> &'static str {
    // Timeline captions from the original (SRT embedded).
    r#"1
00:00:00,120 --> 00:00:54,570
DIRTY MELODY RECORDS - Owns All Rights

2
00:00:53,970 --> 00:01:04,980
J'me lève | I get up / I'm getting up

3
00:01:02,210 --> 00:01:33,059
J'm'ennuie | I'm bored / I'm getting bored “La trêve monotone” feels like a ceasefire that offers no relief—peace without renewal.

4
00:01:25,530 --> 00:01:38,639
J'm'apprête | I'm getting ready. “L’impasse qui sonne” is striking: a dead end that rings, implying an alarm, echo, or realization that won’t stop.

5
00:01:38,840 --> 00:01:40,620
Je rêve de mon lit | I dream of my bed

6
00:01:40,620 --> 00:01:49,680
Elle s'évapore | It evaporates | feelings of being stuck in a loop—dreaming without escape.

7
00:01:49,680 --> 00:01:53,439
it does indeed feel like a dream

8
00:01:53,439 --> 00:01:57,000
the one where you do as you please

9
00:01:57,000 --> 00:02:00,240
not exactly reality

10
00:02:00,240 --> 00:02:03,520
just slightly off when you touch trees

11
00:02:03,520 --> 00:02:06,600
i hope we don't drown in the cream

12
00:02:06,600 --> 00:02:13,520
better ways to let off steam; doesn't matter what you say

13
00:02:13,520 --> 00:02:23,840
you're not coming to my first day

14
00:02:24,860 --> 00:02:36,139
“Rêves monochromes” suggests dreams drained of color—routine, emotional numbness, or repetition.

15
00:02:38,220 --> 00:02:49,629
“L’impasse qui sonne” is striking: a dead end that rings, implying an alarm, echo, or realization that won’t stop.

16
00:02:51,680 --> 00:03:07,719
“M’assomment jusqu’à l’aube” carries both exhaustion and surrender: being overwhelmed all night, not resting but dulled.

17
00:03:07,470 --> 00:03:16,610
Monochrome dreams -
Knock me out until dawn |
The monotonous truce -
The dead end that rings

18
00:03:16,840 --> 00:03:19,370
so

19
00:03:19,370 --> 00:03:24,669
Monochrome dreams -
Knock me out until dawn

20
00:03:24,820 --> 00:03:30,960
The monotonous truce -
The dead end that rings

21
00:03:31,760 --> 00:03:33,670
you

22
00:03:33,670 --> 00:03:37,080
[Music]

23
00:03:42,670 --> 00:03:58,840
[Music]

24
00:03:58,840 --> 00:04:01,840
sounds

25
00:04:05,410 --> 00:04:08,569
[DIRTY MELODY RECORDS - Owns All Rights]

26
00:04:25,230 --> 00:04:30,089
[Music]

27
00:04:33,470 --> 00:05:12,410
FANTASMAGORIA

28
00:05:15,420 --> 00:05:21,230
FANTASMAGORIA [Music]

29
00:05:22,840 --> 00:05:26,840
J'me lève J'm'ennuie J'm'apprête
I get up, I’m bored, I get ready.

30
00:05:26,900 --> 00:05:32,110
Je rêve de mon lit
Elle s'évapore |
I dream of my bed,
it evaporates

31
00:05:37,199 --> 00:05:41,840
Sans cesse
Coule de mes pores
Cette ritournelle
Endlessly,
it flows from my pores,
this little refrain.

32
00:05:41,700 --> 00:05:47,279
Ça éclabousse, ça m'fout la frousse (J'ai la chair de poule)
It splashes, it freaks me out (I’ve got goosebumps).

33
00:05:50,690 --> 00:05:56,010
Ça éclabousse, ça m'fout la frousse (J'ai vraiment très peur)
It splashes, it really scares me (I’m really, really scared).

34
00:06:04,570 --> 00:06:09,620
C'est pas palpable, j'suis malléable |
It’s not tangible, I’m malleable.

35
00:06:09,670 --> 00:06:16,560
Salem ghostly face;
Peaking through the haze

36
00:06:16,560 --> 00:06:21,840
i see the space; at the end of the track

37
00:06:24,660 --> 00:06:35,190
I promise its worth it
We might find an end to yesterday

38
00:06:35,190 --> 00:06:38,430
[Applause]

39
00:06:41,320 --> 00:06:44,370
I'm down
You're up

40
00:06:46,840 --> 00:06:49,440
Here's an idea you should

41
00:06:49,440 --> 00:06:58,720
stand up so you can fall down

42
00:07:00,840 --> 00:07:03,120
faster.

43
00:07:03,120 --> 00:07:06,240
Faster, so we can slow down

44
00:07:11,400 --> 00:07:14,840
Stand up, so we can fall down

45
00:07:14,840 --> 00:07:18,800
so we can fall down

46
00:07:20,840 --> 00:07:31,840
Wake up! should we go downtown?

47
00:07:32,580 --> 00:07:47,520
I'm Down.  You're Up.

48
00:07:47,520 --> 00:07:59,280
i'm down; you're up.

49
00:07:59,280 --> 00:08:13,840
I'm down.  (we're done)

50
00:08:18,320 --> 00:08:29,769
*phone beeps*
Salut Cap' !
Yo Mat', j'suis arrivé, tu me rejoins ?
I’ve arrived—are you coming to meet me?

51
00:08:30,299 --> 00:08:38,519
Bah j'suis déjà à l'intérieur. Y a du brouillard partout, y fait super sombre. Mais tu vas voir, c'est très cool
Well, I’m already inside. There’s fog everywhere, it’s super dark. But you’ll see—it’s really cool.
Okay, j'arrive.

52
00:08:38,599 --> 00:08:41,599
Okay, I’m on my way.
dave horner 12/2025
"#
}
fn lyric_cues() -> &'static [Cue] {
    use std::sync::OnceLock;
    static CUES: OnceLock<Vec<Cue>> = OnceLock::new();
    CUES.get_or_init(|| parse_srt(lyric_srt_text()))
}
fn find_cue_index(cues: &[Cue], t: f32) -> i32 {
    for (i, c) in cues.iter().enumerate() {
        if t >= c.start && t <= c.end {
            return i as i32;
        }
    }
    -1
}
fn parse_srt(srt: &str) -> Vec<Cue> {
    let srt = srt.replace('\r', "");
    let blocks: Vec<&str> = srt
        .split("\n\n")
        .map(str::trim)
        .filter(|b| !b.is_empty())
        .collect();
    let mut out = Vec::new();

    for block in blocks {
        let lines: Vec<&str> = block
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect();
        if lines.len() < 2 {
            continue;
        }
        let time_line = lines.iter().copied().find(|l| l.contains("-->"));
        let Some(time_line) = time_line else { continue };
        let mut parts = time_line.split("-->").map(str::trim);
        let Some(a) = parts.next() else { continue };
        let Some(b) = parts.next() else { continue };

        let (Some(start), Some(end)) = (parse_timecode(a), parse_timecode(b)) else {
            continue;
        };

        let text_lines: Vec<&str> = lines
            .iter()
            .copied()
            .filter(|l| *l != time_line && l.parse::<u32>().is_err())
            .collect();
        let text = text_lines.join(" ").trim().to_string();
        if text.is_empty() {
            continue;
        }
        let is_meta = matches!(text.to_lowercase().as_str(), "[music]" | "[applause]");
        out.push(Cue {
            start,
            end,
            text,
            is_meta,
        });
    }

    out.sort_by(|a, b| a.start.total_cmp(&b.start));
    out
}
fn parse_timecode(tc: &str) -> Option<f32> {
    let (hms, ms) = tc.split_once(',')?;
    let mut parts = hms.split(':');
    let hh: f32 = parts.next()?.parse::<f32>().ok()?;
    let mm: f32 = parts.next()?.parse::<f32>().ok()?;
    let ss: f32 = parts.next()?.parse::<f32>().ok()?;
    let ms: f32 = ms.parse::<f32>().ok()?;
    Some(hh * 3600.0 + mm * 60.0 + ss + (ms / 1000.0))
}
