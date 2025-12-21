#![cfg_attr(target_arch = "wasm32", no_main)]

use bevy::asset::RenderAssetUsages;
use bevy::asset::embedded_asset;
use bevy::asset::io::embedded::EmbeddedAssetRegistry;
use bevy::ecs::system::SystemParam;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::pbr::{DistanceFog, FogFalloff, Material, MaterialPlugin};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::AsBindGroup;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::shader::ShaderRef;
#[cfg(not(target_arch = "wasm32"))]
use bevy::window::PrimaryWindow;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

use bevy_burn_human::BurnHumanSource;
use bevy_burn_human::{BurnHumanAssets, BurnHumanInput, BurnHumanPlugin};

#[cfg(not(target_arch = "wasm32"))]
mod native_assets;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

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
                if looks_like_youtube {
                    if has_ytdlp {
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
                        for line in reader.lines().flatten().take(400) {
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
struct NativeYoutubeSync;

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
struct NativeMpvSync;

#[cfg(not(all(windows, not(target_arch = "wasm32"), feature = "native-mpv")))]
#[derive(Resource, Clone)]
struct NativeMpvConfig;

#[cfg(all(windows, not(target_arch = "wasm32"), feature = "native-mpv"))]
impl Drop for NativeMpvSync {
    fn drop(&mut self) {
        let _ = self.tx.send(native_mpv::Command::Shutdown);
        if let Ok(mut slot) = self.join.lock() {
            if let Some(handle) = slot.take() {
                let _ = handle.join();
            }
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
    Ball,
}

impl SubjectMode {
    fn label(self) -> &'static str {
        match self {
            SubjectMode::Auto => "Subject: auto",
            SubjectMode::Random => "Subject: random",
            SubjectMode::Human => "Subject: human",
            SubjectMode::Ball => "Subject: ball",
        }
    }

    fn short_label(self) -> &'static str {
        match self {
            SubjectMode::Auto => "auto",
            SubjectMode::Random => "random",
            SubjectMode::Human => "human",
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
        }
    }

    fn short_label_from_value(v: u32) -> &'static str {
        match v % 4 {
            0 => "orange/white",
            1 => "NIN",
            2 => "black/white",
            _ => "random grey",
        }
    }

    fn from_value(v: u32) -> Self {
        match v % 4 {
            0 => ColorSchemeMode::OrangeWhite,
            1 => ColorSchemeMode::Nin,
            2 => ColorSchemeMode::BlackWhite,
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
}

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

#[derive(Resource, Clone)]
struct AppearanceTextures {
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
        }
    }

    fn short_label_from_value(v: u32) -> &'static str {
        match v % 4 {
            0 => "stripe",
            1 => "swirl",
            2 => "stripe wire",
            _ => "swirl wire",
        }
    }

    fn from_value(v: u32) -> Self {
        match v % 4 {
            0 => TexturePatternMode::Stripe,
            1 => TexturePatternMode::Swirl,
            2 => TexturePatternMode::StripeWire,
            _ => TexturePatternMode::SwirlWire,
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
            current: SubjectMode::Human,
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
            SubjectMode::Ball => SubjectMode::Human,
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
        let prev = self.pattern_current % 4;
        let mut next = prev;
        for _ in 0..8 {
            next = self.lcg_next() % 4;
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
            next = 2 + (self.lcg_next() % 2);
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
    console_error_panic_hook::set_once();

    #[cfg(not(target_arch = "wasm32"))]
    {
        fn truthy_env(name: &str) -> bool {
            std::env::var(name)
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        }

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
            assets_self_test();
            return;
        }

        // Env-var preflight: download/resolve + print paths, then exit.
        let preflight = truthy_env("MCBAISE_ASSETS_PREFLIGHT");
        if preflight {
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
            return;
        }
    }

    let burn_plugin = {
        #[cfg(target_arch = "wasm32")]
        {
            let tensor_vec = lz4_flex::decompress_size_prepended(TENSOR_LZ4_BYTES)
                .expect("decompress embedded burn_human tensor (lz4)");
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
        .insert_resource(OverlayText::default())
        // Native: we create the primary egui context ourselves (on a dedicated overlay camera)
        // so it isn't constrained by the first 3D camera's viewport.
        // WASM: keep the default auto-created primary context.
        .insert_resource(bevy_egui::EguiGlobalSettings {
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
        .init_resource::<MultiView>()
        .init_resource::<MultiViewHint>()
        .add_plugins(plugins)
        .add_plugins(bevy::pbr::wireframe::WireframePlugin::default());

    #[cfg(target_arch = "wasm32")]
    app.insert_resource(VideoVisibility { show: true });

    app.add_plugins(EguiPlugin::default())
        // Ensure embedded assets are registered before materials request shaders.
        .add_plugins(local_embedded_asset_plugin)
        .add_plugins(MaterialPlugin::<TubeMaterial>::default())
        .add_plugins(burn_plugin)
        .add_systems(Startup, setup_scene)
        // Diagnostic: print embedded registry contents at startup to verify registered paths
        .add_systems(
            Startup,
            (register_embedded_asset_source, print_embedded_registry),
        )
        .add_systems(Update, sync_view_cameras)
        .add_systems(Update, update_multiview_viewports.after(sync_view_cameras))
        .add_systems(Update, ensure_subject_normals)
        .add_systems(Update, update_subject_appearance)
        .add_systems(Update, update_tube_and_subject)
        .add_systems(Update, apply_subject_mode)
        .add_systems(Update, update_overlays)
        .add_systems(EguiPrimaryContextPass, ui_overlay);

    #[cfg(not(target_arch = "wasm32"))]
    app.add_systems(Update, update_overlay_ui_camera);

    #[cfg(target_arch = "wasm32")]
    app.add_systems(Update, apply_js_input);

    #[cfg(not(target_arch = "wasm32"))]
    app.add_systems(Update, (advance_time_native, native_controls));

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

    println!(
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
    mut layout: ResMut<NativeYoutubeWindowLayout>,
    mut primary: Query<(Entity, &mut Window), With<PrimaryWindow>>,
    winit_windows: Option<NonSend<bevy::winit::WinitWindows>>,
) {
    let Some(sync) = sync else {
        return;
    };
    if !sync.enabled {
        return;
    }

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

        if layout.force_topmost
            && let Some(winit_windows) = winit_windows.as_ref()
            && let Some(w) = winit_windows.get_window(primary_entity)
        {
            let hwnd: Option<HWND> =
                w.window_handle()
                    .ok()
                    .and_then(|handle| match handle.as_raw() {
                        RawWindowHandle::Win32(h) => Some(h.hwnd.get() as HWND),
                        _ => None,
                    });

            if let Some(hwnd) = hwnd {
                unsafe {
                    SetWindowPos(
                        hwnd,
                        HWND_TOPMOST,
                        0,
                        0,
                        0,
                        0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
                    );
                    // If minimized, restore. (Doesn't force foreground.)
                    ShowWindow(hwnd, SW_RESTORE);
                }
            }
        }
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

    if cookie_file.is_none() {
        if let Ok(v) = std::env::var("MCBAISE_MPV_COOKIES_FROM_BROWSER") {
            let v = v.trim();
            if !v.is_empty() {
                ytdl_raw_options.push(format!("cookies-from-browser={v}"));
            }
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
    mut app_exit: EventReader<bevy::app::AppExit>,
    mut window_close: EventReader<bevy::window::WindowCloseRequested>,
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
                    disable = true;
                    disable_reason = Some(msg);
                    break;
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
    assets: Res<BurnHumanAssets>,
) {
    let polkadot = images.add(make_polkadot_texture());
    commands.insert_resource(AppearanceTextures { polkadot });

    commands.insert_resource(AmbientLight {
        color: Color::srgb(1.0, 1.0, 1.0),
        brightness: 0.15,
        affects_lightmapped_meshes: true,
    });

    // Key + fill + rim lighting to help the subject read as 3D at all angles.
    commands.spawn((
        DirectionalLight {
            illuminance: 35_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(10.0, 18.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

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

    commands.spawn((
        Mesh3d(tube_mesh),
        MeshMaterial3d(tube_mat.clone()),
        Transform::default(),
        TubeTag,
        Name::new("tube"),
    ));

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
fn native_controls(
    keys: Res<ButtonInput<KeyCode>>,
    mut playback: ResMut<Playback>,
    mut settings: ResMut<TubeSettings>,
    mut scheme_mode: ResMut<ColorSchemeMode>,
    mut pattern_mode: ResMut<TexturePatternMode>,
    _native_youtube: Option<Res<NativeYoutubeSync>>,
    _native_mpv: Option<Res<NativeMpvSync>>,
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
            if let Some(mpv) = _native_mpv.as_ref() {
                if mpv.enabled {
                    let _ = mpv
                        .tx
                        .send(crate::native_mpv::Command::SetPlaying(playback.playing));
                }
            }
        }
    }
    if keys.just_pressed(KeyCode::Digit1) {
        settings.scheme = (settings.scheme + 1) % 4;
        *scheme_mode = ColorSchemeMode::from_value(settings.scheme);
    }
    if keys.just_pressed(KeyCode::Digit2) {
        settings.pattern = (settings.pattern + 1) % 4;
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
            settings.pattern = (settings.pattern + 1) % 4;
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

                auto_style.pattern_current % 4
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
            auto_style.pattern_current % 4
        }
        TexturePatternMode::Stripe => 0,
        TexturePatternMode::Swirl => 1,
        TexturePatternMode::StripeWire => 2,
        TexturePatternMode::SwirlWire => 3,
    };

    auto_style.scheme_current = scheme;
    auto_style.pattern_current = pattern;
    settings.scheme = scheme;
    settings.pattern = pattern;

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
        SubjectMode::Human | SubjectMode::Ball => {
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
        // Roughly once per second, re-anchor to the current progress.
        if auto_cam.pass_reanchor_since_sec >= 1.0 {
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

    Image::new_fill(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    )
}

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

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
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
    }: UiOverlayParams,
) {
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };

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
                    multi_view_hint.show(now_sec, "right click to remove");
                }
            } else if (resp.clicked_by(egui::PointerButton::Secondary) || resp.long_touched())
                && !multi_view.decrement()
            {
                multi_view_hint.show(now_sec, "left click to add more");
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
                                        *subject_mode = SubjectMode::Human
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
                            if let Ok(slot) = mpv.last_error.lock() {
                                if let Some(err) = slot.as_ref() {
                                    ui.label(format!("mpv: {err}"));
                                }
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
                            if let Some(mpv) = _native_mpv.as_ref() {
                                if mpv.enabled {
                                    let _ = mpv.tx.send(crate::native_mpv::Command::SetPlaying(
                                        desired_playing,
                                    ));
                                }
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
            | CameraMode::PassingBottom => (0.9, 1.2),
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

// ---------------------------- tube material ----------------------------

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct TubeMaterial {
    #[uniform(0)]
    // Pack everything into a single uniform buffer: WebGPU has a low per-stage uniform-buffer limit.
    // Layout: [params0, params1, orange, white, dark_inside, dark_outside]
    u: [Vec4; 6],
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
        }
    }
}

impl TubeMaterial {
    fn set_time(&mut self, t: f32) {
        self.u[0].x = t;
    }

    fn set_scheme(&mut self, scheme: u32, time_sec: f32) {
        match scheme % 4 {
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
            _ => {
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
        }
    }

    fn set_pattern(&mut self, pattern: u32) {
        self.u[1].w = (pattern % 4) as f32;
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
