#![cfg(not(target_arch = "wasm32"))]

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use bevy_burn_human::BurnHumanSource;

const DEFAULT_TENSOR_REL: &str = "assets/model/fullbody_default.safetensors";
const DEFAULT_META_REL: &str = "assets/model/fullbody_default.meta.json";

const DEFAULT_RELEASE_ZIP_URL: &str = "https://github.com/davehorner/bevy_Mcbaise_Pale_Regard_Absurdia_Fantasmagoria/releases/download/assets/mcbaise_assets.zip";

fn file_exists(path: &Path) -> bool {
    fs::metadata(path).map(|m| m.is_file()).unwrap_or(false)
}

fn ensure_dir(path: &Path) -> io::Result<()> {
    fs::create_dir_all(path)
}

fn cache_root_dir() -> io::Result<PathBuf> {
    let proj = directories::ProjectDirs::from("io", "mcbaise", "bevy_mcbaise_fantasmagoria")
        .ok_or_else(|| io::Error::other("failed to resolve user data dir"))?;
    Ok(proj.data_local_dir().to_path_buf())
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    let mut s = String::with_capacity(out.len() * 2);
    for b in out {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn download_to_bytes(url: &str) -> io::Result<Vec<u8>> {
    let resp = ureq::get(url)
        .set("User-Agent", "bevy_mcbaise_fantasmagoria")
        .call()
        .map_err(|e| io::Error::other(format!("download failed: {e}")))?;

    if resp.status() < 200 || resp.status() >= 300 {
        return Err(io::Error::other(format!(
            "download failed: HTTP {}",
            resp.status()
        )));
    }

    let mut reader = resp.into_reader();
    let mut buf = Vec::new();
    reader
        .read_to_end(&mut buf)
        .map_err(|e| io::Error::other(format!("read download: {e}")))?;
    Ok(buf)
}

fn unzip_into(zip_bytes: &[u8], out_dir: &Path) -> io::Result<()> {
    let cursor = std::io::Cursor::new(zip_bytes);
    let mut archive =
        zip::ZipArchive::new(cursor).map_err(|e| io::Error::other(format!("open zip: {e}")))?;

    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| io::Error::other(format!("zip entry: {e}")))?;

        let Some(name) = file.enclosed_name().map(|p| p.to_owned()) else {
            continue;
        };

        let out_path = out_dir.join(name);
        if file.is_dir() {
            ensure_dir(&out_path)?;
            continue;
        }

        if let Some(parent) = out_path.parent() {
            ensure_dir(parent)?;
        }

        let mut out_file = fs::File::create(&out_path)?;
        io::copy(&mut file, &mut out_file)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Some(mode) = file.unix_mode() {
                fs::set_permissions(&out_path, fs::Permissions::from_mode(mode))?;
            }
        }
    }

    Ok(())
}

/// Resolve where to load the BurnHuman reference model from.
///
/// Priority:
/// 1) `MCBAISE_BURN_HUMAN_TENSOR` + `MCBAISE_BURN_HUMAN_META`
/// 2) `./assets/model/...` (repo checkout)
/// 3) cached assets under user data dir (auto-download from GitHub Release if missing)
pub fn resolve_burn_human_source() -> BurnHumanSource {
    let override_tensor = std::env::var("MCBAISE_BURN_HUMAN_TENSOR").ok();
    let override_meta = std::env::var("MCBAISE_BURN_HUMAN_META").ok();
    if let (Some(t), Some(m)) = (override_tensor, override_meta) {
        return BurnHumanSource::Paths {
            tensor: PathBuf::from(t),
            meta: PathBuf::from(m),
        };
    }

    let local_tensor = PathBuf::from(DEFAULT_TENSOR_REL);
    let local_meta = PathBuf::from(DEFAULT_META_REL);
    if file_exists(&local_tensor) && file_exists(&local_meta) {
        return BurnHumanSource::Paths {
            tensor: local_tensor,
            meta: local_meta,
        };
    }

    let zip_url = std::env::var("MCBAISE_ASSETS_ZIP_URL")
        .ok()
        .filter(|v| !v.trim().is_empty());
    let zip_url = zip_url.as_deref().unwrap_or(DEFAULT_RELEASE_ZIP_URL);

    let expected_sha = std::env::var("MCBAISE_ASSETS_ZIP_SHA256")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty());

    let cache_root = cache_root_dir().unwrap_or_else(|e| {
        eprintln!("[mcbaise] warning: {e}; falling back to current directory assets");
        PathBuf::from(".")
    });

    let cache_assets_root = cache_root.join("assets");
    let cached_tensor = cache_assets_root.join("model/fullbody_default.safetensors");
    let cached_meta = cache_assets_root.join("model/fullbody_default.meta.json");

    let force_download = std::env::var("MCBAISE_ASSETS_FORCE_DOWNLOAD")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if !force_download && file_exists(&cached_tensor) && file_exists(&cached_meta) {
        return BurnHumanSource::Paths {
            tensor: cached_tensor,
            meta: cached_meta,
        };
    }

    let auto = std::env::var("MCBAISE_ASSETS_AUTO_DOWNLOAD")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);

    if !auto {
        panic!(
            "BurnHuman assets missing. Place {} and {} next to the executable (relative to CWD), or set MCBAISE_BURN_HUMAN_TENSOR/MCBAISE_BURN_HUMAN_META, or set MCBAISE_ASSETS_AUTO_DOWNLOAD=1.",
            DEFAULT_TENSOR_REL, DEFAULT_META_REL
        );
    }

    if force_download {
        eprintln!("[mcbaise] forcing assets download: {zip_url}");
    } else {
        eprintln!("[mcbaise] BurnHuman assets not found; downloading {zip_url} ...");
    }
    eprintln!("[mcbaise] cache dir: {}", cache_root.display());

    ensure_dir(&cache_assets_root).unwrap_or_else(|e| {
        panic!(
            "failed to create cache dir {}: {e}",
            cache_assets_root.display()
        )
    });

    let zip_bytes = download_to_bytes(zip_url).unwrap_or_else(|e| {
        panic!("failed to download assets zip: {e}");
    });

    if let Some(expected) = expected_sha {
        let got = sha256_hex(&zip_bytes);
        if got != expected {
            panic!("assets zip sha256 mismatch: expected {expected}, got {got}");
        }
    }

    // Extract into cache root (zip should contain `assets/...`).
    unzip_into(&zip_bytes, &cache_root).unwrap_or_else(|e| {
        panic!("failed to extract assets zip: {e}");
    });

    if !(file_exists(&cached_tensor) && file_exists(&cached_meta)) {
        panic!(
            "assets zip extracted but model files still missing at {} and {}",
            cached_tensor.display(),
            cached_meta.display()
        );
    }

    BurnHumanSource::Paths {
        tensor: cached_tensor,
        meta: cached_meta,
    }
}
