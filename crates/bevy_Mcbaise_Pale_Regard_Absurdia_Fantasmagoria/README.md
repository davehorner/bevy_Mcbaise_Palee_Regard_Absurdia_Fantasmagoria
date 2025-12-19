# bevy_Mcbaise_Pale_Regard_Absurdia_Fantasmagoria

<https://davehorner.github.io/bevy_Mcbaise_Pale_Regard_Absurdia_Fantasmagoria/>

Forked from https://github.com/mosure/burn_human.git

Bevy port of the CodePen <https://codepen.io/Dave-Horner/pen/zxqgOOo> “absurdia-stuck in an endless loop in mcdonalds playground slide” visualization, using `bevy_burn_human` as the moving subject.

<p align="center">
	<img alt="Screenshot" src="https://raw.githubusercontent.com/davehorner/bevy_Mcbaise_Pale_Regard_Absurdia_Fantasmagoria/main/crates/bevy_Mcbaise_Pale_Regard_Absurdia_Fantasmagoria/screenshot.jpg" />
</p>

## Run (native)

```cmd
task mcbaise:native
```

### Native YouTube sync (optional)

This opens the YouTube watch page in a real browser and syncs the Bevy playback time/playing state to it via WebDriver.

Prereqs:
- Install a WebDriver, e.g. Chrome + `chromedriver`.
- The `task mcbaise:native:youtube` task will try to auto-launch `chromedriver` if it's on your `PATH`.
- If you prefer to start it yourself, run: `chromedriver --port=9515`.

Run:

```cmd
task mcbaise:native:youtube
```

Optional:
- Set `MCBAISE_WEBDRIVER_URL` to point at a different WebDriver endpoint.
- Set `MCBAISE_LAUNCH_WEBDRIVER=0` to disable auto-launch.
- To import a repo-root Netscape cookies file into the automated Chrome session (avoids needing your real Chrome profile), use `cookies.txt` at repo root or set:
	- `MCBAISE_CHROME_COOKIES_TXT=cookies.txt`
	- Disable with `MCBAISE_YOUTUBE_DISABLE_COOKIE_TXT=1`
- To reuse your existing logged-in Chrome profile (cookies/account), set:
	- `MCBAISE_CHROME_USER_DATA_DIR` (example: `C:\Users\<you>\AppData\Local\Google\Chrome\User Data`)
	- `MCBAISE_CHROME_PROFILE_DIR` (example: `Default` or `Profile 1`)

Notes when using your real profile:
- Close all Chrome windows first (Chrome locks the profile directory).
- On Windows, Chrome may keep running in the background even after you close the last window; fully exit it (or end `chrome.exe`) to release the profile lock.
- If the browser never opens, check that your `chromedriver` major version matches your installed Chrome major version.
- Using a real profile can change ad behavior vs a fresh automation profile.

Notes:
- In this mode, YouTube is authoritative: Bevy time/play state follow the browser.

Getting `cookies.txt` (Chrome extension):
- Install “Get cookies.txt locally”:
	- https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc
- In Chrome, make sure you are logged into YouTube, then open a YouTube page (e.g. `https://www.youtube.com/`).
- Click the extension icon and export/download cookies in Netscape `cookies.txt` format.
- Save the file as repo-root `cookies.txt` (next to `Cargo.toml`).
	- This project will auto-detect `cookies.txt` for both `native-mpv` and `native-youtube`.

Security note: `cookies.txt` contains your logged-in session cookies; treat it like a password and don’t commit it.

### Native mpv sync (optional)

This runs `mpv` and syncs the Bevy playback time/playing state from mpv via IPC.

Prereqs:
- Windows only (currently).
- Install `mpv` and ensure it is on your `PATH`.
- For YouTube URLs, install `yt-dlp` and ensure it is on your `PATH` (recommended; `youtube-dl` often fails on modern YouTube).

Run:

```cmd
task mcbaise:native:mpv
```

Optional:
- Use a different URL:
	- `task mcbaise:native:mpv MPV_URL=https://www.youtube.com/watch?v=...`
- Use a specific mpv executable:
	- `task mcbaise:native:mpv MPV_PATH=C:\\path\\to\\mpv.exe`
- Pass extra mpv args (space-separated):
	- `task mcbaise:native:mpv MPV_EXTRA_ARGS=--fullscreen`

If YouTube shows “Sign in to confirm you’re not a bot”, prefer a repo-root `cookies.txt` (Netscape cookie file):
	- Place `cookies.txt` at the repo root (auto-detected), or pass it explicitly:
		- `task mcbaise:native:mpv MPV_COOKIES_FILE=cookies.txt`

Alternate (less reliable on Windows if Chrome is running): pass cookies from your browser:
	- `task mcbaise:native:mpv MPV_COOKIES_FROM_BROWSER=chrome`

To disable all automatic cookie behavior:
	- Set `MCBAISE_MPV_DISABLE_AUTO_COOKIES=1`

Advanced (yt-dlp raw options, comma-separated):
	- `task mcbaise:native:mpv MPV_YTDL_RAW_OPTIONS=cookies=cookies.txt`

Notes:
- In this mode, mpv is authoritative: Bevy time/play state follow mpv.

Troubleshooting (YouTube):
- If you see errors like “Only images are available” or “n challenge solving failed”, yt-dlp is failing YouTube’s JS challenge (EJS) and will not provide real audio/video formats.
	- Install `deno` and ensure it is on `PATH`.
	- Enable EJS remote components and force the web client:
		- `task mcbaise:native:mpv MPV_YTDL_RAW_OPTIONS=remote-components=ejs:github,extractor-args=youtube:player_client=web`
	- To disable the app’s default YouTube extractor args behavior:
		- Set `MCBAISE_MPV_DISABLE_DEFAULT_YOUTUBE_EXTRACTOR_ARGS=1`

## Run (web)

This repo uses go-task (https://taskfile.dev/) to build and serve the wasm demo:

```cmd
task mcbaise:wasm:serve
```

Open the printed URL (served from repo root so `/assets/...` is available).

Notes:
- The published crates.io package name for this demo is `bevy_mcbaise_fantasmagoria`.
- The Taskfile runs `wasm-bindgen` and writes output to `crates/bevy_Mcbaise_Pale_Regard_Absurdia_Fantasmagoria/www/pkg`.

## GitHub Pages (wasm hosting)

The repo root includes a GitHub Actions workflow that builds and publishes this demo to GitHub Pages:

- Workflow: `.github/workflows/deploy.yml`
- Pages URL: `https://davehorner.github.io/bevy_Mcbaise_Pale_Regard_Absurdia_Fantasmagoria/`

Enable Pages (create if missing; uses GitHub Actions as the build source):

```cmd
gh api -X POST repos/davehorner/bevy_Mcbaise_Pale_Regard_Absurdia_Fantasmagoria/pages -f build_type=workflow
```

Or update an existing Pages config:

```cmd
gh api -X PUT repos/davehorner/bevy_Mcbaise_Pale_Regard_Absurdia_Fantasmagoria/pages -f build_type=workflow
```
