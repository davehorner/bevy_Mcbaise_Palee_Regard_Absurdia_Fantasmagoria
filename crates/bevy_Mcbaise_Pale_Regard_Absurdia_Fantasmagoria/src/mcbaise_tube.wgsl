#import bevy_pbr::forward_io::VertexOutput

// Pack everything into a single uniform buffer: WebGPU has a low per-stage uniform-buffer limit.
// Layout matches Rust `TubeMaterial.u: [Vec4; 6]`:
// [params0, params1, orange, white, dark_inside, dark_outside]
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> u: array<vec4<f32>, 6>;

const TAU: f32 = 6.28318530718;

fn aa_band(phase: f32, aa_mul: f32) -> f32 {
    let s = 0.5 + 0.5 * sin(phase);
    let w = fwidth(phase) * aa_mul;
    return smoothstep(0.5 - w, 0.5 + w, s);
}

// Robust grid helper (safe, self-contained). Use this instead of the
// corrupted `aa_grid` if the original became invalid during edits.
fn aa_grid_safe(u: f32, v: f32, velocity: vec2<f32>, time: f32, flow: f32, is_swirl: bool) -> f32 {
    let gu = fract(u * 96.0);
    let gv = fract(v * 48.0);
    let du_val = min(gu, 1.0 - gu);
    let dv_val = min(gv, 1.0 - gv);
    let w_u = max(fwidth(u * 96.0), 1e-6) * 1.5;
    let w_v = max(fwidth(v * 48.0), 1e-6) * 1.5;
    let line_u = 1.0 - smoothstep(0.0, w_u, du_val);
    let line_v = 1.0 - smoothstep(0.0, w_v, dv_val);
    return max(line_u, line_v);
}

// Antialiased swirl-contour wire helper. Draws thin contour lines following
// the phase (useful for swirl patterns). Conservative, stable math using
// `fwidth` to produce screen-space-aware line thickness.
fn aa_swirl_wire(phase: f32, aa_mul: f32) -> f32 {
    let s = 0.5 + 0.5 * sin(phase);
    let d = abs(s - 0.5);
    let w = max(fwidth(phase), 1e-6) * aa_mul * 0.6;
    return 1.0 - smoothstep(0.0, w, d);
}

fn aa_grid(u: f32, v: f32, velocity: vec2<f32>, time: f32, flow: f32, is_swirl: bool) -> f32 {
    // Procedural "wireframe" look: thin grid lines in UV space.
    let fu = fract(u * 96.0);
    let fv = fract(v * 48.0);
    let du_val = min(fu, 1.0 - fu);
    let dv_val = min(fv, 1.0 - fv);
    let w_u = max(fwidth(u * 96.0), 1e-6) * 1.5;
    let w_v = max(fwidth(v * 48.0), 1e-6) * 1.5;
    let line_u = 1.0 - smoothstep(0.0, w_u, du_val);
    let line_v = 1.0 - smoothstep(0.0, w_v, dv_val);
    return max(line_u, line_v);
}

fn aa_hoop_wire(coord: f32, rings: f32, aa_mul: f32) -> f32 {
    // Thin circular (hoop) wire lines around the tube — lines that follow
    // the angular UV coordinate so they form closed rings.
    let f = fract(coord * rings);
    let d = min(f, 1.0 - f);
    let w = fwidth(coord * rings) * aa_mul;
    return 1.0 - smoothstep(0.0, w, d);
}

fn aa_vertical_lines(coord: f32, count: f32, aa_mul: f32) -> f32 {
    // Thin vertical lines at constant angular positions (run along the tube axis).
    let f = fract(coord * count);
    let d = min(f, 1.0 - f);
    let w = fwidth(coord * count) * aa_mul;
    return 1.0 - smoothstep(0.0, w, d);
}

fn hash_f32(x: f32) -> f32 {
    return fract(sin(x) * 43758.5453);
}

@fragment
fn fragment(mesh: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let params0 = u[0];
    let params1 = u[1];
    let orange = u[2];
    let white = u[3];
    let dark_inside = u[4];
    let dark_outside = u[5];

    let time = params0.x;
    let bands = params0.y;
    let turns = params0.z;
    let spin = params0.w;

    let flow = params1.x;
    let aa = params1.y;
    let white_bias = params1.z;
    let pattern = params1.w;

    let ang = mesh.uv.y * TAU;
    let s = mesh.uv.x;

    let s_warp = pow(s, 1.18);

    // pattern 0: stripe
    // pattern 1: swirl
    // pattern 2: stripe (wire)
    // pattern 3: swirl (wire)
    // pattern 4: wave
    // pattern 5: fractal
    // pattern 6: particle
    // pattern 7: grid
    // pattern 8: hoop (wire)
    // pattern 9: hoop (alt)
    var phase: f32;
    var base_col: vec3<f32>;
    var final_alpha: f32 = 1.0;
    var skip_depth_mix: bool = false;
    let p = i32(round(pattern));
    if (p == 0 || p == 2) {
        // stripes: fewer turns, more axial flow
        let theta = ang + s_warp * TAU * 16.0;
        phase = theta * (bands * 0.75) + time * (flow * 3.0);
        let t = smoothstep(white_bias, 1.0, aa_band(phase, aa));
        base_col = mix(white.rgb, orange.rgb, t);
    } else if (p == 4) {
        // Wave: layered sine waves (procedural)
        let uv = vec2<f32>(mesh.uv.x, mesh.uv.y);
        let v = vec2<f32>(
            sin((uv.y * 3.0 + time * 0.35) * TAU),
            cos((uv.x * 2.0 - time * 0.27) * TAU),
        ) * 0.04;
        let adv = uv + v;

        // Two crossing wave fields for interference
        let w1 = sin((adv.x * 12.0 + time * 1.6) * TAU) * 0.5 + 0.5;
        let w2 = sin((adv.y * 18.0 - time * 1.1) * TAU + adv.x * 0.6) * 0.5 + 0.5;
        let wave = pow(w1 * 0.6 + w2 * 0.4, 1.2);

        phase = adv.x * bands * 1.5 + time * flow * 0.6;
        let tt = smoothstep(white_bias, 1.0, aa_band(phase, aa));
        // modulate by wave interference for a trippy look
        base_col = mix(white.rgb, orange.rgb, mix(tt, wave, 0.6));

    } else if (p == 5) {
        // Fractal: simple multi-octave sine "fbm" for psychedelic texture
        let uv = vec2<f32>(mesh.uv.x, mesh.uv.y);
        var f = 0.0;
        var amp = 1.0;
        var freq = 1.0;
        for (var i: i32 = 0; i < 5; i = i + 1) {
            let a = sin((uv.x * freq + uv.y * freq * 0.6 + time * (0.2 * f32(i + 1))) * TAU);
            f = f + a * amp;
            amp = amp * 0.5;
            freq = freq * 2.0;
        }
        let fbm = 0.5 + 0.5 * f;
        phase = fbm * bands * 1.2 + time * flow * 0.5;
        let tt = smoothstep(white_bias, 1.0, aa_band(phase, aa));
        // push colors more into magenta/blue by using the mix factor nonlinearly
        base_col = mix(white.rgb, orange.rgb, pow(tt * fbm, 0.8));

    } else if (p == 6) {
        // Particle: spotty particle-like field that drifts over time
        let uv = vec2<f32>(mesh.uv.x, mesh.uv.y);
        let adv = uv + vec2<f32>(time * 0.02, -time * 0.015);

        // Create high-frequency spot field using product of sines
        let s = sin(adv.x * 120.0 * TAU) * sin(adv.y * 120.0 * TAU);
        let spots = smoothstep(0.98, 1.0, s);
        let brightness = spots * 0.9;
        base_col = vec3<f32>(0.02) + mix(vec3<f32>(0.0), mix(white.rgb, orange.rgb, 0.9), brightness);

        phase = adv.x * bands + time * flow * 0.5;

    } else if (p == 7) {
        // Grid: compute lines for both UV axis orientations and combine them
        // This ensures the grid wraps correctly regardless of which UV axis is
        // the tube's circumference.
        // Separate high-resolution hoop constant (for internal AA) from the
        // grid cell size: define `cell_size` in UV space (fraction of circumference)
        // and compute `hoop_count` from it. This lets users pick a physical
        // cell size, not a raw number of hoops.
        let hoop_resolution = 4800.0; // high-res internal sampling
        // Make grid hoops much denser and suppress vertical line density
        // aggressively for a clearer hoop-dominant grid. 4x denser again.
        let cell_size = 1.0 / 1536.0; // much denser than original
        let hoop_count = max(round(1.0 / cell_size), 1.0);

        // --- Grid A: U = mesh.uv.x, V = mesh.uv.y ---
        let du_a = max(fwidth(mesh.uv.x), 1e-6);
        let dv_a = max(fwidth(mesh.uv.y), 1e-6);
        // Reduce vertical density inside the grid calculation so axis-aligned
        // vertical lines are sparser relative to hoops (approx 1/4 density)
        var grid_v_a = hoop_count * (dv_a / du_a) * 0.25;
        grid_v_a = clamp(grid_v_a, 1.0, 512.0);
        let fu_a = fract(mesh.uv.x * hoop_count);
        let fv_a = fract(mesh.uv.y * grid_v_a);
        let du_val_a = min(fu_a, 1.0 - fu_a);
        let dv_val_a = min(fv_a, 1.0 - fv_a);
        let w_u_a = fwidth(mesh.uv.x * hoop_count) * 1.5;
        let w_v_a = fwidth(mesh.uv.y * grid_v_a) * 1.5;
        let line_u_a = 1.0 - smoothstep(0.0, w_u_a, du_val_a);
        let line_v_a = 1.0 - smoothstep(0.0, w_v_a, dv_val_a);
        let line_a = max(line_u_a, line_v_a);

        // --- Grid B: swapped axes (U = mesh.uv.y, V = mesh.uv.x) ---
        let du_b = max(fwidth(mesh.uv.y), 1e-6);
        let dv_b = max(fwidth(mesh.uv.x), 1e-6);
        var grid_v_b = hoop_count * (dv_b / du_b) * 0.25;
        grid_v_b = clamp(grid_v_b, 1.0, 512.0);
        let fu_b = fract(mesh.uv.y * hoop_count);
        let fv_b = fract(mesh.uv.x * grid_v_b);
        let du_val_b = min(fu_b, 1.0 - fu_b);
        let dv_val_b = min(fv_b, 1.0 - fv_b);
        let w_u_b = fwidth(mesh.uv.y * hoop_count) * 1.5;
        let w_v_b = fwidth(mesh.uv.x * grid_v_b) * 1.5;
        let line_u_b = 1.0 - smoothstep(0.0, w_u_b, du_val_b);
        let line_v_b = 1.0 - smoothstep(0.0, w_v_b, dv_val_b);
        let line_b = max(line_u_b, line_v_b);

        // Combine both grid calculations so lines along both orientations show up.
        // Using max preserves thin wire appearance from either calculation.
        let axis_lines = max(line_a, line_b);

        // Add hoop/ring wires (closed circles around the tube) to reinforce
        // the grid look. Draw visible hoops using `visible_hoops` while the
        // high-resolution `hoop_resolution` is available for other uses.
        let hoops = aa_hoop_wire(mesh.uv.x, hoop_count, 1.5);

        // Blend axis-aligned lines and hoops. Make hoops dominate: lower
        // axis_lines contribution so the grid reads as hoops with sparse axes.
        let line = max(axis_lines * 0.3, hoops);

        let dark = dark_inside.rgb;
        let wire_col = mix(white.rgb, orange.rgb, 0.5);
        base_col = mix(dark, wire_col, line);
        phase = mesh.uv.x * bands + time * flow;
    } else if (p == 8) {
        // HoopWire: pure wireframe — dark faces with hoop rings and vertical
        // grey lines. No colored faces beneath the wires.
        let hoop_resolution = 4800.0;
        // Increase hoop density for HoopWire (denser rings)
        // 4x denser than original
        let cell_size = 1.0 / 384.0;
        let hoop_count = max(round(1.0 / cell_size), 1.0);
        let hoops = aa_hoop_wire(mesh.uv.x, hoop_count, 1.5);

        // Vertical lines (run along tube axis) with per-line random grey shades
        // Determine vertical_count from base_grid_u and UV derivative ratio so
        // spacing between vertical lines matches hoop spacing.
        // Use a small fixed number of verticals so the HoopWire mode shows
        // ~4-5 verticals; make them the same color as the hoop wires.
        var vertical_count = 5.0;
        let lines_a = aa_vertical_lines(mesh.uv.y, vertical_count, 1.0);
        let idx = floor(mesh.uv.y * vertical_count);
        let r = hash_f32(idx);

        // Vertical lines use the same wire color as hoops
        let wire_col = mix(white.rgb, orange.rgb, 0.5);
        let grey_col = wire_col;

        // Show vertical lines only between hoops
        let between = 1.0 - clamp(hoops, 0.0, 1.0);
        // Reduce alpha so verticals don't occlude the mesh; keep subtle strength
        let lines_alpha = lines_a * between * 0.12;

        // Compose: start dark, overlay vertical lines, then overlay hoop wires
        let dark_face = vec3<f32>(0.03, 0.03, 0.03);
        var tmp = mix(dark_face, grey_col, lines_alpha);
        let wire_alpha = clamp(hoops, 0.0, 1.0);
        base_col = mix(tmp, wire_col, wire_alpha);
        // Make faces transparent where there are no wires/lines
        final_alpha = clamp(max(wire_alpha, lines_alpha), 0.0, 1.0);
        skip_depth_mix = true;
        phase = mesh.uv.x * bands + time * flow;

    } else if (p == 9) {
        // HoopAlt: behave the same as Grid — axis-aligned grid plus hoop rings.
        // Increase hoop density for HoopAlt to match denser default
        let base_grid_u = 192.0;

        // --- Grid A: U = mesh.uv.x, V = mesh.uv.y ---
        let du_a = max(fwidth(mesh.uv.x), 1e-6);
        let dv_a = max(fwidth(mesh.uv.y), 1e-6);
        var grid_v_a = base_grid_u * (dv_a / du_a);
        grid_v_a = clamp(grid_v_a, 8.0, 512.0);
        let fu_a = fract(mesh.uv.x * base_grid_u);
        let fv_a = fract(mesh.uv.y * grid_v_a);
        let du_val_a = min(fu_a, 1.0 - fu_a);
        let dv_val_a = min(fv_a, 1.0 - fv_a);
        let w_u_a = fwidth(mesh.uv.x * base_grid_u) * 1.5;
        let w_v_a = fwidth(mesh.uv.y * grid_v_a) * 1.5;
        let line_u_a = 1.0 - smoothstep(0.0, w_u_a, du_val_a);
        let line_v_a = 1.0 - smoothstep(0.0, w_v_a, dv_val_a);
        let line_a = max(line_u_a, line_v_a);

        // --- Grid B: swapped axes (U = mesh.uv.y, V = mesh.uv.x) ---
        let du_b = max(fwidth(mesh.uv.y), 1e-6);
        let dv_b = max(fwidth(mesh.uv.x), 1e-6);
        var grid_v_b = base_grid_u * (dv_b / du_b);
        grid_v_b = clamp(grid_v_b, 8.0, 512.0);
        let fu_b = fract(mesh.uv.y * base_grid_u);
        let fv_b = fract(mesh.uv.x * grid_v_b);
        let du_val_b = min(fu_b, 1.0 - fu_b);
        let dv_val_b = min(fv_b, 1.0 - fv_b);
        let w_u_b = fwidth(mesh.uv.y * base_grid_u) * 1.5;
        let w_v_b = fwidth(mesh.uv.x * grid_v_b) * 1.5;
        let line_u_b = 1.0 - smoothstep(0.0, w_u_b, du_val_b);
        let line_v_b = 1.0 - smoothstep(0.0, w_v_b, dv_val_b);
        let line_b = max(line_u_b, line_v_b);

        // Combine both grid calculations so lines along both orientations show up.
        let axis_lines = max(line_a, line_b);

        // Add hoop/ring wires (closed circles around the tube) to reinforce
        // the grid look. Name suggestion for this effect: "HoopWire".
        let hoops = aa_hoop_wire(mesh.uv.x, base_grid_u, 1.5);

        let line = max(axis_lines * 0.75, hoops);

        let dark = dark_inside.rgb;
        let wire_col = mix(white.rgb, orange.rgb, 0.5);
        base_col = mix(dark, wire_col, line);
        phase = mesh.uv.x * bands + time * flow;
    } else {
        // swirl (default)
        let theta = ang + time * spin + s_warp * TAU * turns;
        phase = theta * bands + time * flow;
        let t = smoothstep(white_bias, 1.0, aa_band(phase, aa));
        base_col = mix(white.rgb, orange.rgb, t);
    }

    let band = aa_band(phase, aa);

    let t = smoothstep(white_bias, 1.0, band);
    var col = base_col;

    if (p == 2 || p == 3) {
        // Wireframe variants: show mostly dark faces with bright grid lines.
        var wire = 0.0;
        if (p == 3) {
            // Swirl wire: use contours of the swirl pattern
            wire = aa_swirl_wire(phase, aa);
        } else {
            // Stripe wire: use traditional grid
            wire = aa_grid_safe(mesh.uv.y, mesh.uv.x, vec2<f32>(0.0), time, flow, false);
        }
        col = mix(vec3<f32>(0.03, 0.03, 0.03), base_col, wire);
    }

    // If this variant requested skipping depth-based darkening (wire-only),
    // don't mix with the dark inside/outside color. Otherwise apply depth mix.
    if (!skip_depth_mix) {
        let depth = smoothstep(0.0, 1.0, s);
        let dark = select(dark_inside.rgb, dark_outside.rgb, is_front);
        col = mix(col, dark, depth * 0.40);
    }

    // Final alpha: use `final_alpha` (1.0 by default, <1 for wire-only variants)
    return vec4<f32>(col, final_alpha);
}
