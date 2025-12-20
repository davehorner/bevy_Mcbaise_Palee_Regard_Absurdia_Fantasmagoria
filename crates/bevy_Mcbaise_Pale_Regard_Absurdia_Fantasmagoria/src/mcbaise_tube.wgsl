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

fn aa_grid(u: f32, v: f32) -> f32 {
    // Procedural "wireframe" look: thin grid lines in UV space.
    // u: around (0..1), v: along (0..1)
    let grid_u = 28.0;
    let grid_v = 64.0;

    let fu = fract(u * grid_u);
    let fv = fract(v * grid_v);
    let du = min(fu, 1.0 - fu);
    let dv = min(fv, 1.0 - fv);
    let d = min(du, dv);

    let w = max(fwidth(u * grid_u), fwidth(v * grid_v)) * 1.75;
    // 1 on lines, 0 in faces.
    return 1.0 - smoothstep(0.0, w, d);
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
    var phase: f32;
    let p = i32(round(pattern));
    if (p == 0 || p == 2) {
        // stripes: fewer turns, more axial flow
        let theta = ang + s_warp * TAU * 16.0;
        phase = theta * (bands * 0.75) + time * (flow * 3.0);
    } else {
        // swirl
        let theta = ang + time * spin + s_warp * TAU * turns;
        phase = theta * bands + time * flow;
    }

    let band = aa_band(phase, aa);

    let t = smoothstep(white_bias, 1.0, band);
    let base_col = mix(white.rgb, orange.rgb, t);
    var col = base_col;

    if (p == 2 || p == 3) {
        // Wireframe variants: show mostly dark faces with bright grid lines.
        let wire = aa_grid(mesh.uv.y, mesh.uv.x);
        col = mix(vec3<f32>(0.03, 0.03, 0.03), base_col, wire);
    }

    let depth = smoothstep(0.0, 1.0, s);
    let dark = select(dark_inside.rgb, dark_outside.rgb, is_front);
    col = mix(col, dark, depth * 0.40);

    return vec4<f32>(col, 1.0);
}
