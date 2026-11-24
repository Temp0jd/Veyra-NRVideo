// Heuristic reactive mask for 2D video / capture-card inputs fed to FSR 3.1 / 4.x.
//
// Unlike a 3D game engine, HDMI capture and video streams lack per-object depth,
// transparency masks, and exact sub-pixel G-buffer motion vectors. Without a
// reactive mask, FSR aggressively accumulates stale history across disocclusions,
// 8x8 optical-flow block boundaries, and static HUD elements over moving scenes,
// producing ghosting and smearing.
//
// Signals combined into R8_UNORM [0, 1] at render extent:
//   1. Warped temporal color error & 3x3 neighborhood clamping excursion
//      (detects disocclusions, particles, flashes, and optical-flow misses).
//   2. Motion-vector boundary shear (detects 8x8 AMD optical-flow block edges).
//   3. High-contrast static HUD over moving background (prevents reticle/UI smear).
//
// Also copies current render-extent color into NextPrev (u1) so the graph keeps
// a dedicated, unconditional previous-frame reference independent of retainReferences.

Texture2D<float4> CurrColor : register(t0);
Texture2D<float4> PrevColor : register(t1);
Texture2D<float2> FlowTex   : register(t2);
RWTexture2D<float> ReactiveOut : register(u0);
RWTexture2D<float4> NextPrevOut : register(u1);

cbuffer Params : register(b0) {
    uint Width;
    uint Height;
    float Strength;    // [0..1] user/graph reactive scaling (default ~0.75)
    uint HasHistory;   // 0 on reset / first frame
    uint FlowValid;    // 1 if FlowTex contains estimated motion
    float3 Pad;
};

float3 Perceptual(float3 linearRgb) {
    float3 c = max(linearRgb, 0.0.xxx);
    float peak = max(c.r, max(c.g, c.b));
    return c / (1.0 + peak);
}

int2 ClampPos(int2 p) {
    return clamp(p, int2(0, 0), int2(int(Width) - 1, int(Height) - 1));
}

float3 SamplePrevBilinear(float2 pos) {
    float2 clamped = clamp(pos, float2(0.0, 0.0), float2(float(Width) - 1.0, float(Height) - 1.0));
    int2 p0 = int2(floor(clamped));
    int2 p1 = min(p0 + int2(1, 1), int2(int(Width) - 1, int(Height) - 1));
    float2 f = frac(clamped);

    float3 c00 = Perceptual(PrevColor.Load(int3(p0.x, p0.y, 0)).rgb);
    float3 c10 = Perceptual(PrevColor.Load(int3(p1.x, p0.y, 0)).rgb);
    float3 c01 = Perceptual(PrevColor.Load(int3(p0.x, p1.y, 0)).rgb);
    float3 c11 = Perceptual(PrevColor.Load(int3(p1.x, p1.y, 0)).rgb);
    return lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= Width || id.y >= Height) return;
    const int2 p = int2(id.xy);
    const float4 currRaw = CurrColor.Load(int3(p, 0));
    NextPrevOut[p] = currRaw;

    if (HasHistory == 0 || Strength <= 0.001 || !all(isfinite(currRaw.rgb))) {
        ReactiveOut[p] = (HasHistory == 0) ? 1.0 : 0.0;
        return;
    }

    const float3 currP = Perceptual(currRaw.rgb);
    float3 cMin = currP;
    float3 cMax = currP;
    [unroll] for (int dy = -1; dy <= 1; ++dy) {
        [unroll] for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            float3 n = Perceptual(CurrColor.Load(int3(ClampPos(p + int2(dx, dy)), 0)).rgb);
            cMin = min(cMin, n);
            cMax = max(cMax, n);
        }
    }
    const float localContrast = length(cMax - cMin);

    float2 flow = float2(0.0, 0.0);
    if (FlowValid != 0) {
        flow = FlowTex.Load(int3(p, 0));
        if (!all(isfinite(flow))) flow = float2(0.0, 0.0);
    }

    // Reproject into previous frame using FSR / Veyra current -> previous convention.
    const float2 prevPos = float2(p) + flow;
    const bool outOfBounds = (prevPos.x < 0.0 || prevPos.y < 0.0 ||
                              prevPos.x > float(Width) - 1.0 || prevPos.y > float(Height) - 1.0);
    if (outOfBounds) {
        ReactiveOut[p] = saturate(Strength);
        return;
    }

    const float3 prevWarped = SamplePrevBilinear(prevPos);
    const float warpDiff = length(currP - prevWarped);
    const float3 clampedPrev = clamp(prevWarped, cMin, cMax);
    const float clampExcursion = length(prevWarped - clampedPrev);

    // Signal 1: warped disocclusion / flow-prediction mismatch.
    const float rWarp = max(smoothstep(0.045, 0.22, warpDiff),
                            smoothstep(0.025, 0.14, clampExcursion));

    // Signal 2: motion-vector discontinuity (8x8 optical-flow block boundary shear).
    float rShear = 0.0;
    if (FlowValid != 0) {
        float2 fL = FlowTex.Load(int3(ClampPos(p + int2(-2, 0)), 0));
        float2 fR = FlowTex.Load(int3(ClampPos(p + int2( 2, 0)), 0));
        float2 fU = FlowTex.Load(int3(ClampPos(p + int2( 0,-2)), 0));
        float2 fD = FlowTex.Load(int3(ClampPos(p + int2( 0, 2)), 0));
        float shear = max(max(length(flow - fL), length(flow - fR)),
                          max(length(flow - fU), length(flow - fD)));
        rShear = smoothstep(1.5, 6.0, shear) * 0.78;
    }

    // Signal 3: static sharp HUD element over moving scene (dragged by coarse flow).
    const float3 prevStatic = Perceptual(PrevColor.Load(int3(p, 0)).rgb);
    const float staticDiff = length(currP - prevStatic);
    const float flowMag = length(flow);
    float rHud = 0.0;
    if (FlowValid != 0 && localContrast > 0.10 && staticDiff < 0.018 && flowMag > 1.25) {
        rHud = smoothstep(1.25, 4.0, flowMag) * 0.85;
    }

    const float reactive = saturate(max(max(rWarp, rShear), rHud) * Strength);
    ReactiveOut[p] = reactive;
}
