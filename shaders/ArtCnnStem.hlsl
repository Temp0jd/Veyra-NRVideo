#include "ArtCnn.hlsli"

// conv0: 1 -> C over the scalar luma buffer. The weight block is one float4 per
// (output quad, tap) holding the four output channels of that quad, so a single
// mad per tap produces four channels at once. No activation, and the result is
// kept as the long-skip source for the fused residual add.
StructuredBuffer<float4> Weights : register(t0);
StructuredBuffer<float4> In : register(t1); // luma, .x only (InQuads == 1)
RWStructuredBuffer<float4> Out : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID)
{
    const uint index = dispatchId.x;
    if (index >= InW * InH * OutQuads) return;
    const uint pixel = index / OutQuads;
    const uint q = index - pixel * OutQuads;
    const uint y = pixel / InW;
    const uint x = pixel - y * InW;

    float4 acc = Weights[BiasOffset + q];
    [unroll]
    for (uint tap = 0; tap < 9; ++tap)
    {
        const int iy = int(y) + int(tap / 3u) - 1;
        const int ix = int(x) + int(tap % 3u) - 1;
        // SAME_UPPER zero padding for a 3x3 kernel: one texel of zeros on each side.
        float v = 0.0;
        if (iy >= 0 && iy < int(InH) && ix >= 0 && ix < int(InW))
            v = In[uint(iy) * InW + uint(ix)].x;
        acc = mad(Weights[WeightOffset + q * 9u + tap], float4(v, v, v, v), acc);
    }
    Out[index] = acc;
}
