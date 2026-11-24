#include "ArtCnn.hlsli"

// conv6 (C -> 4), DepthToSpace(2, DCR), Clip(0, 1), then chroma recombination.
// One thread per input pixel: it produces the 2x2 output block from the four
// predicted subpixels plus bilinearly interpolated sRGB chroma taken from the
// original source, and writes linear BT.709 RGBA with alpha forced to 1.
StructuredBuffer<float4> Weights : register(t0);
StructuredBuffer<float4> In : register(t1); // conv5 output + conv0 output (the fused add)
Texture2D<float4> Source : register(t2);
RWTexture2D<float4> Out : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID)
{
    const uint pixel = dispatchId.x;
    if (pixel >= InW * InH) return;
    const uint y = pixel / InW;
    const uint x = pixel - y * InW;

    // Single output quad, so q == 0 and the bias is one float4.
    float4 acc = Weights[BiasOffset];
    [unroll]
    for (uint tap = 0; tap < 9; ++tap)
    {
        const int iy = int(y) + int(tap / 3u) - 1;
        const int ix = int(x) + int(tap % 3u) - 1;
        if (iy < 0 || iy >= int(InH) || ix < 0 || ix >= int(InW)) continue;
        const uint texel = (uint(iy) * InW + uint(ix)) * InQuads;
        for (uint iq = 0; iq < InQuads; ++iq)
        {
            const float4 v = In[texel + iq];
            const uint wb = WeightOffset + (iq * 9u + tap) * 4u;
            acc = mad(Weights[wb + 0u], v.xxxx, acc);
            acc = mad(Weights[wb + 1u], v.yyyy, acc);
            acc = mad(Weights[wb + 2u], v.zzzz, acc);
            acc = mad(Weights[wb + 3u], v.wwww, acc);
        }
    }
    // Clip(min=0, max=1) from the graph, in sRGB-encoded luma space.
    acc = saturate(acc);

    // DepthToSpace(blocksize=2, DCR): channel i*2+j goes to (2x+j, 2y+i), i.e.
    // .x -> (0,0), .y -> (1,0), .z -> (0,1), .w -> (1,1).
    const float subpixels[4] = { acc.x, acc.y, acc.z, acc.w };
    [unroll]
    for (uint i = 0; i < 2; ++i)
    {
        [unroll]
        for (uint j = 0; j < 2; ++j)
        {
            const uint2 dst = uint2(2u * x + j, 2u * y + i);
            const float2 chroma = SampleSrgbChroma(Source, float2(dst));
            // Network luma is sRGB-encoded, matching the chroma space.
            const float3 srgb = saturate(YcbcrToSrgb(float3(subpixels[i * 2 + j], chroma)));
            Out[dst] = float4(SrgbToLinear(srgb), 1.0);
        }
    }
}
