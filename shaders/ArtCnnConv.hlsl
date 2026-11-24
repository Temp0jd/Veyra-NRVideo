#include "ArtCnn.hlsli"

// Interior C -> C convolutions, one thread per (pixel, output quad).
//
// An input quad float4 dotted into an output quad at one tap is a 4x4
// matrix-vector product, so the weight block holds four float4 rows per
// (output quad, input quad, tap) and each row is folded in with one mad. The
// same pass serves all four interior layers plus the no-activation layer before
// the residual add, and it fuses that add when Flags bit 1 is set: the added
// tensor is the conv0 output, which is exactly what the graph adds before the
// final convolution.
StructuredBuffer<float4> Weights : register(t0);
StructuredBuffer<float4> In : register(t1);
StructuredBuffer<float4> Skip : register(t2); // conv0 output for the fused long skip
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
        // SAME_UPPER zero padding for a 3x3 kernel.
        if (iy < 0 || iy >= int(InH) || ix < 0 || ix >= int(InW)) continue;
        // Parentheses around (iy * InW + ix) are load bearing: the tensor is
        // channel-last, so the pixel index scales by the quad count.
        const uint texel = (uint(iy) * InW + uint(ix)) * InQuads;
        for (uint iq = 0; iq < InQuads; ++iq)
        {
            const float4 v = In[texel + iq];
            const uint wb = WeightOffset + ((q * InQuads + iq) * 9u + tap) * 4u;
            acc = mad(Weights[wb + 0u], v.xxxx, acc);
            acc = mad(Weights[wb + 1u], v.yyyy, acc);
            acc = mad(Weights[wb + 2u], v.zzzz, acc);
            acc = mad(Weights[wb + 3u], v.wwww, acc);
        }
    }
    // Activation first, then the graph's Add(conv5, conv0). The activation layer
    // and the add layer are different nodes, so the order is unambiguous.
    if ((Flags & 1u) != 0) acc = max(acc, 0.0);
    if ((Flags & 2u) != 0) acc += Skip[index];
    Out[index] = acc;
}
