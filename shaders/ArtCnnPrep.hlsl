#include "ArtCnn.hlsli"

// Linear BT.709 working colour -> sRGB-encoded BT.709 luma, one float4 per pixel
// with the luma in .x. ArtCNN was trained on sRGB luma, so the transfer
// function matters: doing the conversion here keeps the network input exact and
// every convolution pass at a single scalar per pixel.
Texture2D<float4> Source : register(t0);
RWStructuredBuffer<float4> LumaOut : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID)
{
    const uint pixel = dispatchId.x;
    if (pixel >= InW * InH) return;
    const uint y = pixel / InW;
    const uint x = pixel - y * InW;
    const float3 srgb = LinearToSrgb(Source.Load(int3(int(x), int(y), 0)).rgb);
    const float luma = dot(srgb, float3(Kr, Kg, Kb));
    LumaOut[pixel] = float4(luma, 0.0, 0.0, 0.0);
}
