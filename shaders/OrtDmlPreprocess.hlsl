// SPDX-License-Identifier: GPL-3.0-only
// M7 super-resolution input: linear BT.709 scRGB (RGBA16F) -> NCHW float32.
//
// The model sees what its training data looked like: a display-referred sRGB
// picture. Channels == 1 encodes BT.709 luma Y' (ArtCNN and similar luma-only
// networks); channels == 3 writes the R'/G'/B' planes in that order. Values are
// saturated into [0, 1] because these models are SDR only: HDR working space is
// rejected by the backend instead of silently clipping.
//
// The source texture may have any extent; sampling is done with a bilinear
// clamp sampler at texel centres of the model grid so a 1x model still reads
// the right pixel and a scaling model gets a box-free, smooth resample.

cbuffer OrtDmlConstants : register(b0)
{
    uint Width;    // model input width
    uint Height;   // model input height
    uint Channels; // 1 (luma) or 3 (RGB)
    uint Reserved;
};

Texture2D<float4> Source : register(t0);
SamplerState BilinearClamp : register(s1);
RWStructuredBuffer<float> Out : register(u0);

// ITU-R BT.709 luma weights, applied to sRGB-encoded values (the space the
// models and the BT.709 matrix are defined in).
static const float3 KrKgKb = float3(0.2126, 0.7152, 0.0722);

float SrgbEncode(float c)
{
    c = saturate(c);
    return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint plane = Width * Height;
    const uint index = id.x;
    if (index >= plane * Channels) return;

    const uint channel = index / plane;
    const uint pixel = index - channel * plane;
    const uint x = pixel % Width;
    const uint y = pixel / Width;

    const float2 uv = float2((x + 0.5) / Width, (y + 0.5) / Height);
    const float3 linearRgb = Source.SampleLevel(BilinearClamp, uv, 0).rgb;
    const float3 srgb = float3(SrgbEncode(linearRgb.r), SrgbEncode(linearRgb.g), SrgbEncode(linearRgb.b));

    float value;
    if (Channels == 1) {
        value = dot(srgb, KrKgKb);
    } else if (channel == 0) {
        value = srgb.r;
    } else if (channel == 1) {
        value = srgb.g;
    } else {
        value = srgb.b;
    }
    Out[index] = value;
}
