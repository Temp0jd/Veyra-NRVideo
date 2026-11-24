// SPDX-License-Identifier: GPL-3.0-only
// M7 super-resolution output: NCHW float32 -> linear BT.709 scRGB (RGBA16F).
//
// Channels == 1 (luma-only model): the network replaced Y' and nothing else, so
// chroma is taken from the same linear frame the preprocess consumed. Because
// that frame may have a different extent than the model output, chroma is
// sampled with a bilinear clamp sampler at the output pixel centre, converted
// to sRGB and reduced to BT.709 Cb/Cr; the result is recombined around the new
// luma and decoded back to linear.
//
// Channels == 3 (RGB model): the three planes are the new R'/G'/B'.

cbuffer OrtDmlConstants : register(b0)
{
    uint Width;    // output width
    uint Height;   // output height
    uint Channels; // 1 (luma) or 3 (RGB)
    uint Reserved;
};

StructuredBuffer<float> In : register(t0);
Texture2D<float4> Source : register(t1);
SamplerState BilinearClamp : register(s1);
RWTexture2D<float4> Output : register(u0);

static const float3 KrKgKb = float3(0.2126, 0.7152, 0.0722);
static const float CbScale = 1.8556; // 2 * (1 - Kb)
static const float CrScale = 1.5748; // 2 * (1 - Kr)
// Exact inverse coefficients: Kr*1.5748/Kg and Kb*1.8556/Kg.
static const float GFromCr = 0.46812427;
static const float GFromCb = 0.18732427;

float SrgbEncode(float c)
{
    c = saturate(c);
    return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

float SrgbDecode(float c)
{
    c = saturate(c);
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

float3 YcbcrToSrgb(float3 ycbcr)
{
    const float r = ycbcr.x + CrScale * ycbcr.z;
    const float b = ycbcr.x + CbScale * ycbcr.y;
    const float g = ycbcr.x - GFromCr * ycbcr.z - GFromCb * ycbcr.y;
    return float3(r, g, b);
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;

    const uint plane = Width * Height;
    const uint index = id.y * Width + id.x;

    float3 srgb;
    if (Channels == 1) {
        const float luma = saturate(In[index]);
        const float2 uv = float2((id.x + 0.5) / Width, (id.y + 0.5) / Height);
        const float3 linearSource = Source.SampleLevel(BilinearClamp, uv, 0).rgb;
        const float3 srgbSource =
            float3(SrgbEncode(linearSource.r), SrgbEncode(linearSource.g), SrgbEncode(linearSource.b));
        const float sourceLuma = dot(srgbSource, KrKgKb);
        const float cb = (srgbSource.b - sourceLuma) / CbScale;
        const float cr = (srgbSource.r - sourceLuma) / CrScale;
        srgb = YcbcrToSrgb(float3(luma, cb, cr));
    } else {
        srgb = float3(In[index], In[plane + index], In[2 * plane + index]);
    }

    Output[int2(id.x, id.y)] =
        float4(SrgbDecode(srgb.r), SrgbDecode(srgb.g), SrgbDecode(srgb.b), 1.0);
}
