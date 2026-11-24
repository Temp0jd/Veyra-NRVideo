#ifndef ARTCNN_HLSLI
#define ARTCNN_HLSLI

// Shared root constants (b0, 16 dwords) for every ArtCNN pass. The layout is
// identical across ArtCnnPrep/Stem/Conv/Finish so one upload path drives them
// all; unused fields are simply zero.
cbuffer ArtCnnConstants : register(b0)
{
    uint InW;          // input tensor width
    uint InH;          // input tensor height
    uint OutW;         // output tensor width
    uint OutH;         // output tensor height
    uint InQuads;      // InW*InH tensor channel quads (1 for the scalar luma input)
    uint OutQuads;     // output tensor channel quads (1 for the 4-channel final conv)
    uint WeightOffset; // float4 index of this layer's weight block
    uint BiasOffset;   // float4 index of this layer's bias block
    uint Flags;        // bit0 = ReLU, bit1 = add the conv0 long skip
    uint Pad0;
    uint Pad1;
    uint Pad2;
    uint Pad3;
    uint Pad4;
    uint Pad5;
    uint Pad6;
};

// ITU-R BT.709 luma/chroma basis. The network sees sRGB-encoded values, so the
// conversions below run on sRGB RGB and only the linear<->sRGB transfer differs.
static const float Kr = 0.2126;
static const float Kg = 0.7152;
static const float Kb = 0.0722;
static const float CbScale = 1.8556; // 2 * (1 - Kb)
static const float CrScale = 1.5748; // 2 * (1 - Kr)
// Exact inverse coefficients: Kr*1.5748/Kg and Kb*1.8556/Kg.
static const float GFromCr = 0.46812427;
static const float GFromCb = 0.18732427;

float SrgbEncode(float c)
{
    c = saturate(c);
    if (c <= 0.0031308) return c * 12.92;
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

float SrgbDecode(float c)
{
    c = saturate(c);
    if (c <= 0.04045) return c / 12.92;
    return pow((c + 0.055) / 1.055, 2.4);
}

float3 LinearToSrgb(float3 linearRgb)
{
    return float3(SrgbEncode(linearRgb.r), SrgbEncode(linearRgb.g), SrgbEncode(linearRgb.b));
}

float3 SrgbToLinear(float3 srgbRgb)
{
    return float3(SrgbDecode(srgbRgb.r), SrgbDecode(srgbRgb.g), SrgbDecode(srgbRgb.b));
}

float3 SrgbToYcbcr(float3 srgbRgb)
{
    const float luma = dot(srgbRgb, float3(Kr, Kg, Kb));
    return float3(luma, (srgbRgb.b - luma) / CbScale, (srgbRgb.r - luma) / CrScale);
}

float3 YcbcrToSrgb(float3 ycbcr)
{
    const float r = ycbcr.x + CrScale * ycbcr.z;
    const float b = ycbcr.x + CbScale * ycbcr.y;
    const float g = ycbcr.x - GFromCr * ycbcr.z - GFromCb * ycbcr.y;
    return float3(r, g, b);
}

// Manual bilinear chroma fetch for one output pixel. Chroma is interpolated in
// sRGB space (the space the network and BT.709 matrix are defined in) with a
// texel-centre-correct mapping and clamp-to-edge addressing, so no sampler or
// sampler state is involved at all.
float2 SampleSrgbChroma(Texture2D<float4> source, float2 outPixel)
{
    // OutW == 2*InW, so an output pixel centre maps back to
    // (outPixel + 0.5) / 2 - 0.5 in input pixel coordinates.
    const float2 inPixel = (outPixel + 0.5) * 0.5 - 0.5;
    const float2 base = floor(inPixel);
    const float2 cell = inPixel - base;   // not `frac`: that is an HLSL intrinsic
    const int2 maxCoord = int2(int(InW) - 1, int(InH) - 1);
    float2 chroma = 0.0;
    [unroll]
    for (int j = 0; j < 2; ++j)
    {
        [unroll]
        for (int i = 0; i < 2; ++i)
        {
            const int2 coord = clamp(int2(base) + int2(i, j), int2(0, 0), maxCoord);
            const float3 srgb = LinearToSrgb(source.Load(int3(coord, 0)).rgb);
            const float2 tapChroma = SrgbToYcbcr(srgb).yz;
            const float weight = (i == 0 ? 1.0 - cell.x : cell.x) * (j == 0 ? 1.0 - cell.y : cell.y);
            chroma += tapChroma * weight;
        }
    }
    return chroma;
}

#endif // ARTCNN_HLSLI
