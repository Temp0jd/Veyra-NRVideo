#pragma once

// ArtCNN 2x luma super resolution (Artoriuz/ArtCNN, MIT).
//
// The backend splits cleanly in two halves:
//   * a pure, platform independent model reader/packer (this header plus the
//     matching code in src/pipeline/ArtCnnBackend.cpp) that turns an upstream
//     ArtCNN ONNX file into the float4 weight layout the kernels consume, and
//   * a D3D12 runtime that uploads those weights and records four compute
//     passes (prep -> stem -> interior convs -> finish).
// The pure half is unit tested on Linux/Wine; only the runtime half needs
// D3D12, so this header stays includable without <d3d12.h> when
// VEYRA_ARTCNN_NO_D3D12 is defined.
//
// Graph (C4F16: C=16, C4F32: C=32; any C4F* with C % 4 == 0):
//   conv0  3x3 1->C   no activation      (the long skip source S)
//   conv1..convN 3x3 C->C + ReLU         (N >= 1; upstream N = 4)
//   convN+1 3x3 C->C  no activation
//   Add(convN+1, conv0)                  <- skip BEFORE the final conv
//   conv6  3x3 C->4   no activation
//   DepthToSpace(blocksize=2, DCR), Clip(0, 1)
// Interior tensors are channel-last float4: index (y * inW + x) * quads + q.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32) && !defined(VEYRA_ARTCNN_NO_D3D12)
#include <d3d12.h>

#include "veyra/pipeline/GpuPassUtils.h"
#define VEYRA_ARTCNN_HAS_D3D12 1
#else
#define VEYRA_ARTCNN_HAS_D3D12 0
#endif

namespace veyra::pipeline {

// One packed convolution. Offsets are in float4 units inside
// ArtCnnPackedModel::weightsF4.
struct ArtCnnLayerSpec {
    uint32_t inChannels = 0;
    uint32_t outChannels = 0;
    uint32_t weightOffsetF4 = 0;
    uint32_t biasOffsetF4 = 0;
    bool relu = false;
};

struct ArtCnnPackedModel {
    uint32_t channels = 0;      // C, always a multiple of 4
    uint32_t quads = 0;         // C / 4
    // conv0, conv1 .. convN+1, conv6 (the graph order).
    std::vector<ArtCnnLayerSpec> layers;
    // Flattened float4 groups: weightsF4.size() == totalFloat4s * 4. Layouts:
    //   stem (1 -> C)    : W[wOff + q * 9 + (ky*3+kx)]        = (w[4q+0][0][ky][kx], .., w[4q+3][0][ky][kx])
    //   conv (Cin -> Cout): W[wOff + ((q*inQuads+iq)*9+tap)*4 + c]
    //                                                        = (w[4q+0][4iq+c][ky][kx], .., w[4q+3][4iq+c][ky][kx])
    //   bias             : W[bOff + q]                        = (b[4q+0], .., b[4q+3])
    std::vector<float> weightsF4;
    // Kernel coefficients only (biases are stored separately and are not counted):
    // 12240 for C4F16, 47520 for C4F32.
    uint64_t paramCount = 0;

    uint64_t weightBytes() const { return uint64_t(weightsF4.size()) * 4; }
    // Luma input (Y) plus the three float4 intermediates (S, A, B).
    uint64_t workingSetBytes(uint32_t inW, uint32_t inH) const
    {
        const uint64_t pixels = uint64_t(inW) * uint64_t(inH);
        return pixels * 16ull + 3ull * pixels * uint64_t(quads) * 16ull;
    }
    uint64_t multiplyAccumulates(uint32_t inW, uint32_t inH) const
    {
        const uint64_t pixels = uint64_t(inW) * uint64_t(inH);
        uint64_t perPixel = 0;
        for (const auto& layer : layers)
            perPixel += 9ull * uint64_t(layer.inChannels) * uint64_t(layer.outChannels);
        return pixels * perPixel;
    }
};

// Reads and validates an ArtCNN ONNX file, then packs its weights.
// Anything outside the graph shape above (wrong strides/dilations/group/kernel,
// missing residual add, blocksize != 2, non-DCR or missing DepthToSpace, a
// channel count that is not a multiple of 4 in [4, 64], non-float32 tensors,
// unsupported operators) is rejected with a message instead of guessed.
bool parseAndPackArtCnnModel(const std::vector<uint8_t>& onnxBytes, ArtCnnPackedModel& out,
                             std::string& error);
bool loadAndPackArtCnnModel(const std::filesystem::path& modelPath, ArtCnnPackedModel& out,
                            std::string& error);

class ArtCnnBackend {
public:
    ArtCnnBackend();
    ~ArtCnnBackend();
    ArtCnnBackend(const ArtCnnBackend&) = delete;
    ArtCnnBackend& operator=(const ArtCnnBackend&) = delete;

    // Only integral 2x in each axis, with a non-empty input.
    static bool supportedRatio(uint32_t inW, uint32_t inH, uint32_t outW, uint32_t outH)
    {
        return inW > 0 && inH > 0 && outW == inW * 2 && outH == inH * 2;
    }

    bool created() const;
    const ArtCnnPackedModel& model() const;
    uint64_t weightBytes() const;
    uint64_t workingSetBytes() const;
    uint64_t multiplyAccumulates() const;
    void shutdown();

#if VEYRA_ARTCNN_HAS_D3D12
    // hdrWorking == true is refused: the network is trained on sRGB luma, so an
    // HDR working space would need a proxy token, which is not implemented.
    bool initialize(ID3D12Device* device, const std::filesystem::path& modelPath,
                    uint32_t inW, uint32_t inH, uint32_t outW, uint32_t outH,
                    bool hdrWorking, std::string& error);
    // Records prep -> stem -> interior convolutions -> finish. `color` is the
    // linear BT.709 working texture (RGBA16F/RGBA32F) at inW x inH, `output` the
    // 2x target. Weights are copied into their default-heap buffer on the first
    // call (inside this list, no CPU fence wait). Returns false while latched.
    bool run(ID3D12GraphicsCommandList* list, StateTracker& tracker, ID3D12Resource* color,
             ID3D12Resource* output);
#endif

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace veyra::pipeline
