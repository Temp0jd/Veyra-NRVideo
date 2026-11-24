// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <d3d12.h>
#include <cstdint>
#include <memory>
#include <string>

namespace veyra::gfx {
// AMD AMF HQScaler: the driver's Video SR 1.0 / 1.1 upscaler plus the classical
// bilinear/bicubic/point filters, from the AMF runtime the AMD driver installs
// (System32\amfrt64.dll; built against the AMF SDK headers, MIT). The component
// is created from the System32 copy of the runtime and scales anything the AMF
// surface maps accept (core/HQScaler.h doc: BGRA, RGBA, R10G10B10A2, RGBA_F16,
// NV12, P010); the output format equals the input format, the component never
// converts. Video SR 1.1 is the experimental DX11/DX12-only model and rejects
// NV12/P010 input; Video SR 1.0 (upstream enum value 2, shared with the
// deprecated FSR entry) takes every format above. SHARPNESS (0..2, default 0.5)
// only applies to the two Video SR models.
//
// Per frame: capture() records a copy into an input texture, the owner submits
// that list, process() hands the frame to AMF and drains the result, copyOutput()
// makes the queue wait for AMF and records the copy into the caller's output,
// the owner submits that list and calls submitted(). No CPU waits on the GPU
// while frames flow; textures cross queues through AMF's fence protocol
// (core/D3D12AMF.h), exactly as AmfFrcBackend does.
class AmfHqScalerBackend {
public:
    AmfHqScalerBackend();
    ~AmfHqScalerBackend();
    AmfHqScalerBackend(const AmfHqScalerBackend&) = delete;
    AmfHqScalerBackend& operator=(const AmfHqScalerBackend&) = delete;
    // An AMD driver's AMF runtime is installed (file check only, nothing loaded).
    static bool runtimeInstalled();
    // This build was compiled with the AMF SDK headers.
    static bool compiledIn();
    // AMF_HQ_SCALER_ALGORITHM values, in upstream order. VideoSr1_0 also answers
    // to the deprecated AMF_HQ_SCALER_ALGORITHM_FSR name, whose value it shares.
    enum class Algorithm : std::uint8_t { Bilinear = 0, Bicubic = 1, VideoSr1_0 = 2, Point = 3, VideoSr1_1 = 4 };
    struct Config {
        Algorithm algorithm = Algorithm::VideoSr1_0;
        // 0..2; used by VideoSr1_0 and VideoSr1_1 only.
        float sharpness = 0.5f;
        // Both flags are always set explicitly: the header defaults (true/true)
        // contradict the component documentation (false/true).
        bool keepAspectRatio = false;
        bool fromSrgb = false;
    };
    // True when the format is one of the AMF-mapped textures and the algorithm
    // accepts it: VideoSr1_1 rejects NV12/P010, everything else accepts them.
    static bool supportsFormat(DXGI_FORMAT format, Algorithm algorithm);
    // 0..2, NaN becomes the 0.5 default.
    static float clampSharpness(float value);
    bool initialize(ID3D12Device*, ID3D12CommandQueue*, uint32_t inputWidth, uint32_t inputHeight,
                    uint32_t outputWidth, uint32_t outputHeight, DXGI_FORMAT, const Config& config);
    // `input` is in COPY_SOURCE; the list runs on the queue given to initialize().
    bool capture(ID3D12GraphicsCommandList*, ID3D12Resource* input);
    // After the list holding capture() was submitted. reset flushes the component
    // first (seek, cut, resize, pause resume); the output AMF then returns for the
    // first frame is still returned and shown.
    bool process(bool reset);
    // Queue wait for AMF, then the copy into `output` (COPY_DEST) at the output
    // extent. False when AMF returned no frame for this input.
    bool copyOutput(ID3D12GraphicsCommandList*, ID3D12Resource* output);
    // After the list holding copyOutput() was submitted: hands AMF's surface back.
    bool submitted();
    bool ready() const;
    // Empty when the runtime never loaded.
    const std::string& runtimeVersion() const;
    // Human-readable reason of the last failed call ("" when none).
    const char* lastError() const;
    void shutdown();
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}
