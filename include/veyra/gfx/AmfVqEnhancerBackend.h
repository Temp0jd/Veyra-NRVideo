// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <d3d12.h>
#include <cstdint>
#include <memory>
#include <string>

namespace veyra::gfx {
// AMD AMF VQEnhancer: the runtime's FCR enhancement (component id AMFVQEnhancer
// from components/VQEnhancer.h), the same block the driver exposes as the video
// quality enhancement. Surfaces go in as BGRA, RGBA, R10G10B10A2, RGBA_F16, NV12
// or P010 and come back in the same format; OUTPUT_SIZE alone switches the
// enhancement on (the API document: setting it triggers the effect).
//
// The engine-type property name comes from the SDK macro and is never written
// out here: upstream the macro value carries a typo (L"AMF_VIDEI_ENHANCER_ENGINE
// _TYPE"), so a hand-written name would silently not configure the component.
// ATTENUATION is a Float in [0.02, 0.4] (default 0.1); RADIUS is an amf_int64
// in [1, 4] that the API document marks deprecated without effect (still set,
// the property exists). SPLIT_VIEW is the experimental 0/1 side-by-side view.
//
// Per frame: capture() records a copy into an input texture, the owner submits
// that list, process() hands the frame to AMF and drains the result, copyOutput()
// makes the queue wait for AMF and records the copy into the caller's output,
// the owner submits that list and calls submitted(). No CPU waits on the GPU
// while frames flow; textures cross queues through AMF's fence protocol
// (core/D3D12AMF.h), exactly as AmfFrcBackend does.
class AmfVqEnhancerBackend {
public:
    AmfVqEnhancerBackend();
    ~AmfVqEnhancerBackend();
    AmfVqEnhancerBackend(const AmfVqEnhancerBackend&) = delete;
    AmfVqEnhancerBackend& operator=(const AmfVqEnhancerBackend&) = delete;
    // An AMD driver's AMF runtime is installed (file check only, nothing loaded).
    static bool runtimeInstalled();
    // This build was compiled with the AMF SDK headers.
    static bool compiledIn();
    // The surface formats the component maps: BGRA, RGBA, R10G10B10A2, RGBA_F16,
    // NV12 and P010; anything else is rejected.
    static bool supportsFormat(DXGI_FORMAT format);
    // 0.02..0.4, NaN becomes the 0.1 default (VE_FCR_DEFAULT_ATTENUATION).
    static float clampAttenuation(float value);
    // 1..4, the range the header documents for AMF_VE_FCR_RADIUS.
    static std::int32_t clampRadius(std::int32_t value);
    struct Config {
        float fcrAttenuation = 0.1f;
        std::int32_t fcrRadius = 2;
        bool splitView = false;
    };
    // One extent: the component enhances at the size it is initialized with.
    bool initialize(ID3D12Device*, ID3D12CommandQueue*, uint32_t width, uint32_t height, DXGI_FORMAT,
                    const Config& config);
    // `input` is in COPY_SOURCE; the list runs on the queue given to initialize().
    bool capture(ID3D12GraphicsCommandList*, ID3D12Resource* input);
    // After the list holding capture() was submitted. reset flushes the component
    // first (seek, cut, resize, pause resume).
    bool process(bool reset);
    // Queue wait for AMF, then the copy into `output` (COPY_DEST). False when AMF
    // returned no frame for this input.
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
