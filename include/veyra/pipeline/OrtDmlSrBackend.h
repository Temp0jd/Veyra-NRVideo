// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// M7: ONNX Runtime v20 + DirectML super-resolution backend with zero-readback
// GPU IoBinding.
//
// Layout of one frame:
//   linear BT.709 scRGB (RGBA16F)
//     -> OrtDmlPreprocess.hlsl   NCHW float32 structured buffer (sRGB encoded)
//     -> DirectML session (RunWithBinding, input/output tensors are D3D12
//        buffers shared through IDMLDevice created on the graph's own device)
//     -> OrtDmlPostprocess.hlsl  linear BT.709 scRGB (RGBA16F)
//
// No pixel ever leaves the GPU on the fast path: both the input and the output
// are bound as D3D12 resources wrapped by the DirectML execution provider's
// allocator. When ORT refuses the pre-allocated output binding the backend
// falls back to BindOutputToDevice + GetBoundOutputValues and resolves the
// D3D12 resource the runtime allocated, which is still zero-readback.
//
// The runtime DLL is loaded by name (application directory / System32) or from
// an explicit absolute path, so the user can drop in a different signed or
// self-built onnxruntime.dll without a rebuild; the loader never hash-locks it.
// The DirectML EP is mandatory: a session that cannot get DirectML is a hard
// failure, never a silent CPU fallback.

#include "veyra/pipeline/GpuPassUtils.h"
#include "veyra/pipeline/OrtDmlApi.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace veyra::pipeline {

class OrtDmlSrBackend final {
public:
    struct Config {
        std::filesystem::path modelPath;
        // Empty -> LoadLibraryExW(L"onnxruntime.dll") with application-dir and
        // system32 search. Non-empty -> must be absolute; loaded with
        // LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32 so the
        // DLL's own dependency folder is searched but the CWD is not.
        std::filesystem::path runtimeDllPath;
        int deviceId = 0;
        // 1: luma-only model (ArtCNN and similar). 3: RGB NCHW model.
        uint32_t channels = 3;
        // Memory-info name registered by the DirectML execution provider.
        std::string dmlMemoryInfoName = "DML";
        // Used only when no IDMLDevice is available: DML2 (device filter) instead
        // of the legacy device-id overload.
        bool useDml2 = false;
        // SDR models only. HDR working space is rejected instead of silently
        // running a model trained on SDR values.
        bool hdrWorking = false;
        // Create the IDMLDevice with DirectML.dll DMLCreateDevice. When the
        // DirectML runtime does not hand out a usable interface the backend logs
        // a warning and continues with DML2/DML(deviceId) instead.
        bool createDmlDevice = true;
        // Caller-owned IDMLDevice (IUnknown*). Takes precedence over createDmlDevice.
        void* dmlDevice = nullptr;
        // Unit-test injection: replaces the DLL load and the DML device.
        const ort::OrtApiBase* testApiBase = nullptr;
        const ort::OrtDmlApi* testDmlApi = nullptr;
        void* testDmlDevice = nullptr;
        // Unit-test injection: skip every D3D12 resource, pipeline and shader so
        // the session plumbing can run on a machine without a D3D12 device.
        bool skipD3dResourcesForTest = false;
    };

    struct Stats {
        uint64_t runs = 0;
        bool usedFallback = false;
    };

    OrtDmlSrBackend();
    ~OrtDmlSrBackend();
    OrtDmlSrBackend(const OrtDmlSrBackend&) = delete;
    OrtDmlSrBackend& operator=(const OrtDmlSrBackend&) = delete;

    // Loads the runtime, resolves OrtApi/OrtDmlApi, creates env + session
    // options, appends the DirectML EP and creates the session.
    bool open(const Config& config, ID3D12Device* device = nullptr, ID3D12CommandQueue* queue = nullptr);
    // Allocates the GPU tensor buffers and binds them. Called again after a
    // resolution change; the previous binding is torn down first.
    bool prepare(uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight);
    // Runs the DirectML session on the current binding.
    bool run();
    // Source: linear BT.709 scRGB RGBA16F, must be in
    // D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.
    bool recordPreprocess(ID3D12GraphicsCommandList* list, ID3D12Resource* linearSource);
    // linearOutput: RGBA16F UAV, must be in D3D12_RESOURCE_STATE_UNORDERED_ACCESS.
    // sourceForChroma: the same linear frame the preprocess consumed; required
    // for channels == 1 (chroma is carried through untouched), ignored
    // otherwise, may be null.
    bool recordPostprocess(ID3D12GraphicsCommandList* list, ID3D12Resource* linearOutput,
                           ID3D12Resource* sourceForChroma);
    void shutdown() noexcept;

    bool ready() const { return session_ != nullptr; }
    bool prepared() const { return prepared_; }
    bool usesFallback() const { return fallback_; }
    const std::string& error() const { return error_; }
    const std::string& warning() const { return warning_; }
    const std::string& runtimeVersion() const { return runtimeVersion_; }
    const std::string& inputName() const { return inputName_; }
    const std::string& outputName() const { return outputName_; }
    uint32_t channels() const { return config_.channels; }
    uint32_t inputWidth() const { return inputWidth_; }
    uint32_t inputHeight() const { return inputHeight_; }
    uint32_t outputWidth() const { return outputWidth_; }
    uint32_t outputHeight() const { return outputHeight_; }
    // The GPU tensor buffers. inputBuffer() is always ours; outputResource() is
    // ours on the primary path and the runtime-allocated buffer on the fallback
    // path (only valid once run() has executed at least once).
    ID3D12Resource* inputBuffer() const { return inputBuffer_.Get(); }
    ID3D12Resource* outputBuffer() const { return outputBuffer_.Get(); }
    ID3D12Resource* outputResource() const;
    const Stats& stats() const { return stats_; }
    const Config& config() const { return config_; }
    static const wchar_t* moduleName();

private:
    // fail() records the reason, logs it and tears the runtime down; reject()
    // records the reason for a recoverable per-frame condition without
    // dropping the session.
    bool fail(const std::string& message);
    bool reject(const std::string& message);
    bool loadRuntime();
    bool resolveApi();
    bool createDmlDevice();
    bool createSessionOptions();
    bool createSession();
    bool introspect();
    bool introspectTensor(const char* what, size_t index, bool isInput, std::string& name,
                          std::vector<int64_t>& dims);
    bool resolveFallbackOutput();
    void releaseBinding() noexcept;
    void releaseSessionChain() noexcept;
    void releaseAllocations() noexcept;
    bool loadPasses();
    bool transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                    D3D12_RESOURCE_STATES& tracked, D3D12_RESOURCE_STATES to);

    Config config_;
    Stats stats_;

    // Loaded modules. Kept loaded until every runtime object is gone.
    HMODULE runtimeModule_ = nullptr;
    HMODULE directmlModule_ = nullptr;

    const ort::OrtApiBase* apiBase_ = nullptr;
    ort::OrtApiView api_;
    const ort::OrtDmlApi* dmlApi_ = nullptr;

    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    void* dmlDevice_ = nullptr;
    bool dmlDeviceOwned_ = false;

    ort::OrtEnv* env_ = nullptr;
    ort::OrtSessionOptions* sessionOptions_ = nullptr;
    ort::OrtSession* session_ = nullptr;
    ort::OrtIoBinding* ioBinding_ = nullptr;
    ort::OrtValue* inputValue_ = nullptr;
    ort::OrtValue* outputValue_ = nullptr;
    ort::OrtValue* fallbackOutputValue_ = nullptr;
    ort::OrtMemoryInfo* memoryInfo_ = nullptr;
    ort::OrtAllocator* dmlAllocator_ = nullptr;
    ort::OrtAllocator* hostAllocator_ = nullptr;

    void* inputAllocation_ = nullptr;
    void* outputAllocation_ = nullptr;
    ID3D12Resource* resolvedOutput_ = nullptr;

    ComPtr<ID3D12Resource> inputBuffer_;
    ComPtr<ID3D12Resource> outputBuffer_;
    std::vector<uint8_t> hostInput_;
    std::vector<uint8_t> hostOutput_;

    static constexpr UINT kPostDescriptorSlots = 8;
    ComputePass preprocess_;
    ComputePass postprocess_;
    DescriptorStager stager_;
    ID3D12Resource* cachedPreSource_ = nullptr;
    ID3D12Resource* cachedPreInputBuf_ = nullptr;
    ID3D12Resource* cachedPostModelOut_ = nullptr;
    ID3D12Resource* cachedPostChroma_ = nullptr;
    ID3D12Resource* cachedPostLinearOut_ = nullptr;
    UINT cachedPostSlot_ = 0;
    UINT nextPostSlot_ = 0;
    bool postDescriptorValid_ = false;

    D3D12_RESOURCE_STATES inputBufferState_ = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES outputResourceState_ = D3D12_RESOURCE_STATE_COMMON;

    std::string inputName_;
    std::string outputName_;
    std::vector<int64_t> inputDims_;
    std::vector<int64_t> outputDims_;
    std::string runtimeVersion_;
    std::string error_;
    std::string warning_;

    uint32_t inputWidth_ = 0;
    uint32_t inputHeight_ = 0;
    uint32_t outputWidth_ = 0;
    uint32_t outputHeight_ = 0;
    bool fallback_ = false;
    bool prepared_ = false;
};

} // namespace veyra::pipeline
