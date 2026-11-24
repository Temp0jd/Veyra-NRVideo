// SPDX-License-Identifier: GPL-3.0-only
#include "veyra/gfx/AmfHqScalerBackend.h"

#include "veyra/Log.h"

#include <windows.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <mutex>

#ifdef VEYRA_HAS_AMF_HQSCALER
#include "core/Factory.h"
#include "core/D3D12AMF.h"
#include "components/HQScaler.h"
#endif

namespace veyra::gfx {
using Microsoft::WRL::ComPtr;

constexpr const char* kLog = "amf-hqscaler";

#ifdef VEYRA_HAS_AMF_HQSCALER
namespace {
// AMFInit hands out one factory per process; the runtime stays loaded with it.
// Same protocol as AmfFrcBackend.cpp: the two files cannot share a helper
// without a header the other backends do not have.
struct Runtime {
    amf::AMFFactory* factory = nullptr;
    std::string version;
    std::wstring compiler;
    bool systemCompiler = false;
};
const Runtime& runtime() {
    static Runtime r;
    static std::once_flag once;
    std::call_once(once, [] {
        // AMF reflects its D3D12 kernels with d3dcompiler_47.dll, loaded by name. The copy Qt
        // deploys next to the program (6.3.9600, Windows 8.1) cannot reflect them: D3DReflect
        // returns E_NOINTERFACE and the component Init fails with AMF_DIRECTX_FAILED (AMF
        // trace, 2026-11-14). Bind the name to the system copy before AMF asks for it.
        HMODULE compiler = GetModuleHandleW(L"d3dcompiler_47.dll");
        if (!compiler) compiler = LoadLibraryExW(L"d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        wchar_t compilerPath[MAX_PATH]{}, systemPath[MAX_PATH]{};
        if (compiler) GetModuleFileNameW(compiler, compilerPath, MAX_PATH);
        GetSystemDirectoryW(systemPath, MAX_PATH);
        r.compiler = compilerPath;
        std::error_code ec;
        r.systemCompiler = compiler && std::filesystem::equivalent(std::filesystem::path(compilerPath).parent_path(), systemPath, ec);
        if (!r.systemCompiler)
            log::warn(kLog, "d3dcompiler_47.dll is not the system copy; the AMF components may fail to load: " +
                      std::filesystem::path(compilerPath).string());
        // Only the driver's copy in System32, never one next to the program.
        HMODULE module = LoadLibraryExW(AMF_DLL_NAME, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!module) { log::warn(kLog, std::format("AMF runtime not loaded err={}", GetLastError())); return; }
        auto query = reinterpret_cast<AMFQueryVersion_Fn>(GetProcAddress(module, AMF_QUERY_VERSION_FUNCTION_NAME));
        auto init = reinterpret_cast<AMFInit_Fn>(GetProcAddress(module, AMF_INIT_FUNCTION_NAME));
        amf_uint64 installed = 0;
        if (query) query(&installed);
        r.version = std::format("{}.{}.{}.{}", AMF_GET_MAJOR_VERSION(installed), AMF_GET_MINOR_VERSION(installed),
                                AMF_GET_SUBMINOR_VERSION(installed), AMF_GET_BUILD_VERSION(installed));
        // A runtime older than these headers is asked for its own version.
        const amf_uint64 requested = installed && installed < AMF_FULL_VERSION ? installed : AMF_FULL_VERSION;
        const AMF_RESULT result = init ? init(requested, &r.factory) : AMF_FAIL;
        if (result != AMF_OK || !r.factory) {
            r.factory = nullptr;
            log::error(kLog, std::format("AMFInit failed result={} runtime={}", int(result), r.version));
            return;
        }
        // Diagnostics only: VEYRA_AMF_TRACE=<file> writes AMF's own debug trace there.
        wchar_t tracePath[MAX_PATH]{};
        amf::AMFTrace* trace = nullptr;
        if (GetEnvironmentVariableW(L"VEYRA_AMF_TRACE", tracePath, MAX_PATH) && r.factory->GetTrace(&trace) == AMF_OK && trace) {
            trace->SetPath(tracePath);
            trace->EnableWriter(AMF_TRACE_WRITER_FILE, true);
            trace->SetGlobalLevel(AMF_TRACE_DEBUG);
            trace->SetWriterLevel(AMF_TRACE_WRITER_FILE, AMF_TRACE_DEBUG);
            log::info(kLog, "AMF trace enabled (VEYRA_AMF_TRACE)");
        }
    });
    return r;
}
UINT64 fenceValue(ID3D12Fence* fence) {
    UINT64 value = 0; UINT size = sizeof(value);
    return SUCCEEDED(fence->GetPrivateData(AMFFenceValueGUID, &size, &value)) ? value : 0;
}
void setFenceValue(ID3D12Fence* fence, UINT64 value) { fence->SetPrivateData(AMFFenceValueGUID, sizeof(value), &value); }
ComPtr<ID3D12Fence> resourceFence(ID3D12Resource* resource) {
    IUnknown* unknown = nullptr; UINT size = sizeof(unknown);
    ComPtr<ID3D12Fence> fence;
    if (SUCCEEDED(resource->GetPrivateData(AMFFenceGUID, &size, &unknown)) && unknown) {
        unknown->QueryInterface(IID_PPV_ARGS(&fence));
        unknown->Release();
    }
    return fence;
}
D3D12_RESOURCE_STATES resourceState(ID3D12Resource* resource) {
    UINT value = 0; UINT size = sizeof(value);
    resource->GetPrivateData(AMFResourceStateGUID, &size, &value);
    return D3D12_RESOURCE_STATES(value);
}
void setResourceState(ID3D12Resource* resource, D3D12_RESOURCE_STATES state) {
    const UINT value = UINT(state);
    resource->SetPrivateData(AMFResourceStateGUID, sizeof(value), &value);
}
void barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    if (before == after) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    list->ResourceBarrier(1, &b);
}
// The surface formats core/HQScaler.h lists; the component never converts, so
// the output keeps the input format.
amf::AMF_SURFACE_FORMAT surfaceFormat(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM: return amf::AMF_SURFACE_RGBA;
    case DXGI_FORMAT_B8G8R8A8_UNORM: return amf::AMF_SURFACE_BGRA;
    case DXGI_FORMAT_R10G10B10A2_UNORM: return amf::AMF_SURFACE_R10G10B10A2;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return amf::AMF_SURFACE_RGBA_F16;
    case DXGI_FORMAT_NV12: return amf::AMF_SURFACE_NV12;
    case DXGI_FORMAT_P010: return amf::AMF_SURFACE_P010;
    default: return amf::AMF_SURFACE_UNKNOWN;
    }
}
AMF_HQ_SCALER_ALGORITHM_ENUM algorithmValue(AmfHqScalerBackend::Algorithm algorithm) {
    switch (algorithm) {
    case AmfHqScalerBackend::Algorithm::Bilinear: return AMF_HQ_SCALER_ALGORITHM_BILINEAR;
    case AmfHqScalerBackend::Algorithm::Bicubic: return AMF_HQ_SCALER_ALGORITHM_BICUBIC;
    case AmfHqScalerBackend::Algorithm::Point: return AMF_HQ_SCALER_ALGORITHM_POINT;
    case AmfHqScalerBackend::Algorithm::VideoSr1_1: return AMF_HQ_SCALER_ALGORITHM_VIDEOSR1_1;
    case AmfHqScalerBackend::Algorithm::VideoSr1_0: break;
    }
    return AMF_HQ_SCALER_ALGORITHM_VIDEOSR1_0;
}
// SHARPNESS only applies to the two Video SR models; the filters ignore it.
bool isVideoSr(AmfHqScalerBackend::Algorithm algorithm) {
    return algorithm == AmfHqScalerBackend::Algorithm::VideoSr1_0 || algorithm == AmfHqScalerBackend::Algorithm::VideoSr1_1;
}
// Shutdown only: the textures may still be read on AMF's queue.
void waitFence(ID3D12Fence* fence, UINT64 value) {
    if (!fence || !value || fence->GetCompletedValue() >= value) return;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (event && SUCCEEDED(fence->SetEventOnCompletion(value, event))) WaitForSingleObject(event, 2000);
    if (event) CloseHandle(event);
}
}
#endif

struct AmfHqScalerBackend::Impl {
#ifdef VEYRA_HAS_AMF_HQSCALER
    // Two input slots: capture() fills one while AMF still reads the other.
    static constexpr int kInputs = 2;
    amf::AMFContextPtr context;
    amf::AMFComponentPtr scaler;
    ComPtr<ID3D12CommandQueue> queue;
    std::array<ComPtr<ID3D12Resource>, kInputs> input;
    std::array<ComPtr<ID3D12Fence>, kInputs> inputFence;
    int next = 0, captured = -1;
    // The frame AMF returned for the latest input, then the one being copied out.
    amf::AMFSurfacePtr out, held;
    ID3D12Resource* outTexture = nullptr;
    ComPtr<ID3D12Fence> outFence, heldFence;
    D3D12_RESOURCE_STATES outState = D3D12_RESOURCE_STATE_COMMON;
    uint32_t inWidth = 0, inHeight = 0, outWidth = 0, outHeight = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    uint64_t inputs = 0, outputs = 0;
#endif
    bool ready = false;
    std::string version;
    std::string error;
};

AmfHqScalerBackend::AmfHqScalerBackend() : p_(std::make_unique<Impl>()) {}
AmfHqScalerBackend::~AmfHqScalerBackend() { shutdown(); }

bool AmfHqScalerBackend::runtimeInstalled() {
    wchar_t system[MAX_PATH]{};
    if (!GetSystemDirectoryW(system, MAX_PATH)) return false;
    std::error_code ec;
    return std::filesystem::is_regular_file(std::filesystem::path(system) / L"amfrt64.dll", ec);
}

bool AmfHqScalerBackend::compiledIn() {
#ifdef VEYRA_HAS_AMF_HQSCALER
    return true;
#else
    return false;
#endif
}

bool AmfHqScalerBackend::supportsFormat(DXGI_FORMAT format, Algorithm algorithm) {
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        return true;
    case DXGI_FORMAT_NV12:
    case DXGI_FORMAT_P010:
        // Video SR 1.1 is the experimental DX11/DX12 model and takes no NV12/P010.
        return algorithm != Algorithm::VideoSr1_1;
    default:
        return false;
    }
}

float AmfHqScalerBackend::clampSharpness(float value) {
    if (std::isnan(value)) return 0.5f;
    return std::clamp(value, 0.0f, 2.0f);
}

bool AmfHqScalerBackend::initialize(ID3D12Device* device, ID3D12CommandQueue* queue, uint32_t inputWidth,
                                    uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight,
                                    DXGI_FORMAT format, const Config& config) {
#ifdef VEYRA_HAS_AMF_HQSCALER
    auto& p = *p_;
    shutdown();
    p.error.clear();
    const auto fail = [&p](std::string message) {
        p.error = std::move(message);
        log::error(kLog, p.error);
        return false;
    };
    // Arguments are validated before the runtime is loaded: an unusable call must
    // not depend on (or disturb) the driver's DLL.
    if (!device || !queue || !inputWidth || !inputHeight || !outputWidth || !outputHeight) return fail("invalid device, queue or extent");
    if (!supportsFormat(format, config.algorithm))
        return fail(std::format("format {} unsupported by algorithm {}", int(format), int(config.algorithm)));
    const amf::AMF_SURFACE_FORMAT amfFormat = surfaceFormat(format);
    if (amfFormat == amf::AMF_SURFACE_UNKNOWN) return fail(std::format("unsupported texture format {}", int(format)));
    const auto& rt = runtime();
    p.version = rt.version;
    if (!rt.factory) return fail("AMF runtime unavailable; AMD Video SR needs an AMD GPU driver");
    const auto start = std::chrono::steady_clock::now();
    AMF_RESULT result = rt.factory->CreateContext(&p.context);
    amf::AMFContext2Ptr context2(p.context);
    if (result != AMF_OK || !context2) { shutdown(); return fail(std::format("CreateContext failed result={}", int(result))); }
    // Fails on a device of another vendor: the AMF runtime drives AMD GPUs only.
    result = context2->InitDX12(device, amf::AMF_DX12);
    if (result != AMF_OK) { shutdown(); return fail(std::format("InitDX12 failed result={} (AMD GPU required)", int(result))); }
    result = rt.factory->CreateComponent(p.context, AMFHQScaler, &p.scaler);
    if (result != AMF_OK || !p.scaler) {
        shutdown();
        return fail(std::format("CreateComponent(AMFHQScaler) failed result={} runtime={}", int(result), p.version));
    }
    // Everything goes in before Init: the component reads its properties there.
    const auto set = [&p](const wchar_t* name, const amf::AMFVariant& value) { p.scaler->SetProperty(name, value); };
    set(AMF_HQ_SCALER_ENGINE_TYPE, amf_int64(amf::AMF_MEMORY_DX12));
    set(AMF_HQ_SCALER_OUTPUT_SIZE, AMFConstructSize(amf_int32(outputWidth), amf_int32(outputHeight)));
    set(AMF_HQ_SCALER_ALGORITHM, amf_int64(algorithmValue(config.algorithm)));
    // Both flags are always written: the header defaults (true/true) contradict
    // the component documentation (false/true), so leaving them unset would be
    // ambiguous.
    set(AMF_HQ_SCALER_KEEP_ASPECT_RATIO, config.keepAspectRatio);
    set(AMF_HQ_SCALER_FROM_SRGB, config.fromSrgb);
    if (isVideoSr(config.algorithm)) set(AMF_HQ_SCALER_SHARPNESS, clampSharpness(config.sharpness));
    result = p.scaler->Init(amfFormat, amf_int32(inputWidth), amf_int32(inputHeight));
    if (result != AMF_OK) {
        if (config.algorithm == Algorithm::VideoSr1_1) {
            // The experimental model is not in every driver build: report once and
            // stay out of the way. The caller keeps its previous upscaler; the
            // algorithm is never switched silently here.
            log::warn(kLog, std::format("Video SR 1.1 unavailable in this runtime result={}; caller keeps its previous upscaler", int(result)));
            p.error.clear();
            shutdown();
            return false;
        }
        shutdown();
        return fail(std::format("Init failed result={} {}x{} -> {}x{} format={} algorithm={}", int(result),
                                inputWidth, inputHeight, outputWidth, outputHeight, int(format), int(config.algorithm)));
    }
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width = inputWidth; desc.Height = inputHeight;
    desc.DepthOrArraySize = 1; desc.MipLevels = 1; desc.Format = format; desc.SampleDesc.Count = 1;
    for (int i = 0; i < Impl::kInputs; ++i) {
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&p.input[i]))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&p.inputFence[i])))) {
            shutdown();
            return fail("input texture/fence allocation failed");
        }
        p.input[i]->SetName(std::format(L"Veyra AMF HQScaler input {}", i).c_str());
        const UINT state = D3D12_RESOURCE_STATE_COMMON;
        p.input[i]->SetPrivateData(AMFResourceStateGUID, sizeof(state), &state);
        setFenceValue(p.inputFence[i].Get(), 0);
        p.input[i]->SetPrivateDataInterface(AMFFenceGUID, p.inputFence[i].Get());
    }
    p.queue = queue;
    p.inWidth = inputWidth; p.inHeight = inputHeight; p.outWidth = outputWidth; p.outHeight = outputHeight;
    p.format = format; p.ready = true;
    log::info(kLog, std::format("ready runtime={} {}x{} -> {}x{} format={} algorithm={} sharpness={:.2f} initMs={:.1f}",
        p.version, inputWidth, inputHeight, outputWidth, outputHeight, int(format), int(config.algorithm),
        isVideoSr(config.algorithm) ? clampSharpness(config.sharpness) : 0.0f,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()));
    return true;
#else
    (void)device; (void)queue; (void)inputWidth; (void)inputHeight; (void)outputWidth; (void)outputHeight;
    (void)format; (void)config;
    p_->error = "this build has no AMF SDK headers (VEYRA_HAS_AMF_HQSCALER undefined)";
    log::error(kLog, p_->error);
    return false;
#endif
}

bool AmfHqScalerBackend::capture(ID3D12GraphicsCommandList* list, ID3D12Resource* input) {
#ifdef VEYRA_HAS_AMF_HQSCALER
    auto& p = *p_;
    if (!p.ready || !list || !input) return false;
    const int slot = p.next;
    p.next = (p.next + 1) % Impl::kInputs;
    auto* fence = p.inputFence[slot].Get();
    // AMF raises this fence when it no longer reads the texture; the copy waits for it.
    if (const UINT64 wait = fenceValue(fence)) p.queue->Wait(fence, wait);
    // AMF leaves the texture in the state it read it in (shader resource) and records
    // that state on the texture; start from it and record the state left behind.
    auto* texture = p.input[slot].Get();
    barrier(list, texture, resourceState(texture), D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyResource(texture, input);
    barrier(list, texture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    setResourceState(texture, D3D12_RESOURCE_STATE_COMMON);
    p.captured = slot;
    return true;
#else
    (void)list; (void)input; return false;
#endif
}

bool AmfHqScalerBackend::process(bool reset) {
#ifdef VEYRA_HAS_AMF_HQSCALER
    auto& p = *p_;
    if (!p.ready || p.captured < 0) return false;
    const auto fail = [&p](std::string message) {
        p.error = std::move(message);
        log::error(kLog, p.error);
        return false;
    };
    const int slot = p.captured;
    p.captured = -1;
    // The list holding the copy is submitted: AMF may read once this signals.
    auto* fence = p.inputFence[slot].Get();
    const UINT64 written = fenceValue(fence) + 1;
    setFenceValue(fence, written);
    if (FAILED(p.queue->Signal(fence, written))) return fail("input fence signal failed");
    p.out = nullptr; p.outTexture = nullptr; p.outFence.Reset();
    // Without a flush AMF would keep the previous frame's history across the cut.
    if (reset && p.inputs) p.scaler->Flush();
    amf::AMFContext2Ptr context2(p.context);
    amf::AMFSurfacePtr surface;
    AMF_RESULT result = context2 ? context2->CreateSurfaceFromDX12Native(p.input[slot].Get(), &surface, nullptr) : AMF_FAIL;
    if (result != AMF_OK || !surface) return fail(std::format("CreateSurfaceFromDX12Native failed result={}", int(result)));
    result = p.scaler->SubmitInput(surface);
    // Bounded retry: a full input queue means older outputs are still unread; drain
    // one and retry. They belong to earlier inputs and are dropped on purpose.
    for (int retry = 0; (result == AMF_INPUT_FULL || result == AMF_REPEAT) && retry < 4; ++retry) {
        amf::AMFDataPtr stale;
        p.scaler->QueryOutput(&stale);
        result = p.scaler->SubmitInput(surface);
    }
    if (result != AMF_OK && result != AMF_NEED_MORE_INPUT)
        return fail(std::format("SubmitInput failed result={}", int(result)));
    ++p.inputs;
    for (int i = 0; i < 4; ++i) {
        amf::AMFDataPtr data;
        result = p.scaler->QueryOutput(&data);
        if (!data) break;
        p.out = amf::AMFSurfacePtr(data); // the latest output belongs to this input
    }
    if (p.out && p.out->GetPlaneAt(0)) {
        p.outTexture = static_cast<ID3D12Resource*>(p.out->GetPlaneAt(0)->GetNative());
        p.outFence = resourceFence(p.outTexture);
        p.outState = resourceState(p.outTexture);
    }
    if (p.inputs <= 3 || p.inputs % 600 == 0)
        log::info(kLog, std::format("input={} reset={} output={} result={}", p.inputs, reset, p.outTexture != nullptr, int(result)));
    return true;
#else
    (void)reset; return false;
#endif
}

bool AmfHqScalerBackend::copyOutput(ID3D12GraphicsCommandList* list, ID3D12Resource* output) {
#ifdef VEYRA_HAS_AMF_HQSCALER
    auto& p = *p_;
    if (!p.ready || !p.out || !p.outTexture || !list || !output) return false;
    if (p.outFence) {
        if (const UINT64 wait = fenceValue(p.outFence.Get())) p.queue->Wait(p.outFence.Get(), wait);
    }
    barrier(list, p.outTexture, p.outState, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource = output; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource = p.outTexture; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    const D3D12_BOX box{0, 0, 0, p.outWidth, p.outHeight, 1};
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
    barrier(list, p.outTexture, D3D12_RESOURCE_STATE_COPY_SOURCE, p.outState);
    p.held = std::move(p.out); p.heldFence = std::move(p.outFence); p.outTexture = nullptr;
    ++p.outputs;
    return true;
#else
    (void)list; (void)output; return false;
#endif
}

bool AmfHqScalerBackend::submitted() {
#ifdef VEYRA_HAS_AMF_HQSCALER
    auto& p = *p_;
    if (!p.held) return false;
    // AMF may write this frame again once the copy is done.
    if (p.heldFence) {
        const UINT64 value = fenceValue(p.heldFence.Get()) + 1;
        setFenceValue(p.heldFence.Get(), value);
        p.queue->Signal(p.heldFence.Get(), value);
    }
    p.held = nullptr; p.heldFence.Reset();
    return true;
#else
    return false;
#endif
}

bool AmfHqScalerBackend::ready() const { return p_->ready; }
const std::string& AmfHqScalerBackend::runtimeVersion() const { return p_->version; }
const char* AmfHqScalerBackend::lastError() const { return p_->error.c_str(); }

void AmfHqScalerBackend::shutdown() {
#ifdef VEYRA_HAS_AMF_HQSCALER
    auto& p = *p_;
    if (p.ready) log::info(kLog, std::format("shutdown inputs={} outputs={}", p.inputs, p.outputs));
    if (p.held && p.heldFence) waitFence(p.heldFence.Get(), fenceValue(p.heldFence.Get()));
    p.held = nullptr; p.heldFence.Reset();
    p.out = nullptr; p.outTexture = nullptr; p.outFence.Reset();
    if (p.scaler) { p.scaler->Terminate(); p.scaler = nullptr; }
    for (int i = 0; i < Impl::kInputs; ++i) {
        if (p.inputFence[i]) waitFence(p.inputFence[i].Get(), fenceValue(p.inputFence[i].Get()));
        p.input[i].Reset(); p.inputFence[i].Reset();
    }
    if (p.context) { p.context->Terminate(); p.context = nullptr; }
    p.queue.Reset();
    p.next = 0; p.captured = -1; p.inputs = p.outputs = 0;
    p.inWidth = p.inHeight = p.outWidth = p.outHeight = 0;
    p.format = DXGI_FORMAT_UNKNOWN;
#endif
    p_->ready = false;
}
}
