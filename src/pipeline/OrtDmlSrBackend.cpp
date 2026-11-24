// SPDX-License-Identifier: GPL-3.0-only
#include "veyra/pipeline/OrtDmlSrBackend.h"

#include <windows.h>
#include <d3d12.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

#include "veyra/Log.h"

namespace veyra::pipeline {
namespace {

constexpr const char* kLogTag = "ort-dml";
constexpr uint32_t kMaxExtent = 16384;
constexpr UINT kStagerSlots = 16;

// DirectML device creation. The DirectML SDK header is proprietary and is not
// part of the source tree, so the entry point is resolved dynamically and the
// prototype + documented IDMLDevice IID (6DBD6437-96FD-423F-A98C-AE5E7C2A573F)
// are transcribed here. Any failure here is non-fatal: the session then uses
// the DirectML EP's own device and the caller has to expect a tensor-sharing
// failure instead of a silent fallback.
using DmlCreateDeviceFn = HRESULT(WINAPI*)(ID3D12Device* device, uint32_t flags, const IID& riid, void** out);
constexpr uint32_t kDmlCreateDeviceFlagNone = 0;
constexpr GUID kIidIdmlDevice = {0x6dbd6437, 0x96fd, 0x423f, {0xa9, 0x8c, 0xae, 0x5e, 0x7c, 0x2a, 0x57, 0x3f}};

// Handles used only with skipD3dResourcesForTest: the DirectML wrappers are
// called with non-null stand-ins so the session plumbing can be exercised
// without a D3D12 device. The value is never dereferenced.
constexpr uintptr_t kFakeResourceHandle = 0x1000;

// GetProcAddress hands back FARPROC, so every export lookup crosses a
// function-pointer type. GCC warns about that by design; the cast is the
// documented way to bind a dynamic export, so the diagnostic is silenced here
// and nowhere else.
template <typename Fn>
Fn resolveExport(HMODULE module, const char* name)
{
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
    return reinterpret_cast<Fn>(GetProcAddress(module, name));
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

std::string describePath(const std::filesystem::path& path) { return path.string(); }

ComPtr<ID3D12Resource> makeGpuBuffer(ID3D12Device* device, uint64_t bytes)
{
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> resource;
    const HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                       D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                       IID_PPV_ARGS(&resource));
    if (FAILED(hr)) {
        veyra::log::error(kLogTag, std::format("GPU buffer allocation failed bytes={} hr=0x{:08X}",
                                               bytes, static_cast<unsigned>(hr)));
        return nullptr;
    }
    return resource;
}

void makeBufferSrvDesc(D3D12_SHADER_RESOURCE_VIEW_DESC& desc, UINT elements, UINT stride)
{
    desc = {};
    // UNKNOWN format + StructureByteStride selects the structured buffer view;
    // the shader reads it as StructuredBuffer<float>/StructuredBuffer<float4>.
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.Buffer.FirstElement = 0;
    desc.Buffer.NumElements = elements;
    desc.Buffer.StructureByteStride = stride;
    desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
}

void makeBufferUav(ID3D12Device* device, ID3D12Resource* resource, UINT elements, UINT stride,
                   const D3D12_CPU_DESCRIPTOR_HANDLE& handle)
{
    D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    desc.Buffer.FirstElement = 0;
    desc.Buffer.NumElements = elements;
    desc.Buffer.StructureByteStride = stride;
    desc.Buffer.CounterOffsetInBytes = 0;
    desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    device->CreateUnorderedAccessView(resource, nullptr, &desc, handle);
}

D3D12_SHADER_RESOURCE_VIEW_DESC makeTextureSrvDesc(DXGI_FORMAT format)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = format;
    desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.Texture2D.MostDetailedMip = 0;
    desc.Texture2D.MipLevels = 1;
    desc.Texture2D.PlaneSlice = 0;
    desc.Texture2D.ResourceMinLODClamp = 0.0f;
    return desc;
}

// Owns an OrtTypeInfo for the length of one introspection; the runtime releases
// it exactly once even on an early error return.
class TypeInfoOwner {
public:
    TypeInfoOwner(ort::FnReleaseTypeInfo release, ort::OrtTypeInfo* info) : release_(release), info_(info) {}
    TypeInfoOwner(const TypeInfoOwner&) = delete;
    TypeInfoOwner& operator=(const TypeInfoOwner&) = delete;
    ~TypeInfoOwner()
    {
        if (info_ && release_)
            release_(info_);
    }

    const ort::OrtTypeInfo* get() const { return info_; }

private:
    ort::FnReleaseTypeInfo release_ = nullptr;
    ort::OrtTypeInfo* info_ = nullptr;
};

std::string describeDims(const std::vector<int64_t>& dims)
{
    if (dims.empty())
        return "<none>";
    std::string text;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i)
            text += 'x';
        text += dims[i] < 0 ? std::string("?") : std::to_string(dims[i]);
    }
    return text;
}

// True when the model shape cannot serve the prepared extent. Symbolic
// dimensions (-1) accept anything; fixed ones must match exactly.
bool dimsMismatch(const std::vector<int64_t>& dims, uint32_t channels, uint32_t height, uint32_t width)
{
    if (dims.size() != 4)
        return true;
    const int64_t expected[4] = {1, static_cast<int64_t>(channels), static_cast<int64_t>(height),
                                 static_cast<int64_t>(width)};
    for (size_t i = 0; i < 4; ++i) {
        if (dims[i] >= 0 && dims[i] != expected[i])
            return true;
    }
    return false;
}

} // namespace

OrtDmlSrBackend::OrtDmlSrBackend() = default;
OrtDmlSrBackend::~OrtDmlSrBackend() { shutdown(); }

const wchar_t* OrtDmlSrBackend::moduleName() { return L"onnxruntime.dll"; }

bool OrtDmlSrBackend::fail(const std::string& message)
{
    error_ = message;
    veyra::log::error(kLogTag, message);
    shutdown();
    return false;
}

bool OrtDmlSrBackend::reject(const std::string& message)
{
    error_ = message;
    veyra::log::error(kLogTag, message);
    return false;
}

bool OrtDmlSrBackend::open(const Config& config, ID3D12Device* device, ID3D12CommandQueue* queue)
{
    shutdown();
    error_.clear();
    warning_.clear();
    runtimeVersion_.clear();
    inputName_.clear();
    outputName_.clear();
    inputDims_.clear();
    outputDims_.clear();
    stats_ = Stats{};
    config_ = config;

    if (config_.hdrWorking)
        return fail("OrtDmlSrBackend: HDR working space is not supported by the DirectML super-resolution "
                    "path; the model is SDR only, so convert to SDR before it instead of feeding it scRGB");
    if (config_.channels != 1 && config_.channels != 3)
        return fail(std::format("OrtDmlSrBackend: channels must be 1 (luma) or 3 (RGB), got {}", config_.channels));
    if (config_.modelPath.empty())
        return fail("OrtDmlSrBackend: modelPath is empty");
    if (config_.deviceId < 0)
        return fail(std::format("OrtDmlSrBackend: deviceId must be >= 0, got {}", config_.deviceId));
    if (config_.dmlMemoryInfoName.empty())
        return fail("OrtDmlSrBackend: dmlMemoryInfoName is empty");
    {
        std::error_code ec;
        if (!std::filesystem::is_regular_file(config_.modelPath, ec))
            return fail(std::format("OrtDmlSrBackend: model file not found: {}", describePath(config_.modelPath)));
    }

    if (device)
        device_ = device;
    if (queue)
        queue_ = queue;

    if (!loadRuntime())
        return false;
    if (!resolveApi())
        return false;
    if (!createDmlDevice())
        return false;
    if (!createSessionOptions())
        return false;
    if (!createSession())
        return false;

    veyra::log::info(kLogTag, std::format("opened runtime={} model={} input={}({}) output={}({}) channels={}",
                                          runtimeVersion_, describePath(config_.modelPath), inputName_,
                                          describeDims(inputDims_), outputName_, describeDims(outputDims_),
                                          config_.channels));
    return true;
}

bool OrtDmlSrBackend::loadRuntime()
{
    if (config_.testApiBase) {
        apiBase_ = config_.testApiBase;
        return true;
    }

    if (!config_.runtimeDllPath.empty()) {
        std::error_code ec;
        if (!config_.runtimeDllPath.is_absolute())
            return fail(std::format("OrtDmlSrBackend: runtimeDllPath must be absolute: {}",
                                    describePath(config_.runtimeDllPath)));
        if (!std::filesystem::is_regular_file(config_.runtimeDllPath, ec))
            return fail(std::format("OrtDmlSrBackend: runtime DLL not found: {}",
                                    describePath(config_.runtimeDllPath)));
        runtimeModule_ = LoadLibraryExW(config_.runtimeDllPath.c_str(), nullptr,
                                        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!runtimeModule_)
            return fail(std::format("OrtDmlSrBackend: LoadLibraryExW failed for {} (win32 {})",
                                    describePath(config_.runtimeDllPath),
                                    static_cast<unsigned long>(GetLastError())));
    } else {
        runtimeModule_ = LoadLibraryExW(L"onnxruntime.dll", nullptr,
                                        LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!runtimeModule_)
            return fail(std::format("OrtDmlSrBackend: onnxruntime.dll was not found in the application directory or "
                                    "System32 (win32 {})",
                                    static_cast<unsigned long>(GetLastError())));
    }

    const auto getApiBase = resolveExport<ort::FnOrtGetApiBase>(runtimeModule_, "OrtGetApiBase");
    if (!getApiBase)
        return fail("OrtDmlSrBackend: onnxruntime.dll does not export OrtGetApiBase");
    apiBase_ = getApiBase();
    if (!apiBase_)
        return fail("OrtDmlSrBackend: OrtGetApiBase returned null");
    return true;
}

bool OrtDmlSrBackend::resolveApi()
{
    if (!apiBase_)
        return fail("OrtDmlSrBackend: the ONNX Runtime entry point is missing");
    if (!apiBase_->GetApi)
        return fail("OrtDmlSrBackend: OrtGetApiBase returned an empty table");
    if (apiBase_->GetVersionString)
        runtimeVersion_ = apiBase_->GetVersionString();

    const void* api = apiBase_->GetApi(ort::kOrtApiVersion);
    if (!api)
        return fail(std::format("OrtDmlSrBackend: onnxruntime does not provide C API version {} (runtime {})",
                                ort::kOrtApiVersion, runtimeVersion_.empty() ? std::string("<unknown>")
                                                                             : runtimeVersion_));
    api_ = ort::OrtApiView(api);

    if (!api_.slot<ort::FnCreateStatus>(ort::kSlot_CreateStatus) ||
        !api_.slot<ort::FnGetErrorCode>(ort::kSlot_GetErrorCode) ||
        !api_.slot<ort::FnGetErrorMessage>(ort::kSlot_GetErrorMessage) ||
        !api_.slot<ort::FnReleaseStatus>(ort::kSlot_ReleaseStatus))
        return fail("OrtDmlSrBackend: the runtime does not expose the OrtStatus entry points");

    if (config_.testDmlApi) {
        dmlApi_ = config_.testDmlApi;
    } else {
        const auto getExecutionProviderApi =
            api_.slot<ort::FnGetExecutionProviderApi>(ort::kSlot_GetExecutionProviderApi);
        if (!getExecutionProviderApi)
            return fail("OrtDmlSrBackend: the runtime does not expose GetExecutionProviderApi");
        const void* dml = nullptr;
        if (ort::OrtStatus* status = getExecutionProviderApi("DML", ort::kOrtApiVersion, &dml))
            return fail(std::format("OrtDmlSrBackend: the DirectML execution provider is unavailable: {}",
                                    ort::consumeStatus(api_, status)));
        if (!dml)
            return fail("OrtDmlSrBackend: the runtime returned a null DirectML API table");
        dmlApi_ = static_cast<const ort::OrtDmlApi*>(dml);
    }

    if (!dmlApi_)
        return fail("OrtDmlSrBackend: the DirectML API table is missing");
    if (!dmlApi_->SessionOptionsAppendExecutionProvider_DML ||
        !dmlApi_->CreateGPUAllocationFromD3DResource || !dmlApi_->FreeGPUAllocation ||
        !dmlApi_->GetD3D12ResourceFromAllocation)
        return fail("OrtDmlSrBackend: the DirectML API table is incomplete");
    return true;
}

bool OrtDmlSrBackend::createDmlDevice()
{
    if (config_.testDmlDevice) {
        dmlDevice_ = config_.testDmlDevice;
        dmlDeviceOwned_ = false;
        return true;
    }
    if (config_.dmlDevice) {
        dmlDevice_ = config_.dmlDevice;
        dmlDeviceOwned_ = false;
        return true;
    }
    if (!config_.createDmlDevice || !device_)
        return true;

    if (!config_.runtimeDllPath.empty()) {
        const std::filesystem::path sideBySide = config_.runtimeDllPath.parent_path() / L"DirectML.dll";
        std::error_code ec;
        if (std::filesystem::is_regular_file(sideBySide, ec)) {
            directmlModule_ = LoadLibraryExW(sideBySide.c_str(), nullptr,
                                             LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        }
    }
    if (!directmlModule_) {
        directmlModule_ = LoadLibraryExW(L"DirectML.dll", nullptr,
                                         LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    }
    if (!directmlModule_) {
        warning_ = std::format("OrtDmlSrBackend: DirectML.dll could not be loaded (win32 {}); the session uses "
                               "the DirectML execution provider's own device",
                               static_cast<unsigned long>(GetLastError()));
        veyra::log::warn(kLogTag, warning_);
        return true;
    }

    const auto createDevice = resolveExport<DmlCreateDeviceFn>(directmlModule_, "DMLCreateDevice");
    if (!createDevice) {
        warning_ = "OrtDmlSrBackend: DirectML.dll does not export DMLCreateDevice; the session uses the DirectML "
                   "execution provider's own device";
        veyra::log::warn(kLogTag, warning_);
        FreeLibrary(directmlModule_);
        directmlModule_ = nullptr;
        return true;
    }

    void* dmlDevice = nullptr;
    const HRESULT hr = createDevice(device_.Get(), kDmlCreateDeviceFlagNone, kIidIdmlDevice, &dmlDevice);
    if (FAILED(hr) || !dmlDevice) {
        warning_ = std::format("OrtDmlSrBackend: DMLCreateDevice failed (hr=0x{:08X}); the session uses the "
                               "DirectML execution provider's own device and same-device tensor sharing is "
                               "not guaranteed",
                               static_cast<unsigned>(hr));
        veyra::log::warn(kLogTag, warning_);
        FreeLibrary(directmlModule_);
        directmlModule_ = nullptr;
        return true;
    }

    dmlDevice_ = dmlDevice;
    dmlDeviceOwned_ = true;
    return true;
}

bool OrtDmlSrBackend::createSessionOptions()
{
    const auto createEnv = api_.slot<ort::FnCreateEnv>(ort::kSlot_CreateEnv);
    if (!createEnv)
        return fail("OrtDmlSrBackend: the runtime does not expose CreateEnv");
    // (1) environment. WARNING keeps the runtime from logging per-run noise.
    if (ort::OrtStatus* status = createEnv(ort::ORT_LOGGING_LEVEL_WARNING, "veyra-ort-dml", &env_))
        return fail(std::format("OrtDmlSrBackend: CreateEnv failed: {}", ort::consumeStatus(api_, status)));

    // (2) session options.
    const auto createSessionOptions = api_.slot<ort::FnCreateSessionOptions>(ort::kSlot_CreateSessionOptions);
    if (!createSessionOptions)
        return fail("OrtDmlSrBackend: the runtime does not expose CreateSessionOptions");
    if (ort::OrtStatus* status = createSessionOptions(&sessionOptions_))
        return fail(std::format("OrtDmlSrBackend: CreateSessionOptions failed: {}",
                                ort::consumeStatus(api_, status)));

    // (3) memory pattern off: mandatory for the DirectML execution provider.
    const auto disableMemPattern = api_.slot<ort::FnDisableMemPattern>(ort::kSlot_DisableMemPattern);
    if (!disableMemPattern)
        return fail("OrtDmlSrBackend: the runtime does not expose DisableMemPattern");
    if (ort::OrtStatus* status = disableMemPattern(sessionOptions_))
        return fail(std::format("OrtDmlSrBackend: DisableMemPattern failed: {}", ort::consumeStatus(api_, status)));

    // (4) sequential execution: mandatory for the DirectML execution provider.
    const auto setExecutionMode = api_.slot<ort::FnSetSessionExecutionMode>(ort::kSlot_SetSessionExecutionMode);
    if (!setExecutionMode)
        return fail("OrtDmlSrBackend: the runtime does not expose SetSessionExecutionMode");
    if (ort::OrtStatus* status = setExecutionMode(sessionOptions_, ort::ORT_SEQUENTIAL))
        return fail(std::format("OrtDmlSrBackend: SetSessionExecutionMode failed: {}",
                                ort::consumeStatus(api_, status)));

    // (5) DirectML execution provider.
    if (dmlDevice_ && queue_) {
        if (!dmlApi_->SessionOptionsAppendExecutionProvider_DML1)
            return fail("OrtDmlSrBackend: the DirectML API table has no DML1 entry point, which is required to "
                        "share the graph's D3D12 device and command queue");
        if (ort::OrtStatus* status =
                dmlApi_->SessionOptionsAppendExecutionProvider_DML1(sessionOptions_, dmlDevice_, queue_.Get()))
            return fail(std::format("OrtDmlSrBackend: SessionOptionsAppendExecutionProvider_DML1 failed: {}",
                                    ort::consumeStatus(api_, status)));
        return true;
    }

    if (config_.useDml2) {
        if (!dmlApi_->SessionOptionsAppendExecutionProvider_DML2)
            return fail("OrtDmlSrBackend: the DirectML API table has no DML2 entry point");
        ort::OrtDmlDeviceOptions deviceOptions{};
        deviceOptions.Preference = ort::DmlPreferenceHighPerformance;
        deviceOptions.Filter = ort::DmlFilterGpu;
        if (ort::OrtStatus* status =
                dmlApi_->SessionOptionsAppendExecutionProvider_DML2(sessionOptions_, &deviceOptions))
            return fail(std::format("OrtDmlSrBackend: SessionOptionsAppendExecutionProvider_DML2 failed: {}",
                                    ort::consumeStatus(api_, status)));
        return true;
    }

    if (!dmlApi_->SessionOptionsAppendExecutionProvider_DML)
        return fail("OrtDmlSrBackend: the DirectML API table has no DML entry point");
    if (ort::OrtStatus* status =
            dmlApi_->SessionOptionsAppendExecutionProvider_DML(sessionOptions_, config_.deviceId))
        return fail(std::format("OrtDmlSrBackend: SessionOptionsAppendExecutionProvider_DML failed: {}",
                                ort::consumeStatus(api_, status)));
    return true;
}

bool OrtDmlSrBackend::createSession()
{
    const auto createSession = api_.slot<ort::FnCreateSession>(ort::kSlot_CreateSession);
    if (!createSession)
        return fail("OrtDmlSrBackend: the runtime does not expose CreateSession");
    if (ort::OrtStatus* status = createSession(env_, config_.modelPath.c_str(), sessionOptions_, &session_))
        return fail(std::format("OrtDmlSrBackend: CreateSession failed for {}: {}",
                                describePath(config_.modelPath), ort::consumeStatus(api_, status)));
    if (!session_)
        return fail("OrtDmlSrBackend: CreateSession returned a null session");
    return introspect();
}

bool OrtDmlSrBackend::introspect()
{
    const auto getAllocator = api_.slot<ort::FnGetAllocatorWithDefaultOptions>(ort::kSlot_GetAllocatorWithDefaultOptions);
    if (!getAllocator)
        return fail("OrtDmlSrBackend: the runtime does not expose GetAllocatorWithDefaultOptions");
    if (ort::OrtStatus* status = getAllocator(&hostAllocator_))
        return fail(std::format("OrtDmlSrBackend: GetAllocatorWithDefaultOptions failed: {}",
                                ort::consumeStatus(api_, status)));
    if (!hostAllocator_)
        return fail("OrtDmlSrBackend: the runtime returned a null default allocator");

    const auto inputCount = api_.slot<ort::FnSessionGetInputCount>(ort::kSlot_SessionGetInputCount);
    const auto outputCount = api_.slot<ort::FnSessionGetOutputCount>(ort::kSlot_SessionGetOutputCount);
    if (!inputCount || !outputCount)
        return fail("OrtDmlSrBackend: the runtime does not expose the session shape queries");

    size_t count = 0;
    if (ort::OrtStatus* status = inputCount(session_, &count))
        return fail(std::format("OrtDmlSrBackend: SessionGetInputCount failed: {}", ort::consumeStatus(api_, status)));
    if (count < 1)
        return fail("OrtDmlSrBackend: the model has no inputs");
    if (!introspectTensor("input", 0, true, inputName_, inputDims_))
        return false;

    count = 0;
    if (ort::OrtStatus* status = outputCount(session_, &count))
        return fail(std::format("OrtDmlSrBackend: SessionGetOutputCount failed: {}",
                                ort::consumeStatus(api_, status)));
    if (count < 1)
        return fail("OrtDmlSrBackend: the model has no outputs");
    return introspectTensor("output", 0, false, outputName_, outputDims_);
}

bool OrtDmlSrBackend::introspectTensor(const char* what, size_t index, bool isInput, std::string& name,
                                       std::vector<int64_t>& dims)
{
    const auto getName = api_.slot<ort::FnSessionGetInputName>(ort::kSlot_SessionGetInputName);
    const auto getNameOut = api_.slot<ort::FnSessionGetOutputName>(ort::kSlot_SessionGetOutputName);
    const auto getTypeInfo = api_.slot<ort::FnSessionGetInputTypeInfo>(ort::kSlot_SessionGetInputTypeInfo);
    const auto getTypeInfoOut = api_.slot<ort::FnSessionGetOutputTypeInfo>(ort::kSlot_SessionGetOutputTypeInfo);
    const auto allocatorFree = api_.slot<ort::FnAllocatorFree>(ort::kSlot_AllocatorFree);
    const auto releaseTypeInfo = api_.slot<ort::FnReleaseTypeInfo>(ort::kSlot_ReleaseTypeInfo);
    const auto castToTensorInfo = api_.slot<ort::FnCastTypeInfoToTensorInfo>(ort::kSlot_CastTypeInfoToTensorInfo);
    const auto getElementType = api_.slot<ort::FnGetTensorElementType>(ort::kSlot_GetTensorElementType);
    const auto getDimCount = api_.slot<ort::FnGetDimensionsCount>(ort::kSlot_GetDimensionsCount);
    const auto getDims = api_.slot<ort::FnGetDimensions>(ort::kSlot_GetDimensions);
    const auto nameFn = isInput ? getName : getNameOut;
    const auto typeInfoFn = isInput ? getTypeInfo : getTypeInfoOut;
    if (!nameFn || !typeInfoFn || !allocatorFree || !releaseTypeInfo || !castToTensorInfo || !getElementType ||
        !getDimCount || !getDims)
        return fail(std::format("OrtDmlSrBackend: the runtime does not expose the {} introspection slots", what));

    char* rawName = nullptr;
    if (ort::OrtStatus* status = nameFn(session_, index, hostAllocator_, &rawName))
        return fail(std::format("OrtDmlSrBackend: reading the {} name failed: {}", what,
                                ort::consumeStatus(api_, status)));
    if (!rawName)
        return fail(std::format("OrtDmlSrBackend: the model {} {} has no name", what, index));
    name = rawName;
    if (ort::OrtStatus* status = allocatorFree(hostAllocator_, rawName))
        veyra::log::warn(kLogTag, std::format("releasing the {} name failed: {}", what,
                                              ort::consumeStatus(api_, status)));

    ort::OrtTypeInfo* typeInfo = nullptr;
    if (ort::OrtStatus* status = typeInfoFn(session_, index, &typeInfo))
        return fail(std::format("OrtDmlSrBackend: reading the {} type failed: {}", what,
                                ort::consumeStatus(api_, status)));
    if (!typeInfo)
        return fail(std::format("OrtDmlSrBackend: the model {} {} has no type information", what, index));
    const TypeInfoOwner typeInfoOwner(releaseTypeInfo, typeInfo);

    const ort::OrtTensorTypeAndShapeInfo* shape = nullptr;
    if (ort::OrtStatus* status = castToTensorInfo(typeInfo, &shape)) {
        const std::string message = ort::consumeStatus(api_, status);
        return fail(std::format("OrtDmlSrBackend: {} `{}` is not a tensor: {}", what, name, message));
    }
    if (!shape)
        return fail(std::format("OrtDmlSrBackend: {} `{}` is not a tensor", what, name));

    ort::ONNXTensorElementDataType elementType = ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    if (ort::OrtStatus* status = getElementType(shape, &elementType))
        return fail(std::format("OrtDmlSrBackend: reading the {} element type failed: {}", what,
                                ort::consumeStatus(api_, status)));
    if (elementType != ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        return fail(std::format("OrtDmlSrBackend: {} `{}` must be float32, ONNX element type {} is not supported",
                                what, name, static_cast<int>(elementType)));

    size_t rank = 0;
    if (ort::OrtStatus* status = getDimCount(shape, &rank))
        return fail(std::format("OrtDmlSrBackend: reading the {} rank failed: {}", what,
                                ort::consumeStatus(api_, status)));
    if (rank != 4)
        return fail(std::format("OrtDmlSrBackend: {} `{}` must be NCHW (rank 4), the model declares rank {}", what,
                                name, rank));

    dims.assign(rank, 0);
    if (ort::OrtStatus* status = getDims(shape, dims.data(), rank))
        return fail(std::format("OrtDmlSrBackend: reading the {} shape failed: {}", what,
                                ort::consumeStatus(api_, status)));
    return true;
}

bool OrtDmlSrBackend::prepare(uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth,
                              uint32_t outputHeight)
{
    if (!ready())
        return fail("OrtDmlSrBackend: prepare() before a successful open()");
    if (inputWidth == 0 || inputHeight == 0 || outputWidth == 0 || outputHeight == 0)
        return fail(std::format("OrtDmlSrBackend: tensor dimensions must be non-zero (in {}x{}, out {}x{})",
                                inputWidth, inputHeight, outputWidth, outputHeight));
    if (inputWidth > kMaxExtent || inputHeight > kMaxExtent || outputWidth > kMaxExtent ||
        outputHeight > kMaxExtent)
        return fail(std::format("OrtDmlSrBackend: unsupported size (in {}x{}, out {}x{}; limit {} per axis)",
                                inputWidth, inputHeight, outputWidth, outputHeight, kMaxExtent));
    if (dimsMismatch(inputDims_, config_.channels, inputHeight, inputWidth))
        return fail(std::format("OrtDmlSrBackend: model input `{}` is {}, cannot serve {}x{}x{}x{}", inputName_,
                                describeDims(inputDims_), 1, config_.channels, inputHeight, inputWidth));
    if (dimsMismatch(outputDims_, config_.channels, outputHeight, outputWidth))
        return fail(std::format("OrtDmlSrBackend: model output `{}` is {}, cannot serve {}x{}x{}x{}", outputName_,
                                describeDims(outputDims_), 1, config_.channels, outputHeight, outputWidth));

    // Resolution change: drop the previous binding before allocating again.
    prepared_ = false;
    releaseBinding();
    releaseAllocations();
    stats_.usedFallback = false;
    fallback_ = false;
    inputBufferState_ = D3D12_RESOURCE_STATE_COMMON;
    outputResourceState_ = D3D12_RESOURCE_STATE_COMMON;

    inputWidth_ = inputWidth;
    inputHeight_ = inputHeight;
    outputWidth_ = outputWidth;
    outputHeight_ = outputHeight;

    const uint32_t channels = config_.channels;
    const uint64_t inputElements = static_cast<uint64_t>(channels) * inputHeight * inputWidth;
    const uint64_t outputElements = static_cast<uint64_t>(channels) * outputHeight * outputWidth;
    const size_t inputBytes = static_cast<size_t>(inputElements * sizeof(float));
    const size_t outputBytes = static_cast<size_t>(outputElements * sizeof(float));

    ID3D12Resource* inputResource = nullptr;
    ID3D12Resource* outputResource = nullptr;
    if (config_.skipD3dResourcesForTest) {
        hostInput_.assign(inputBytes, 0);
        hostOutput_.assign(outputBytes, 0);
        inputResource = reinterpret_cast<ID3D12Resource*>(kFakeResourceHandle);
        outputResource = reinterpret_cast<ID3D12Resource*>(kFakeResourceHandle + 0x10);
    } else {
        if (!device_)
            return fail("OrtDmlSrBackend: prepare() needs the graph's ID3D12Device for the DirectML tensors");
        inputBuffer_ = makeGpuBuffer(device_.Get(), inputBytes);
        if (!inputBuffer_)
            return fail(std::format("OrtDmlSrBackend: input tensor buffer allocation failed ({} bytes)", inputBytes));
        outputBuffer_ = makeGpuBuffer(device_.Get(), outputBytes);
        if (!outputBuffer_)
            return fail(std::format("OrtDmlSrBackend: output tensor buffer allocation failed ({} bytes)",
                                    outputBytes));
        inputResource = inputBuffer_.Get();
        outputResource = outputBuffer_.Get();
    }

    const auto createMemoryInfo = api_.slot<ort::FnCreateMemoryInfo>(ort::kSlot_CreateMemoryInfo);
    if (!createMemoryInfo)
        return fail("OrtDmlSrBackend: the runtime does not expose CreateMemoryInfo");
    if (ort::OrtStatus* status = createMemoryInfo(config_.dmlMemoryInfoName.c_str(), ort::OrtDeviceAllocator,
                                                 config_.deviceId, ort::OrtMemTypeDefault, &memoryInfo_))
        return fail(std::format("OrtDmlSrBackend: CreateMemoryInfo(\"{}\") failed: {}",
                                config_.dmlMemoryInfoName, ort::consumeStatus(api_, status)));
    if (!memoryInfo_)
        return fail(std::format("OrtDmlSrBackend: CreateMemoryInfo(\"{}\") returned null",
                                config_.dmlMemoryInfoName));

    // Wrap the D3D12 buffers in DirectML allocations. The resulting handles are
    // the tensor data pointers, so the runtime never touches system memory.
    if (ort::OrtStatus* status = dmlApi_->CreateGPUAllocationFromD3DResource(inputResource, &inputAllocation_))
        return fail(std::format("OrtDmlSrBackend: CreateGPUAllocationFromD3DResource(input) failed: {}",
                                ort::consumeStatus(api_, status)));
    if (!inputAllocation_)
        return fail("OrtDmlSrBackend: the DirectML input allocation is null");
    if (ort::OrtStatus* status = dmlApi_->CreateGPUAllocationFromD3DResource(outputResource, &outputAllocation_))
        return fail(std::format("OrtDmlSrBackend: CreateGPUAllocationFromD3DResource(output) failed: {}",
                                ort::consumeStatus(api_, status)));
    if (!outputAllocation_)
        return fail("OrtDmlSrBackend: the DirectML output allocation is null");

    const auto createTensor = api_.slot<ort::FnCreateTensorWithDataAsOrtValue>(ort::kSlot_CreateTensorWithDataAsOrtValue);
    if (!createTensor)
        return fail("OrtDmlSrBackend: the runtime does not expose CreateTensorWithDataAsOrtValue");
    const int64_t inputShape[4] = {1, static_cast<int64_t>(channels), static_cast<int64_t>(inputHeight),
                                   static_cast<int64_t>(inputWidth)};
    if (ort::OrtStatus* status = createTensor(memoryInfo_, inputAllocation_, inputBytes, inputShape, 4,
                                              ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputValue_))
        return fail(std::format("OrtDmlSrBackend: creating the input tensor failed: {}",
                                ort::consumeStatus(api_, status)));
    const int64_t outputShape[4] = {1, static_cast<int64_t>(channels), static_cast<int64_t>(outputHeight),
                                    static_cast<int64_t>(outputWidth)};
    if (ort::OrtStatus* status = createTensor(memoryInfo_, outputAllocation_, outputBytes, outputShape, 4,
                                              ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &outputValue_))
        return fail(std::format("OrtDmlSrBackend: creating the output tensor failed: {}",
                                ort::consumeStatus(api_, status)));

    const auto createIoBinding = api_.slot<ort::FnCreateIoBinding>(ort::kSlot_CreateIoBinding);
    const auto bindInput = api_.slot<ort::FnBindInput>(ort::kSlot_BindInput);
    const auto bindOutput = api_.slot<ort::FnBindOutput>(ort::kSlot_BindOutput);
    if (!createIoBinding || !bindInput || !bindOutput)
        return fail("OrtDmlSrBackend: the runtime does not expose the IoBinding entry points");
    if (ort::OrtStatus* status = createIoBinding(session_, &ioBinding_))
        return fail(std::format("OrtDmlSrBackend: CreateIoBinding failed: {}", ort::consumeStatus(api_, status)));
    if (!ioBinding_)
        return fail("OrtDmlSrBackend: CreateIoBinding returned null");
    if (ort::OrtStatus* status = bindInput(ioBinding_, inputName_.c_str(), inputValue_))
        return fail(std::format("OrtDmlSrBackend: BindInput(\"{}\") failed: {}", inputName_,
                                ort::consumeStatus(api_, status)));

    std::string bindOutputError;
    if (ort::OrtStatus* status = bindOutput(ioBinding_, outputName_.c_str(), outputValue_)) {
        bindOutputError = ort::consumeStatus(api_, status);
        fallback_ = true;
    }
    if (fallback_) {
        // The execution provider refused a caller-owned output tensor. Binding
        // the device instead still keeps the result on the GPU: the runtime
        // allocates the output and we resolve its ID3D12Resource afterwards.
        const auto bindOutputToDevice = api_.slot<ort::FnBindOutputToDevice>(ort::kSlot_BindOutputToDevice);
        const auto createAllocator = api_.slot<ort::FnCreateAllocator>(ort::kSlot_CreateAllocator);
        if (!bindOutputToDevice || !createAllocator)
            return fail(std::format("OrtDmlSrBackend: BindOutput(\"{}\") failed ({}) and the fallback entry "
                                    "points are missing",
                                    outputName_, bindOutputError));
        if (ort::OrtStatus* status = bindOutputToDevice(ioBinding_, outputName_.c_str(), memoryInfo_))
            return fail(std::format("OrtDmlSrBackend: BindOutput(\"{}\") failed ({}) and BindOutputToDevice "
                                    "failed too: {}",
                                    outputName_, bindOutputError, ort::consumeStatus(api_, status)));
        if (ort::OrtStatus* status = createAllocator(session_, memoryInfo_, &dmlAllocator_))
            return fail(std::format("OrtDmlSrBackend: CreateAllocator for the DirectML output failed: {}",
                                    ort::consumeStatus(api_, status)));
        if (!dmlAllocator_)
            return fail("OrtDmlSrBackend: CreateAllocator returned a null DirectML allocator");
        warning_ = std::format("OrtDmlSrBackend: the DirectML execution provider refused the pre-allocated output "
                               "binding ({}); binding the device and resolving the runtime-allocated resource "
                               "instead (still zero readback)",
                               bindOutputError);
        veyra::log::warn(kLogTag, warning_);
    }
    stats_.usedFallback = fallback_;

    if (!config_.skipD3dResourcesForTest && !loadPasses())
        return false;

    prepared_ = true;
    veyra::log::info(kLogTag, std::format("prepared in={}x{}x{} out={}x{}x{} bytes={}/{} output={}",
                                          channels, inputWidth_, inputHeight_, channels, outputWidth_,
                                          outputHeight_, inputBytes, outputBytes,
                                          fallback_ ? "runtime-allocated" : "pre-allocated"));
    return true;
}

bool OrtDmlSrBackend::loadPasses()
{
    if (!device_)
        return fail("OrtDmlSrBackend: the shader pipelines need an ID3D12Device");

    std::vector<uint8_t> code;
    if (!preprocess_.loadShader("OrtDmlPreprocess.dxil", code))
        return fail("OrtDmlSrBackend: shader OrtDmlPreprocess.dxil is missing; the build must compile "
                    "shaders/OrtDmlPreprocess.hlsl with dxc into the shader directory");
    if (!preprocess_.create(device_.Get(), code, 2, 1, 1, 4, 0, true))
        return fail("OrtDmlSrBackend: OrtDmlPreprocess pipeline creation failed");
    if (!postprocess_.loadShader("OrtDmlPostprocess.dxil", code))
        return fail("OrtDmlSrBackend: shader OrtDmlPostprocess.dxil is missing; the build must compile "
                    "shaders/OrtDmlPostprocess.hlsl with dxc into the shader directory");
    if (!postprocess_.create(device_.Get(), code, 3 * kPostDescriptorSlots, 2, 1, 4, 0, true))
        return fail("OrtDmlSrBackend: OrtDmlPostprocess pipeline creation failed");
    if (!stager_.initialize(device_.Get(), kStagerSlots))
        return fail("OrtDmlSrBackend: descriptor staging heap creation failed");
    return true;
}

bool OrtDmlSrBackend::run()
{
    if (!prepared_)
        return fail("OrtDmlSrBackend: run() before prepare()");
    const auto runWithBinding = api_.slot<ort::FnRunWithBinding>(ort::kSlot_RunWithBinding);
    if (!runWithBinding)
        return fail("OrtDmlSrBackend: the runtime does not expose RunWithBinding");
    if (ort::OrtStatus* status = runWithBinding(session_, nullptr, ioBinding_))
        return fail(std::format("OrtDmlSrBackend: RunWithBinding failed: {}", ort::consumeStatus(api_, status)));

    ++stats_.runs;
    stats_.usedFallback = fallback_;
    if (fallback_ && !resolveFallbackOutput())
        return false;
    return true;
}

bool OrtDmlSrBackend::resolveFallbackOutput()
{
    const auto getBoundOutputValues = api_.slot<ort::FnGetBoundOutputValues>(ort::kSlot_GetBoundOutputValues);
    const auto getTensorMutableData = api_.slot<ort::FnGetTensorMutableData>(ort::kSlot_GetTensorMutableData);
    const auto releaseValue = api_.slot<ort::FnReleaseValue>(ort::kSlot_ReleaseValue);
    const auto allocatorFree = api_.slot<ort::FnAllocatorFree>(ort::kSlot_AllocatorFree);
    if (!getBoundOutputValues || !getTensorMutableData || !releaseValue || !allocatorFree)
        return fail("OrtDmlSrBackend: the runtime does not expose the bound-output entry points");

    ort::OrtValue** values = nullptr;
    size_t count = 0;
    if (ort::OrtStatus* status = getBoundOutputValues(ioBinding_, hostAllocator_, &values, &count))
        return fail(std::format("OrtDmlSrBackend: GetBoundOutputValues failed: {}",
                                ort::consumeStatus(api_, status)));

    std::string message;
    ID3D12Resource* resource = nullptr;
    if (!values || count < 1 || !values[0]) {
        message = "OrtDmlSrBackend: GetBoundOutputValues returned no output tensor";
    } else {
        void* data = nullptr;
        if (ort::OrtStatus* status = getTensorMutableData(values[0], &data)) {
            message = std::format("OrtDmlSrBackend: reading the runtime-allocated output failed: {}",
                                  ort::consumeStatus(api_, status));
        } else if (ort::OrtStatus* status =
                       dmlApi_->GetD3D12ResourceFromAllocation(dmlAllocator_, data, &resource)) {
            message = std::format("OrtDmlSrBackend: resolving the runtime-allocated output resource failed: {}",
                                  ort::consumeStatus(api_, status));
        } else if (!resource) {
            message = "OrtDmlSrBackend: the runtime-allocated output resource is null";
        }
    }

    // Keep values[0] alive in fallbackOutputValue_ until the next run() or
    // releaseBinding() so the DirectML allocator cannot reclaim the pooled
    // backing memory before recordPostprocess() executes on the GPU.
    ort::OrtValue* adoptedValue = nullptr;
    if (message.empty() && values && count >= 1 && values[0]) {
        adoptedValue = values[0];
        values[0] = nullptr;
    }
    if (values) {
        for (size_t i = 0; i < count; ++i) {
            if (values[i])
                releaseValue(values[i]);
        }
        if (ort::OrtStatus* status = allocatorFree(hostAllocator_, values))
            veyra::log::warn(kLogTag, std::format("releasing the bound output array failed: {}",
                                                  ort::consumeStatus(api_, status)));
    }
    if (!message.empty()) {
        if (adoptedValue)
            releaseValue(adoptedValue);
        return fail(message);
    }

    // OrtDmlApi::GetD3D12ResourceFromAllocation returns an AddRef'd COM pointer
    // (dml_provider_factory.cc calls (*d3d_resource)->AddRef()). Adopt ownership
    // directly: releasing the previous pointer first also drops our prior ref
    // when DirectML hands back the same pooled allocation on subsequent frames.
    if (resolvedOutput_)
        resolvedOutput_->Release();
    if (fallbackOutputValue_)
        releaseValue(fallbackOutputValue_);
    fallbackOutputValue_ = adoptedValue;
    resolvedOutput_ = resource;
    outputResourceState_ = D3D12_RESOURCE_STATE_COMMON;
    return true;
}

bool OrtDmlSrBackend::transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                                 D3D12_RESOURCE_STATES& tracked, D3D12_RESOURCE_STATES to)
{
    if (!resource || !list)
        return false;
    if (tracked == to)
        return true;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = tracked;
    barrier.Transition.StateAfter = to;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list->ResourceBarrier(1, &barrier);
    tracked = to;
    return true;
}

bool OrtDmlSrBackend::recordPreprocess(ID3D12GraphicsCommandList* list, ID3D12Resource* linearSource)
{
    if (!prepared_)
        return reject("OrtDmlSrBackend: recordPreprocess() before prepare()");
    if (!list)
        return reject("OrtDmlSrBackend: recordPreprocess() without a command list");
    if (!linearSource)
        return reject("OrtDmlSrBackend: recordPreprocess() without a source texture");
    if (!preprocess_.rootSig || !inputBuffer_)
        return reject("OrtDmlSrBackend: the preprocess pipeline is not available (missing shader or D3D12 device)");

    const DXGI_FORMAT format = linearSource->GetDesc().Format;
    if (format != DXGI_FORMAT_R16G16B16A16_FLOAT)
        return reject(std::format("OrtDmlSrBackend: the preprocess source must be RGBA16F, got DXGI format {}",
                                  static_cast<unsigned>(format)));

    const uint32_t channels = config_.channels;
    const uint64_t elements = static_cast<uint64_t>(channels) * inputWidth_ * inputHeight_;

    transition(list, inputBuffer_.Get(), inputBufferState_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Cache preprocess views so steady-state frames never overwrite descriptors
    // in the shader-visible heap while an earlier command list is still in flight.
    if (linearSource != cachedPreSource_ || inputBuffer_.Get() != cachedPreInputBuf_) {
        const D3D12_SHADER_RESOURCE_VIEW_DESC sourceView = makeTextureSrvDesc(format);
        stager_.stageSrv(linearSource, &sourceView, preprocess_.heap.Get(), 0);
        makeBufferUav(device_.Get(), inputBuffer_.Get(), static_cast<UINT>(elements), sizeof(float),
                      cpuHandleOf(preprocess_, 1));
        cachedPreSource_ = linearSource;
        cachedPreInputBuf_ = inputBuffer_.Get();
    }

    float constants[4] = {std::bit_cast<float>(inputWidth_), std::bit_cast<float>(inputHeight_),
                          std::bit_cast<float>(channels), 0.0f};
    preprocess_.bind(list, constants, gpuHandleOf(preprocess_, 0).ptr, gpuHandleOf(preprocess_, 1).ptr);
    list->Dispatch(static_cast<UINT>((elements + 63) / 64), 1, 1);

    // Make the dispatch visible, then hand the buffer back in COMMON: that is the
    // only state the DirectML execution provider accepts for a bound tensor.
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = inputBuffer_.Get();
    list->ResourceBarrier(1, &barrier);
    transition(list, inputBuffer_.Get(), inputBufferState_, D3D12_RESOURCE_STATE_COMMON);
    return true;
}

bool OrtDmlSrBackend::recordPostprocess(ID3D12GraphicsCommandList* list, ID3D12Resource* linearOutput,
                                        ID3D12Resource* sourceForChroma)
{
    if (!prepared_)
        return reject("OrtDmlSrBackend: recordPostprocess() before prepare()");
    if (!list)
        return reject("OrtDmlSrBackend: recordPostprocess() without a command list");
    if (!linearOutput)
        return reject("OrtDmlSrBackend: recordPostprocess() without an output texture");
    if (!postprocess_.rootSig)
        return reject("OrtDmlSrBackend: the postprocess pipeline is not available (missing shader or D3D12 device)");
    ID3D12Resource* modelOutput = outputResource();
    if (!modelOutput)
        return reject("OrtDmlSrBackend: the postprocess needs the model output; run() has not produced one");
    if (config_.channels == 1 && !sourceForChroma)
        return reject("OrtDmlSrBackend: a luma-only model needs the linear frame for chroma recombination");

    const DXGI_FORMAT format = linearOutput->GetDesc().Format;
    if (format != DXGI_FORMAT_R16G16B16A16_FLOAT)
        return reject(std::format("OrtDmlSrBackend: the postprocess output must be RGBA16F, got DXGI format {}",
                                  static_cast<unsigned>(format)));
    if (sourceForChroma && sourceForChroma->GetDesc().Format != DXGI_FORMAT_R16G16B16A16_FLOAT)
        return reject("OrtDmlSrBackend: the chroma source must be RGBA16F");

    const uint32_t channels = config_.channels;
    const uint64_t elements = static_cast<uint64_t>(channels) * outputWidth_ * outputHeight_;

    transition(list, modelOutput, outputResourceState_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // Cache postprocess views and advance through a ring of descriptor triplets
    // whenever a resource pointer changes (e.g. rotating runtime-allocated DML
    // fallback outputs), avoiding shader-visible descriptor heap races in flight.
    if (!postDescriptorValid_ || modelOutput != cachedPostModelOut_ ||
        sourceForChroma != cachedPostChroma_ || linearOutput != cachedPostLinearOut_) {
        const UINT slot = nextPostSlot_;
        nextPostSlot_ = (nextPostSlot_ + 1) % kPostDescriptorSlots;
        const UINT base = slot * 3;
        D3D12_SHADER_RESOURCE_VIEW_DESC modelView{};
        makeBufferSrvDesc(modelView, static_cast<UINT>(elements), sizeof(float));
        stager_.stageSrv(modelOutput, &modelView, postprocess_.heap.Get(), base);
        const D3D12_SHADER_RESOURCE_VIEW_DESC chromaView = makeTextureSrvDesc(DXGI_FORMAT_R16G16B16A16_FLOAT);
        // RGB models never read t1; staging a typed null SRV keeps the descriptor
        // table valid in D3D12 without bypassing the non-shader-visible staging heap.
        stager_.stageSrv(sourceForChroma, &chromaView, postprocess_.heap.Get(), base + 1);
        makeUav(device_.Get(), linearOutput, format, cpuHandleOf(postprocess_, base + 2));
        cachedPostSlot_ = slot;
        cachedPostModelOut_ = modelOutput;
        cachedPostChroma_ = sourceForChroma;
        cachedPostLinearOut_ = linearOutput;
        postDescriptorValid_ = true;
    }

    const UINT base = cachedPostSlot_ * 3;
    float constants[4] = {std::bit_cast<float>(outputWidth_), std::bit_cast<float>(outputHeight_),
                          std::bit_cast<float>(channels), 0.0f};
    postprocess_.bind(list, constants, gpuHandleOf(postprocess_, base).ptr, gpuHandleOf(postprocess_, base + 2).ptr);
    list->Dispatch((outputWidth_ + 7) / 8, (outputHeight_ + 7) / 8, 1);

    transition(list, modelOutput, outputResourceState_, D3D12_RESOURCE_STATE_COMMON);
    return true;
}

void OrtDmlSrBackend::releaseBinding() noexcept
{
    const auto releaseIoBinding = api_.slot<ort::FnReleaseIoBinding>(ort::kSlot_ReleaseIoBinding);
    if (releaseIoBinding && ioBinding_)
        releaseIoBinding(ioBinding_);
    ioBinding_ = nullptr;

    const auto releaseValue = api_.slot<ort::FnReleaseValue>(ort::kSlot_ReleaseValue);
    if (releaseValue) {
        if (fallbackOutputValue_)
            releaseValue(fallbackOutputValue_);
        if (outputValue_)
            releaseValue(outputValue_);
        if (inputValue_)
            releaseValue(inputValue_);
    }
    fallbackOutputValue_ = nullptr;
    outputValue_ = nullptr;
    inputValue_ = nullptr;

    const auto releaseMemoryInfo = api_.slot<ort::FnReleaseMemoryInfo>(ort::kSlot_ReleaseMemoryInfo);
    if (releaseMemoryInfo && memoryInfo_)
        releaseMemoryInfo(memoryInfo_);
    memoryInfo_ = nullptr;

    const auto releaseAllocator = api_.slot<ort::FnReleaseAllocator>(ort::kSlot_ReleaseAllocator);
    if (releaseAllocator && dmlAllocator_)
        releaseAllocator(dmlAllocator_);
    dmlAllocator_ = nullptr;
}

void OrtDmlSrBackend::releaseSessionChain() noexcept
{
    const auto releaseSession = api_.slot<ort::FnReleaseSession>(ort::kSlot_ReleaseSession);
    if (releaseSession && session_)
        releaseSession(session_);
    session_ = nullptr;

    const auto releaseSessionOptions = api_.slot<ort::FnReleaseSessionOptions>(ort::kSlot_ReleaseSessionOptions);
    if (releaseSessionOptions && sessionOptions_)
        releaseSessionOptions(sessionOptions_);
    sessionOptions_ = nullptr;

    const auto releaseEnv = api_.slot<ort::FnReleaseEnv>(ort::kSlot_ReleaseEnv);
    if (releaseEnv && env_)
        releaseEnv(env_);
    env_ = nullptr;

    // The default allocator is owned by the runtime and must not be released.
    hostAllocator_ = nullptr;
}

void OrtDmlSrBackend::releaseAllocations() noexcept
{
    if (dmlApi_ && dmlApi_->FreeGPUAllocation) {
        if (outputAllocation_)
            dmlApi_->FreeGPUAllocation(outputAllocation_);
        if (inputAllocation_)
            dmlApi_->FreeGPUAllocation(inputAllocation_);
    }
    outputAllocation_ = nullptr;
    inputAllocation_ = nullptr;

    if (resolvedOutput_) {
        resolvedOutput_->Release();
        resolvedOutput_ = nullptr;
    }
    inputBuffer_.Reset();
    outputBuffer_.Reset();
    hostInput_.clear();
    hostOutput_.clear();
    inputBufferState_ = D3D12_RESOURCE_STATE_COMMON;
    outputResourceState_ = D3D12_RESOURCE_STATE_COMMON;
    cachedPreSource_ = nullptr;
    cachedPreInputBuf_ = nullptr;
    cachedPostModelOut_ = nullptr;
    cachedPostChroma_ = nullptr;
    cachedPostLinearOut_ = nullptr;
    cachedPostSlot_ = 0;
    nextPostSlot_ = 0;
    postDescriptorValid_ = false;
    prepared_ = false;
    fallback_ = false;
    stats_.usedFallback = false;
    inputWidth_ = 0;
    inputHeight_ = 0;
    outputWidth_ = 0;
    outputHeight_ = 0;
}

void OrtDmlSrBackend::shutdown() noexcept
{
    // Reverse creation order: binding -> session chain -> allocations -> the
    // device and modules the module's objects depend on.
    releaseBinding();
    releaseSessionChain();
    releaseAllocations();

    if (dmlDeviceOwned_ && dmlDevice_)
        static_cast<IUnknown*>(dmlDevice_)->Release();
    dmlDevice_ = nullptr;
    dmlDeviceOwned_ = false;

    if (directmlModule_) {
        FreeLibrary(directmlModule_);
        directmlModule_ = nullptr;
    }
    if (runtimeModule_) {
        FreeLibrary(runtimeModule_);
        runtimeModule_ = nullptr;
    }

    preprocess_ = ComputePass{};
    postprocess_ = ComputePass{};
    stager_ = DescriptorStager{};
    device_.Reset();
    queue_.Reset();

    apiBase_ = nullptr;
    api_ = ort::OrtApiView{};
    dmlApi_ = nullptr;

    inputName_.clear();
    outputName_.clear();
    inputDims_.clear();
    outputDims_.clear();
    runtimeVersion_.clear();
}

ID3D12Resource* OrtDmlSrBackend::outputResource() const
{
    return fallback_ ? resolvedOutput_ : outputBuffer_.Get();
}

} // namespace veyra::pipeline
