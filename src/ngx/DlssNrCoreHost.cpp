// SPDX-License-Identifier: GPL-3.0-only
// Line B: the NGX D3D12 Feature-18 host for dlssnr_core.dll. Every entry point is resolved
// with GetProcAddress from the module itself, so no NVIDIA import library is linked; the SDK
// header is needed only to call NVSDK_NGX_Parameter::Set through the module's own vtable.
#include "veyra/ngx/DlssNrCoreHost.h"

#if defined(VEYRA_EXPERIMENTAL_LOCAL_DLSSNR)

#include <d3d12.h>
#include <nvsdk_ngx.h>
#include "veyra/Log.h"
#include "veyra/NgxResult.h"
#include "veyra/ngx/DlssNrParameters.h"
#include "veyra/ngx/NgxParameters.h"

#include <algorithm>
#include <format>

namespace veyra::ngx {
namespace {
using InitExtFn = NVSDK_NGX_Result (*)(unsigned long long, const wchar_t*, ID3D12Device*, NVSDK_NGX_Version, const NVSDK_NGX_Parameter*);
using AllocateParametersFn = NVSDK_NGX_Result (*)(NVSDK_NGX_Parameter**);
using DestroyParametersFn = NVSDK_NGX_Result (*)(NVSDK_NGX_Parameter*);
using CreateFeatureFn = NVSDK_NGX_Result (*)(ID3D12GraphicsCommandList*, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
using EvaluateFeatureFn = NVSDK_NGX_Result (*)(ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback);
using ReleaseFeatureFn = NVSDK_NGX_Result (*)(NVSDK_NGX_Handle*);
using Shutdown1Fn = NVSDK_NGX_Result (*)(ID3D12Device*);
constexpr unsigned long long kAppId = 0x0876232Cull; // same signed snippet id the NVIDIA NR path uses
// Feature ID 18 is not named in NVIDIA DLSS SDK 310.7 (<nvsdk_ngx.h> has no such enumerator), so
// it is spelled as the audited numeric value shared with DlssNrRuntimeAdapter.h:50.
constexpr auto kFeatureId = static_cast<NVSDK_NGX_Feature>(18);
}

struct DlssNrCoreHost::Impl {
    HMODULE module = nullptr;
    InitExtFn init = nullptr;
    AllocateParametersFn allocate = nullptr;
    DestroyParametersFn destroyParameters = nullptr;
    CreateFeatureFn createFeature = nullptr;
    EvaluateFeatureFn evaluateFeature = nullptr;
    ReleaseFeatureFn releaseFeature = nullptr;
    Shutdown1Fn shutdown = nullptr;
    NVSDK_NGX_Parameter* parameters = nullptr;
    NVSDK_NGX_Handle* handle = nullptr;
    ID3D12Device* device = nullptr;
    DlssNrCoreDescription desc{};
    std::string error;
    bool inited = false;
};

bool DlssNrCoreHost::compiledIn() { return true; }
const wchar_t* DlssNrCoreHost::moduleName() { return L"dlssnr_core.dll"; }

DlssNrCoreHost::DlssNrCoreHost() : impl_(std::make_unique<Impl>()) {}
DlssNrCoreHost::~DlssNrCoreHost() { close(); }
bool DlssNrCoreHost::ready() const noexcept { return impl_ && impl_->handle != nullptr; }
const std::string& DlssNrCoreHost::error() const { return impl_->error; }

bool DlssNrCoreHost::open(const std::filesystem::path& modulePath, ID3D12Device* device, const std::filesystem::path& appDataPath, const DlssNrCoreDescription& desc) {
    close();
    auto& s = *impl_;
    s.desc = desc;
    if (!device || !desc.width || !desc.height) { s.error = "dlssnr_core requires a device and a non-zero extent"; log::error("amd-nr", s.error); return false; }
    const auto normalized = modulePath.lexically_normal();
    if (!normalized.is_absolute() || normalized.filename() != moduleName()) {
        s.error = std::format("{} is not an absolute dlssnr_core.dll path", normalized.string());
        log::error("amd-nr", s.error); return false;
    }
    s.module = LoadLibraryExW(normalized.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!s.module) { s.error = std::format("dlssnr_core.dll load failed win32={}", GetLastError()); log::error("amd-nr", s.error); return false; }
    const auto resolve = [this](const char* name) { return GetProcAddress(impl_->module, name); };
    s.init = reinterpret_cast<InitExtFn>(resolve("NVSDK_NGX_D3D12_Init_Ext"));
    s.allocate = reinterpret_cast<AllocateParametersFn>(resolve("NVSDK_NGX_D3D12_AllocateParameters"));
    s.destroyParameters = reinterpret_cast<DestroyParametersFn>(resolve("NVSDK_NGX_D3D12_DestroyParameters"));
    s.createFeature = reinterpret_cast<CreateFeatureFn>(resolve("NVSDK_NGX_D3D12_CreateFeature"));
    s.evaluateFeature = reinterpret_cast<EvaluateFeatureFn>(resolve("NVSDK_NGX_D3D12_EvaluateFeature"));
    s.releaseFeature = reinterpret_cast<ReleaseFeatureFn>(resolve("NVSDK_NGX_D3D12_ReleaseFeature"));
    s.shutdown = reinterpret_cast<Shutdown1Fn>(resolve("NVSDK_NGX_D3D12_Shutdown1"));
    if (!s.init || !s.allocate || !s.destroyParameters || !s.createFeature || !s.evaluateFeature || !s.releaseFeature) {
        s.error = "dlssnr_core.dll is missing a required NGX D3D12 entry point"; log::error("amd-nr", s.error); close(); return false;
    }
    s.device = device;
    const auto appData = appDataPath.wstring();
    const auto initResult = s.init(kAppId, appData.c_str(), device, NVSDK_NGX_Version_API, nullptr);
    if (NVSDK_NGX_FAILED(initResult)) { s.error = std::format("NVSDK_NGX_D3D12_Init_Ext failed: {}", ngxResultString(initResult)); log::error("amd-nr", s.error); close(); return false; }
    s.inited = true;
    const auto allocResult = s.allocate(&s.parameters);
    if (NVSDK_NGX_FAILED(allocResult) || !s.parameters) { s.error = std::format("NVSDK_NGX_D3D12_AllocateParameters failed: {}", ngxResultString(allocResult)); log::error("amd-nr", s.error); close(); return false; }
    ParameterBlock params(s.parameters);
    params.setU32(dlssnr::kWidth, desc.width);
    params.setU32(dlssnr::kHeight, desc.height);
    params.setI32(dlssnr::kHintRenderPreset, desc.style);
    // The upstream nr_ngx_core.cpp read_controls() reads DLSSNR.Style; setting only the hint left
    // the style parameter absent and the network fell back to its own default.
    params.setI32(dlssnr::kStyle, desc.style);
    params.setI32(dlssnr::kUICorrection, desc.uiCorrection);
    params.setU32(dlssnr::kEnabled, 1);
    log::info("amd-nr", std::format("dlssnr_core NGX Feature 18 host opened {}x{} style={} uiCorrection={}", desc.width, desc.height, desc.style, desc.uiCorrection));
    return true;
}

bool DlssNrCoreHost::create(ID3D12GraphicsCommandList* list) {
    auto& s = *impl_;
    if (!s.parameters || s.handle || !list) { s.error = "dlssnr_core create requires an open host, a parameter block and a command list"; log::error("amd-nr", s.error); return false; }
    const auto result = s.createFeature(list, kFeatureId, s.parameters, &s.handle);
    if (NVSDK_NGX_FAILED(result) || !s.handle) {
        s.handle = nullptr;
        s.error = std::format("NVSDK_NGX_D3D12_CreateFeature(18) failed: {}", ngxResultString(result));
        log::error("amd-nr", s.error); return false;
    }
    log::info("amd-nr", "dlssnr_core Feature 18 created; the create frame stays pass-through while the submission epoch starts");
    return true;
}

bool DlssNrCoreHost::evaluate(ID3D12GraphicsCommandList* list, const DlssNrCoreFrame& frame) {
    auto& s = *impl_;
    if (!s.handle || !list || !frame.color || !frame.output) { s.error = "dlssnr_core evaluate requires a created feature, a list, colour and output"; log::error("amd-nr", s.error); return false; }
    ParameterBlock params(s.parameters);
    params.setU32(dlssnr::kWidth, s.desc.width);
    params.setU32(dlssnr::kHeight, s.desc.height);
    params.setD3D12Resource(dlssnr::kColor, frame.color);
    params.setD3D12Resource(dlssnr::kOutput, frame.output);
    if (frame.depth) params.setD3D12Resource(dlssnr::kDepth, frame.depth);
    if (frame.motion) params.setD3D12Resource(dlssnr::kMVec, frame.motion);
    // The module can only express an origin-zero colour/output rect; the subrect fields carry the
    // extent instead of the allocation size (a larger allocation with a smaller rect is rejected).
    params.setU32(dlssnr::kColorSubrectBaseX, 0);
    params.setU32(dlssnr::kColorSubrectBaseY, 0);
    params.setU32(dlssnr::kOutputSubrectBaseX, 0);
    params.setU32(dlssnr::kOutputSubrectBaseY, 0);
    params.setU32(dlssnr::kMVecSubrectBaseX, 0);
    params.setU32(dlssnr::kMVecSubrectBaseY, 0);
    params.setU32(dlssnr::kDepthSubrectBaseX, 0);
    params.setU32(dlssnr::kDepthSubrectBaseY, 0);
    params.setU32(dlssnr::kColorSubrectWidth, s.desc.width);
    params.setU32(dlssnr::kColorSubrectHeight, s.desc.height);
    params.setU32(dlssnr::kOutputSubrectWidth, s.desc.width);
    params.setU32(dlssnr::kOutputSubrectHeight, s.desc.height);
    if (frame.motion) {
        params.setU32(dlssnr::kMVecSubrectWidth, s.desc.width);
        params.setU32(dlssnr::kMVecSubrectHeight, s.desc.height);
    }
    if (frame.depth) {
        params.setU32(dlssnr::kDepthSubrectWidth, s.desc.width);
        params.setU32(dlssnr::kDepthSubrectHeight, s.desc.height);
    }
    params.setF32(dlssnr::kMVecScaleX, 1.0f);
    params.setF32(dlssnr::kMVecScaleY, 1.0f);
    params.setI32(dlssnr::kDepthInverted, 1);
    params.setU32(dlssnr::kReset, frame.reset ? 1u : 0u);
    params.setU32(dlssnr::kEnabled, 1);
    // read_controls() also reads DLSSNR.Style every evaluate; a missing key would reset the style.
    params.setI32(dlssnr::kStyle, s.desc.style);
    params.setF32(dlssnr::kIntensity, std::clamp(frame.intensity, 0.0f, 1.0f));
    params.setF32(dlssnr::kLocalToneStrength, frame.tone);
    params.setF32(dlssnr::kLocalStructureStrength, frame.structure);
    params.setF32(dlssnr::kSkinStructureStrength, frame.skin);
    params.setI32(dlssnr::kUseAutoMask, frame.autoMask);
    const auto result = s.evaluateFeature(list, s.handle, s.parameters, nullptr);
    if (NVSDK_NGX_FAILED(result)) { s.error = std::format("NVSDK_NGX_D3D12_EvaluateFeature(18) failed: {}", ngxResultString(result)); log::error("amd-nr", s.error); return false; }
    return true;
}

void DlssNrCoreHost::close() noexcept {
    if (!impl_) return;
    auto& s = *impl_;
    if (s.module) {
        if (s.handle && s.releaseFeature) s.releaseFeature(s.handle);
        s.handle = nullptr;
        if (s.parameters && s.destroyParameters) s.destroyParameters(s.parameters);
        s.parameters = nullptr;
        if (s.inited && s.shutdown && s.device) s.shutdown(s.device);
        s.inited = false;
        FreeLibrary(s.module);
        s.module = nullptr;
    }
    s.device = nullptr;
}

} // namespace veyra::ngx

#else // !VEYRA_EXPERIMENTAL_LOCAL_DLSSNR

namespace veyra::ngx {
struct DlssNrCoreHost::Impl { std::string error{ "dlssnr_core host not built (VEYRA_ENABLE_EXPERIMENTAL_DLSSNR=OFF)" }; };
bool DlssNrCoreHost::compiledIn() { return false; }
const wchar_t* DlssNrCoreHost::moduleName() { return L"dlssnr_core.dll"; }
DlssNrCoreHost::DlssNrCoreHost() : impl_(std::make_unique<Impl>()) {}
DlssNrCoreHost::~DlssNrCoreHost() = default;
bool DlssNrCoreHost::ready() const noexcept { return false; }
const std::string& DlssNrCoreHost::error() const { return impl_->error; }
bool DlssNrCoreHost::open(const std::filesystem::path&, ID3D12Device*, const std::filesystem::path&, const DlssNrCoreDescription&) { return false; }
bool DlssNrCoreHost::create(ID3D12GraphicsCommandList*) { return false; }
bool DlssNrCoreHost::evaluate(ID3D12GraphicsCommandList*, const DlssNrCoreFrame&) { return false; }
void DlssNrCoreHost::close() noexcept {}
}
#endif
