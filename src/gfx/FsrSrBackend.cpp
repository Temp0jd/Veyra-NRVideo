#include "veyra/gfx/FsrSrBackend.h"

#include "veyra/gfx/FfxVersionSelect.h"
#include "veyra/Log.h"
#include "veyra/RuntimePaths.h"

#include <algorithm>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#ifdef VEYRA_HAS_FSR
#include <ffx_api_loader.h>
#include <ffx_upscale.h>
#include <dx12/ffx_api_dx12.h>
#endif

namespace veyra::gfx {

#ifdef VEYRA_HAS_FSR
namespace {

FfxApiSurfaceFormat surfaceFormatFromDxgi(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM: return FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_R10G10B10A2_UNORM: return FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R16G16_FLOAT: return FFX_API_SURFACE_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R32_FLOAT: return FFX_API_SURFACE_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R16_FLOAT: return FFX_API_SURFACE_FORMAT_R16_FLOAT;
    case DXGI_FORMAT_R8_UNORM: return FFX_API_SURFACE_FORMAT_R8_UNORM;
    default: return FFX_API_SURFACE_FORMAT_UNKNOWN;
    }
}

FfxApiResource makeResource(ID3D12Resource* resource, uint32_t state) {
    FfxApiResource api{};
    if (resource == nullptr) return api;
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    api.resource = resource;
    api.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    api.description.format = surfaceFormatFromDxgi(desc.Format);
    api.description.width = uint32_t(desc.Width);
    api.description.height = desc.Height;
    api.description.depth = desc.DepthOrArraySize;
    api.description.mipCount = desc.MipLevels;
    api.description.flags = 0;
    api.description.usage = FFX_API_RESOURCE_USAGE_READ_ONLY;
    if ((desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0) {
        api.description.usage = FfxApiResourceUsage(api.description.usage | FFX_API_RESOURCE_USAGE_UAV);
    }
    api.state = state;
    return api;
}

const char* returnName(ffxReturnCode_t code) {
    switch (code) {
    case FFX_API_RETURN_OK: return "OK";
    case FFX_API_RETURN_ERROR: return "ERROR";
    case FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE: return "ERROR_UNKNOWN_DESCTYPE";
    case FFX_API_RETURN_ERROR_RUNTIME_ERROR: return "ERROR_RUNTIME_ERROR";
    case FFX_API_RETURN_NO_PROVIDER: return "NO_PROVIDER";
    case FFX_API_RETURN_ERROR_MEMORY: return "ERROR_MEMORY";
    case FFX_API_RETURN_ERROR_PARAMETER: return "ERROR_PARAMETER";
    default: return "UNKNOWN";
    }
}

void fsrMessage(uint32_t type, const wchar_t* message) {
    if (message == nullptr) return;
    std::string text;
    for (const wchar_t* p = message; *p != 0; ++p) {
        text.push_back(*p < 128 ? char(*p) : '?');
    }
    if (type == FFX_API_MESSAGE_TYPE_ERROR) {
        log::error("fsr-sr", text);
    } else {
        log::warn("fsr-sr", text);
    }
}

// Keep the provider directory on the DLL search path only while the loader
// resolves amd_fidelityfx_upscaler_dx12.dll, then restore the process state.
class DirectoryGuard {
public:
    explicit DirectoryGuard(const std::filesystem::path& directory) {
        const auto abs = std::filesystem::absolute(directory);
        cookie_ = AddDllDirectory(abs.c_str());
    }
    ~DirectoryGuard() {
        if (cookie_ != nullptr) RemoveDllDirectory(cookie_);
    }

private:
    DLL_DIRECTORY_COOKIE cookie_ = nullptr;
};

} // namespace
#endif

struct FsrSrBackend::Impl {
    bool available = false;
    uint64_t evaluateCount = 0;
    uint64_t failureCount = 0;
    uint64_t providerVersionId = 0;
    std::string providerVersion;
#ifdef VEYRA_HAS_FSR
    HMODULE loader = nullptr;
    HMODULE provider = nullptr;
    ffxFunctions functions{};
    ffxContext context = nullptr;
    uint32_t maxRenderWidth = 0, maxRenderHeight = 0;
    uint32_t maxUpscaleWidth = 0, maxUpscaleHeight = 0;
#endif
};

FsrSrBackend::FsrSrBackend() : p_(std::make_unique<Impl>()) {}

FsrSrBackend::~FsrSrBackend() { release(); }

bool FsrSrBackend::initialize(ID3D12Device* device, uint32_t maxRenderWidth, uint32_t maxRenderHeight,
                              uint32_t maxUpscaleWidth, uint32_t maxUpscaleHeight, bool hdr,
                              uint32_t preferredMajor) {
#ifdef VEYRA_HAS_FSR
    auto& p = *p_;
    if (p.available) return true;
    const auto root = runtime::localDataDirectory() / "amd" / "fidelityfx";
    const auto loaderPath = std::filesystem::absolute(root / "amd_fidelityfx_loader_dx12.dll");
    const auto providerPath = std::filesystem::absolute(root / "amd_fidelityfx_upscaler_dx12.dll");
    if (!std::filesystem::exists(loaderPath)) {
        log::warn("fsr-sr", std::format("loader missing at {}", loaderPath.string()));
        return false;
    }
    DirectoryGuard guard(root);
    p.loader = LoadLibraryExW(loaderPath.c_str(), nullptr,
                              LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (p.loader == nullptr) {
        log::error("fsr-sr", std::format("loader load failed win32={}", GetLastError()));
        return false;
    }
    if (std::filesystem::exists(providerPath)) {
        p.provider = LoadLibraryExW(providerPath.c_str(), nullptr,
                                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!p.provider) {
            log::warn("fsr-sr", std::format("provider preload failed win32={}", GetLastError()));
        }
    }
    ffxLoadFunctions(&p.functions, p.loader);
    if (p.functions.CreateContext == nullptr || p.functions.DestroyContext == nullptr ||
        p.functions.Dispatch == nullptr || p.functions.Query == nullptr) {
        log::error("fsr-sr", "loader is missing the FidelityFX API entry points");
        release();
        return false;
    }

    std::vector<uint64_t> ids;
    std::vector<const char*> names;
    {
        ffxQueryDescGetVersions versions{};
        versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
        versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        versions.device = device;
        uint64_t count = 0;
        versions.outputCount = &count;
        const auto countResult = p.functions.Query(nullptr, &versions.header);
        if (countResult == FFX_API_RETURN_OK && count > 0 && count <= 64) {
            ids.assign(size_t(count), 0);
            names.assign(size_t(count), nullptr);
            uint64_t capacity = count;
            versions.outputCount = &capacity;
            versions.versionIds = ids.data();
            versions.versionNames = names.data();
            const auto listResult = p.functions.Query(nullptr, &versions.header);
            if (listResult == FFX_API_RETURN_OK && capacity <= ids.size()) {
                ids.resize(size_t(capacity));
                names.resize(size_t(capacity));
                for (size_t i = 0; i < ids.size(); ++i) {
                    log::info("fsr-sr", std::format("inventory id=0x{:X} name={} preferredMajor={}",
                                                    ids[i], names[i] ? names[i] : "unknown", preferredMajor));
                }
            } else {
                ids.clear();
                names.clear();
            }
        } else {
            log::warn("fsr-sr", std::format("provider inventory query result={} count={}", returnName(countResult), count));
        }
    }

    // Build ordered candidate overrides:
    // - preferredMajor == 0 (Auto): try highest overall semver (e.g. 4.1.2 on RX 9000 / SDK 2.4),
    //   then highest 3.x if the top candidate was 4.x, then default loader selection (0).
    // - preferredMajor == 4 or 3: try highest matching that major first; if 4 fails in Auto/explicit,
    //   only fall back to 3.x when preferredMajor == 0.
    std::vector<FfxVersionSelection> candidates;
    const auto pushUnique = [&](const FfxVersionSelection& sel) {
        if (!sel.matched || sel.id == 0) return;
        for (const auto& existing : candidates) {
            if (existing.id == sel.id) return;
        }
        candidates.push_back(sel);
    };
    if (!ids.empty()) {
        if (preferredMajor != 0) {
            pushUnique(selectBestFfxVersion(ids, names, preferredMajor));
        } else {
            const auto bestAny = selectBestFfxVersion(ids, names, 0);
            pushUnique(bestAny);
            if (bestAny.matched && bestAny.semver.valid && bestAny.semver.major != 3) {
                pushUnique(selectBestFfxVersion(ids, names, 3));
            }
        }
    }

    auto tryCreate = [&](uint64_t overrideId, bool includeApiVersion) -> ffxReturnCode_t {
        ffxOverrideVersion overrideDesc{};
        overrideDesc.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
        overrideDesc.versionId = overrideId;

        ffxCreateContextDescUpscaleVersion versionDesc{};
        versionDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
        versionDesc.version = FFX_UPSCALER_VERSION;

        ffxCreateBackendDX12Desc backendDesc{};
        backendDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
        backendDesc.device = device;

        ffxApiHeader* tail = nullptr;
        if (overrideId != 0) {
            overrideDesc.header.pNext = tail;
            tail = &overrideDesc.header;
        }
        if (includeApiVersion) {
            versionDesc.header.pNext = tail;
            tail = &versionDesc.header;
        }
        backendDesc.header.pNext = tail;

        ffxCreateContextDescUpscale upscaleDesc{};
        upscaleDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        upscaleDesc.header.pNext = &backendDesc.header;
        upscaleDesc.flags = hdr ? FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE : 0u;
        upscaleDesc.maxRenderSize = {maxRenderWidth, maxRenderHeight};
        upscaleDesc.maxUpscaleSize = {maxUpscaleWidth, maxUpscaleHeight};
        upscaleDesc.fpMessage = fsrMessage;

        const auto res = p.functions.CreateContext(&p.context, &upscaleDesc.header, nullptr);
        if (res != FFX_API_RETURN_OK && p.context != nullptr) {
            const auto cleanup = p.functions.DestroyContext(&p.context, nullptr);
            if (cleanup != FFX_API_RETURN_OK) {
                log::error("fsr-sr", "cannot release partial context");
            }
            p.context = nullptr;
        }
        return res;
    };

    ffxReturnCode_t result = FFX_API_RETURN_NO_PROVIDER;
    uint64_t chosenOverrideId = 0;
    std::string chosenOverrideName;

    for (const auto& candidate : candidates) {
        result = tryCreate(candidate.id, true);
        if (result != FFX_API_RETURN_OK || p.context == nullptr) {
            log::warn("fsr-sr", std::format("context id=0x{:X} ({}) with apiVersion failed result={}; retrying without apiVersion",
                                            candidate.id, candidate.name, returnName(result)));
            result = tryCreate(candidate.id, false);
        }
        if (result == FFX_API_RETURN_OK && p.context != nullptr) {
            chosenOverrideId = candidate.id;
            chosenOverrideName = candidate.name;
            break;
        }
        log::warn("fsr-sr", std::format("provider candidate id=0x{:X} ({}) rejected result={}",
                                        candidate.id, candidate.name, returnName(result)));
    }

    if ((result != FFX_API_RETURN_OK || p.context == nullptr) && preferredMajor == 0) {
        result = tryCreate(0, true);
        if (result != FFX_API_RETURN_OK || p.context == nullptr) {
            log::warn("fsr-sr", std::format("default context with apiVersion failed result={}; retrying without it",
                                            returnName(result)));
            result = tryCreate(0, false);
        }
    }

    if (result != FFX_API_RETURN_OK || p.context == nullptr) {
        log::error("fsr-sr", std::format("context creation failed result={} maxRender={}x{} maxUpscale={}x{} preferredMajor={}",
                                         returnName(result), maxRenderWidth, maxRenderHeight, maxUpscaleWidth,
                                         maxUpscaleHeight, preferredMajor));
        release();
        return false;
    }

    p.providerVersionId = chosenOverrideId;
    p.providerVersion = chosenOverrideName;
    ffxQueryGetProviderVersion actual{};
    actual.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    if (p.functions.Query(&p.context, &actual.header) == FFX_API_RETURN_OK) {
        p.providerVersionId = actual.versionId;
        if (actual.versionName && actual.versionName[0] != '\0') {
            p.providerVersion = actual.versionName;
        }
    }
    if (p.providerVersion.empty() && !names.empty() && names[0] != nullptr) {
        p.providerVersion = names[0];
    }

    p.maxRenderWidth = maxRenderWidth;
    p.maxRenderHeight = maxRenderHeight;
    p.maxUpscaleWidth = maxUpscaleWidth;
    p.maxUpscaleHeight = maxUpscaleHeight;
    p.available = true;
    log::info("fsr-sr", std::format("context ready provider={} id=0x{:X} fsr4={} maxRender={}x{} maxUpscale={}x{} hdr={}",
                                    p.providerVersion.empty() ? "unknown" : p.providerVersion,
                                    p.providerVersionId, isFsr4() ? 1 : 0,
                                    maxRenderWidth, maxRenderHeight, maxUpscaleWidth, maxUpscaleHeight,
                                    hdr ? 1 : 0));
    return true;
#else
    (void)device; (void)maxRenderWidth; (void)maxRenderHeight; (void)maxUpscaleWidth; (void)maxUpscaleHeight;
    (void)hdr; (void)preferredMajor;
    log::warn("fsr-sr", "built without FidelityFX SDK headers");
    return false;
#endif
}

bool FsrSrBackend::evaluate(ID3D12GraphicsCommandList* list, ID3D12Resource* color, ID3D12Resource* motion,
                            ID3D12Resource* depth, ID3D12Resource* output, uint32_t renderWidth, uint32_t renderHeight,
                            uint32_t upscaleWidth, uint32_t upscaleHeight, bool reset, float frameTimeMs,
                            ID3D12Resource* reactive, bool enableSharpening, float sharpness) {
#ifdef VEYRA_HAS_FSR
    auto& p = *p_;
    if (!p.available || list == nullptr || color == nullptr || motion == nullptr || depth == nullptr ||
        output == nullptr) {
        return false;
    }
    if (renderWidth > p.maxRenderWidth || renderHeight > p.maxRenderHeight ||
        upscaleWidth > p.maxUpscaleWidth || upscaleHeight > p.maxUpscaleHeight) {
        log::error("fsr-sr", std::format("dispatch extent exceeds context limits render={}x{} upscale={}x{}", renderWidth,
                                         renderHeight, upscaleWidth, upscaleHeight));
        ++p.failureCount;
        return false;
    }
    const auto inputOk = [&](ID3D12Resource* resource) {
        const auto desc = resource->GetDesc();
        return desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && desc.SampleDesc.Count == 1 &&
               desc.Width >= renderWidth && desc.Height >= renderHeight &&
               surfaceFormatFromDxgi(desc.Format) != FFX_API_SURFACE_FORMAT_UNKNOWN;
    };
    const auto target = output->GetDesc();
    const bool reactiveOk = reactive == nullptr || inputOk(reactive);
    if (!renderWidth || !renderHeight || !upscaleWidth || !upscaleHeight ||
        !inputOk(color) || !inputOk(motion) || !inputOk(depth) || !reactiveOk || output == color ||
        target.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || target.SampleDesc.Count != 1 ||
        target.Width < upscaleWidth || target.Height < upscaleHeight ||
        !(target.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) ||
        surfaceFormatFromDxgi(target.Format) == FFX_API_SURFACE_FORMAT_UNKNOWN) {
        const auto cd = color->GetDesc();
        const auto md = motion->GetDesc();
        const auto dd = depth->GetDesc();
        const auto rd = reactive ? reactive->GetDesc() : D3D12_RESOURCE_DESC{};
        log::error("fsr-sr", std::format(
            "dispatch rejected render={}x{} upscale={}x{} color={}x{}(fmt={}) motion={}x{}(fmt={}) depth={}x{}(fmt={}) reactive={}x{}(fmt={}) out={}x{}(fmt={})",
            renderWidth, renderHeight, upscaleWidth, upscaleHeight,
            uint64_t(cd.Width), cd.Height, int(cd.Format),
            uint64_t(md.Width), md.Height, int(md.Format),
            uint64_t(dd.Width), dd.Height, int(dd.Format),
            uint64_t(rd.Width), rd.Height, int(rd.Format),
            uint64_t(target.Width), target.Height, int(target.Format)));
        ++p.failureCount;
        return false;
    }

    ffxDispatchDescUpscale dispatch{};
    dispatch.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    dispatch.commandList = list;
    // Declared states are the states the caller actually set: a mismatch here
    // is what produced runtime errors in the frame-generation path.
    dispatch.color = makeResource(color, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    dispatch.motionVectors = makeResource(motion, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    dispatch.depth = makeResource(depth, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    dispatch.exposure = {};
    dispatch.reactive = reactive ? makeResource(reactive, FFX_API_RESOURCE_STATE_COMPUTE_READ) : FfxApiResource{};
    dispatch.transparencyAndComposition = {};
    dispatch.output = makeResource(output, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    dispatch.jitterOffset = {0.0f, 0.0f};
    // Guidance motion is render-extent pixel space, so the factor that converts
    // it to UV inside the provider is 1:1 (same rule as the frame generator).
    dispatch.motionVectorScale = {1.0f, 1.0f};
    dispatch.renderSize = {renderWidth, renderHeight};
    dispatch.upscaleSize = {upscaleWidth, upscaleHeight};
    const float clampedSharpness = std::clamp(sharpness, 0.0f, 1.0f);
    dispatch.enableSharpening = enableSharpening && clampedSharpness > 0.001f;
    dispatch.sharpness = dispatch.enableSharpening ? clampedSharpness : 0.0f;
    dispatch.frameTimeDelta = frameTimeMs > 0.0f ? frameTimeMs : 16.6f;
    dispatch.preExposure = 1.0f;
    dispatch.reset = reset;
    dispatch.cameraNear = 0.1f;
    dispatch.cameraFar = 1000.0f;
    dispatch.cameraFovAngleVertical = 1.0f;
    dispatch.viewSpaceToMetersFactor = 1.0f;
    dispatch.flags = 0;

    const auto result = p.functions.Dispatch(&p.context, &dispatch.header);
    if (result != FFX_API_RETURN_OK) {
        log::warn("fsr-sr", std::format("dispatch result={} render={}x{} upscale={}x{} reset={} reactive={} rcas={:.2f}",
                                        returnName(result), renderWidth, renderHeight, upscaleWidth, upscaleHeight,
                                        reset, reactive != nullptr ? 1 : 0, dispatch.sharpness));
        ++p.failureCount;
        return false;
    }
    ++p.evaluateCount;
    return true;
#else
    (void)list; (void)color; (void)motion; (void)depth; (void)output; (void)renderWidth; (void)renderHeight;
    (void)upscaleWidth; (void)upscaleHeight; (void)reset; (void)frameTimeMs;
    (void)reactive; (void)enableSharpening; (void)sharpness;
    return false;
#endif
}

void FsrSrBackend::release() {
#ifdef VEYRA_HAS_FSR
    auto& p = *p_;
    if (p.context != nullptr) {
        const auto result = p.functions.DestroyContext(&p.context, nullptr);
        log::info("fsr-sr", std::format("context destroyed result={} evaluations={} failures={}", returnName(result),
                                        p.evaluateCount, p.failureCount));
        p.context = nullptr;
    }
    if (p.loader != nullptr) {
        FreeLibrary(p.loader);
        p.loader = nullptr;
    }
    if (p.provider != nullptr) {
        FreeLibrary(p.provider);
        p.provider = nullptr;
    }
    p.available = false;
    p.providerVersionId = 0;
#endif
}

bool FsrSrBackend::created() const { return p_->available; }
uint64_t FsrSrBackend::evaluateCount() const { return p_->evaluateCount; }
uint64_t FsrSrBackend::failureCount() const { return p_->failureCount; }
const char* FsrSrBackend::providerVersion() const { return p_->providerVersion.c_str(); }
uint64_t FsrSrBackend::providerVersionId() const { return p_->providerVersionId; }
bool FsrSrBackend::isFsr4() const {
    const auto sv = parseFfxSemver(p_->providerVersion);
    return sv.valid && sv.major >= 4;
}

} // namespace veyra::gfx
