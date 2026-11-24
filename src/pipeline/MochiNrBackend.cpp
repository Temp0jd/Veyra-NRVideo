// SPDX-License-Identifier: GPL-3.0-only
#include "veyra/pipeline/MochiNrBackend.h"
#include "veyra/Log.h"
#include <algorithm>
#include <cmath>
#include <format>

namespace veyra::pipeline {
namespace {
// MochizukiNrControls.h: optional exports beside the API table. struct_size is read first, so a
// runtime that answers an older/newer layout stays usable; Veyra patches only fields it has
// proven to fit.
constexpr uint32_t kControlsRenodxOffset = 176;
constexpr uint32_t kControlsRenodxEnd = 180;
constexpr uint32_t kControlsVeyraSize = 520;
}

MochiNrBackend::MochiNrBackend() = default;
MochiNrBackend::~MochiNrBackend() { close(); }

const wchar_t* MochiNrBackend::moduleName() { return L"MochizukiNrRuntime.dll"; }

bool MochiNrBackend::runtimeAvailable(const std::filesystem::path& directory) {
    std::error_code ec;
    return std::filesystem::is_regular_file(directory / moduleName(), ec);
}

bool MochiNrBackend::result(int32_t code, const char* operation) {
    if (code == MOCHI_NR_OK) return true;
    char detail[2048]{};
    if (api_.GetLastError) api_.GetLastError(detail, sizeof(detail));
    error_ = std::format("{} code={} {}", operation, code, detail);
    log::error("amd-nr", error_);
    return false;
}

bool MochiNrBackend::open(const std::filesystem::path& directory, ID3D12Device* device, ID3D12CommandQueue* queue, uint32_t vendor, bool performance) {
    close(); error_.clear();
    if (vendor != 0x1002 || !device || !queue || !directory.is_absolute()) {
        error_ = "Mochi NR requires an AMD adapter, a D3D12 queue and an absolute runtime directory";
        log::error("amd-nr", error_); return false;
    }
    const auto path = directory / moduleName();
    module_ = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!module_) { error_ = std::format("{} load failed win32={}", path.filename().string(), GetLastError()); log::error("amd-nr", error_); return false; }
    // The Mochi source ships the same 16-slot v1 table under either spelling; the lmxxf-compatible
    // build keeps the old export so existing hosts keep working.
    auto getApi = reinterpret_cast<int32_t (*)(uint32_t, MochiNrApi*)>(GetProcAddress(module_, "MochiNrGetApi"));
    const bool nativeExport = getApi != nullptr;
    if (!getApi) getApi = reinterpret_cast<int32_t (*)(uint32_t, MochiNrApi*)>(GetProcAddress(module_, "LmxxfNrGetApi"));
    if (!getApi) { error_ = "MochiNrGetApi/LmxxfNrGetApi export is missing"; log::error("amd-nr", error_); close(); return false; }
    api_.struct_size = sizeof(api_);
    int32_t status = getApi(MOCHI_NR_ABI_VERSION, &api_);
    if (status == MOCHI_NR_INVALID_ARGUMENT) { api_ = {}; api_.struct_size = MOCHI_NR_API_V1_SIZE; status = getApi(MOCHI_NR_ABI_VERSION, &api_); }
    if (status != MOCHI_NR_OK || api_.struct_size < MOCHI_NR_API_V1_SIZE || api_.abi_version != MOCHI_NR_ABI_VERSION ||
        !api_.QueryCapabilities || !api_.Create || !api_.Destroy || !api_.PrepareSession ||
        !api_.PrepareFrame || !api_.RecordInputs || !api_.EnqueueHip || !api_.RecordOutputs ||
        !api_.CancelUnsubmitted || !api_.Retire || !api_.ResetHistory || !api_.Drain || !api_.GetLastError) {
        error_ = std::format("invalid Mochi ABI/function table code={}", status); log::error("amd-nr", error_); close(); return false;
    }
    caps_.struct_size = sizeof(caps_);
    if (!result(api_.QueryCapabilities(&caps_), "QueryCapabilities") || caps_.abi_version != MOCHI_NR_ABI_VERSION || !caps_.max_input_width || !caps_.max_input_height) { close(); return false; }
    const auto assets = directory / L"assets";
    MochiNrCreateInfo info{};
    info.struct_size = sizeof(info);
    info.device = device;
    info.queue = queue;
    info.assets_directory = assets.c_str();
    // ACO is mochizuki-only; the lmxxf-compatible build may reject it. ZERO_OUTPUT_FALLBACK stays
    // CLEAR on both attempts: a failing frame keeps the original colour, never a black surface.
    info.flags = MOCHI_NR_CREATE_FLAG_ACO;
    int32_t createStatus = api_.Create(&info, &context_);
    if (createStatus == MOCHI_NR_INVALID_ARGUMENT) {
        if (context_) { api_.Destroy(context_); context_ = nullptr; }
        log::info("amd-nr", "Mochi Create rejected MOCHI_NR_CREATE_FLAG_ACO; retrying with no create flags");
        info.flags = 0u;
        createStatus = api_.Create(&info, &context_);
    }
    if (!result(createStatus, "Create") || !context_) { close(); return false; }
    performance_ = performance;
    applyControls();
    if (!result(api_.PrepareSession(context_), "PrepareSession")) { close(); return false; }
    queue_ = queue;
    job_.struct_size = sizeof(job_);
    log::info("amd-nr", std::format("Mochi {} ABI={} struct={} maxInput={}x{} history={} gfx1201={} perfModel={}; Vulkan/ACO inference initializes on the first frame",
        nativeExport ? "MochizukiNrRuntime" : "MochizukiNrRuntime(LmxxfNrGetApi)", api_.abi_version, api_.struct_size,
        caps_.max_input_width, caps_.max_input_height, caps_.history_supported, caps_.gfx1201_target, performance ? 0 : 1));
    return true;
}

// The upstream RenoDX stage composes the network into RenoDX's encode, which is not Veyra's
// colour pipeline. Its default is 1; patch it to 0 before PrepareSession. A runtime older than
// the controls header simply keeps its own default and the log says so once.
void MochiNrBackend::applyControls() {
    const auto defaults = reinterpret_cast<int32_t (*)(void*)>(GetProcAddress(module_, "MochizukiNrGetControlDefaults"));
    const auto set = reinterpret_cast<int32_t (*)(void*, const void*)>(GetProcAddress(module_, "MochizukiNrSetControls"));
    if (!defaults || !set) { log::warn("amd-nr", "MochizukiNrControls exports are missing; the runtime keeps its own RenoDX default"); return; }
    alignas(8) unsigned char storage[kControlsVeyraSize]{};
    // MochizukiNrGetControlDefaults only fills up to the struct_size the caller declares; the
    // buffer's first field is the size, so it must be set before the call.
    *reinterpret_cast<uint32_t*>(storage) = kControlsVeyraSize;
    if (defaults(storage) != MOCHI_NR_OK) { log::warn("amd-nr", "MochizukiNrGetControlDefaults failed; the runtime keeps its own RenoDX default"); return; }
    const auto size = *reinterpret_cast<const uint32_t*>(storage);
    if (size < kControlsRenodxEnd) { log::warn("amd-nr", std::format("MochizukiNrControls struct_size={} has no renodx field; the runtime keeps its own default", size)); return; }
    // renodx@176 is a uint32_t toggle in MochizukiNrControls.h, not a float.
    *reinterpret_cast<uint32_t*>(storage + kControlsRenodxOffset) = 0u;
    const auto status = set(context_, storage);
    if (status != MOCHI_NR_OK) log::warn("amd-nr", std::format("MochizukiNrSetControls code={}; RenoDX stage stays at the runtime default", status));
    else log::info("amd-nr", "Mochi RenoDX stage disabled (pure network output for Veyra's own colour pipeline)");
}

bool MochiNrBackend::ready() const { return context_ != nullptr; }

bool MochiNrBackend::admits(uint32_t width, uint32_t height) const {
    // The runtime decides the ceiling; a 4K session is admitted when its caps say so.
    return ready() && width && height && width <= caps_.max_input_width && height <= caps_.max_input_height &&
        uint64_t(width) * height <= uint64_t(caps_.max_input_width) * caps_.max_input_height;
}

bool MochiNrBackend::recordReset(ID3D12GraphicsCommandList* list) {
    (void)list;
    return result(api_.ResetHistory(context_), "ResetHistory");
}

bool MochiNrBackend::recordInputs(ID3D12GraphicsCommandList* list, ID3D12Resource* color, uint64_t frame, bool reset, float intensity) {
    if (!context_ || !list || !color) { error_ = "Mochi recordInputs requires a live session, a command list and a colour input"; log::error("amd-nr", error_); return false; }
    if (job_.handle) { error_ = "Mochi recordInputs called with an unretired job"; log::error("amd-nr", error_); return false; }
    const auto desc = color->GetDesc();
    if (desc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT) { error_ = std::format("Mochi NR needs linear RGBA16F input, got format {}", unsigned(desc.Format)); log::error("amd-nr", error_); return false; }
    // D3D12_RESOURCE_DESC::Height is 32-bit already; only Width can exceed the uint32_t extent the
    // C ABI carries, so the guard is checked on Width alone to keep -Wtype-limits quiet.
    if (desc.Width > 0xFFFFFFFFull || !admits(uint32_t(desc.Width), desc.Height)) {
        error_ = std::format("Mochi NR does not admit {}x{} (runtime max {}x{})", desc.Width, desc.Height, caps_.max_input_width, caps_.max_input_height);
        log::error("amd-nr", error_); return false;
    }
    if (reset && !recordReset(list)) return false;
    MochiNrFrameInfo info{};
    info.struct_size = sizeof(info);
    info.session_id = session_;
    info.frame_id = frame;
    info.list_generation = frame;
    info.command_list = list;
    info.color = color;
    info.color_width = unsigned(desc.Width);
    info.color_height = desc.Height;
    info.color_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    // The performance model is the absence of ORIGINAL_MODEL, matching the lmxxf SF setting.
    info.flags = MOCHI_NR_FRAME_FLAG_STRENGTH | (performance_ ? 0u : MOCHI_NR_FRAME_FLAG_ORIGINAL_MODEL);
    info.transfer_strength = std::clamp(intensity, 0.0f, 1.0f);
    info.color_strength = 1.0f;
    info.model_scale = 1.0f;
    info.motion_scale_x = 1.0f;
    info.motion_scale_y = 1.0f;
    info.depth_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    job_.struct_size = sizeof(job_);
    int32_t status = api_.PrepareFrame(context_, &info, &job_);
    // Ladder for older runtimes: drop the Original-model request first (an unknown flag is an
    // INVALID_ARGUMENT), then shorten the frame struct to the V1 prefix they grew from.
    if (status == MOCHI_NR_INVALID_ARGUMENT && !performance_) {
        log::warn("amd-nr", "Mochi PrepareFrame rejected MOCHI_NR_FRAME_FLAG_ORIGINAL_MODEL; retrying with the performance model");
        info.flags = MOCHI_NR_FRAME_FLAG_STRENGTH;
        status = api_.PrepareFrame(context_, &info, &job_);
    }
    if (status == MOCHI_NR_INVALID_ARGUMENT) {
        info.struct_size = uint32_t(offsetof(MochiNrFrameInfo, passes));
        status = api_.PrepareFrame(context_, &info, &job_);
    }
    if (status != MOCHI_NR_OK) return result(status, "PrepareFrame");
    if (!job_.handle || !job_.private_output) { error_ = "Mochi PrepareFrame returned no job handle or private output"; log::error("amd-nr", error_); return false; }
    const auto out = static_cast<ID3D12Resource*>(job_.private_output)->GetDesc();
    if (out.Format != DXGI_FORMAT_R16G16B16A16_FLOAT || out.Width != desc.Width || out.Height != desc.Height) {
        error_ = std::format("Mochi private output {}x{} format {} does not match the {}x{} RGBA16F input",
            out.Width, out.Height, unsigned(out.Format), desc.Width, desc.Height);
        log::error("amd-nr", error_); return false;
    }
    if (!result(api_.RecordInputs(context_, job_.handle, list), "RecordInputs")) return false;
    enqueued_ = outputsRecorded_ = false;
    return true;
}

bool MochiNrBackend::enqueue() {
    if (!context_ || !job_.handle || enqueued_) { error_ = "Mochi enqueue without a prepared job"; log::error("amd-nr", error_); return false; }
    if (!result(api_.EnqueueHip(context_, job_.handle, queue_), "EnqueueHip")) return false;
    enqueued_ = true;
    if (!diagnosticsLogged_) {
        diagnosticsLogged_ = true;
        char status[512]{};
        if (api_.GetStatus && api_.GetStatus(context_, status, sizeof(status)) == MOCHI_NR_OK)
            log::info("amd-nr", std::format("Mochi session status: {}", status));
    }
    return true;
}

bool MochiNrBackend::recordOutputs(ID3D12GraphicsCommandList* list, ID3D12Resource* destination) {
    if (!context_ || !job_.handle || !job_.private_output || !list || !destination) { error_ = "Mochi recordOutputs requires a prepared job, a list and a destination"; log::error("amd-nr", error_); return false; }
    if (!result(api_.RecordOutputs(context_, job_.handle, list), "RecordOutputs")) return false;
    auto* source = static_cast<ID3D12Resource*>(job_.private_output);
    D3D12_RESOURCE_BARRIER toCopy{};
    toCopy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    toCopy.Transition.pResource = source;
    toCopy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    toCopy.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    toCopy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &toCopy);
    list->CopyResource(destination, source);
    auto back = toCopy;
    back.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    back.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    list->ResourceBarrier(1, &back);
    outputsRecorded_ = true;
    return true;
}

bool MochiNrBackend::retireSubmitted() {
    if (!context_ || !job_.handle) return true;
    if (!outputsRecorded_) return true;
    if (!result(api_.Retire(context_, job_.handle), "Retire")) return false;
    job_ = {};
    job_.struct_size = sizeof(job_);
    enqueued_ = outputsRecorded_ = false;
    return true;
}

const std::string& MochiNrBackend::error() const { return error_; }

void MochiNrBackend::close() noexcept {
    if (context_) {
        if (job_.handle && !enqueued_) api_.CancelUnsubmitted(context_, job_.handle);
        if (job_.handle && api_.Drain) api_.Drain(context_);
        api_.Destroy(context_);
    }
    context_ = nullptr;
    queue_ = nullptr;
    api_ = {};
    job_ = {};
    session_ = 0;
    performance_ = true;
    enqueued_ = outputsRecorded_ = diagnosticsLogged_ = false;
    if (module_) { FreeLibrary(module_); module_ = nullptr; }
}
}
