// SPDX-License-Identifier: GPL-3.0-only
#include "veyra/pipeline/DlssNrCoreBackend.h"
#include "veyra/Log.h"
#include <d3d12.h>
#include <format>

namespace veyra::pipeline {

DlssNrCoreBackend::DlssNrCoreBackend() = default;
DlssNrCoreBackend::~DlssNrCoreBackend() { close(); }

const wchar_t* DlssNrCoreBackend::moduleName() { return L"dlssnr_core.dll"; }

bool DlssNrCoreBackend::runtimeAvailable(const std::filesystem::path& directory) {
    std::error_code ec;
    return std::filesystem::is_regular_file(directory / moduleName(), ec) && ngx::DlssNrCoreHost::compiledIn();
}

bool DlssNrCoreBackend::open(const std::filesystem::path& directory, ID3D12Device* device, ID3D12CommandQueue* queue,
    uint32_t vendor, uint32_t width, uint32_t height, int32_t style, int32_t uiCorrection,
    ID3D12Resource* zeroMotion, ID3D12Resource* zeroDepth) {
    close(); error_.clear();
    if (vendor != 0x1002 || !device || !queue || !directory.is_absolute() || !width || !height || !zeroMotion || !zeroDepth) {
        error_ = "dlssnr_core requires an AMD adapter, a D3D12 queue, an absolute runtime directory, an extent and zeroed guidance";
        log::error("amd-nr", error_); return false;
    }
    host_ = std::make_unique<ngx::DlssNrCoreHost>();
    ngx::DlssNrCoreDescription description{};
    description.width = width;
    description.height = height;
    description.style = style;
    description.uiCorrection = uiCorrection;
    if (!host_->open(directory / moduleName(), device, directory, description)) { error_ = host_->error(); host_.reset(); return false; }
    // The private NR target is the surface EvaluateFeature writes; the graph only ever sees the
    // copy into its own layer output, so no extra readback and no CPU wait are introduced.
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    Microsoft::WRL::ComPtr<ID3D12Resource> output;
    const auto hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&output));
    if (FAILED(hr) || !output) { error_ = std::format("dlssnr_core private output allocation hr=0x{:X}", unsigned(hr)); log::error("amd-nr", error_); close(); return false; }
    privateOutput_ = std::move(output);
    privateState_ = D3D12_RESOURCE_STATE_COMMON;
    zeroMotion_ = zeroMotion;
    zeroDepth_ = zeroDepth;
    width_ = width;
    height_ = height;
    log::info("amd-nr", std::format("dlssnr_core NGX Feature 18 backend opened {}x{}; the first frame after CreateFeature stays pass-through", width, height));
    return true;
}

bool DlssNrCoreBackend::ready() const { return host_ && privateOutput_ && !degraded_; }

bool DlssNrCoreBackend::admits(uint32_t width, uint32_t height) const {
    return ready() && width && height && width <= 3840 && height <= 2160;
}

void DlssNrCoreBackend::transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES after) {
    if (privateState_ == after) return;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = privateState_;
    barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
    privateState_ = after;
}

bool DlssNrCoreBackend::recordInputs(ID3D12GraphicsCommandList* list, ID3D12Resource* color, uint64_t frame, bool reset, float intensity) {
    (void)frame;
    if (!host_ || !list || !color) { error_ = "dlssnr_core recordInputs requires an open host, a command list and a colour input"; log::error("amd-nr", error_); return false; }
    const auto desc = color->GetDesc();
    if (desc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT || desc.Width != width_ || desc.Height != height_) {
        error_ = std::format("dlssnr_core needs {}x{} linear RGBA16F, got {}x{} format {}", width_, height_,
            desc.Width, desc.Height, unsigned(desc.Format));
        log::error("amd-nr", error_); return false;
    }
    frameColor_ = color;
    (void)intensity; // per-frame strength stays a full-network control; the AMD path composites residuals in the graph
    passThrough_ = true;
    outputsRecorded_ = false;
    if (degraded_) return true;
    if (!createAttempted_) {
        createAttempted_ = true;
        if (!host_->create(list)) {
            degraded_ = true;
            error_ = host_->error();
            log::warn("amd-nr", std::format("dlssnr_core stays pass-through: {}", error_));
            return true;
        }
        return true; // the module records nothing into the create list; this frame passes through
    }
    if (!host_->ready()) { passThrough_ = true; return true; }
    transition(list, privateOutput_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ngx::DlssNrCoreFrame frameInfo{};
    frameInfo.color = color;
    frameInfo.depth = zeroDepth_;
    frameInfo.motion = zeroMotion_;
    frameInfo.output = privateOutput_.Get();
    // outputsRecorded_ is cleared by retireSubmitted() on every frame, so it can never tell the
    // module whether a previous frame was really evaluated: only hasEvaluatedFrame_ can.
    frameInfo.reset = reset || !hasEvaluatedFrame_;
    frameInfo.intensity = intensity;
    if (!host_->evaluate(list, frameInfo)) {
        error_ = host_->error();
        if (!degradeLogged_) { degradeLogged_ = true; log::warn("amd-nr", std::format("dlssnr_core evaluation degraded to pass-through: {}", error_)); }
        return true;
    }
    hasEvaluatedFrame_ = true;
    passThrough_ = false;
    return true;
}

bool DlssNrCoreBackend::enqueue() {
    // The NGX Feature 18 session owns its submission epoch on its own queue; the recorded work
    // stays ordered behind this frame's list on Veyra's queue.
    return true;
}

bool DlssNrCoreBackend::recordOutputs(ID3D12GraphicsCommandList* list, ID3D12Resource* destination) {
    if (!list || !destination) { error_ = "dlssnr_core recordOutputs requires a list and a destination"; log::error("amd-nr", error_); return false; }
    if (passThrough_ || !privateOutput_) {
        // Degraded or create frame: publish the untouched colour so the composer sees a valid
        // frame instead of an unwritten surface. The graph owns the colour state, so it is
        // restored exactly.
        if (frameColor_) {
            D3D12_RESOURCE_BARRIER b[2]{};
            b[0].Type = b[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b[0].Transition.pResource = b[1].Transition.pResource = frameColor_;
            b[0].Transition.Subresource = b[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            b[0].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            b[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            b[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            b[1].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            list->ResourceBarrier(1, &b[0]);
            list->CopyResource(destination, frameColor_);
            list->ResourceBarrier(1, &b[1]);
        }
        outputsRecorded_ = true;
        return true;
    }
    transition(list, privateOutput_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyResource(destination, privateOutput_.Get());
    outputsRecorded_ = true;
    return true;
}

bool DlssNrCoreBackend::retireSubmitted() {
    outputsRecorded_ = false;
    frameColor_ = nullptr;
    return true;
}

const std::string& DlssNrCoreBackend::error() const { return error_; }

void DlssNrCoreBackend::close() noexcept {
    host_.reset();
    privateOutput_.Reset();
    zeroMotion_ = zeroDepth_ = frameColor_ = nullptr;
    width_ = height_ = 0;
    privateState_ = D3D12_RESOURCE_STATE_COMMON;
    createAttempted_ = degraded_ = degradeLogged_ = false;
    hasEvaluatedFrame_ = false;
    passThrough_ = true;
    outputsRecorded_ = false;
}
}
