// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "veyra/pipeline/AmdNrBackend.h"
#include "veyra/ngx/DlssNrCoreHost.h"
#include <d3d12.h>
#include <wrl/client.h>
#include <filesystem>
#include <memory>
#include <string>
namespace veyra::pipeline {
// Line B of the AMD NR probe: dlssnr_core.dll (NGX Feature 18 over D3D12, shared NGX D3D12
// runtime, no ROCm self-test). The NGX parameter block is a vendor vtable, so every SDK-typed
// call lives in veyra_ngx (src/ngx/DlssNrCoreHost.cpp); this backend only speaks the plain
// contract declared in veyra/ngx/DlssNrCoreHost.h.
class DlssNrCoreBackend final : public AmdNrBackend {
public:
    DlssNrCoreBackend();
    ~DlssNrCoreBackend() override;
    DlssNrCoreBackend(const DlssNrCoreBackend&) = delete;
    DlssNrCoreBackend& operator=(const DlssNrCoreBackend&) = delete;
    // zeroMotion/zeroDepth are the graph's shared constant guidance textures; they stay owned by
    // EnhanceGraph and are only read here.
    bool open(const std::filesystem::path& directory, ID3D12Device*, ID3D12CommandQueue*, uint32_t vendor,
              uint32_t width, uint32_t height, int32_t style, int32_t uiCorrection,
              ID3D12Resource* zeroMotion, ID3D12Resource* zeroDepth);
    void close() noexcept;
    bool ready() const override;
    bool admits(uint32_t width, uint32_t height) const override;
    bool recordInputs(ID3D12GraphicsCommandList*, ID3D12Resource*, uint64_t, bool, float) override;
    bool enqueue() override;
    bool recordOutputs(ID3D12GraphicsCommandList*, ID3D12Resource*) override;
    bool retireSubmitted() override;
    bool needsGuidance() const override { return true; }
    const std::string& error() const override;
    static bool runtimeAvailable(const std::filesystem::path& directory);
    static const wchar_t* moduleName();
private:
    void transition(ID3D12GraphicsCommandList*, ID3D12Resource*, D3D12_RESOURCE_STATES after);
    std::unique_ptr<ngx::DlssNrCoreHost> host_;
    Microsoft::WRL::ComPtr<ID3D12Resource> privateOutput_;
    ID3D12Resource* zeroMotion_ = nullptr;
    ID3D12Resource* zeroDepth_ = nullptr;
    ID3D12Resource* frameColor_ = nullptr;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    D3D12_RESOURCE_STATES privateState_ = D3D12_RESOURCE_STATE_COMMON;
    bool createAttempted_ = false;
    bool hasEvaluatedFrame_ = false;
    bool degraded_ = false;
    bool passThrough_ = true;
    bool outputsRecorded_ = false;
    bool degradeLogged_ = false;
    std::string error_;
};
}
