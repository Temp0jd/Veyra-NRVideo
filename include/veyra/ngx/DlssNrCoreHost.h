// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// SDK-free face of the NGX Feature-18 host for dlssnr_core.dll (Line B of the AMD NR probe).
// veyra_pipeline cannot include <nvsdk_ngx.h>: the parameter block is a hand-built vtable whose
// slot order is NVIDIA's, so only a translation unit compiled against the real SDK header may
// call it. This declaration carries plain types; the implementation lives in
// src/ngx/DlssNrCoreHost.cpp (veyra_ngx closes over the SDK and its include path).
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
struct ID3D12Device;
struct ID3D12CommandQueue;
struct ID3D12GraphicsCommandList;
struct ID3D12Resource;
namespace veyra::ngx {
struct DlssNrCoreDescription {
    uint32_t width = 0;
    uint32_t height = 0;
    int32_t style = 0;          // DLSSNR.Hint.Render.Preset
    int32_t uiCorrection = 0;   // DLSSNR.UICorrection
};
struct DlssNrCoreFrame {
    ID3D12Resource* color = nullptr;   // linear RGBA16F, NON_PIXEL_SHADER_RESOURCE
    ID3D12Resource* depth = nullptr;   // R32_FLOAT, NON_PIXEL_SHADER_RESOURCE
    ID3D12Resource* motion = nullptr;  // R16G16_FLOAT, NON_PIXEL_SHADER_RESOURCE
    ID3D12Resource* output = nullptr;  // linear RGBA16F, UNORDERED_ACCESS on entry
    bool reset = false;
    float intensity = 1.0f;
    float tone = 1.0f;
    float structure = 1.0f;
    float skin = -1.0f;
    int32_t autoMask = 0;
};
class DlssNrCoreHost {
public:
    // False when Veyra was built without VEYRA_ENABLE_EXPERIMENTAL_DLSSNR: the probe then skips
    // this line instead of failing, and the legacy runtime still serves AMD NR.
    static bool compiledIn();
    static const wchar_t* moduleName();
    DlssNrCoreHost();
    ~DlssNrCoreHost();
    DlssNrCoreHost(const DlssNrCoreHost&) = delete;
    DlssNrCoreHost& operator=(const DlssNrCoreHost&) = delete;
    bool open(const std::filesystem::path& modulePath, ID3D12Device*, const std::filesystem::path& appDataPath, const DlssNrCoreDescription&);
    void close() noexcept;
    bool ready() const noexcept;
    // Creates the feature on the first frame's list; the module records nothing into the list at
    // create, so callers must treat the create frame as pass-through.
    bool create(ID3D12GraphicsCommandList*);
    bool evaluate(ID3D12GraphicsCommandList*, const DlssNrCoreFrame&);
    const std::string& error() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
