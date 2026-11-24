// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "veyra/pipeline/AmdNrBackend.h"
#include "veyra/pipeline/MochiNrAbi.h"
#include <filesystem>
#include <string>
namespace veyra::pipeline {
// Line A of the AMD NR probe: MochizukiNrRuntime.dll (Vulkan/ACO runtime, exports
// MochiNrGetApi, or LmxxfNrGetApi when built from the lmxxf-compatible source). HIP is the
// command-queue transport only; there is no ROCm runtime to self-test, so this backend never
// runs AmdHipPreflight.
class MochiNrBackend final : public AmdNrBackend {
public:
    MochiNrBackend();
    ~MochiNrBackend() override;
    MochiNrBackend(const MochiNrBackend&) = delete;
    MochiNrBackend& operator=(const MochiNrBackend&) = delete;
    bool open(const std::filesystem::path& directory, ID3D12Device*, ID3D12CommandQueue*, uint32_t vendor, bool performance);
    void close() noexcept;
    bool ready() const override;
    bool admits(uint32_t width, uint32_t height) const override;
    bool recordInputs(ID3D12GraphicsCommandList*, ID3D12Resource*, uint64_t, bool, float) override;
    bool enqueue() override;
    bool recordOutputs(ID3D12GraphicsCommandList*, ID3D12Resource*) override;
    bool retireSubmitted() override;
    const std::string& error() const override;
    // Probed by Hardware.cpp and AmdNrBackendFactory.cpp before any load is attempted.
    static bool runtimeAvailable(const std::filesystem::path& directory);
    static const wchar_t* moduleName();
private:
    bool result(int32_t code, const char* operation);
    void applyControls();
    bool recordReset(ID3D12GraphicsCommandList* list);
    HMODULE module_ = nullptr;
    MochiNrApi api_{};
    MochiNrCapabilities caps_{};
    MochiNrJob job_{};
    void* context_ = nullptr;
    ID3D12CommandQueue* queue_ = nullptr;
    uint64_t session_ = 0;
    bool performance_ = true;
    bool enqueued_ = false;
    bool outputsRecorded_ = false;
    bool diagnosticsLogged_ = false;
    std::string error_;
};
}
