// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "veyra/pipeline/AmdNrBackend.h"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
struct ID3D12Device;
struct ID3D12CommandQueue;
struct ID3D12Resource;
namespace veyra::pipeline {
// Everything the AMD NR lines need to open, in one place: the graph never names a backend.
struct AmdNrOpenRequest {
    std::filesystem::path runtimeAbsPath; // desc.runtimeAbsPath: <root>\runtime_local\nvidia
    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    uint32_t vendor = 0;
    std::wstring_view adapterDescription;
    uint32_t width = 0;
    uint32_t height = 0;
    int32_t style = 0;
    int32_t uiCorrection = 0;
    bool performance = true;
    ID3D12Resource* zeroMotion = nullptr; // graph-owned constant guidance, borrowed
    ID3D12Resource* zeroDepth = nullptr;
};
// RX 7000 keeps its ZLUDA worker first; when that package is absent the same probe runs, so a
// Mochi/dlssnr-core package also serves RDNA3. RX 9000 probes Mochi -> dlssnr_core -> lmxxf.
std::unique_ptr<AmdNrBackend> openAmdNrBackend(const AmdNrOpenRequest&, std::string& error);
}
