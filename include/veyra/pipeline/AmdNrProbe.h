// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Probe order for the AMD NR runtime, shared by the runner (which opens the first present
// module) and by the UI availability gate (which must not promise a runtime that is not there).
// Pure header: no D3D12, no Windows, so a Linux unit test can pin the order and the HIP rule.
#include "veyra/pipeline/AmdNrProvider.h"
#include <array>
#include <cstdint>
#include <string_view>
namespace veyra::pipeline {
enum class AmdNrLine : uint32_t {
    None = 0,      // nothing installed
    Rdna3 = 1,     // RX 7000: ZLUDA worker, chosen by adapter and not part of the probe
    Mochi = 2,     // Line A: MochizukiNrRuntime.dll (Vulkan + optional ACO)
    DlssNrCore = 3,// Line B: dlssnr_core.dll, NGX Feature 18 over D3D12
    Lmxxf = 4      // Line C: legacy lmxxf 0.44 (HIP/ROCm)
};
struct AmdNrCandidate {
    AmdNrLine line;
    const wchar_t* module;   // bare file name inside the directory below
    const wchar_t* directory;// sibling of the NVIDIA runtime directory
    bool hipPreflight;       // only the ROCm-based lmxxf runtime needs the self-test
};
// One runtime directory (`<runtimeAbsPath>/../amd-nr`) holds all three modules; a package ships
// whichever lines it was cleared for and the first present, loadable module wins.
inline constexpr std::array<AmdNrCandidate, 3> kAmdNrProbeOrder{{
    {AmdNrLine::Mochi, L"MochizukiNrRuntime.dll", L"amd-nr", false},
    {AmdNrLine::DlssNrCore, L"dlssnr_core.dll", L"amd-nr", false},
    {AmdNrLine::Lmxxf, L"LmxxfNrRuntime.dll", L"amd-nr", true},
}};
inline constexpr bool amdNrNeedsHipPreflight(AmdNrLine line) {
    for (const auto& candidate : kAmdNrProbeOrder) if (candidate.line == line) return candidate.hipPreflight;
    return false;
}
// Log tag only; kept narrow so the graph and the probe log the same token on every platform.
inline constexpr std::string_view amdNrLineName(AmdNrLine line) {
    switch (line) {
    case AmdNrLine::Rdna3: return "rdna3";
    case AmdNrLine::Mochi: return "mochi";
    case AmdNrLine::DlssNrCore: return "dlssnr-core";
    case AmdNrLine::Lmxxf: return "lmxxf";
    case AmdNrLine::None: break;
    }
    return "none";
}
// RX 7000 is routed before the probe: that runtime is a separate worker + ZLUDA translation
// layer, not a module in the probe list.
inline constexpr bool amdNrUsesProbe(AmdNrProvider provider) { return provider == AmdNrProvider::LmxxfRdna4; }
// exists(name) is any callable taking const wchar_t*; the runner passes a filesystem check, the
// test passes a canned set.
template <class Exists>
inline constexpr AmdNrLine amdNrFirstPresent(Exists&& exists) {
    for (const auto& candidate : kAmdNrProbeOrder) if (exists(candidate.module)) return candidate.line;
    return AmdNrLine::None;
}
inline constexpr const AmdNrCandidate* amdNrCandidate(AmdNrLine line) {
    for (const auto& candidate : kAmdNrProbeOrder) if (candidate.line == line) return &candidate;
    return nullptr;
}
}
