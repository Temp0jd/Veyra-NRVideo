// SPDX-License-Identifier: GPL-3.0-only
// Pin the AMD NR probe order (Mochi -> dlssnr_core -> lmxxf), the HIP-preflight rule and the
// hardware routing. Runs on Linux: AmdNrProbe.h is pure and AmdNrProvider.h is constexpr.
#include "veyra/pipeline/AmdNrProbe.h"
#include "veyra/pipeline/MochiNrAbi.h"
#include <cstddef>
#include <cstdio>
#include <string>

namespace {
int failures = 0;
int checks = 0;
void check(const char* what, bool ok) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL %s\n", what); }
}
using provider = veyra::pipeline::AmdNrProvider;
using line = veyra::pipeline::AmdNrLine;
}

int main() {
    using namespace veyra::pipeline;
    check("probe has three lines", kAmdNrProbeOrder.size() == 3);
    check("1st is Mochi", kAmdNrProbeOrder[0].line == line::Mochi);
    check("2nd is DlssNrCore", kAmdNrProbeOrder[1].line == line::DlssNrCore);
    check("3rd is Lmxxf", kAmdNrProbeOrder[2].line == line::Lmxxf);
    check("1st module is MochizukiNrRuntime.dll", std::wstring(kAmdNrProbeOrder[0].module) == L"MochizukiNrRuntime.dll");
    check("2nd module is dlssnr_core.dll", std::wstring(kAmdNrProbeOrder[1].module) == L"dlssnr_core.dll");
    check("3rd module is LmxxfNrRuntime.dll", std::wstring(kAmdNrProbeOrder[2].module) == L"LmxxfNrRuntime.dll");
    for (const auto& candidate : kAmdNrProbeOrder) {
        check("module is a bare file name", std::wstring(candidate.module).find(L'/') == std::wstring::npos &&
            std::wstring(candidate.module).find(L'\\') == std::wstring::npos);
        check("module is a dll", std::wstring(candidate.module).ends_with(L".dll"));
        check("shared runtime directory", std::wstring(candidate.directory) == L"amd-nr");
    }
    check("no HIP preflight for Mochi", !amdNrNeedsHipPreflight(line::Mochi));
    check("no HIP preflight for dlssnr_core", !amdNrNeedsHipPreflight(line::DlssNrCore));
    check("HIP preflight stays on lmxxf", amdNrNeedsHipPreflight(line::Lmxxf));
    check("no HIP preflight for RX 7000", !amdNrNeedsHipPreflight(line::Rdna3));
    check("no HIP preflight for none", !amdNrNeedsHipPreflight(line::None));
    check("mochi name", amdNrLineName(line::Mochi) == "mochi");
    check("dlssnr-core name", amdNrLineName(line::DlssNrCore) == "dlssnr-core");
    check("lmxxf name", amdNrLineName(line::Lmxxf) == "lmxxf");
    check("none name", amdNrLineName(line::None) == "none");
    check("rdna3 name", amdNrLineName(line::Rdna3) == "rdna3");
    // Kept opaque so the compiler cannot fold the lookup to a non-null constant.
    volatile int which = 1;
    const line lookup = static_cast<line>(which == 0 ? 3u : 3u);
    check("candidate lookup", amdNrCandidate(lookup) != nullptr && amdNrCandidate(lookup)->line == lookup);
    check("candidate lookup misses", amdNrCandidate(static_cast<line>(which + 0u)) == nullptr);

    // First present wins, in order.
    check("all present picks Mochi", amdNrFirstPresent([](const wchar_t*) { return true; }) == line::Mochi);
    check("no Mochi picks dlssnr_core", amdNrFirstPresent([](const wchar_t* name) {
        return std::wstring(name) != L"MochizukiNrRuntime.dll"; }) == line::DlssNrCore);
    check("only lmxxf picks lmxxf", amdNrFirstPresent([](const wchar_t* name) {
        return std::wstring(name) == L"LmxxfNrRuntime.dll"; }) == line::Lmxxf);
    check("nothing present", amdNrFirstPresent([](const wchar_t*) { return false; }) == line::None);

    // Hardware routing: the probe only runs for the RDNA4 provider; RX 7000 keeps its worker.
    check("RX 9070 XT routes to RDNA4 provider", amdNrProvider(0x1002, L"AMD Radeon RX 9070 XT") == provider::LmxxfRdna4);
    check("RX 9060 XT routes to RDNA4 provider", amdNrProvider(0x1002, L"AMD Radeon RX 9060 XT") == provider::LmxxfRdna4);
    check("RX 7900 XTX routes to RedDuke", amdNrProvider(0x1002, L"AMD Radeon RX 7900 XTX") == provider::RedDukeRdna3);
    check("probe used for RDNA4", amdNrUsesProbe(provider::LmxxfRdna4));
    check("probe skipped for RedDuke", !amdNrUsesProbe(provider::RedDukeRdna3));
    check("probe skipped for NVIDIA", !amdNrUsesProbe(amdNrProvider(0x10DE, L"NVIDIA GeForce RTX 5070")));

    // The C ABI is transcribed by hand from the upstream headers, so its sizes, offsets and flag
    // values are pinned here: an upstream layout change must fail loudly instead of corrupting
    // a frame inside the module.
    check("MOCHI_NR_API_V1_SIZE is 136", MOCHI_NR_API_V1_SIZE == 136u);
    check("sizeof(MochiNrApi) is 136", sizeof(MochiNrApi) == 136u);
    check("sizeof(MochiNrCapabilities) is 36", sizeof(MochiNrCapabilities) == 36u);
    check("sizeof(MochiNrCreateInfo) is 40", sizeof(MochiNrCreateInfo) == 40u);
    check("sizeof(MochiNrFrameInfo) is 152", sizeof(MochiNrFrameInfo) == 152u);
    check("passes offset is 80", offsetof(MochiNrFrameInfo, passes) == 80u);
    check("sizeof(MochiNrJob) is 32", sizeof(MochiNrJob) == 32u);
    check("MOCHI_NR_CREATE_FLAG_ACO is 2", MOCHI_NR_CREATE_FLAG_ACO == 2u);
    check("MOCHI_NR_FRAME_FLAG_ORIGINAL_MODEL is 1<<13", MOCHI_NR_FRAME_FLAG_ORIGINAL_MODEL == (1u << 13));

    std::printf("MochiNrBackendTests: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
