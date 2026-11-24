// SPDX-License-Identifier: GPL-3.0-only
#include "veyra/pipeline/AmdNrBackendFactory.h"
#include "veyra/Log.h"
#include "veyra/ngx/DlssNrCoreHost.h"
#include "veyra/pipeline/AmdNrProbe.h"
#include "veyra/pipeline/AmdNrProvider.h"
#include "veyra/pipeline/DlssNrCoreBackend.h"
#include "veyra/pipeline/LmxxfNrBackend.h"
#include "veyra/pipeline/MochiNrBackend.h"
#include "veyra/pipeline/Rdna3NrBackend.h"
#include <format>
#include <vector>

namespace veyra::pipeline {
namespace {
constexpr wchar_t kRuntimeDirectory[] = L"amd-nr";
constexpr wchar_t kRdna3Directory[] = L"amd-nr-rdna3";

std::filesystem::path siblingDirectory(const std::filesystem::path& runtimeAbsPath, const wchar_t* name) {
    return (runtimeAbsPath / L".." / name).lexically_normal();
}

bool filePresent(const std::filesystem::path& directory, const wchar_t* module) {
    std::error_code ec;
    return std::filesystem::is_regular_file(directory / module, ec);
}
}

std::unique_ptr<AmdNrBackend> openAmdNrBackend(const AmdNrOpenRequest& request, std::string& error) {
    error.clear();
    const auto provider = amdNrProvider(request.vendor, request.adapterDescription);
    if (provider == AmdNrProvider::Unsupported) { error = "AMD NR needs an AMD RX 7000 / 9000 adapter"; return nullptr; }
    if (!request.device || !request.queue || !request.width || !request.height) { error = "AMD NR needs a D3D12 device, a queue and a non-zero extent"; return nullptr; }
    if (!request.runtimeAbsPath.is_absolute()) { error = std::format("AMD NR runtime path {} is not absolute", request.runtimeAbsPath.string()); return nullptr; }
    std::vector<std::string> failures;
    // RX 7000: the validated ZLUDA worker stays first. Its absence is not fatal any more: the
    // probe below serves RDNA3 with Mochi or dlssnr_core instead.
    if (provider == AmdNrProvider::RedDukeRdna3) {
        auto backend = std::make_unique<Rdna3NrBackend>();
        if (backend->open(siblingDirectory(request.runtimeAbsPath, kRdna3Directory), request.device, request.queue, request.width, request.height, request.performance))
            return backend;
        failures.push_back(std::format("rdna3: {}", backend->error()));
        log::warn("amd-nr", std::format("RX 7000 worker unavailable ({}); probing the shared AMD NR runtime directory", backend->error()));
    }
    const auto directory = siblingDirectory(request.runtimeAbsPath, kRuntimeDirectory);
    for (const auto& candidate : kAmdNrProbeOrder) {
        if (!filePresent(directory, candidate.module)) continue;
        std::unique_ptr<AmdNrBackend> backend;
        std::string failure;
        switch (candidate.line) {
        case AmdNrLine::Mochi: {
            auto line = std::make_unique<MochiNrBackend>();
            if (line->open(directory, request.device, request.queue, request.vendor, request.performance)) backend = std::move(line);
            else failure = line->error();
            break;
        }
        case AmdNrLine::DlssNrCore: {
            // The module is shipped but this build has no SDK-typed host compiled in: skip the
            // line instead of reporting a load failure the user cannot act on.
            if (!ngx::DlssNrCoreHost::compiledIn()) {
                log::info("amd-nr", "dlssnr_core.dll present but skipped (built with VEYRA_ENABLE_EXPERIMENTAL_DLSSNR=OFF)");
                continue;
            }
            auto line = std::make_unique<DlssNrCoreBackend>();
            if (line->open(directory, request.device, request.queue, request.vendor, request.width, request.height,
                    request.style, request.uiCorrection, request.zeroMotion, request.zeroDepth)) backend = std::move(line);
            else failure = line->error();
            break;
        }
        case AmdNrLine::Lmxxf: {
            auto line = std::make_unique<LmxxfNrBackend>();
            if (line->open(directory, request.device, request.queue, request.vendor, request.performance)) backend = std::move(line);
            else failure = line->error();
            break;
        }
        case AmdNrLine::Rdna3:
        case AmdNrLine::None:
            break;
        }
        if (backend) {
            log::info("amd-nr", std::format("AMD NR line={} module={}", amdNrLineName(candidate.line), std::filesystem::path(candidate.module).string()));
            return backend;
        }
        failures.push_back(std::format("{}: {}", std::filesystem::path(candidate.module).string(), failure.empty() ? "open failed" : failure));
        log::warn("amd-nr", std::format("{} is present but did not open: {}; trying the next AMD NR line",
            std::filesystem::path(candidate.module).string(), failure.empty() ? "open failed" : failure));
    }
    error = std::format("no usable AMD NR runtime in {} (tried {} lines{})", directory.string(), kAmdNrProbeOrder.size(),
        failures.empty() ? std::string{} : std::format(", failures: {}", [&]{ std::string all; for (const auto& f : failures) { if (!all.empty()) all += "; "; all += f; } return all; }()));
    log::error("amd-nr", error);
    return nullptr;
}
}
