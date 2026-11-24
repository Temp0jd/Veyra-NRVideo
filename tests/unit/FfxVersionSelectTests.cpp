// SPDX-License-Identifier: GPL-3.0-only
#include "veyra/gfx/FfxVersionSelect.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
int checks = 0;
int failures = 0;

void expect(bool cond, const char* msg) {
    ++checks;
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}
} // namespace

int main() {
    using namespace veyra::gfx;

    // 1. Semver parsing
    {
        const auto v = parseFfxSemver("4.1.2");
        expect(v.valid && v.major == 4 && v.minor == 1 && v.patch == 2 && v.build == 0, "parse 4.1.2");
        const auto v2 = parseFfxSemver("4.0.2.2740");
        expect(v2.valid && v2.major == 4 && v2.minor == 0 && v2.patch == 2 && v2.build == 2740, "parse 4.0.2.2740");
        const auto v3 = parseFfxSemver("FSR 3.1.4");
        expect(v3.valid && v3.major == 3 && v3.minor == 1 && v3.patch == 4, "parse prefixed FSR 3.1.4");
        const auto bad = parseFfxSemver("unknown");
        expect(!bad.valid, "parse unknown invalid");
        expect(v3 < v2 && v2 < v, "ordering 3.1.4 < 4.0.2 < 4.1.2");
    }

    // 2. Auto-select highest overall version (SDK 2.4 exposing 3.1.4, 4.0.2, 4.1.2)
    {
        std::vector<uint64_t> ids = {0x314, 0x402, 0x412, 0x313};
        std::vector<const char*> names = {"3.1.4", "4.0.2", "4.1.2", "3.1.3"};
        const auto bestAny = selectBestFfxVersion(ids, names, 0);
        expect(bestAny.matched && bestAny.id == 0x412 && bestAny.name == "4.1.2", "auto selects 4.1.2");

        const auto best4 = selectBestFfxVersion(ids, names, 4);
        expect(best4.matched && best4.id == 0x412 && best4.semver.major == 4, "major 4 selects 4.1.2");

        const auto best3 = selectBestFfxVersion(ids, names, 3);
        expect(best3.matched && best3.id == 0x314 && best3.name == "3.1.4", "major 3 selects 3.1.4 over 3.1.3");
    }

    // 3. FSR FG SDK 2.4 exposing 4.0.1 and 4.0.2 alongside 3.1.5
    {
        std::vector<uint64_t> ids = {0x315, 0x401, 0x402};
        std::vector<const char*> names = {"3.1.5", "4.0.1", "4.0.2"};
        const auto bestMl = selectBestFfxVersion(ids, names, 4);
        expect(bestMl.matched && bestMl.id == 0x402 && bestMl.name == "4.0.2", "FG ML picks 4.0.2 over 4.0.1");

        const auto bestOf = selectBestFfxVersion(ids, names, 3);
        expect(bestOf.matched && bestOf.id == 0x315 && bestOf.name == "3.1.5", "FG 3.x picks 3.1.5");
    }

    // 4. Missing requested major returns !matched (no silent fallback)
    {
        std::vector<uint64_t> ids = {0x314};
        std::vector<const char*> names = {"3.1.4"};
        const auto missing4 = selectBestFfxVersion(ids, names, 4);
        expect(!missing4.matched && missing4.id == 0, "missing major 4 fails cleanly");
    }

    // 5. Null names do not crash
    {
        std::vector<uint64_t> ids = {0x1, 0x2};
        std::vector<const char*> names = {nullptr, "3.1.4"};
        const auto res = selectBestFfxVersion(ids, names, 3);
        expect(res.matched && res.id == 0x2, "handles null provider name safely");
    }

    std::printf("FfxVersionSelectTests: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
