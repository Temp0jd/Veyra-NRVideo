// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace veyra::gfx {

struct FfxSemver {
    uint32_t major = 0;
    uint32_t minor = 0;
    uint32_t patch = 0;
    uint32_t build = 0;
    bool valid = false;

    bool operator<(const FfxSemver& o) const {
        if (valid != o.valid) return !valid && o.valid;
        if (major != o.major) return major < o.major;
        if (minor != o.minor) return minor < o.minor;
        if (patch != o.patch) return patch < o.patch;
        return build < o.build;
    }
    bool operator>(const FfxSemver& o) const { return o < *this; }
    bool operator==(const FfxSemver& o) const = default;
};

// Parse dotted numeric version segments from FidelityFX provider version strings
// such as "4.1.2", "4.0.2", "3.1.4", or "FSR 4.1.2.1234".
inline FfxSemver parseFfxSemver(std::string_view text) {
    FfxSemver out;
    size_t pos = 0;
    while (pos < text.size() && (text[pos] < '0' || text[pos] > '9')) ++pos;
    if (pos >= text.size()) return out;

    uint32_t parts[4] = {0, 0, 0, 0};
    int count = 0;
    while (pos < text.size() && count < 4) {
        if (text[pos] < '0' || text[pos] > '9') break;
        uint64_t val = 0;
        while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
            val = val * 10 + static_cast<uint64_t>(text[pos] - '0');
            if (val > 0xFFFFFFFFull) val = 0xFFFFFFFFull;
            ++pos;
        }
        parts[count++] = static_cast<uint32_t>(val);
        if (pos < text.size() && text[pos] == '.') {
            ++pos;
        } else {
            break;
        }
    }
    if (count == 0) return out;
    out.major = parts[0];
    out.minor = parts[1];
    out.patch = parts[2];
    out.build = parts[3];
    out.valid = true;
    return out;
}

struct FfxVersionSelection {
    uint64_t id = 0;
    std::string name;
    FfxSemver semver;
    bool matched = false;
};

// Selects the highest semantic version matching `requiredMajor` (0 = any major).
// Among multiple versions in the same family (e.g. SDK 2.4 exposing 4.1.2 and
// 3.1.4, or 4.0.2 alongside 4.0.1), always prefers the newest release.
inline FfxVersionSelection selectBestFfxVersion(const std::vector<uint64_t>& ids,
                                                const std::vector<const char*>& names,
                                                uint32_t requiredMajor) {
    FfxVersionSelection best;
    const size_t n = ids.size() < names.size() ? ids.size() : names.size();
    for (size_t i = 0; i < n; ++i) {
        const char* raw = names[i] ? names[i] : "";
        const FfxSemver sv = parseFfxSemver(raw);
        if (requiredMajor != 0) {
            if (sv.valid) {
                if (sv.major != requiredMajor) continue;
            } else {
                const std::string prefix = std::to_string(requiredMajor) + ".";
                if (std::string_view(raw).find(prefix) == std::string_view::npos) continue;
            }
        }
        if (!best.matched || best.semver < sv) {
            best.id = ids[i];
            best.name = raw;
            best.semver = sv;
            best.matched = true;
        }
    }
    return best;
}

} // namespace veyra::gfx
