// SPDX-License-Identifier: GPL-3.0-only
// Tests for the AMF HQScaler (Video SR 1.0/1.1) and AMF VQEnhancer (FCR) backends.
// Only the parts that do not need an AMD GPU and the AMF runtime are covered: the
// format matrix, the property clamps and the shutdown/capture/process contract.
// Both builds are exercised: without the SDK headers (stub) and with the mock AMF
// 1.5.3 headers (VEYRA_HAS_AMF_HQSCALER / VEYRA_HAS_AMF_VQENHANCER), where the
// component can never initialize unless a real AMD runtime is installed.
#include "veyra/gfx/AmfHqScalerBackend.h"
#include "veyra/gfx/AmfVqEnhancerBackend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

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

bool approx(float a, float b) { return std::fabs(a - b) < 1e-6f; }

bool contains(const char* text, const char* needle) {
    return text && needle && std::strstr(text, needle) != nullptr;
}
} // namespace

int main() {
    using namespace veyra::gfx;

    // 1. Defaults match the documented component defaults
    {
        const AmfHqScalerBackend::Config hq;
        expect(hq.algorithm == AmfHqScalerBackend::Algorithm::VideoSr1_0, "HQScaler default algorithm VideoSR 1.0");
        expect(approx(hq.sharpness, 0.5f), "HQScaler default sharpness 0.5");
        expect(!hq.keepAspectRatio, "HQScaler default keepAspectRatio false");
        expect(!hq.fromSrgb, "HQScaler default fromSrgb false");

        const AmfVqEnhancerBackend::Config vq;
        expect(approx(vq.fcrAttenuation, 0.1f), "VQEnhancer default attenuation 0.1");
        expect(vq.fcrRadius == 2, "VQEnhancer default radius 2");
        expect(!vq.splitView, "VQEnhancer default splitView false");
    }

    // 2. Algorithm ids follow AMF_HQ_SCALER_ALGORITHM_ENUM (FSR is the deprecated name of 2)
    {
        expect(int(AmfHqScalerBackend::Algorithm::Bilinear) == 0, "algorithm Bilinear id 0");
        expect(int(AmfHqScalerBackend::Algorithm::Bicubic) == 1, "algorithm Bicubic id 1");
        expect(int(AmfHqScalerBackend::Algorithm::VideoSr1_0) == 2, "algorithm VideoSR 1.0 id 2");
        expect(int(AmfHqScalerBackend::Algorithm::Point) == 3, "algorithm Point id 3");
        expect(int(AmfHqScalerBackend::Algorithm::VideoSr1_1) == 4, "algorithm VideoSR 1.1 id 4");
    }

    // 3. HQScaler format matrix: VideoSR 1.1 is DX11/DX12 only and rejects the
    // semi-planar formats; every other algorithm accepts what AMF maps.
    {
        expect(AmfHqScalerBackend::supportsFormat(DXGI_FORMAT_R8G8B8A8_UNORM, AmfHqScalerBackend::Algorithm::VideoSr1_0), "HQScaler RGBA8 accepted for VideoSR 1.0");
        expect(AmfHqScalerBackend::supportsFormat(DXGI_FORMAT_B8G8R8A8_UNORM, AmfHqScalerBackend::Algorithm::VideoSr1_1), "HQScaler BGRA8 accepted for VideoSR 1.1");
        expect(AmfHqScalerBackend::supportsFormat(DXGI_FORMAT_R16G16B16A16_FLOAT, AmfHqScalerBackend::Algorithm::VideoSr1_1), "HQScaler RGBA16F accepted for VideoSR 1.1");
        expect(AmfHqScalerBackend::supportsFormat(DXGI_FORMAT_R10G10B10A2_UNORM, AmfHqScalerBackend::Algorithm::VideoSr1_0), "HQScaler R10G10B10A2 accepted for VideoSR 1.0");
        expect(AmfHqScalerBackend::supportsFormat(DXGI_FORMAT_R10G10B10A2_UNORM, AmfHqScalerBackend::Algorithm::Bicubic), "HQScaler R10G10B10A2 accepted for Bicubic");

        expect(!AmfHqScalerBackend::supportsFormat(DXGI_FORMAT_NV12, AmfHqScalerBackend::Algorithm::VideoSr1_1), "HQScaler NV12 rejected for VideoSR 1.1");
        expect(!AmfHqScalerBackend::supportsFormat(DXGI_FORMAT_P010, AmfHqScalerBackend::Algorithm::VideoSr1_1), "HQScaler P010 rejected for VideoSR 1.1");
        expect(AmfHqScalerBackend::supportsFormat(DXGI_FORMAT_NV12, AmfHqScalerBackend::Algorithm::VideoSr1_0), "HQScaler NV12 accepted for VideoSR 1.0");
        expect(AmfHqScalerBackend::supportsFormat(DXGI_FORMAT_P010, AmfHqScalerBackend::Algorithm::Bilinear), "HQScaler P010 accepted for Bilinear");

        expect(!AmfHqScalerBackend::supportsFormat(DXGI_FORMAT_UNKNOWN, AmfHqScalerBackend::Algorithm::VideoSr1_0), "HQScaler UNKNOWN rejected");
        expect(!AmfHqScalerBackend::supportsFormat(DXGI_FORMAT_R8_UNORM, AmfHqScalerBackend::Algorithm::Bicubic), "HQScaler R8_UNORM rejected");
        expect(!AmfHqScalerBackend::supportsFormat(DXGI_FORMAT_R32G32B32A32_FLOAT, AmfHqScalerBackend::Algorithm::VideoSr1_1), "HQScaler RGBA32F rejected");
    }

    // 4. VQEnhancer format matrix: the semi-planar formats are supported here
    {
        expect(AmfVqEnhancerBackend::supportsFormat(DXGI_FORMAT_R8G8B8A8_UNORM), "VQEnhancer RGBA8 accepted");
        expect(AmfVqEnhancerBackend::supportsFormat(DXGI_FORMAT_B8G8R8A8_UNORM), "VQEnhancer BGRA8 accepted");
        expect(AmfVqEnhancerBackend::supportsFormat(DXGI_FORMAT_R16G16B16A16_FLOAT), "VQEnhancer RGBA16F accepted");
        expect(AmfVqEnhancerBackend::supportsFormat(DXGI_FORMAT_R10G10B10A2_UNORM), "VQEnhancer R10G10B10A2 accepted");
        expect(AmfVqEnhancerBackend::supportsFormat(DXGI_FORMAT_NV12), "VQEnhancer NV12 accepted");
        expect(AmfVqEnhancerBackend::supportsFormat(DXGI_FORMAT_P010), "VQEnhancer P010 accepted");
        expect(!AmfVqEnhancerBackend::supportsFormat(DXGI_FORMAT_UNKNOWN), "VQEnhancer UNKNOWN rejected");
        expect(!AmfVqEnhancerBackend::supportsFormat(DXGI_FORMAT_R8_UNORM), "VQEnhancer R8_UNORM rejected");
        expect(!AmfVqEnhancerBackend::supportsFormat(DXGI_FORMAT_R32G32B32A32_FLOAT), "VQEnhancer RGBA32F rejected");
    }

    // 5. Sharpness range 0..2, NaN falls back to the documented default
    {
        expect(approx(AmfHqScalerBackend::clampSharpness(-1.0f), 0.0f), "clampSharpness negative -> 0");
        expect(approx(AmfHqScalerBackend::clampSharpness(std::numeric_limits<float>::quiet_NaN()), 0.5f), "clampSharpness NaN -> 0.5");
        expect(approx(AmfHqScalerBackend::clampSharpness(std::numeric_limits<float>::infinity()), 2.0f), "clampSharpness +inf -> 2");
        expect(approx(AmfHqScalerBackend::clampSharpness(-std::numeric_limits<float>::infinity()), 0.0f), "clampSharpness -inf -> 0");
        expect(approx(AmfHqScalerBackend::clampSharpness(0.0f), 0.0f), "clampSharpness 0 stays 0");
        expect(approx(AmfHqScalerBackend::clampSharpness(0.75f), 0.75f), "clampSharpness 0.75 stays 0.75");
        expect(approx(AmfHqScalerBackend::clampSharpness(2.0f), 2.0f), "clampSharpness 2 stays 2");
        expect(approx(AmfHqScalerBackend::clampSharpness(5.0f), 2.0f), "clampSharpness 5 -> 2");
    }

    // 6. FCR attenuation range 0.02..0.4
    {
        expect(approx(AmfVqEnhancerBackend::clampAttenuation(0.001f), 0.02f), "clampAttenuation 0.001 -> 0.02");
        expect(approx(AmfVqEnhancerBackend::clampAttenuation(0.02f), 0.02f), "clampAttenuation 0.02 stays 0.02");
        expect(approx(AmfVqEnhancerBackend::clampAttenuation(std::numeric_limits<float>::quiet_NaN()), 0.1f), "clampAttenuation NaN -> 0.1");
        expect(approx(AmfVqEnhancerBackend::clampAttenuation(0.1f), 0.1f), "clampAttenuation 0.1 stays 0.1");
        expect(approx(AmfVqEnhancerBackend::clampAttenuation(0.4f), 0.4f), "clampAttenuation 0.4 stays 0.4");
        expect(approx(AmfVqEnhancerBackend::clampAttenuation(5.0f), 0.4f), "clampAttenuation 5 -> 0.4");
        expect(approx(AmfVqEnhancerBackend::clampAttenuation(-2.0f), 0.02f), "clampAttenuation negative -> 0.02");
    }

    // 7. FCR radius range 1..4
    {
        expect(AmfVqEnhancerBackend::clampRadius(0) == 1, "clampRadius 0 -> 1");
        expect(AmfVqEnhancerBackend::clampRadius(1) == 1, "clampRadius 1 stays 1");
        expect(AmfVqEnhancerBackend::clampRadius(2) == 2, "clampRadius 2 stays 2");
        expect(AmfVqEnhancerBackend::clampRadius(4) == 4, "clampRadius 4 stays 4");
        expect(AmfVqEnhancerBackend::clampRadius(99) == 4, "clampRadius 99 -> 4");
        expect(AmfVqEnhancerBackend::clampRadius(-7) == 1, "clampRadius negative -> 1");
    }

    // 8. HQScaler lifecycle without a usable component: init fails with a message,
    // nothing else claims success, shutdown is idempotent.
    {
        AmfHqScalerBackend scaler;
        expect(!scaler.ready(), "HQScaler not ready before initialize");
        expect(scaler.lastError() != nullptr, "HQScaler lastError readable before initialize");

        AmfHqScalerBackend::Config config;
        config.algorithm = AmfHqScalerBackend::Algorithm::VideoSr1_1;
        const bool initialized = scaler.initialize(nullptr, nullptr, 1920, 1080, 3840, 2160, DXGI_FORMAT_R8G8B8A8_UNORM, config);
        expect(!initialized, "HQScaler initialize without device fails");
        expect(!scaler.ready(), "HQScaler not ready after failed initialize");
        expect(scaler.lastError() && scaler.lastError()[0] != '\0', "HQScaler failed initialize sets lastError");
        if (!AmfHqScalerBackend::compiledIn())
            expect(contains(scaler.lastError(), "no AMF SDK"), "stub build reports the missing AMF SDK");

        expect(!scaler.capture(nullptr, nullptr), "HQScaler capture without initialize fails");
        expect(!scaler.process(false), "HQScaler process without initialize fails");
        expect(!scaler.copyOutput(nullptr, nullptr), "HQScaler copyOutput without initialize fails");
        expect(!scaler.submitted(), "HQScaler submitted without a copied frame fails");

        scaler.shutdown();
        scaler.shutdown();
        expect(!scaler.ready(), "HQScaler shutdown twice leaves it not ready");

        // A rejected format must fail before anything is created.
        AmfHqScalerBackend other;
        const bool rejected = other.initialize(nullptr, nullptr, 1920, 1080, 3840, 2160, DXGI_FORMAT_R8_UNORM, config);
        expect(!rejected, "HQScaler initialize rejects an unsupported format");
        expect(other.lastError() && other.lastError()[0] != '\0', "HQScaler rejected format sets lastError");
    }

    // 9. Same lifecycle contract for the VQEnhancer
    {
        AmfVqEnhancerBackend enhancer;
        expect(!enhancer.ready(), "VQEnhancer not ready before initialize");

        AmfVqEnhancerBackend::Config config;
        config.fcrAttenuation = 0.3f;
        config.fcrRadius = 3;
        config.splitView = true;
        const bool initialized = enhancer.initialize(nullptr, nullptr, 1920, 1080, DXGI_FORMAT_NV12, config);
        expect(!initialized, "VQEnhancer initialize without device fails");
        expect(!enhancer.ready(), "VQEnhancer not ready after failed initialize");
        expect(enhancer.lastError() && enhancer.lastError()[0] != '\0', "VQEnhancer failed initialize sets lastError");
        if (!AmfVqEnhancerBackend::compiledIn())
            expect(contains(enhancer.lastError(), "no AMF SDK"), "stub build reports the missing AMF SDK");

        expect(!enhancer.capture(nullptr, nullptr), "VQEnhancer capture without initialize fails");
        expect(!enhancer.process(true), "VQEnhancer process without initialize fails");
        expect(!enhancer.copyOutput(nullptr, nullptr), "VQEnhancer copyOutput without initialize fails");
        expect(!enhancer.submitted(), "VQEnhancer submitted without a copied frame fails");

        enhancer.shutdown();
        enhancer.shutdown();
        expect(!enhancer.ready(), "VQEnhancer shutdown twice leaves it not ready");

        AmfVqEnhancerBackend other;
        const bool rejected = other.initialize(nullptr, nullptr, 1920, 1080, DXGI_FORMAT_R32G32B32A32_FLOAT, config);
        expect(!rejected, "VQEnhancer initialize rejects an unsupported format");
        expect(other.lastError() && other.lastError()[0] != '\0', "VQEnhancer rejected format sets lastError");
    }

    // 10. The runtime check reflects System32 only and never throws
    {
        const bool installedHq = AmfHqScalerBackend::runtimeInstalled();
        const bool installedVq = AmfVqEnhancerBackend::runtimeInstalled();
        expect(installedHq == installedVq, "both backends see the same AMF runtime");
        // Without a runtime the version string stays empty; with one it is dotted.
        AmfHqScalerBackend versionProbe;
        AmfVqEnhancerBackend versionProbe2;
        expect(installedHq || versionProbe.runtimeVersion().empty(), "HQScaler runtimeVersion empty without a runtime");
        expect(installedVq || versionProbe2.runtimeVersion().empty(), "VQEnhancer runtimeVersion empty without a runtime");
        if (!installedHq) {
            AmfHqScalerBackend probe;
            AmfHqScalerBackend::Config config;
            expect(!probe.initialize(nullptr, nullptr, 1280, 720, 2560, 1440, DXGI_FORMAT_B8G8R8A8_UNORM, config),
                   "HQScaler initialize fails without an AMF runtime");
        }
    }

    std::printf("AmfVideoEnhanceTests: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
