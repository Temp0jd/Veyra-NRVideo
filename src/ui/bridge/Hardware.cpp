// QML bridge: Hardware responsibilities.
#include "../QmlPlayerBridge_p.h"
#include "veyra/pipeline/Rdna3NrBackend.h"
#include "veyra/pipeline/AmdNrProbe.h"
#include "veyra/pipeline/AmdNrProvider.h"
#include "veyra/ngx/DlssNrCoreHost.h"
#include "veyra/pipeline/VfxSrBackend.h"
#include "veyra/ngx/AnrProvider.h"
#include "veyra/gfx/AmfFrcBackend.h"
#include "veyra/gfx/AmfHqScalerBackend.h"
#include "veyra/gfx/AmfVqEnhancerBackend.h"

namespace veyra::ui {

// The first hardware adapter, the one the engine opens (gfx adapter[0]): 0x10DE NVIDIA,
// 0x1002 AMD, 0x8086 Intel, 0 unknown.
uint32_t primaryGpuVendor() {
    return effectGpu().vendor;
}
// FSR 4 ML frame generation needs an AMD RX 9000; on other GPUs the provider crashed the
// whole program (field report 2026-10-01). Known non-AMD adapters are refused up front.
bool fsr4Possible() { return currentEffectCapabilities().value("fsr4").toMap().value("available").toBool(); }
// The adapter the engine opens: the user's compute GPU when one is chosen
// (gfx::preferredAdapterLuid) and usable, else the first D3D12 hardware
// adapter in DXGI's high-performance order, as D3D12DeviceContext picks it.
// UI thread only; cached per preference.
const DXGI_ADAPTER_DESC1* engineAdapterDesc(){
    static uint64_t cachedFor=~0ull;static DXGI_ADAPTER_DESC1 cached{};static bool found=false;
    const uint64_t preferred=gfx::preferredAdapterLuid();
    if(cachedFor==preferred)return found?&cached:nullptr;
    cachedFor=preferred;found=false;
    Microsoft::WRL::ComPtr<IDXGIFactory6> factory;
    if(FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))return nullptr;
    for(int pass=preferred?0:1;pass<2&&!found;++pass)
        for(UINT i=0;;++i){Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
            if(FAILED(factory->EnumAdapterByGpuPreference(i,DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,IID_PPV_ARGS(&adapter)))||!adapter)break;
            DXGI_ADAPTER_DESC1 d{};
            if(FAILED(adapter->GetDesc1(&d))||(d.Flags&DXGI_ADAPTER_FLAG_SOFTWARE))continue;
            const uint64_t luid=uint64_t(uint32_t(d.AdapterLuid.HighPart))<<32|d.AdapterLuid.LowPart;
            if(pass==0&&luid!=preferred)continue;
            if(FAILED(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,__uuidof(ID3D12Device),nullptr)))continue;
            cached=d;found=true;break;
        }
    return found?&cached:nullptr;
}
// The menu checks the same adapter as the engine. The native backend then
// verifies CUDA compute capability and the exact adapter LUID.
bool vfgHardwarePossible(){
    const auto* d=engineAdapterDesc();
    if(!d)return false;
    const auto name=QString::fromWCharArray(d->Description);
    return d->VendorId==0x10DE&&(name.contains(QStringLiteral("GeForce RTX 30"),Qt::CaseInsensitive)||name.contains(QStringLiteral("GeForce RTX 40"),Qt::CaseInsensitive)||
        name.contains(QStringLiteral("GeForce RTX 50"),Qt::CaseInsensitive)||name.contains(QStringLiteral("Ada"),Qt::CaseInsensitive)||name.contains(QStringLiteral("Blackwell"),Qt::CaseInsensitive));
}
unsigned normalizeFgMultiplier(unsigned value,engine::FrameGenerationBackend backend,unsigned cap){
    value=std::clamp(value,2u,cap);
    return backend==engine::FrameGenerationBackend::Dlss&&value==5?4u:value;
}
bool QmlPlayerBridge::amdNrGpu() const { return primaryGpuVendor()==0x1002; }

ui::EffectGpu effectGpu() {
    ui::EffectGpu result;
    const auto* d = engineAdapterDesc();
    if (!d) return result;
    const auto name = QString::fromWCharArray(d->Description);
    result.vendor = d->VendorId;
    result.deviceId = d->DeviceId;
    result.rtx = name.contains("RTX", Qt::CaseInsensitive);
    result.blackwell = name.contains("RTX 50", Qt::CaseInsensitive) || name.contains("Blackwell", Qt::CaseInsensitive);
    result.ada = name.contains("RTX 40", Qt::CaseInsensitive) || name.contains("Ada", Qt::CaseInsensitive);
    result.ampere = name.contains("GeForce RTX 30", Qt::CaseInsensitive);
    result.rx9000 = name.contains("RX 90", Qt::CaseInsensitive);
    result.rx7000 = pipeline::amdNrProvider(d->VendorId,d->Description)==pipeline::AmdNrProvider::RedDukeRdna3;
    return result;
}

namespace {
constexpr DWORD kProbePathChars = 32768;
// The directory holding the running executable, the last place a model or a
// runtime component may live (next to Veyra.exe, not the working directory).
std::filesystem::path executableDirectory() {
    wchar_t exe[kProbePathChars]{};
    const DWORD length = GetModuleFileNameW(nullptr, exe, kProbePathChars);
    return length && length < kProbePathChars ? std::filesystem::path(exe).parent_path() : std::filesystem::current_path();
}
// Spatial SR weights, in the order FaceTracker locates its model: the packaged
// runtime directory, the local override, then beside the executable.
bool spatialSrModelExists(const wchar_t* fileName) {
    const auto root = runtime::applicationRoot();
    const std::filesystem::path folders[] = {root / "runtime" / "models", root / "runtime_local" / "models", executableDirectory() / "models"};
    std::error_code ec;
    for (const auto& folder : folders)
        if (std::filesystem::is_regular_file(folder / fileName, ec)) return true;
    return false;
}
// ONNX Runtime + DirectML for the DirectML SR backends. VEYRA_ONNXRUNTIME_DIR
// wins only as an absolute path: a relative override would resolve against the
// working directory and break the restricted absolute-only loading rule.
bool ortDmlRuntimeExists() {
    wchar_t overrideDir[kProbePathChars]{};
    const DWORD length = GetEnvironmentVariableW(L"VEYRA_ONNXRUNTIME_DIR", overrideDir, kProbePathChars);
    const std::filesystem::path overridePath(overrideDir);
    if (length && length < kProbePathChars && overridePath.is_absolute()) {
        std::error_code ec;
        if (std::filesystem::is_regular_file(overridePath / L"onnxruntime.dll", ec)) return true;
    }
    const auto root = runtime::applicationRoot();
    const std::filesystem::path candidates[] = {
        root / "runtime" / "models" / "onnxruntime" / "onnxruntime.dll",
        root / "runtime_local" / "models" / "onnxruntime" / "onnxruntime.dll",
        executableDirectory() / "models" / "onnxruntime" / "onnxruntime.dll"};
    std::error_code ec;
    for (const auto& candidate : candidates)
        if (std::filesystem::is_regular_file(candidate, ec)) return true;
    return false;
}
} // namespace

static QVariantMap scanEffectCapabilities() {
    using H = ui::HardwareEffect;
    const auto gpu = effectGpu();
    const auto nr = runtime::localRuntimeDirectory();
    const auto data = runtime::localDataDirectory();
    auto exists = [](const std::filesystem::path& path) { std::error_code ec; return std::filesystem::is_regular_file(path, ec); };
    const auto fsr = data / "amd/fidelityfx";
    const bool fsrLoader = exists(fsr / "amd_fidelityfx_loader_dx12.dll");
    QVariantMap out;
    auto add = [&](const QString& id, H hardware, bool files, const QString& requirement) {
        const bool supported = ui::hardwareSupports(hardware, gpu);
        out[id] = QVariantMap{{"available", supported && files}, {"reason", !supported ? requirement : !files ? QObject::tr("当前包缺少此功能的运行组件") : QString{}}};
    };
    add("nr2", H::NvidiaNrSf, exists(nr / "nr-ampere/nvngx_dlssnr.dll"), QObject::tr("需要 NVIDIA RTX 显卡"));
    out["nr0"]=out["nr2"]; // Retired preset ID; no retired runtime is loaded.
    out["nrAuto"]=QVariantMap{{"available",gpu.vendor==0x10DE&&gpu.deviceId==0x2f04&&out.value("nr0").toMap().value("available").toBool()},
        {"reason",gpu.deviceId!=0x2f04?QObject::tr("自动 NR 当前仅验证 RTX 5070"):!out.value("nr0").toMap().value("available").toBool()?QObject::tr("当前包缺少此功能的运行组件"):QString{}}};
    add("nr3", H::NvidiaNr50, exists(nr / "nr-original/nvngx_dlssnr.dll"), QObject::tr("需要 NVIDIA RTX 50 / Blackwell"));
    const auto rdna3Directory=(nr/"../amd-nr-rdna3").lexically_normal();
    const auto amdNrDirectory=(nr/"../amd-nr").lexically_normal();
    // RX 7000 keeps its ZLUDA worker, but a Mochi/dlssnr_core package still serves RDNA3, and
    // dlssnr_core only counts when this build compiled the NGX host in.
    const auto amdNrLine=pipeline::amdNrFirstPresent([&](const wchar_t* module){
        if (!exists(amdNrDirectory / module)) return false;
        if (std::wstring(module) == ngx::DlssNrCoreHost::moduleName() && !ngx::DlssNrCoreHost::compiledIn()) return false;
        return true;
    });
    const bool rdna3Ready=gpu.rx7000&&pipeline::Rdna3NrBackend::runtimeAvailable(rdna3Directory);
    const bool rdna3Usable=rdna3Ready&&pipeline::Rdna3NrBackend::hipRuntimeAvailable();
    const bool amdNrFiles=rdna3Ready||amdNrLine!=pipeline::AmdNrLine::None;
    add("nr4", H::AmdNr, amdNrFiles, QObject::tr("需要 AMD RX 7000 / 9000"));
    if(gpu.rx7000&&!amdNrFiles){auto cap=out.value("nr4").toMap();cap["reason"]=QObject::tr("当前包缺少 RX 7000 NR 运行组件");out["nr4"]=cap;}
    wchar_t systemDir[MAX_PATH]{};
    const bool systemHipPresent=GetSystemDirectoryW(systemDir, MAX_PATH)&&(exists(std::filesystem::path(systemDir)/"amdhip64_7.dll")||GetModuleHandleW(L"amdhip64_7.dll"));
    const bool probeUsable=amdNrLine!=pipeline::AmdNrLine::None&&(!pipeline::amdNrNeedsHipPreflight(amdNrLine)||systemHipPresent);
    const bool hipMissing=amdNrFiles&&!rdna3Usable&&!probeUsable;
    if (hipMissing) {
        auto cap = out.value("nr4").toMap();
        if (cap.value("available").toBool()) { cap["available"] = false; cap["reason"] = QObject::tr("需要 AMD HIP 7 驱动运行组件"); out["nr4"] = cap; }
    }
    add("sr0", H::Nvidia, exists(nr / "nvngx_dlss.dll"), QObject::tr("需要 NVIDIA RTX 显卡"));
    for (int i = 1; i <= 4; ++i) add("sr" + QString::number(i), H::Nvidia, exists(nr / "nvngx_vsr.dll"), QObject::tr("需要 NVIDIA RTX 显卡"));
    // RTX Video SR high bitrate / streaming: Video Effects VideoSuperRes, from the VFG runtime directory.
    const bool vfxSr = pipeline::VfxSrBackend::runtimeAvailable(nr.wstring());
    for (int q : {16, 17, 18, 19}) add("sr" + QString::number(q), H::Nvidia, vfxSr, QObject::tr("需要 NVIDIA RTX 显卡"));
    for (int q : {21, 23}) add("sr" + QString::number(q), H::NvidiaAmpere, vfxSr, QObject::tr("需要 NVIDIA RTX 30 及更新显卡"));
    add("sr5", H::CrossVendor, fsrLoader && exists(fsr / "amd_fidelityfx_upscaler_dx12.dll"), QObject::tr("需要支持 DirectX 12 的硬件显卡"));
    // Vendor-neutral spatial SR (IDs 24..29): no motion vectors, run on the
    // NVIDIA, AMD and Intel adapters the existing CrossVendor / AmdFrc gates
    // already describe. The model and runtime files decide availability, and a
    // supported adapter with the component missing says which file it wants.
    auto addWithMissingReason = [&](const QString& id, H hardware, bool files,
                                    const QString& requirement, const QString& missingReason) {
        const bool supported = ui::hardwareSupports(hardware, gpu);
        out[id] = QVariantMap{
            {"available", supported && files},
            {"reason", !supported ? requirement : !files ? missingReason : QString{}}};
    };
    addWithMissingReason("sr24", H::CrossVendor, spatialSrModelExists(L"artcnn_c4f16.onnx"),
        QObject::tr("需要支持 DirectX 12 的硬件显卡"), QObject::tr("缺少 ArtCNN C4F16 权重文件"));
    addWithMissingReason("sr25", H::CrossVendor, spatialSrModelExists(L"artcnn_c4f32.onnx"),
        QObject::tr("需要支持 DirectX 12 的硬件显卡"), QObject::tr("缺少 ArtCNN C4F32 权重文件"));
    // The AMF backends use the runtime the AMD driver installs; nothing ships.
    add("sr26", H::AmdFrc, gfx::AmfHqScalerBackend::compiledIn() && gfx::AmfHqScalerBackend::runtimeInstalled(), QObject::tr("需要 AMD 显卡（驱动自带 AMF 运行库）"));
    add("sr27", H::AmdFrc, gfx::AmfVqEnhancerBackend::compiledIn() && gfx::AmfVqEnhancerBackend::runtimeInstalled(), QObject::tr("需要 AMD 显卡（驱动自带 AMF 运行库）"));
    const bool ortRuntime = ortDmlRuntimeExists();
    addWithMissingReason("sr28", H::CrossVendor, ortRuntime && spatialSrModelExists(L"span.onnx"),
        QObject::tr("需要支持 DirectX 12 的硬件显卡"),
        !ortRuntime ? QObject::tr("缺少 ONNX Runtime / DirectML 运行组件") : QObject::tr("缺少 SPAN ONNX 模型文件"));
    addWithMissingReason("sr29", H::CrossVendor, ortRuntime && spatialSrModelExists(L"realplksr.onnx"),
        QObject::tr("需要支持 DirectX 12 的硬件显卡"),
        !ortRuntime ? QObject::tr("缺少 ONNX Runtime / DirectML 运行组件") : QObject::tr("缺少 RealPLKSR ONNX 模型文件"));
    add("anr", H::CrossVendor, ngx::AnrProvider::available(nr.wstring()), QObject::tr("需要支持 DirectX 12 的硬件显卡"));
    add("dlss", H::Nvidia, exists(nr / "nvngx_dlssg.dll"), QObject::tr("需要 NVIDIA RTX 显卡"));
    add("vfg", H::Vfg, pipeline::VfgBackend::runtimeAvailable(nr.wstring()), QObject::tr("需要 NVIDIA RTX 30 / 40 / 50（30 系为实验兼容）"));
    add("xess", H::CrossVendor, exists(data / "intel/experimental/libxess_fg.dll") && exists(data / "intel/experimental/libxell.dll"), QObject::tr("需要支持 DirectX 12 的硬件显卡"));
    add("fsr3", H::CrossVendor, fsrLoader && exists(fsr / "amd_fidelityfx_framegeneration_dx12.dll"), QObject::tr("需要支持 DirectX 12 的硬件显卡"));
    add("fsr4", H::Fsr4, fsrLoader && exists(fsr / "amd_fidelityfx_framegeneration_dx12.dll"), QObject::tr("需要 AMD RX 9000"));
    // AMD FRC uses the AMF runtime the AMD driver installs; nothing ships with Veyra.
    add("amdfrc", H::AmdFrc, gfx::AmfFrcBackend::runtimeInstalled(), QObject::tr("需要 AMD 显卡（驱动自带 AMF 运行库）"));
    add("hdr", H::Nvidia, exists(nr / "nvngx_truehdr.dll"), QObject::tr("需要 NVIDIA RTX 显卡"));
    add("flow0", H::Nvidia, true, QObject::tr("需要 NVIDIA RTX 显卡及 NVOF 驱动"));
    add("flow1", H::CrossVendor, true, QObject::tr("需要支持 DirectX 12 的硬件显卡"));
    return out;
}
// All callers are on the bridge/UI thread. Snapshot reads must not perform
// filesystem probes or rebuild capability maps. Refresh on actions, language
// changes and a slow timer so user-replaced runtime files remain discoverable.
QVariantMap currentEffectCapabilities(bool refresh) {
    static QVariantMap cached;
    static QString language;
    const auto currentLanguage=i18n::current();
    if(refresh||cached.isEmpty()||language!=currentLanguage){
        cached=scanEffectCapabilities();language=currentLanguage;
    }
    return cached;
}
QString fgEffectId(engine::FrameGenerationBackend backend) {
    switch (backend) {
    case engine::FrameGenerationBackend::Dlss: return "dlss";
    case engine::FrameGenerationBackend::XeSS: return "xess";
    case engine::FrameGenerationBackend::Fsr: return "fsr3";
    case engine::FrameGenerationBackend::Fsr4: return "fsr4";
    case engine::FrameGenerationBackend::Vfg: return "vfg";
    case engine::FrameGenerationBackend::Anr: return "anr";
    case engine::FrameGenerationBackend::AmdFrc: return "amdfrc";
    }
    return {};
}
QString unavailableEffects(const engine::EnhancementSettings& settings) {
    const auto caps = currentEffectCapabilities(true);
    QStringList requested;
    if (settings.nr) requested << "nr" + QString::number(int(engine::currentNrRuntime(settings.nrRuntime)));
    if (settings.sr) requested << "sr" + QString::number(settings.videoSrQuality);
    // Double SR runs a second pass only when neither pass is a Phase-1 spatial SR method
    // (spatial SR runs single-stage even if a saved preset carries a dormant dualSrEnabled flag).
    if (settings.sr && settings.dualSrEnabled &&
        !engine::spatialCnnVideoSr(settings.videoSrQuality) &&
        !engine::spatialCnnVideoSr(settings.dualSrQuality))
        requested << "sr" + QString::number(settings.dualSrQuality);
    if (settings.multiplier > 1) requested << fgEffectId(settings.frameGenerationBackend);
    if (settings.videoHdr.enabled) requested << "hdr";
    for (const auto& id : requested) {
        const auto cap = caps.value(id).toMap();
        if (!cap.value("available").toBool()) return id + ": " + cap.value("reason").toString();
    }
    return {};
}
QString disableUnavailableEffects(engine::ChainConfiguration& configuration) {
    const auto caps = currentEffectCapabilities();
    const auto available = [&](const char* id) { return caps.value(id).toMap().value("available").toBool(); };
    const engine::FrameGenerationAvailability fgAvailable{
        available("dlss"),available("xess"),available("fsr3"),available("fsr4"),available("vfg"),available("anr"),available("amdfrc")};
    QStringList reasons;
    const auto sanitize = [&](engine::EffectChain& chain, engine::ChainGlobalSettings& globals) {
        // Also migrate disabled stages: keeping an unavailable DLSS selection
        // left the AMD dropdown at DLSS even though a fresh profile chose FSR.
        const auto savedBackend = globals.fgBackend;
        globals.fgBackend = engine::restoredFrameGenerationBackend(savedBackend,fgAvailable);
        if (globals.fgBackend != savedBackend) {
            if (engine::twoXFrameGeneration(globals.fgBackend)) chain.fgMultiplier=2;
            else if (globals.fgBackend==engine::FrameGenerationBackend::XeSS)
                chain.fgMultiplier=normalizeFgMultiplier(chain.fgMultiplier,globals.fgBackend,2);
            else chain.fgMultiplier=normalizeFgMultiplier(chain.fgMultiplier,globals.fgBackend,6);
            veyra::log::warn("fg-default",std::format("unavailable saved backend={} selected={} rememberedMultiplier={} enabled flags preserved",
                engine::frameGenerationBackendName(savedBackend),engine::frameGenerationBackendName(globals.fgBackend),chain.fgMultiplier));
        }
        for (uint32_t i = 0; i < chain.nodeCount; ++i) {
            auto& node = chain.nodes[i];
            QString id;
            switch (node.type) {
            case engine::EffectType::NrEnhance: id = "nr" + QString::number(int(engine::currentNrRuntime(node.nr.runtime))); break;
            case engine::EffectType::SuperResolution: id = "sr" + QString::number(globals.videoSrQuality); break;
            case engine::EffectType::FrameGeneration: id = fgEffectId(globals.fgBackend); break;
            case engine::EffectType::VideoHdr: id = "hdr"; break;
            default: break;
            }
            if (id.isEmpty() || !node.enabled) continue;
            const auto cap = caps.value(id).toMap();
            if (!cap.value("available").toBool()) {
                node.enabled = false;
                reasons << id + ": " + cap.value("reason").toString();
            }
        }
        if (globals.opticalFlowBackend == engine::OpticalFlowBackend::Nvidia &&
            !caps.value("flow0").toMap().value("available").toBool())
            globals.opticalFlowBackend = engine::OpticalFlowBackend::AmdFidelityFx;
        // A double SR second algorithm this card cannot run (a profile moved from
        // an NVIDIA card) follows the base algorithm when that base supports double SR,
        // or double SR turns off. Phase-1 spatial SR methods never become the second pass.
        const auto runs = [&](uint32_t quality) { return caps.value("sr" + QString::number(quality)).toMap().value("available").toBool(); };
        const auto dualRuns = [&](uint32_t quality) { return !engine::spatialCnnVideoSr(quality) && runs(quality); };
        if (globals.dualSrEnabled && !dualRuns(globals.dualSrQuality)) {
            const auto saved = globals.dualSrQuality;
            if (dualRuns(globals.videoSrQuality)) globals.dualSrQuality = globals.videoSrQuality;
            else globals.dualSrEnabled = false;
            veyra::log::warn("effect-availability", std::format("double SR second quality={} unavailable; second={} enabled={}",
                saved, globals.dualSrQuality, globals.dualSrEnabled));
        }
    };
    sanitize(configuration.chain, configuration);
    if (configuration.editor) {
        auto editor = std::make_shared<engine::NodeEditorDocument>(*configuration.editor);
        auto globals = editor->globals.value_or(static_cast<const engine::ChainGlobalSettings&>(configuration));
        sanitize(editor->nodes, globals);
        editor->globals = globals;
        configuration.editor = std::move(editor);
    }
    reasons.removeDuplicates();
    return reasons.join("; ");
}
QVariantMap QmlPlayerBridge::effectCapabilities() const { return currentEffectCapabilities(); }
QVariantMap QmlPlayerBridge::effectAvailability(const QString& id) const {
    const auto& s = settings();
    const QString key = id == "nr" ? "nr" + QString::number(int(engine::currentNrRuntime(s.nrRuntime))) :
        id == "sr" ? "sr" + QString::number(s.videoSrQuality) : id == "video-hdr" ? QString("hdr") :
        id == "frame-generation" ? fgEffectId(s.frameGenerationBackend) :
        id == "dlss-fg" ? QString("dlss") : id == "xess-fg" ? QString("xess") :
        id == "fsr3-fg" ? QString("fsr3") : id == "fsr4-fg" ? QString("fsr4") : id == "vfg-fg" ? QString("vfg") : id == "anr-fg" ? QString("anr") : id == "amdfrc-fg" ? QString("amdfrc") : id;
    const auto caps = effectCapabilities();
    return caps.contains(key) ? caps.value(key).toMap() : QVariantMap{{"available", true}, {"reason", QString{}}};
}
QVariantList QmlPlayerBridge::nrRuntimeChoices() const {
    QVariantList choices;
    const auto caps = effectCapabilities();
    for (const auto& entry : std::initializer_list<std::pair<int, QString>>{{2, "RTX 20–50 · SF 310.9.16"}, {3, tr("RTX 50 · NVIDIA 原版")}, {4, tr("RX 7000 / 9000 · AMD NR（实验）")}}) {
        const auto cap = caps.value("nr" + QString::number(entry.first)).toMap();
        choices << QVariantMap{{"id", QString::number(entry.first)}, {"label", entry.second}, {"disabled", !cap.value("available").toBool()}, {"note", cap.value("reason")}};
    }
    return choices;
}
QVariantList QmlPlayerBridge::srBackendChoices() const {
    QVariantList choices;
    const auto caps = effectCapabilities();
    // RTX Video SR families take string ids (srAlgorithmOf); the tier is chosen
    // separately. Availability follows each family's runtime.
    struct Entry { QString id; int capability; QString label; };
    for (const auto& entry : std::initializer_list<Entry>{{"0", 0, "DLSS SR"}, {"rtx", 4, tr("RTX 视频超分 · 标准")},
            {"rtx-hb", 16, tr("RTX 视频超分 · 高码率")}, {"rtx-st", 21, tr("RTX 视频超分 · 流媒体")},
            {"5", 5, effectGpu().vendor == 0x1002 ? tr("FSR 3.1 / 4 · 自动兼容") : QString("FSR 3.1")},
            // Spatial single-pass SR (IDs 24..29). "AMD AMF Video SR" is a brand
            // name, kept untranslated like "DLSS SR".
            {"24", 24, tr("ArtCNN C4F16 · 2× 超分")},
            {"25", 25, tr("ArtCNN C4F32 · 2× 超分")},
            {"26", 26, QStringLiteral("AMD AMF Video SR")},
            {"27", 27, tr("AMD AMF 画质增强")},
            {"28", 28, tr("DirectML · SPAN 超分")},
            {"29", 29, tr("DirectML · RealPLKSR 超分")}}) {
        const auto cap = caps.value("sr" + QString::number(entry.capability)).toMap();
        choices << QVariantMap{{"id", entry.id}, {"label", entry.label}, {"disabled", !cap.value("available").toBool()}, {"note", cap.value("reason")}};
    }
    return choices;
}


} // namespace veyra::ui
