#pragma once
#include "veyra/engine/EnhancementSettings.h"

namespace veyra::engine {
enum class FailedBackend { None, Infrastructure, OpticalFlow, NgxCore, Nr, Sr, Fg, VideoHdr, Dlaa };
// NGX 310.9.1 definitions. A failure on 566.07 alone does not establish a
// universal minimum driver. Preserve the real result and suggest recovery.
constexpr const wchar_t* ngxFailureHint(uint64_t result) {
    switch(uint32_t(result)) {
    case 0xBAD00002:return L"平台或驱动初始化失败；请更新 NVIDIA 驱动或选择兼容运行库";
    case 0xBAD00001:return L"显卡、驱动或运行库不支持所选功能";
    default:return L"初始化失败；详细原因见诊断日志";
    }
}
// SR backends that do not need the NVIDIA NGX runtime. FSR is vendor neutral;
// the Phase-1 spatial CNN/vendor methods (ArtCNN 24/25, DirectML 28/29) run on
// their own runtimes on every adapter, and the AMF video SR methods (26/27)
// run on an AMD adapter. An RTX/DLSS SR selection is NGX-only, so it is the
// one that normalization must strip on a non-NVIDIA adapter.
inline bool nonNvidiaSrAllowed(uint32_t quality, bool amdAdapter) {
    return quality == kVideoSrFsr || artcnnVideoSr(quality) || dmlVideoSr(quality) ||
           (amdAdapter && amfVideoSr(quality));
}
// amdNr: the adapter is AMD, where NR runs on the lmxxf runtime (the graph
// selects it for every NR layer; an unsupported RX model fails in its own
// open/admits and is reported like any other NR backend failure).
inline bool disableUnsupportedNvidiaEffects(EnhancementSettings& settings,bool nvidia,bool amdNr=false){
    if(nvidia)return false;
    const auto before=settings;
    if(!amdNr)settings.nr=false;
    // NVOF does not exist on these adapters; NR motion uses AMD FidelityFX flow.
    if(amdNr&&settings.nr&&settings.opticalFlowBackend==OpticalFlowBackend::Nvidia)
        settings.opticalFlowBackend=OpticalFlowBackend::AmdFidelityFx;
    settings.videoHdr.enabled=false;
    // FSR upscaling and the spatial CNN/vendor SR backends are vendor neutral
    // and must survive the NVIDIA-only normalization; DLSS and RTX Video SR are
    // NGX-only. SR survives only when the user turned it on: FSR is the AMD
    // default backend, so assigning the backend test here switched SR on at
    // every graph build (2.1.0 field report: SR could not be turned off on AMD).
    settings.sr=settings.sr&&nonNvidiaSrAllowed(settings.videoSrQuality,amdNr);
    if(settings.frameGenerationBackend==FrameGenerationBackend::Dlss)settings.multiplier=1;
    return settings!=before;
}
// A stage that only an NVIDIA adapter can run is still requested: NR outside
// the AMD runtime, SR other than FSR, and frame generation other than the
// cross-vendor XeSS/FSR providers.
inline bool nvidiaOnlyStagesRequested(bool nr,bool sr,bool fg,const EnhancementSettings& settings,bool amdNr){
    return (nr&&!amdNr)||(sr&&!nonNvidiaSrAllowed(settings.videoSrQuality,amdNr))||
        (fg&&!crossVendorFrameGeneration(settings.frameGenerationBackend));
}
inline const wchar_t* backendFailureName(FailedBackend backend) {
    switch(backend){
    case FailedBackend::OpticalFlow:return L"光流";
    case FailedBackend::NgxCore:return L"NGX 核心";
    case FailedBackend::Nr:return L"NR";
    case FailedBackend::Sr:return L"超分";
    case FailedBackend::Fg:return L"帧生成";
    case FailedBackend::VideoHdr:return L"RTX Video HDR";
    case FailedBackend::Dlaa:return L"DLAA";
    default:return L"GPU / 输入 / 显示资源";
    }
}
// Disable only the failed feature and its consumers. Never substitute zero
// motion for a failed optical-flow backend while claiming normal enhancement.
inline bool disableFailedBackend(EnhancementSettings& settings,FailedBackend backend){
    const auto before=settings;
    const auto disableDlaa=[&] {
        if(settings.antiAlias.usesDlaa())settings.antiAlias.enabled=false;
        for(auto& aa:settings.additionalAntiAlias)
            if(aa.usesDlaa())aa.enabled=false;
    };
    switch(backend){
    case FailedBackend::Dlaa:disableDlaa();break;
    case FailedBackend::VideoHdr:settings.videoHdr.enabled=false;break;
    case FailedBackend::Nr:settings.nr=false;break;
    case FailedBackend::Sr:settings.sr=false;break;
    // FSR 4 ML needs an RX 9000 whose driver provider offers the ML version.
    // Without it FSR 3.1 (any DirectX 12 GPU) keeps 2X frame generation running
    // instead of turning it off; the caller says so. A failing FSR 3.1 then
    // turns frame generation off as before.
    case FailedBackend::Fg:
        if(settings.frameGenerationBackend==FrameGenerationBackend::Fsr4)settings.frameGenerationBackend=FrameGenerationBackend::Fsr;
        else settings.multiplier=1;
        break;
    case FailedBackend::NgxCore:
        disableDlaa();
        settings.videoHdr.enabled=false;
        // Only NGX-backed SR dies with the NGX core. FSR, ArtCNN, AMF and
        // DirectML keep their own runtimes, so a lost NGX core is not a reason
        // to switch the user's surviving SR selection off.
        if(settings.videoSrQuality!=kVideoSrFsr&&!spatialCnnVideoSr(settings.videoSrQuality))settings.sr=false;
        settings.nr=false;
        if(settings.frameGenerationBackend==FrameGenerationBackend::Dlss)settings.multiplier=1;
        break;
    case FailedBackend::OpticalFlow:
        disableDlaa();
        settings.nr=false;settings.multiplier=1;
        // DLSS SR as the base or the second of a double SR.
        if(!settings.videoSrQuality||(settings.dualSrEnabled&&!settings.dualSrQuality))settings.sr=false;
        break;
    default:return false;
    }
    return settings!=before;
}
}
