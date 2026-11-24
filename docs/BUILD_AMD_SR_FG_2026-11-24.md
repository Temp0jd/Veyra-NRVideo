# Windows 构建指南：AMD SR / FG 整合版（2026-11-24）

本分支在 2.2.0 基础上新增：FSR 4/3.1 版本选择、FSR 反应遮罩与 RCAS、AMF FRC 质量档/运动搜索/未来帧、4:2:2 采集色度抗振铃、Mochi AMD NR 双探测、ArtCNN 2× 超分、AMD AMF Video SR / 画质增强、ONNX Runtime + DirectML 超分（SPAN / RealPLKSR），以及对应的持久化、UI 与导出防静默降级。

本机（WSL2 Linux）只能构建 `linux/obs` 滤镜，**不能**产出 `veyra_qml_ui.exe`，因此本仓库的整合结果必须在 Windows x64 机器上编译。

## 1. 必需工具

- Visual Studio 2022（MSVC v143，含 C++ 桌面工作负载）
- Windows 10/11 SDK（提供 `dxc.exe`；CMake 会在 `%ProgramFiles(x86)%\Windows Kits\10\bin\*\x64\dxc.exe` 下探测）
- CMake 3.24+ 与 Ninja
- Git
- Qt 6.8.3 `win64_msvc2022_64`
- OpenSSL 1.1.1+ x64（`find_package(OpenSSL REQUIRED COMPONENTS Crypto)`）
- FFmpeg x64-windows 前缀（vcpkg 安装树即可，需 `include/libavformat/avformat.h`）

## 2. 必须提前放置的第三方 SDK（全部在 `.gitignore` 覆盖的 `third_party_local/` 下，不进源码 Git）

| CMake 变量 | 默认/建议路径 | 关键文件 |
|---|---|---|
| `VEYRA_DLSS_SDK_ROOT` | `third_party_local/nvidia/DLSS_SDK_310.7.0` | `include/nvsdk_ngx.h`、`lib/Windows_x86_64/x64/nvsdk_ngx_s.lib` |
| `VEYRA_NVOF_SDK_ROOT` | `third_party_local/nvidia/Optical_Flow_SDK_5.0.7` | `NvOFInterface/nvOpticalFlowCommon.h` |
| `VEYRA_NVENC_HEADERS_ROOT` | `third_party_local/nvidia/nv-codec-headers-13.0` | `include/ffnvcodec/nvEncodeAPI.h` |
| `VEYRA_FIDELITYFX2_ROOT` | `third_party_local/amd/FidelityFX-SDK-2.3.0/Kits/FidelityFX` | `framegeneration/include/dx12/ffx_api_framegeneration_dx12.h` |
| `VEYRA_AMF_SDK_ROOT` | `third_party_local/amd/AMF-SDK-1.5.3` | `include/components/FRC.h`、`HQScaler.h`、`VQEnhancer.h` |
| `VEYRA_FIDELITYFX_ROOT` | `third_party_local/amd/FidelityFX-SDK/sdk`（1.1.4） | `bin/ffx_sdk/ffx_opticalflow_x64.lib`（AMD 光流，可选但推荐） |
| `VEYRA_LIBDOVI_ROOT` | 自定义 | `include/libdovi/rpu_parser.h`、`lib`、`bin/dolby_vision.dll` |

说明：

- 全量产品构建需要 `VEYRA_ENABLE_EXPERIMENTAL_DLSSNR=ON`，因为 `veyra_engine` 无条件链接 `veyra_ngx` 与 `veyra_nvof`（`CMakeLists.txt:1794`），而 `veyra_ngx` 只在该开关打开时定义。
- 若 FFmpeg 前缀可用，`VEYRA_ENABLE_LIBDOVI` 默认 ON 且缺少 libdovi 会直接 `FATAL_ERROR`；只做精简开发构建时可显式 `-DVEYRA_ENABLE_LIBDOVI=OFF`。
- `VEYRA_ENABLE_DEINTERLACE` 默认 ON 且要求 FFmpeg 前缀含 `libavfilter/avfilter.h`；不需要就去交错时显式 `-DVEYRA_ENABLE_DEINTERLACE=OFF`。
- AMF 组件按头文件独立启用：有 `FRC.h` 才有 AMD FRC，有 `HQScaler.h` 才有 `sr26`，有 `VQEnhancer.h` 才有 `sr27`。

## 3. 配置与编译（PowerShell，一条命令）

```powershell
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

cmake -S . -B out/build/amd-sr -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH="C:\Qt\6.8.3\msvc2022_64;C:\veyra-deps\installed\x64-windows" `
  -DVEYRA_BUILD_QML_UI=ON `
  -DVEYRA_ENABLE_D3D12_DEBUG=OFF `
  -DVEYRA_ENABLE_EXPERIMENTAL_DLSSNR=ON `
  -DVEYRA_DLSS_SDK_ROOT="third_party_local/nvidia/DLSS_SDK_310.7.0" `
  -DVEYRA_NVOF_SDK_ROOT="third_party_local/nvidia/Optical_Flow_SDK_5.0.7" `
  -DVEYRA_NVENC_HEADERS_ROOT="third_party_local/nvidia/nv-codec-headers-13.0" `
  -DVEYRA_FIDELITYFX2_ROOT="third_party_local/amd/FidelityFX-SDK-2.3.0/Kits/FidelityFX" `
  -DVEYRA_AMF_SDK_ROOT="third_party_local/amd/AMF-SDK-1.5.3" `
  -DVEYRA_FIDELITYFX_ROOT="third_party_local/amd/FidelityFX-SDK/sdk" `
  -DVEYRA_FFMPEG_ROOT="C:\veyra-deps\installed\x64-windows" `
  -DVEYRA_LIBDOVI_ROOT="C:\veyra-deps\libdovi" `
  -DVEYRA_DISPLAY_VERSION=2.2.0-amd-sr-test

cmake --build out/build/amd-sr --target veyra_qml_ui --parallel 2
```

一键脚本（会先检查所有依赖，缺什么就列出确切路径后退出）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build-amd-sr.ps1 -QtRoot "C:\Qt\6.8.3\msvc2022_64" -DepsRoot "C:\veyra-deps"
```

补充：

- 也可以用 `scripts/build.ps1 -Root <仓库根> -Preset x64-release`，但该脚本不传 `VEYRA_BUILD_QML_UI=ON`，需要在源码根放好 SDK 后改用上面的显式命令。
- 着色器由 `dxc` 在构建时编译到 `<build>/shaders/*.dxil`；程序通过编译期 `VEYRA_SHADER_DIR` 读取该目录，直接运行 `out/build/amd-sr/veyra_qml_ui.exe` 最省事。
- `src/virtualcam` 会额外用 VS 2022 生成器构建 x86 虚拟摄像机模块，需要本机同时具备 VS 2022 的 Win32 组件。

## 4. 运行前放置的可选运行组件（都不进源码 Git）

```
<程序根>/
  runtime_local/
    amd/fidelityfx/amd_fidelityfx_loader_dx12.dll
    amd/fidelityfx/amd_fidelityfx_upscaler_dx12.dll
    amd/fidelityfx/amd_fidelityfx_framegeneration_dx12.dll
    amd-nr/MochizukiNrRuntime.dll   （或 dlssnr_core.dll；旧 LmxxfNrRuntime.dll 作兜底）
    models/artcnn_c4f32.onnx        （ArtCNN 2× 超分；C4F16 为 artcnn_c4f16.onnx）
    models/span.onnx                （DirectML SPAN 超分）
    models/realplksr.onnx           （DirectML RealPLKSR 超分）
    models/onnxruntime/onnxruntime.dll  （DirectML 版 ONNX Runtime）
```

- 若程序根存在 `runtime/` 目录，AMD NR 目录改为 `runtime/amd-nr`（`localRuntimeDirectory()` 规则）。
- `VEYRA_ONNXRUNTIME_DIR` 可指向 `onnxruntime.dll` 所在绝对目录，优先级高于上述路径。
- AMD AMF 相关功能不需要放任何 DLL：使用系统 `System32/amfrt64.dll`（随 AMD 驱动安装）。

## 5. 首轮冒烟顺序

1. 只开采集（Xbox 输出 1080p60 SDR，采集像素格式 YUY2，色彩空间手动 `SDR · 709`，范围手动 `Limited`），确认画面不再偏黄发暗。
2. 打开 `FSR 超分`，依次验证 `FSR 版本=自动/FSR 4.x`、`FSR 反应遮罩=开`、`FSR RCAS 锐化=开`（20–35%）。
3. 关闭 FSR，改测 `ArtCNN C4F32 · 2× 超分`（仅 SDR、仅 2× 比例）；再测 `DirectML · SPAN/RealPLKSR`。
4. 最后开 `AMD FRC` 补帧（质量档 超高、运动搜索 原生精度、未来帧插值按游戏类型取舍），开启 HUD 锁定。
5. 若放置了 Mochi/`dlssnr_core`，最后再开 AMD NR，Anti-Flicker 建议 `FlowPlus` 或 `Static`，推理复用建议关闭。

任何一步出现“所选超分后端初始化失败……”，请查看 `logs/` 下的任务诊断日志，不要把它当成普通缩放结果。

## 6. 已验证 / 未验证边界

已在本机（Linux + MinGW-w64 + Wine）通过：EffectAvailabilityTests 80、FfxVersionSelectTests 12、MochiNrBackendTests 47、ArtCnnPackTests 297（真实 ONNX 权重）、AmfVideoEnhanceTests 83（stub）、OrtDmlSessionTests 614、VideoAiSettingsTests 468、RepairContractTests 436、PresetLibraryTests、EffectChainTests，以及 9 语言 i18n 提取检查（3212 条、0 缺失）。

**未验证**：MSVC 全量构建、`dxc` 编译全部着色器、RX 9070 XT 上的真实 FSR 4 / AMF FRC / AMF Video SR / ArtCNN / DirectML 推理、Mochi 或 dlssnr_core 在真实 AMD 显卡上的输出质量与延迟、MS2131 采集实机色彩。以上必须在目标机器上逐项确认后才能称为通过。
