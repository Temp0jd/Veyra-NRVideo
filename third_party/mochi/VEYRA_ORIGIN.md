# Mochi / dlssnr_core AMD NR origin

- Upstream (Mochi line, `MochizukiNrRuntime.dll`): https://github.com/mochizuki0323/DLSSNR-AMD
- Upstream (dlssnr_core line, `dlssnr_core.dll`): https://github.com/TheAutomatic/DLSSNR-AMD
- License: MIT for both upstream ports (NVIDIA DLSS NR research ported to AMD).
- Veyra source Git contains only the public C ABI header, this attribution and the
  host-side integration. No DLL, `.f16`/`.f32` weight, HIP module, model package or
  driver capture is committed.
- `include/veyra/pipeline/MochiNrAbi.h` is the Veyra-owned transcription of the
  upstream public C ABI: `MochiNrApi.h` for the create/prepare/submit/collect
  export table, `MochizukiNrControls.h` for the control-block layout, and the
  NGX Feature 18 D3D12 export names (`NVSDK_NGX_D3D12_Init_Ext`,
  `_AllocateParameters`, `_CreateFeature`, `_EvaluateFeature`, `_ReleaseFeature`,
  `_Shutdown1`) for the dlssnr_core variant. Sizes, field offsets and the flag
  values are asserted in `tests/unit/MochiNrBackendTests.cpp` so a drifted upstream
  header is a build/test failure instead of a silent ABI mismatch.
- Veyra-owned host files: `MochiNrBackend.h/.cpp`, `DlssNrCoreBackend.h/.cpp`,
  `src/ngx/DlssNrCoreHost.cpp` (SDK-typed NGX parameter block), `AmdNrProbe.h`,
  `AmdNrBackendFactory.h/.cpp` and the `EnhanceGraph`/`NrInstance` integration.
  They reuse the existing shared enhancement graph, NR composite and history, and
  add no pixel readback or per-pass CPU fence wait on the render thread.
- ABI limits: the Mochi line speaks the flat C API only (no C++ ABI, no ROCm
  self-test on RX 9000); the dlssnr_core line needs Veyra built with
  `VEYRA_ENABLE_EXPERIMENTAL_DLSSNR=ON`, otherwise the probe skips it and the
  legacy AMD NR runtime serves the chain.
- Validation status: host unit tests and Windows syntax gates pass; actual RX 9000
  / RX 9070 XT inference with either module is **not** validated on real hardware
  in this round. Do not report the probe as a verified AMD NR engine.

See `docs/AMD_NR_INTEGRATION_2026-10-03.md` for the legacy asset provenance and
ABI discussion; this probe reuses its NR composite/history contract.
