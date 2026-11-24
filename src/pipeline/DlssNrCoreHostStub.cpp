// SPDX-License-Identifier: GPL-3.0-only
// Build without VEYRA_ENABLE_EXPERIMENTAL_DLSSNR: the dlssnr_core line reports "not built" and
// the probe falls through to the legacy runtime. Compiled into veyra_pipeline so the backend
// never depends on veyra_ngx being present.
#include "veyra/ngx/DlssNrCoreHost.h"

#if !defined(VEYRA_EXPERIMENTAL_LOCAL_DLSSNR)

namespace veyra::ngx {
struct DlssNrCoreHost::Impl {
    std::string error{ "dlssnr_core host not built (VEYRA_ENABLE_EXPERIMENTAL_DLSSNR=OFF)" };
};
bool DlssNrCoreHost::compiledIn() { return false; }
const wchar_t* DlssNrCoreHost::moduleName() { return L"dlssnr_core.dll"; }
DlssNrCoreHost::DlssNrCoreHost() : impl_(std::make_unique<Impl>()) {}
DlssNrCoreHost::~DlssNrCoreHost() = default;
bool DlssNrCoreHost::ready() const noexcept { return false; }
const std::string& DlssNrCoreHost::error() const { return impl_->error; }
bool DlssNrCoreHost::open(const std::filesystem::path&, ID3D12Device*, const std::filesystem::path&, const DlssNrCoreDescription&) { return false; }
bool DlssNrCoreHost::create(ID3D12GraphicsCommandList*) { return false; }
bool DlssNrCoreHost::evaluate(ID3D12GraphicsCommandList*, const DlssNrCoreFrame&) { return false; }
void DlssNrCoreHost::close() noexcept {}
}

#endif
