// SPDX-License-Identifier: GPL-3.0-only
//
// Veyra NRVideo - DirectML memory-info probe.
//
// The DirectML execution provider of ONNX Runtime names its device memory
// information after the OrtDmlApi version and the D3D12 device that backs it
// (for example "DML" on a plain D3D12 device, "DML2" for the DML2 factory).
// OrtDmlSrBackend has to create a matching OrtMemoryInfo to hand a D3D12
// texture to the provider, so this tool asks a real installation what the
// string, allocator id, memory type and device type actually are and prints a
// PASS/FAIL line for each value it can reconstruct with CreateMemoryInfo +
// CompareMemoryInfo.
//
// It is a standalone diagnostic: it only needs onnxruntime.dll and
// DirectML.dll, both looked up next to the executable or in System32 unless a
// path is given on the command line. Nothing here is linked into the product.
//
// Usage:
//   ort_dml_meminfo_probe.exe [onnxruntime.dll] [memory-info-name]
//
//   argv[1]  optional absolute path to onnxruntime.dll; when omitted the
//            loader searches the application directory and System32 only.
//   argv[2]  optional memory-info name to request from the provider, default
//            "DML" (try "DML2" on installations built against a newer
//            DirectML release).
//
// Exit code 0 means every probe passed, 1 means at least one step failed and
// 2 means the command line was rejected.

#include "veyra/pipeline/OrtDmlApi.h"

#include <windows.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

namespace ort = veyra::pipeline::ort;

namespace {

constexpr const char* kProbeEnvName = "veyra-ort-dml-probe";
constexpr uint32_t kProbeApiVersion = ort::kOrtApiVersion;

std::FILE* g_out = stdout;

// GetProcAddress returns FARPROC, so binding an export always crosses a
// function-pointer type; silence the GCC diagnostic at the one place it occurs.
template <typename Fn>
Fn resolveExport(HMODULE module, const char* name)
{
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
    return reinterpret_cast<Fn>(::GetProcAddress(module, name));
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

#if defined(__GNUC__)
void say(const char* format, ...) __attribute__((format(printf, 1, 2)));
#endif

void say(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    std::vfprintf(g_out, format, args);
    va_end(args);
    std::fputc('\n', g_out);
    std::fflush(g_out);
}

// Owns every runtime object the probe creates and tears them down in the exact
// reverse order the API requires. Members are only released when they were
// created, so a partially completed probe still exits cleanly.
class Probe {
public:
    Probe() = default;
    Probe(const Probe&) = delete;
    Probe& operator=(const Probe&) = delete;

    ~Probe()
    {
        if (binding_ && api_.valid())
            api_.slot<ort::FnReleaseIoBinding>(ort::kSlot_ReleaseIoBinding)(binding_);
        if (values_ && hostAllocator_ && api_.valid())
            api_.slot<ort::FnAllocatorFree>(ort::kSlot_AllocatorFree)(hostAllocator_, values_);
        if (outputValue_ && api_.valid())
            api_.slot<ort::FnReleaseValue>(ort::kSlot_ReleaseValue)(outputValue_);
        if (inputValue_ && api_.valid())
            api_.slot<ort::FnReleaseValue>(ort::kSlot_ReleaseValue)(inputValue_);
        if (cpuInfo_ && api_.valid())
            api_.slot<ort::FnReleaseMemoryInfo>(ort::kSlot_ReleaseMemoryInfo)(cpuInfo_);
        if (dmlInfo_ && api_.valid())
            api_.slot<ort::FnReleaseMemoryInfo>(ort::kSlot_ReleaseMemoryInfo)(dmlInfo_);
        if (session_ && api_.valid())
            api_.slot<ort::FnReleaseSession>(ort::kSlot_ReleaseSession)(session_);
        if (options_ && api_.valid())
            api_.slot<ort::FnReleaseSessionOptions>(ort::kSlot_ReleaseSessionOptions)(options_);
        if (env_ && api_.valid())
            api_.slot<ort::FnReleaseEnv>(ort::kSlot_ReleaseEnv)(env_);
        if (module_)
            ::FreeLibrary(module_);
    }

    bool load(const wchar_t* dllPath)
    {
        if (dllPath && *dllPath) {
            const std::filesystem::path path(dllPath);
            if (!path.is_absolute()) {
                say("FAIL onnxruntime path must be absolute: %s", path.string().c_str());
                return false;
            }
            module_ = ::LoadLibraryExW(path.c_str(), nullptr,
                                       LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        } else {
            module_ = ::LoadLibraryExW(L"onnxruntime.dll", nullptr,
                                       LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        }
        if (!module_) {
            const std::string attempted = (dllPath && *dllPath)
                                               ? std::filesystem::path(dllPath).string()
                                               : std::string("onnxruntime.dll");
            say("FAIL LoadLibrary(%s) win32=%lu", attempted.c_str(),
                static_cast<unsigned long>(::GetLastError()));
            return false;
        }

        const auto getApiBase = resolveExport<ort::FnOrtGetApiBase>(module_, "OrtGetApiBase");
        if (!getApiBase) {
            say("FAIL export OrtGetApiBase missing");
            return false;
        }
        base_ = getApiBase();
        if (!base_) {
            say("FAIL OrtGetApiBase() returned null");
            return false;
        }
        say("OK   runtime version: %s", base_->GetVersionString ? base_->GetVersionString() : "<unknown>");
        return true;
    }

    bool resolveApi(const char* memoryInfoName)
    {
        memoryInfoName_ = memoryInfoName;
        api_ = ort::OrtApiView(base_->GetApi(kProbeApiVersion));
        if (!api_.valid()) {
            say("FAIL GetApi(%u) returned null; this runtime is too old for the %u ABI", kProbeApiVersion,
                kProbeApiVersion);
            return false;
        }
        say("OK   OrtApi v%u resolved", kProbeApiVersion);

        const auto getProviderApi = api_.slot<ort::FnGetExecutionProviderApi>(ort::kSlot_GetExecutionProviderApi);
        if (!getProviderApi) {
            say("FAIL GetExecutionProviderApi slot unavailable");
            return false;
        }
        const void* dml = nullptr;
        if (!check("GetExecutionProviderApi(\"DML\")", getProviderApi("DML", kProbeApiVersion, &dml)))
            return false;
        if (!dml) {
            say("FAIL GetExecutionProviderApi(\"DML\") returned a null OrtDmlApi");
            return false;
        }
        dml_ = static_cast<const ort::OrtDmlApi*>(dml);
        say("OK   OrtDmlApi resolved (6 entry points)");
        return true;
    }

    bool createSession()
    {
        if (!check("CreateEnv", api_.slot<ort::FnCreateEnv>(ort::kSlot_CreateEnv)(
                                   ort::ORT_LOGGING_LEVEL_WARNING, kProbeEnvName, &env_)))
            return false;
        if (!check("CreateSessionOptions",
                   api_.slot<ort::FnCreateSessionOptions>(ort::kSlot_CreateSessionOptions)(&options_)))
            return false;
        // Same configuration contract as OrtDmlSrBackend: memory pattern off
        // (the DML EP ring-allocates its own output) and sequential execution.
        if (!check("DisableMemPattern",
                   api_.slot<ort::FnDisableMemPattern>(ort::kSlot_DisableMemPattern)(options_)))
            return false;
        if (!check("SetSessionExecutionMode",
                   api_.slot<ort::FnSetSessionExecutionMode>(ort::kSlot_SetSessionExecutionMode)(
                       options_, ort::ORT_SEQUENTIAL)))
            return false;

        const auto appendDml = dml_ ? dml_->SessionOptionsAppendExecutionProvider_DML : nullptr;
        if (!appendDml) {
            say("FAIL OrtDmlApi::SessionOptionsAppendExecutionProvider_DML is null");
            return false;
        }
        if (!check("SessionOptionsAppendExecutionProvider_DML", appendDml(options_, 0)))
            return false;
        say("OK   DML execution provider appended for device 0");

        const auto createFromArray =
            api_.slot<ort::FnCreateSessionFromArray>(ort::kSlot_CreateSessionFromArray);
        if (!createFromArray) {
            say("FAIL CreateSessionFromArray slot unavailable");
            return false;
        }
        if (!check("CreateSessionFromArray(identity model)", createFromArray(
                       env_, ort::kMinimalIdentityOnnxModel, ort::kMinimalIdentityOnnxModelSize, options_,
                       &session_)))
            return false;
        say("OK   session created from the embedded %llu-byte identity model",
            static_cast<unsigned long long>(ort::kMinimalIdentityOnnxModelSize));

        if (!check("GetAllocatorWithDefaultOptions",
                   api_.slot<ort::FnGetAllocatorWithDefaultOptions>(ort::kSlot_GetAllocatorWithDefaultOptions)(
                       &hostAllocator_)))
            return false;
        return readNames();
    }

    bool probe()
    {
        const auto createMemoryInfo = api_.slot<ort::FnCreateMemoryInfo>(ort::kSlot_CreateMemoryInfo);
        if (!createMemoryInfo) {
            say("FAIL CreateMemoryInfo slot unavailable");
            return false;
        }
        if (!check("CreateMemoryInfo(Cpu)", createMemoryInfo("Cpu", ort::OrtDeviceAllocator, 0,
                                                            ort::OrtMemTypeDefault, &cpuInfo_)))
            return false;

        bool pass = true;
        // The provider may refuse the name we guessed; report the real failure
        // instead of silently falling back so the caller learns the right one.
        ort::OrtStatus* status = createMemoryInfo(
            memoryInfoName_.c_str(), ort::OrtDeviceAllocator, 0,
            ort::OrtMemTypeDefault, &dmlInfo_);
        if (status) {
            std::string message = ort::consumeStatus(api_, status);
            say("FAIL CreateMemoryInfo(\"%s\"): %s", memoryInfoName_.c_str(), message.c_str());
            say("HINT the DirectML EP in this build may report another name, try \"DML2\" or \"DML\"");
            return false;
        }
        say("OK   CreateMemoryInfo(\"%s\") accepted", memoryInfoName_.c_str());

        // Identity model, float32 [1,1,1,1]: a host binding for the input keeps
        // the probe independent of any D3D12 device, while the output is bound
        // to the provider's own device memory, which is what we want to read.
        const int64_t shape[4] = {1, 1, 1, 1};
        hostInput_ = 1.0f;
        if (!check("CreateTensorWithDataAsOrtValue(input)",
                   api_.slot<ort::FnCreateTensorWithDataAsOrtValue>(ort::kSlot_CreateTensorWithDataAsOrtValue)(
                       cpuInfo_, &hostInput_, sizeof(hostInput_), shape, 4,
                       ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputValue_)))
            return false;

        if (!check("CreateIoBinding", api_.slot<ort::FnCreateIoBinding>(ort::kSlot_CreateIoBinding)(
                                          session_, &binding_)))
            return false;
        if (!check("BindInput", api_.slot<ort::FnBindInput>(ort::kSlot_BindInput)(
                                    binding_, inputName_.c_str(), inputValue_)))
            return false;
        if (!check("BindOutputToDevice",
                   api_.slot<ort::FnBindOutputToDevice>(ort::kSlot_BindOutputToDevice)(
                       binding_, outputName_.c_str(), dmlInfo_)))
            return false;
        say("OK   input \"%s\" bound on host, output \"%s\" bound to \"%s\"", inputName_.c_str(),
            outputName_.c_str(), memoryInfoName_.c_str());

        if (!check("RunWithBinding", api_.slot<ort::FnRunWithBinding>(ort::kSlot_RunWithBinding)(
                                         session_, nullptr, binding_)))
            return false;

        if (!check("GetBoundOutputValues",
                   api_.slot<ort::FnGetBoundOutputValues>(ort::kSlot_GetBoundOutputValues)(
                       binding_, hostAllocator_, &values_, &valueCount_)))
            return false;
        if (valueCount_ < 1 || !values_ || !values_[0]) {
            say("FAIL GetBoundOutputValues produced %llu values",
                static_cast<unsigned long long>(valueCount_));
            return false;
        }
        outputValue_ = values_[0];

        const ort::OrtMemoryInfo* probed = nullptr;
        if (!check("GetTensorMemoryInfo(output)",
                   api_.slot<ort::FnGetTensorMemoryInfo>(ort::kSlot_GetTensorMemoryInfo)(outputValue_,
                                                                                       &probed)))
            return false;
        if (!probed) {
            say("FAIL GetTensorMemoryInfo(output) returned null");
            return false;
        }

        const char* name = nullptr;
        int id = -1;
        ort::OrtMemType memType = ort::OrtMemTypeDefault;
        ort::OrtAllocatorType allocatorType = ort::OrtInvalidAllocator;
        ort::OrtMemoryInfoDeviceType deviceType = ort::OrtMemoryInfoDeviceType_CPU;

        pass = check("MemoryInfoGetName",
                     api_.slot<ort::FnMemoryInfoGetName>(ort::kSlot_MemoryInfoGetName)(probed, &name)) &&
               pass;
        pass = check("MemoryInfoGetId",
                     api_.slot<ort::FnMemoryInfoGetId>(ort::kSlot_MemoryInfoGetId)(probed, &id)) &&
               pass;
        pass = check("MemoryInfoGetMemType",
                     api_.slot<ort::FnMemoryInfoGetMemType>(ort::kSlot_MemoryInfoGetMemType)(probed, &memType)) &&
               pass;
        pass = check("MemoryInfoGetType",
                     api_.slot<ort::FnMemoryInfoGetType>(ort::kSlot_MemoryInfoGetType)(probed, &allocatorType)) &&
               pass;
        api_.slot<ort::FnMemoryInfoGetDeviceType>(ort::kSlot_MemoryInfoGetDeviceType)(probed, &deviceType);
        if (!pass)
            return false;

        say("DML memory info: name=\"%s\" id=%d memType=%d allocatorType=%d deviceType=%d",
            name ? name : "<null>", id, static_cast<int>(memType), static_cast<int>(allocatorType),
            static_cast<int>(deviceType));

        ort::OrtMemoryInfo* reconstructed = nullptr;
        if (!check("CreateMemoryInfo(reconstructed)",
                   createMemoryInfo(name ? name : "", allocatorType, id, memType, &reconstructed)))
            return false;
        int compared = -1;
        const bool comparedOk =
            check("CompareMemoryInfo", api_.slot<ort::FnCompareMemoryInfo>(ort::kSlot_CompareMemoryInfo)(
                                           probed, reconstructed, &compared));
        api_.slot<ort::FnReleaseMemoryInfo>(ort::kSlot_ReleaseMemoryInfo)(reconstructed);
        if (!comparedOk)
            return false;
        // OrtApi::CompareMemoryInfo sets *out = 0 when equal, -1 when different.
        if (compared != 0) {
            say("FAIL CompareMemoryInfo reports the reconstructed info differs from the provider's (out=%d)",
                compared);
            return false;
        }
        say("OK   reconstructed memory info compares equal");

        // The backend also checks the device type, which tells it whether the
        // binding really landed on the GPU instead of a host-visible mirror.
        if (deviceType != ort::OrtMemoryInfoDeviceType_GPU) {
            say("FAIL device type is %d, expected GPU (%d)", static_cast<int>(deviceType),
                static_cast<int>(ort::OrtMemoryInfoDeviceType_GPU));
            return false;
        }
        say("OK   device type is GPU");

        // A D3D12 resource is only reachable through the DML allocation table,
        // so report whether the resource bridge is available at all.
        if (!dml_->CreateGPUAllocationFromD3DResource || !dml_->FreeGPUAllocation ||
            !dml_->GetD3D12ResourceFromAllocation) {
            say("FAIL OrtDmlApi is missing the D3D12 allocation bridge entry points");
            return false;
        }
        say("OK   D3D12 allocation bridge present");
        std::fputc('\n', g_out);
        say("PASS DirectML memory info is usable for zero-readback IoBinding");
        return true;
    }

private:
    bool check(const char* what, ort::OrtStatus* status)
    {
        if (!status)
            return true;
        std::string message = ort::consumeStatus(api_, status);
        say("FAIL %s: %s", what, message.c_str());
        return false;
    }

    bool readNames()
    {
        size_t inputCount = 0;
        size_t outputCount = 0;
        if (!check("SessionGetInputCount",
                   api_.slot<ort::FnSessionGetInputCount>(ort::kSlot_SessionGetInputCount)(session_,
                                                                                        &inputCount)))
            return false;
        if (!check("SessionGetOutputCount",
                   api_.slot<ort::FnSessionGetOutputCount>(ort::kSlot_SessionGetOutputCount)(session_,
                                                                                           &outputCount)))
            return false;
        if (inputCount != 1 || outputCount != 1) {
            say("FAIL expected 1 input and 1 output, got %llu and %llu",
                static_cast<unsigned long long>(inputCount), static_cast<unsigned long long>(outputCount));
            return false;
        }

        char* inputName = nullptr;
        char* outputName = nullptr;
        const bool namesOk =
            check("SessionGetInputName",
                  api_.slot<ort::FnSessionGetInputName>(ort::kSlot_SessionGetInputName)(
                      session_, 0, hostAllocator_, &inputName)) &&
            check("SessionGetOutputName",
                  api_.slot<ort::FnSessionGetOutputName>(ort::kSlot_SessionGetOutputName)(
                      session_, 0, hostAllocator_, &outputName));
        if (!namesOk)
            return false;
        inputName_ = inputName ? inputName : "";
        outputName_ = outputName ? outputName : "";
        if (inputName)
            api_.slot<ort::FnAllocatorFree>(ort::kSlot_AllocatorFree)(hostAllocator_, inputName);
        if (outputName)
            api_.slot<ort::FnAllocatorFree>(ort::kSlot_AllocatorFree)(hostAllocator_, outputName);
        say("OK   model IO: input=\"%s\" output=\"%s\"", inputName_.c_str(), outputName_.c_str());
        return true;
    }

    HMODULE module_ = nullptr;
    const ort::OrtApiBase* base_ = nullptr;
    ort::OrtApiView api_;
    const ort::OrtDmlApi* dml_ = nullptr;

    ort::OrtEnv* env_ = nullptr;
    ort::OrtSessionOptions* options_ = nullptr;
    ort::OrtSession* session_ = nullptr;
    ort::OrtAllocator* hostAllocator_ = nullptr;
    ort::OrtMemoryInfo* cpuInfo_ = nullptr;
    ort::OrtMemoryInfo* dmlInfo_ = nullptr;
    ort::OrtIoBinding* binding_ = nullptr;
    ort::OrtValue* inputValue_ = nullptr;
    ort::OrtValue* outputValue_ = nullptr;
    ort::OrtValue** values_ = nullptr;
    size_t valueCount_ = 0;

    float hostInput_ = 0.0f;
    std::string inputName_;
    std::string outputName_;
    std::string memoryInfoName_ = "DML";
};

int usage()
{
    say("usage: ort_dml_meminfo_probe.exe [onnxruntime.dll] [memory-info-name]");
    return 2;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc > 3)
        return usage();

    std::filesystem::path dllPath;
    if (argc >= 2)
        dllPath = std::filesystem::path(argv[1]);
    const char* memoryInfoName = argc >= 3 ? argv[2] : "DML";

    Probe probe;
    say("Veyra NRVideo DirectML memory-info probe (OrtApi v%u)", kProbeApiVersion);
    if (!probe.load(dllPath.empty() ? nullptr : dllPath.c_str()))
        return 1;
    if (!probe.resolveApi(memoryInfoName))
        return 1;
    if (!probe.createSession())
        return 1;
    if (!probe.probe())
        return 1;
    return 0;
}
