// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// Hand-written transcription of the ONNX Runtime C API table at
// ORT_API_VERSION 20 plus the DirectML execution-provider extension table.
//
// Provenance: the slot order and signatures were read from the public
// onnxruntime 1.20.1 headers (onnxruntime_c_api.h / onnxruntime_dml.h). The
// upstream C API header is ~218 KB and the DirectML SDK header is not part of
// the source tree, so this file keeps only the pointers the product uses, each
// at its verified 0-based index inside `struct OrtApi`. The indices are checked
// by tests/unit/OrtDmlSessionTests.cpp against a hard-coded table so a runtime
// built from a different ORT line cannot be mistaken for v20.
//
// Deliberately dependency-free: only <d3d12.h> (for the DirectML resource
// interop signatures), <cstdint>, <cstddef> and <string>. Nothing here pulls in
// the proprietary ONNX Runtime or DirectML SDK headers.
//
// The table is a vtable of plain C function pointers. Function-pointer ->
// void* conversions are used to index it (the same approach ORT's own dynamic
// loader takes); every pointer in the table is called through its exact
// declared type, never through the void* form.

#include <d3d12.h>

#include <cstddef>
#include <cstdint>
#include <string>

#if defined(_WIN32)
#define VEYRA_ORT_API_CALL __stdcall
#else
#define VEYRA_ORT_API_CALL
#endif

namespace veyra::pipeline::ort {

// ---------------------------------------------------------------------------
// Opaque runtime handles. ORT objects are only ever passed back to the runtime,
// so a forward declaration is the complete ABI contract.
// ---------------------------------------------------------------------------
struct OrtEnv;
struct OrtStatus; // nullptr means success
struct OrtMemoryInfo;
struct OrtIoBinding;
struct OrtSession;
struct OrtValue;
struct OrtSessionOptions;
struct OrtRunOptions;
struct OrtTypeInfo;
struct OrtTensorTypeAndShapeInfo;
struct OrtAllocator;

// ---------------------------------------------------------------------------
// Enums, copied verbatim (same enumerator names and values) from ORT 1.20.1.
// Fixed `int32_t`/`uint32_t` underlying types keep the by-value ABI identical
// to the C enums the runtime was built with.
// ---------------------------------------------------------------------------
enum OrtErrorCode : int32_t {
    ORT_OK = 0,
    ORT_FAIL = 1,
    ORT_INVALID_ARGUMENT = 2,
    ORT_NO_SUCHFILE = 3,
    ORT_NO_MODEL = 4,
    ORT_ENGINE_ERROR = 5,
    ORT_RUNTIME_EXCEPTION = 6,
    ORT_INVALID_PROTOBUF = 7,
    ORT_MODEL_LOADED = 8,
    ORT_NOT_IMPLEMENTED = 9,
    ORT_INVALID_GRAPH = 10,
    ORT_EP_FAIL = 11,
};

enum OrtLoggingLevel : int32_t {
    ORT_LOGGING_LEVEL_VERBOSE = 0,
    ORT_LOGGING_LEVEL_INFO = 1,
    ORT_LOGGING_LEVEL_WARNING = 2,
    ORT_LOGGING_LEVEL_ERROR = 3,
    ORT_LOGGING_LEVEL_FATAL = 4,
};

enum ExecutionMode : int32_t {
    ORT_SEQUENTIAL = 0,
    ORT_PARALLEL = 1,
};

// Mirrors ONNX TensorProto.DataType. Only the entries the product names are
// spelled out; the numeric values match the protobuf enum exactly.
enum ONNXTensorElementDataType : int32_t {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED = 0,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 = 2,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 = 3,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 = 4,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 = 5,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 = 6,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 7,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING = 8,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL = 9,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE = 11,
};

enum OrtAllocatorType : int32_t {
    OrtInvalidAllocator = -1,
    OrtDeviceAllocator = 0,
    OrtArenaAllocator = 1,
};

enum OrtMemType : int32_t {
    OrtMemTypeCPUInput = -2,
    OrtMemTypeCPUOutput = -1,
    OrtMemTypeCPU = OrtMemTypeCPUOutput,
    OrtMemTypeDefault = 0,
};

enum OrtMemoryInfoDeviceType : int32_t {
    OrtMemoryInfoDeviceType_CPU = 0,
    OrtMemoryInfoDeviceType_GPU = 1,
    OrtMemoryInfoDeviceType_FPGA = 2,
};

enum GraphOptimizationLevel : int32_t {
    ORT_DISABLE_ALL = 0,
    ORT_ENABLE_BASIC = 1,
    ORT_ENABLE_EXTENDED = 2,
    ORT_ENABLE_ALL = 99,
};

// DirectML extension API enums (onnxruntime_dml.h, 1.20.1).
enum OrtDmlPerformancePreference : uint32_t {
    DmlPreferenceDefault = 0,
    DmlPreferenceHighPerformance = 1,
    DmlPreferenceMinimumPower = 2,
};

enum OrtDmlDeviceFilter : uint32_t {
    DmlFilterAny = 0xffffffffu,
    DmlFilterGpu = 1u,
    DmlFilterNpu = 2u,
};

struct OrtDmlDeviceOptions {
    OrtDmlPerformancePreference Preference = DmlPreferenceHighPerformance;
    OrtDmlDeviceFilter Filter = DmlFilterGpu;
};

// ---------------------------------------------------------------------------
// ABI constants.
// ---------------------------------------------------------------------------
inline constexpr uint32_t kOrtApiVersion = 20;
inline constexpr uint32_t kOrtApiSlotCount = 285;
inline constexpr uint32_t kOrtDmlApiVersion = 1; // version argument of GetExecutionProviderApi

// ---------------------------------------------------------------------------
// Function-pointer types for every slot the product uses. Names mirror the
// upstream member names; signatures are byte-identical to ORT 1.20.1.
// ---------------------------------------------------------------------------
using FnCreateStatus = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtErrorCode code, const char* message);
using FnGetErrorCode = OrtErrorCode(VEYRA_ORT_API_CALL*)(const OrtStatus* status);
using FnGetErrorMessage = const char*(VEYRA_ORT_API_CALL*)(const OrtStatus* status);
using FnReleaseStatus = void(VEYRA_ORT_API_CALL*)(OrtStatus* status);

using FnCreateEnv = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtLoggingLevel level, const char* logid, OrtEnv** out);
using FnReleaseEnv = void(VEYRA_ORT_API_CALL*)(OrtEnv* env);
using FnReleaseSession = void(VEYRA_ORT_API_CALL*)(OrtSession* session);

using FnCreateSession = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtEnv* env, const wchar_t* modelPath,
                                                        const OrtSessionOptions* options, OrtSession** out);
using FnCreateSessionFromArray = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtEnv* env, const void* modelData,
                                                                 size_t modelDataLength,
                                                                 const OrtSessionOptions* options, OrtSession** out);
using FnRun = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtSession* session, const OrtRunOptions* runOptions,
                                             const char* const* inputNames, const OrtValue* const* inputs,
                                             size_t inputCount, const char* const* outputNames,
                                             size_t outputCount, OrtValue** outputs);
using FnCreateSessionOptions = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtSessionOptions** out);
using FnReleaseSessionOptions = void(VEYRA_ORT_API_CALL*)(OrtSessionOptions* options);
using FnSetSessionExecutionMode = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtSessionOptions* options, ExecutionMode mode);
using FnEnableMemPattern = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtSessionOptions* options);
using FnDisableMemPattern = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtSessionOptions* options);
using FnDisableCpuMemArena = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtSessionOptions* options);
using FnSetSessionGraphOptimizationLevel = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtSessionOptions* options,
                                                                         GraphOptimizationLevel level);

using FnSessionGetInputCount = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtSession* session, size_t* out);
using FnSessionGetOutputCount = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtSession* session, size_t* out);
using FnSessionGetInputTypeInfo = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtSession* session, size_t index,
                                                                 OrtTypeInfo** out);
using FnSessionGetOutputTypeInfo = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtSession* session, size_t index,
                                                                  OrtTypeInfo** out);
using FnSessionGetInputName = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtSession* session, size_t index,
                                                             OrtAllocator* allocator, char** out);
using FnSessionGetOutputName = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtSession* session, size_t index,
                                                              OrtAllocator* allocator, char** out);

using FnCreateRunOptions = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtRunOptions** out);
using FnReleaseRunOptions = void(VEYRA_ORT_API_CALL*)(OrtRunOptions* options);

using FnCreateTensorWithDataAsOrtValue = OrtStatus*(VEYRA_ORT_API_CALL*)(
    const OrtMemoryInfo* info, void* data, size_t dataLength, const int64_t* shape, size_t shapeLength,
    ONNXTensorElementDataType elementType, OrtValue** out);
using FnIsTensor = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtValue* value, int* out);
using FnGetTensorMutableData = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtValue* value, void** out);

using FnCastTypeInfoToTensorInfo = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtTypeInfo* info,
                                                                 const OrtTensorTypeAndShapeInfo** out);
using FnGetTensorElementType = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtTensorTypeAndShapeInfo* info,
                                                              ONNXTensorElementDataType* out);
using FnGetDimensionsCount = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtTensorTypeAndShapeInfo* info, size_t* out);
using FnGetDimensions = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtTensorTypeAndShapeInfo* info, int64_t* dims,
                                                        size_t dimCount);
using FnGetSymbolicDimensions = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtTensorTypeAndShapeInfo* info,
                                                               const char** dimParams, size_t dimParamCount);
using FnGetTensorTypeAndShape = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtValue* value,
                                                               OrtTensorTypeAndShapeInfo** out);
using FnReleaseTensorTypeAndShapeInfo = void(VEYRA_ORT_API_CALL*)(OrtTensorTypeAndShapeInfo* info);
using FnReleaseTypeInfo = void(VEYRA_ORT_API_CALL*)(OrtTypeInfo* info);

using FnCreateMemoryInfo = OrtStatus*(VEYRA_ORT_API_CALL*)(const char* name, OrtAllocatorType allocatorType,
                                                          int id, OrtMemType memType, OrtMemoryInfo** out);
using FnCompareMemoryInfo = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtMemoryInfo* lhs, const OrtMemoryInfo* rhs,
                                                           int* out);
using FnMemoryInfoGetName = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtMemoryInfo* info, const char** out);
using FnMemoryInfoGetId = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtMemoryInfo* info, int* out);
using FnMemoryInfoGetMemType = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtMemoryInfo* info, OrtMemType* out);
using FnMemoryInfoGetType = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtMemoryInfo* info, OrtAllocatorType* out);
using FnMemoryInfoGetDeviceType = void(VEYRA_ORT_API_CALL*)(const OrtMemoryInfo* info,
                                                           OrtMemoryInfoDeviceType* out);
using FnReleaseMemoryInfo = void(VEYRA_ORT_API_CALL*)(OrtMemoryInfo* info);

using FnAllocatorAlloc = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtAllocator* allocator, size_t size, void** out);
using FnAllocatorFree = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtAllocator* allocator, void* p);
using FnGetAllocatorWithDefaultOptions = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtAllocator** out);

using FnCreateAllocator = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtSession* session, const OrtMemoryInfo* info,
                                                         OrtAllocator** out);
using FnReleaseAllocator = void(VEYRA_ORT_API_CALL*)(OrtAllocator* allocator);

using FnRunWithBinding = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtSession* session, const OrtRunOptions* runOptions,
                                                        const OrtIoBinding* binding);
using FnCreateIoBinding = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtSession* session, OrtIoBinding** out);
using FnReleaseIoBinding = void(VEYRA_ORT_API_CALL*)(OrtIoBinding* binding);
using FnBindInput = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtIoBinding* binding, const char* name,
                                                   const OrtValue* value);
using FnBindOutput = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtIoBinding* binding, const char* name,
                                                    const OrtValue* value);
using FnBindOutputToDevice = OrtStatus*(VEYRA_ORT_API_CALL*)(OrtIoBinding* binding, const char* name,
                                                             const OrtMemoryInfo* info);
using FnGetBoundOutputNames = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtIoBinding* binding, OrtAllocator* allocator,
                                                             char** buffer, size_t** lengths, size_t* count);
using FnGetBoundOutputValues = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtIoBinding* binding, OrtAllocator* allocator,
                                                              OrtValue*** values, size_t* count);
using FnClearBoundInputs = void(VEYRA_ORT_API_CALL*)(OrtIoBinding* binding);
using FnClearBoundOutputs = void(VEYRA_ORT_API_CALL*)(OrtIoBinding* binding);
using FnReleaseValue = void(VEYRA_ORT_API_CALL*)(OrtValue* value);

using FnGetTensorMemoryInfo = OrtStatus*(VEYRA_ORT_API_CALL*)(const OrtValue* value, const OrtMemoryInfo** out);
using FnGetExecutionProviderApi = OrtStatus*(VEYRA_ORT_API_CALL*)(const char* providerName, uint32_t version,
                                                                 const void** out);

// ---------------------------------------------------------------------------
// Slot indices, 0-based into the `OrtApi` vtable. Verified against
// ort_capi_v1.20.1.h; order here follows the struct so the list can be diffed
// against the header directly.
// ---------------------------------------------------------------------------
inline constexpr size_t kSlot_CreateStatus = 0;
inline constexpr size_t kSlot_GetErrorCode = 1;
inline constexpr size_t kSlot_GetErrorMessage = 2;
inline constexpr size_t kSlot_CreateEnv = 3;
inline constexpr size_t kSlot_CreateSession = 7;
inline constexpr size_t kSlot_CreateSessionFromArray = 8;
inline constexpr size_t kSlot_Run = 9;
inline constexpr size_t kSlot_CreateSessionOptions = 10;
inline constexpr size_t kSlot_SetSessionExecutionMode = 13;
inline constexpr size_t kSlot_EnableMemPattern = 16;
inline constexpr size_t kSlot_DisableMemPattern = 17;
inline constexpr size_t kSlot_DisableCpuMemArena = 19;
inline constexpr size_t kSlot_SetSessionGraphOptimizationLevel = 23;
inline constexpr size_t kSlot_SessionGetInputCount = 30;
inline constexpr size_t kSlot_SessionGetOutputCount = 31;
inline constexpr size_t kSlot_SessionGetInputTypeInfo = 33;
inline constexpr size_t kSlot_SessionGetOutputTypeInfo = 34;
inline constexpr size_t kSlot_SessionGetInputName = 36;
inline constexpr size_t kSlot_SessionGetOutputName = 37;
inline constexpr size_t kSlot_CreateRunOptions = 39;
inline constexpr size_t kSlot_CreateTensorWithDataAsOrtValue = 49;
inline constexpr size_t kSlot_IsTensor = 50;
inline constexpr size_t kSlot_GetTensorMutableData = 51;
inline constexpr size_t kSlot_CastTypeInfoToTensorInfo = 55;
inline constexpr size_t kSlot_GetTensorElementType = 60;
inline constexpr size_t kSlot_GetDimensionsCount = 61;
inline constexpr size_t kSlot_GetDimensions = 62;
inline constexpr size_t kSlot_GetSymbolicDimensions = 63;
inline constexpr size_t kSlot_GetTensorTypeAndShape = 65;
inline constexpr size_t kSlot_CreateMemoryInfo = 68;
inline constexpr size_t kSlot_CompareMemoryInfo = 70;
inline constexpr size_t kSlot_MemoryInfoGetName = 71;
inline constexpr size_t kSlot_MemoryInfoGetId = 72;
inline constexpr size_t kSlot_MemoryInfoGetMemType = 73;
inline constexpr size_t kSlot_MemoryInfoGetType = 74;
inline constexpr size_t kSlot_AllocatorAlloc = 75;
inline constexpr size_t kSlot_AllocatorFree = 76;
inline constexpr size_t kSlot_GetAllocatorWithDefaultOptions = 78;
inline constexpr size_t kSlot_ReleaseEnv = 92;
inline constexpr size_t kSlot_ReleaseStatus = 93;
inline constexpr size_t kSlot_ReleaseMemoryInfo = 94;
inline constexpr size_t kSlot_ReleaseSession = 95;
inline constexpr size_t kSlot_ReleaseValue = 96;
inline constexpr size_t kSlot_ReleaseRunOptions = 97;
inline constexpr size_t kSlot_ReleaseTypeInfo = 98;
inline constexpr size_t kSlot_ReleaseTensorTypeAndShapeInfo = 99;
inline constexpr size_t kSlot_ReleaseSessionOptions = 100;
inline constexpr size_t kSlot_CreateAllocator = 131;
inline constexpr size_t kSlot_ReleaseAllocator = 132;
inline constexpr size_t kSlot_RunWithBinding = 133;
inline constexpr size_t kSlot_CreateIoBinding = 134;
inline constexpr size_t kSlot_ReleaseIoBinding = 135;
inline constexpr size_t kSlot_BindInput = 136;
inline constexpr size_t kSlot_BindOutput = 137;
inline constexpr size_t kSlot_BindOutputToDevice = 138;
inline constexpr size_t kSlot_GetBoundOutputNames = 139;
inline constexpr size_t kSlot_GetBoundOutputValues = 140;
inline constexpr size_t kSlot_ClearBoundInputs = 141;
inline constexpr size_t kSlot_ClearBoundOutputs = 142;
inline constexpr size_t kSlot_GetTensorMemoryInfo = 194;
inline constexpr size_t kSlot_GetExecutionProviderApi = 195;
inline constexpr size_t kSlot_MemoryInfoGetDeviceType = 225;

// `OrtApiBase` is the two-pointer entry point exported by OrtGetApiBase().
struct OrtApiBase {
    const void*(VEYRA_ORT_API_CALL* GetApi)(uint32_t version);
    const char*(VEYRA_ORT_API_CALL* GetVersionString)();
};

using FnOrtGetApiBase = const OrtApiBase*(VEYRA_ORT_API_CALL*)();

// DirectML execution-provider extension: exactly six pointers, byte-identical
// to onnxruntime_dml.h. `dml_device` stays `void*` because the DirectML SDK's
// IDMLDevice is not available at compile time; the runtime only forwards it.
struct OrtDmlApi {
    OrtStatus*(VEYRA_ORT_API_CALL* SessionOptionsAppendExecutionProvider_DML)(OrtSessionOptions* options,
                                                                            int deviceId);
    OrtStatus*(VEYRA_ORT_API_CALL* SessionOptionsAppendExecutionProvider_DML1)(OrtSessionOptions* options,
                                                                              void* dmlDevice,
                                                                              ID3D12CommandQueue* commandQueue);
    OrtStatus*(VEYRA_ORT_API_CALL* CreateGPUAllocationFromD3DResource)(ID3D12Resource* d3dResource,
                                                                      void** dmlResource);
    OrtStatus*(VEYRA_ORT_API_CALL* FreeGPUAllocation)(void* dmlResource);
    OrtStatus*(VEYRA_ORT_API_CALL* GetD3D12ResourceFromAllocation)(OrtAllocator* provider, void* dmlResource,
                                                                  ID3D12Resource** d3dResource);
    OrtStatus*(VEYRA_ORT_API_CALL* SessionOptionsAppendExecutionProvider_DML2)(OrtSessionOptions* options,
                                                                               OrtDmlDeviceOptions* deviceOptions);
};

// ---------------------------------------------------------------------------
// Vtable view: type-safe slot lookup for a raw `const void* const*` table.
// ---------------------------------------------------------------------------
class OrtApiView {
public:
    OrtApiView() = default;
    explicit OrtApiView(const void* const* vtable) : vtable_(vtable) {}
    explicit OrtApiView(const void* api) : vtable_(static_cast<const void* const*>(api)) {}

    bool valid() const { return vtable_ != nullptr; }
    const void* const* table() const { return vtable_; }

    template <typename Fn>
    Fn slot(size_t index) const
    {
        return vtable_ ? reinterpret_cast<Fn>(vtable_[index]) : nullptr;
    }

private:
    const void* const* vtable_ = nullptr;
};

// ---------------------------------------------------------------------------
// Status handling.
// ---------------------------------------------------------------------------
// OrtStatus is the runtime's error object. It must always be released exactly
// once, so every call site funnels through consumeStatus(): it extracts the
// code/message (for logging and for `error_`) and then frees the object. A
// nullptr status means success and yields an empty message.
inline std::string consumeStatus(const OrtApiView& api, OrtStatus* status)
{
    if (!status)
        return {};

    const auto getCode = api.slot<FnGetErrorCode>(kSlot_GetErrorCode);
    const auto getMessage = api.slot<FnGetErrorMessage>(kSlot_GetErrorMessage);
    const auto release = api.slot<FnReleaseStatus>(kSlot_ReleaseStatus);

    std::string message;
    if (getMessage) {
        if (const char* text = getMessage(status); text && *text)
            message = text;
    }
    if (getCode) {
        const OrtErrorCode code = getCode(status);
        if (message.empty())
            message = "ORT error code " + std::to_string(static_cast<int>(code));
        else
            message += " (ORT code " + std::to_string(static_cast<int>(code)) + ")";
    } else if (message.empty()) {
        message = "unknown ONNX Runtime error";
    }
    if (release)
        release(status);
    return message;
}

// RAII owner for an OrtStatus returned by the runtime. Destroying the guard
// releases the status, so early returns cannot leak it.
class OrtStatusGuard {
public:
    OrtStatusGuard() = default;
    OrtStatusGuard(const OrtStatusGuard&) = delete;
    OrtStatusGuard& operator=(const OrtStatusGuard&) = delete;

    OrtStatusGuard(const OrtApiView& api, OrtStatus* status) noexcept : api_(api), status_(status) {}

    ~OrtStatusGuard() { reset(); }

    OrtStatus* get() const noexcept { return status_; }
    bool ok() const noexcept { return status_ == nullptr; }
    explicit operator bool() const noexcept { return status_ != nullptr; }

    // Consumes the owned status and returns its human readable form.
    std::string consume()
    {
        OrtStatus* owned = status_;
        status_ = nullptr;
        return consumeStatus(api_, owned);
    }

    // Releases without inspecting (used when the caller already reported it).
    void reset() noexcept
    {
        if (!status_)
            return;
        OrtStatus* owned = status_;
        status_ = nullptr;
        if (const auto release = api_.slot<FnReleaseStatus>(kSlot_ReleaseStatus))
            release(owned);
    }

    OrtStatus* release() noexcept
    {
        OrtStatus* owned = status_;
        status_ = nullptr;
        return owned;
    }

private:
    OrtApiView api_;
    OrtStatus* status_ = nullptr;
};

// ---------------------------------------------------------------------------
// Minimal ONNX model used by the memory-info probe and the unit test.
//
// A 115 byte serialized ModelProto: opset 11, one `Identity` node (x -> y),
// float32 [1,1,1,1] input and output, producer_name "veyra-ort-dml-probe".
// It is small because CreateSessionFromArray rejects empty graphs, and the
// probe only needs a session that the DirectML EP will accept so that the
// bound-value memory info is queryable. Keeping it inline avoids shipping a
// binary fixture that would have to live under the source tree.
// ---------------------------------------------------------------------------
inline constexpr uint8_t kMinimalIdentityOnnxModel[] = {
    0x08, 0x07, 0x12, 0x13, 0x76, 0x65, 0x79, 0x72, 0x61, 0x2D, 0x6F, 0x72, 0x74, 0x2D, 0x64, 0x6D,
    0x6C, 0x2D, 0x70, 0x72, 0x6F, 0x62, 0x65, 0x3A, 0x56, 0x0A, 0x1A, 0x0A, 0x01, 0x78, 0x12, 0x01,
    0x79, 0x1A, 0x08, 0x69, 0x64, 0x65, 0x6E, 0x74, 0x69, 0x74, 0x79, 0x22, 0x08, 0x49, 0x64, 0x65,
    0x6E, 0x74, 0x69, 0x74, 0x79, 0x5A, 0x1B, 0x0A, 0x01, 0x78, 0x12, 0x16, 0x0A, 0x14, 0x08, 0x01,
    0x12, 0x10, 0x0A, 0x02, 0x08, 0x01, 0x0A, 0x02, 0x08, 0x01, 0x0A, 0x02, 0x08, 0x01, 0x0A, 0x02,
    0x08, 0x01, 0x62, 0x1B, 0x0A, 0x01, 0x79, 0x12, 0x16, 0x0A, 0x14, 0x08, 0x01, 0x12, 0x10, 0x0A,
    0x02, 0x08, 0x01, 0x0A, 0x02, 0x08, 0x01, 0x0A, 0x02, 0x08, 0x01, 0x0A, 0x02, 0x08, 0x01, 0x42,
    0x02, 0x10, 0x0B,
};

inline constexpr size_t kMinimalIdentityOnnxModelSize = sizeof(kMinimalIdentityOnnxModel);

} // namespace veyra::pipeline::ort
