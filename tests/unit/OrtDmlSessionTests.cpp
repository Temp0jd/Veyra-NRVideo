// SPDX-License-Identifier: GPL-3.0-only
//
// Unit tests for OrtDmlSrBackend (M7: ONNX Runtime v20 + DirectML backend).
//
// The ONNX Runtime and DirectML C APIs are replaced by a recording mock: every
// slot the backend uses is a typed stub that forwards to a Mock instance, so the
// session setup order, the ABI table, the IoBinding zero-readback path, the
// runtime-allocated fallback and the teardown order are all verified on a
// machine without onnxruntime.dll, DirectML.dll or a D3D12 device.
//
// Build and run on Linux (MinGW cross build + Wine):
//   x86_64-w64-mingw32-g++ -std=c++20 -Iinclude -DNOMINMAX -DWIN32_LEAN_AND_MEAN
//     tests/unit/OrtDmlSessionTests.cpp src/pipeline/OrtDmlSrBackend.cpp
//     src/base/Log.cpp src/base/RuntimePaths.cpp src/beauty/OnnxLite.cpp
//     -o /tmp/vr_ort_dml_session_tests.exe
//     -ld3d12 -ldxgi -ld3dcompiler -lole32 -luuid -static-libgcc -static-libstdc++
//   wine /tmp/vr_ort_dml_session_tests.exe
//
// What this cannot cover on Linux: whether a real DirectML driver accepts the
// "DML" memory-info name and the D3D12 buffers handed to it. That is what
// tools/ort_dml_meminfo_probe.cpp answers on a Windows box.

#include "veyra/beauty/OnnxLite.h"
#include "veyra/pipeline/OrtDmlApi.h"
#include "veyra/pipeline/OrtDmlSrBackend.h"

#include <windows.h>
#include <d3d12.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace ort = veyra::pipeline::ort;
using veyra::pipeline::OrtDmlSrBackend;

int g_checks = 0;
int g_failures = 0;
const char* g_test = "<none>";

void check(const char* what, bool ok)
{
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL [%s] %s\n", g_test, what);
    }
}

void checkEqual(const char* what, long long got, long long want)
{
    ++g_checks;
    if (got != want) {
        ++g_failures;
        std::printf("  FAIL [%s] %s: got %lld, want %lld\n", g_test, what, got, want);
    }
}

void checkText(const char* what, const std::string& text, const char* needle)
{
    ++g_checks;
    if (text.find(needle) == std::string::npos) {
        ++g_failures;
        std::printf("  FAIL [%s] %s: \"%s\" not in \"%s\"\n", g_test, what, needle, text.c_str());
    }
}

template <typename Fn>
void* fnToVoid(Fn fn)
{
    static_assert(sizeof(Fn) == sizeof(void*), "a function pointer must fit in a table slot");
    void* slot = nullptr;
    std::memcpy(&slot, &fn, sizeof(slot));
    return slot;
}

// ---------------------------------------------------------------------------
// Fake runtime objects. The backend only passes these pointers around, so the
// layouts only have to be good enough to record what was asked; every handle is
// destroyed by the matching release entry point, which makes leaks observable.
// ---------------------------------------------------------------------------
struct FakeStatus {
    ort::OrtErrorCode code = ort::ORT_OK;
    std::string message;
};
struct FakeMemoryInfo {
    std::string name;
    int allocatorType = 0;
    int id = 0;
    int memType = 0;
    int deviceType = 1;
};
struct FakeShape { // stand-in for OrtTensorTypeAndShapeInfo, owned by FakeTypeInfo
    int elementType = ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    std::vector<int64_t> dims;
};
struct FakeTypeInfo {
    FakeShape shape;
};
struct FakeValue {
    void* data = nullptr;
    FakeMemoryInfo* info = nullptr;
};
struct FakeEnv {};
struct FakeSession {};
struct FakeSessionOptions {
    int executionMode = -1;
    bool memPattern = true;
};
struct FakeIoBinding {
    int inputs = 0;
    int boundOutputs = 0;
    int deviceOutputs = 0;
};
struct FakeAllocator {
    int kind = 0; // 0 = runtime default, 1 = DirectML provider allocator
};

// A minimal COM object so the backend can AddRef/Release the ID3D12Resource it
// resolves from a DirectML allocation (and the device/queue it is handed).
struct FakeComObject {
    void** vtable = nullptr;
    long refs = 0;
};

HRESULT WINAPI fakeQueryInterface(FakeComObject* self, const IID&, void** out)
{
    (void)self;
    if (out)
        *out = nullptr;
    return E_NOINTERFACE;
}

ULONG WINAPI fakeAddRef(FakeComObject* self) { return static_cast<ULONG>(++self->refs); }

ULONG WINAPI fakeRelease(FakeComObject* self) { return static_cast<ULONG>(--self->refs); }

void* g_fakeComVtable[3] = {fnToVoid(&fakeQueryInterface), fnToVoid(&fakeAddRef), fnToVoid(&fakeRelease)};

FakeComObject g_fakeResource{g_fakeComVtable, 0};
FakeComObject g_fakeDevice{g_fakeComVtable, 0};
FakeComObject g_fakeQueue{g_fakeComVtable, 0};
FakeComObject g_fakeObject{g_fakeComVtable, 0}; // command list / texture stand-in

ID3D12Resource* fakeResource() { return reinterpret_cast<ID3D12Resource*>(&g_fakeResource); }
ID3D12Device* fakeDevice() { return reinterpret_cast<ID3D12Device*>(&g_fakeDevice); }
ID3D12CommandQueue* fakeQueue() { return reinterpret_cast<ID3D12CommandQueue*>(&g_fakeQueue); }
ID3D12GraphicsCommandList* fakeList() { return reinterpret_cast<ID3D12GraphicsCommandList*>(&g_fakeObject); }

// ---------------------------------------------------------------------------
// Mock
// ---------------------------------------------------------------------------
class Mock;

Mock* g_mock = nullptr;
ort::OrtDmlApi g_dmlApi{};

class Mock {
public:
    // --- installed state -------------------------------------------------
    std::vector<void*> table;
    std::vector<std::string> events;

    void record(const char* name) { events.emplace_back(name); }

    int indexOf(const char* name) const
    {
        for (size_t i = 0; i < events.size(); ++i) {
            if (events[i] == name)
                return static_cast<int>(i);
        }
        return -1;
    }

    int count(const char* name) const
    {
        int hits = 0;
        for (const auto& event : events) {
            if (event == name)
                ++hits;
        }
        return hits;
    }

    bool happenedBefore(const char* first, const char* second) const
    {
        const int a = indexOf(first);
        const int b = indexOf(second);
        return a >= 0 && b >= 0 && a < b;
    }

    // --- model description used by the introspection stubs ---------------
    std::string inputName = "input";
    std::string outputName = "output";
    std::vector<int64_t> inputDims = {1, 3, -1, -1};
    std::vector<int64_t> outputDims = {1, 3, -1, -1};
    int elementType = ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;

    // --- failure injection ----------------------------------------------
    bool acceptApiVersion = true;
    bool failCreateEnv = false;
    bool failCreateSessionOptions = false;
    bool failDisableMemPattern = false;
    bool failSetExecutionMode = false;
    bool failAppendDml = false;
    bool failCreateSession = false;
    bool failCreateMemoryInfo = false;
    bool failCreateTensor = false;
    bool failCreateIoBinding = false;
    bool failBindInput = false;
    bool failBindOutput = false;
    bool failBindOutputToDevice = false;
    bool failCreateAllocator = false;
    bool failRun = false;
    bool failGetBoundOutputValues = false;
    bool failGetProviderApi = false;
    bool stripDmlBridge = false;
    bool omitDml1 = false;
    bool omitDml2 = false;

    // --- recorded values -------------------------------------------------
    int logLevel = -1;
    std::string logId;
    int executionMode = -1;
    bool memPattern = true;
    int appendCalls = 0;
    std::string appendKind;
    int appendDeviceId = -1;
    void* appendDmlDevice = nullptr;
    ID3D12CommandQueue* appendQueue = nullptr;
    int appendPreference = -1;
    int appendFilter = -1;
    std::string memoryInfoName;
    int memoryInfoId = -1;
    int memoryInfoAllocatorType = -1;
    int memoryInfoMemType = -1;
    int memoryInfoCalls = 0;
    std::vector<size_t> tensorBytes;
    std::vector<std::vector<int64_t>> tensorShapes;
    int tensorCalls = 0;
    std::vector<void*> allocations;
    int freeCalls = 0;
    void* outputAllocation = nullptr;
    std::string boundInputName;
    std::string boundOutputName;
    int boundOutputToDeviceCalls = 0;
    int runCalls = 0;
    std::wstring modelPathSeen;
    std::string providerNameSeen;
    int providerVersionSeen = -1;

    // --- live/runtime bookkeeping ---------------------------------------
    int liveStatus = 0;
    int liveValue = 0;
    int liveMemoryInfo = 0;
    int liveBinding = 0;
    int liveSession = 0;
    int liveOptions = 0;
    int liveEnv = 0;
    int liveTypeInfo = 0;
    int liveAllocator = 0;
    int createdStatus = 0;
    int releasedStatus = 0;
    int unexpectedCalls = 0;

    void reset()
    {
        table.assign(ort::kOrtApiSlotCount, fnToVoid(&mockUnused));
        events.clear();
        inputName = "input";
        outputName = "output";
        inputDims = {1, 3, -1, -1};
        outputDims = {1, 3, -1, -1};
        elementType = ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        acceptApiVersion = true;
        failCreateEnv = failCreateSessionOptions = failDisableMemPattern = failSetExecutionMode = false;
        failAppendDml = failCreateSession = failCreateMemoryInfo = failCreateTensor = false;
        failCreateIoBinding = failBindInput = failBindOutput = failBindOutputToDevice = false;
        failCreateAllocator = failRun = failGetBoundOutputValues = failGetProviderApi = false;
        stripDmlBridge = omitDml1 = omitDml2 = false;
        logLevel = -1;
        logId.clear();
        executionMode = -1;
        memPattern = true;
        appendCalls = 0;
        appendKind.clear();
        appendDeviceId = -1;
        appendDmlDevice = nullptr;
        appendQueue = nullptr;
        appendPreference = -1;
        appendFilter = -1;
        memoryInfoName.clear();
        memoryInfoId = -1;
        memoryInfoAllocatorType = -1;
        memoryInfoMemType = -1;
        memoryInfoCalls = 0;
        tensorBytes.clear();
        tensorShapes.clear();
        tensorCalls = 0;
        allocations.clear();
        freeCalls = 0;
        outputAllocation = nullptr;
        boundInputName.clear();
        boundOutputName.clear();
        boundOutputToDeviceCalls = 0;
        runCalls = 0;
        modelPathSeen.clear();
        providerNameSeen.clear();
        providerVersionSeen = -1;
        liveStatus = liveValue = liveMemoryInfo = liveBinding = 0;
        liveSession = liveOptions = liveEnv = liveTypeInfo = liveAllocator = 0;
        createdStatus = releasedStatus = 0;
        unexpectedCalls = 0;
        g_fakeResource.refs = 0;
        g_fakeDevice.refs = 0;
        g_fakeQueue.refs = 0;
        g_fakeObject.refs = 0;
        install();
    }

    void install();

    const void* apiForVersion(uint32_t version) const
    {
        if (!acceptApiVersion || version != ort::kOrtApiVersion)
            return nullptr;
        return table.data();
    }

    ort::OrtStatus* makeStatus(ort::OrtErrorCode code, std::string message)
    {
        auto* fake = new FakeStatus{code, std::move(message)};
        ++liveStatus;
        ++createdStatus;
        return reinterpret_cast<ort::OrtStatus*>(fake);
    }

    static FakeStatus* asStatus(const ort::OrtStatus* status)
    {
        return reinterpret_cast<FakeStatus*>(const_cast<ort::OrtStatus*>(status));
    }

    // ---------------------------------------------------------------
    // OrtApi v20 slots
    // ---------------------------------------------------------------
    ort::OrtStatus* createStatus(ort::OrtErrorCode code, const char* message)
    {
        record("CreateStatus");
        return makeStatus(code, message ? message : "");
    }
    ort::OrtErrorCode getErrorCode(const ort::OrtStatus* status) { return asStatus(status)->code; }
    const char* getErrorMessage(const ort::OrtStatus* status) { return asStatus(status)->message.c_str(); }
    void releaseStatus(ort::OrtStatus* status)
    {
        record("ReleaseStatus");
        --liveStatus;
        ++releasedStatus;
        delete asStatus(status);
    }

    ort::OrtStatus* createEnv(ort::OrtLoggingLevel level, const char* logid, ort::OrtEnv** out)
    {
        record("CreateEnv");
        if (failCreateEnv)
            return makeStatus(ort::ORT_FAIL, "CreateEnv rejected by the mock");
        logLevel = static_cast<int>(level);
        logId = logid ? logid : "";
        ++liveEnv;
        *out = reinterpret_cast<ort::OrtEnv*>(new FakeEnv());
        return nullptr;
    }
    void releaseEnv(ort::OrtEnv* env)
    {
        record("ReleaseEnv");
        --liveEnv;
        delete reinterpret_cast<FakeEnv*>(env);
    }

    ort::OrtStatus* createSessionOptions(ort::OrtSessionOptions** out)
    {
        record("CreateSessionOptions");
        if (failCreateSessionOptions)
            return makeStatus(ort::ORT_FAIL, "CreateSessionOptions rejected by the mock");
        ++liveOptions;
        *out = reinterpret_cast<ort::OrtSessionOptions*>(new FakeSessionOptions());
        return nullptr;
    }
    void releaseSessionOptions(ort::OrtSessionOptions* options)
    {
        record("ReleaseSessionOptions");
        --liveOptions;
        delete reinterpret_cast<FakeSessionOptions*>(options);
    }
    ort::OrtStatus* setExecutionMode(ort::OrtSessionOptions* options, ort::ExecutionMode mode)
    {
        record("SetSessionExecutionMode");
        if (failSetExecutionMode)
            return makeStatus(ort::ORT_FAIL, "SetSessionExecutionMode rejected by the mock");
        executionMode = static_cast<int>(mode);
        reinterpret_cast<FakeSessionOptions*>(options)->executionMode = executionMode;
        return nullptr;
    }
    ort::OrtStatus* disableMemPattern(ort::OrtSessionOptions* options)
    {
        record("DisableMemPattern");
        if (failDisableMemPattern)
            return makeStatus(ort::ORT_FAIL, "DisableMemPattern rejected by the mock");
        memPattern = false;
        reinterpret_cast<FakeSessionOptions*>(options)->memPattern = false;
        return nullptr;
    }

    ort::OrtStatus* createSession(const ort::OrtEnv*, const wchar_t* modelPath, const ort::OrtSessionOptions*,
                                  ort::OrtSession** out)
    {
        record("CreateSession");
        if (failCreateSession)
            return makeStatus(ort::ORT_NO_MODEL, "CreateSession rejected by the mock");
        modelPathSeen = modelPath ? modelPath : L"";
        ++liveSession;
        *out = reinterpret_cast<ort::OrtSession*>(new FakeSession());
        return nullptr;
    }
    void releaseSession(ort::OrtSession* session)
    {
        record("ReleaseSession");
        --liveSession;
        delete reinterpret_cast<FakeSession*>(session);
    }

    ort::OrtStatus* sessionGetInputCount(const ort::OrtSession*, size_t* out)
    {
        record("SessionGetInputCount");
        *out = 1;
        return nullptr;
    }
    ort::OrtStatus* sessionGetOutputCount(const ort::OrtSession*, size_t* out)
    {
        record("SessionGetOutputCount");
        *out = 1;
        return nullptr;
    }
    ort::OrtStatus* sessionGetInputTypeInfo(const ort::OrtSession*, size_t index, ort::OrtTypeInfo** out)
    {
        record("SessionGetInputTypeInfo");
        if (index != 0)
            return makeStatus(ort::ORT_INVALID_ARGUMENT, "only one input");
        return makeTypeInfo(inputDims, out);
    }
    ort::OrtStatus* sessionGetOutputTypeInfo(const ort::OrtSession*, size_t index, ort::OrtTypeInfo** out)
    {
        record("SessionGetOutputTypeInfo");
        if (index != 0)
            return makeStatus(ort::ORT_INVALID_ARGUMENT, "only one output");
        return makeTypeInfo(outputDims, out);
    }
    ort::OrtStatus* makeTypeInfo(const std::vector<int64_t>& dims, ort::OrtTypeInfo** out)
    {
        auto* info = new FakeTypeInfo{};
        info->shape.elementType = elementType;
        info->shape.dims = dims;
        ++liveTypeInfo;
        *out = reinterpret_cast<ort::OrtTypeInfo*>(info);
        return nullptr;
    }
    void releaseTypeInfo(ort::OrtTypeInfo* info)
    {
        record("ReleaseTypeInfo");
        --liveTypeInfo;
        delete reinterpret_cast<FakeTypeInfo*>(info);
    }
    ort::OrtStatus* castTypeInfoToTensorInfo(const ort::OrtTypeInfo* info,
                                             const ort::OrtTensorTypeAndShapeInfo** out)
    {
        record("CastTypeInfoToTensorInfo");
        const auto* fake = reinterpret_cast<const FakeTypeInfo*>(info);
        *out = reinterpret_cast<const ort::OrtTensorTypeAndShapeInfo*>(&fake->shape);
        return nullptr;
    }
    ort::OrtStatus* getTensorElementType(const ort::OrtTensorTypeAndShapeInfo* shape,
                                         ort::ONNXTensorElementDataType* out)
    {
        record("GetTensorElementType");
        *out = static_cast<ort::ONNXTensorElementDataType>(reinterpret_cast<const FakeShape*>(shape)->elementType);
        return nullptr;
    }
    ort::OrtStatus* getDimensionsCount(const ort::OrtTensorTypeAndShapeInfo* shape, size_t* out)
    {
        record("GetDimensionsCount");
        *out = reinterpret_cast<const FakeShape*>(shape)->dims.size();
        return nullptr;
    }
    ort::OrtStatus* getDimensions(const ort::OrtTensorTypeAndShapeInfo* shape, int64_t* dims, size_t count)
    {
        record("GetDimensions");
        const auto* fake = reinterpret_cast<const FakeShape*>(shape);
        if (count != fake->dims.size())
            return makeStatus(ort::ORT_INVALID_ARGUMENT, "dimension count mismatch");
        for (size_t i = 0; i < count; ++i)
            dims[i] = fake->dims[i];
        return nullptr;
    }
    ort::OrtStatus* getSymbolicDimensions(const ort::OrtTensorTypeAndShapeInfo*, const char**, size_t)
    {
        record("GetSymbolicDimensions");
        return nullptr;
    }
    ort::OrtStatus* sessionGetName(const std::string& name, char** out)
    {
        char* buffer = static_cast<char*>(std::malloc(name.size() + 1));
        if (!buffer)
            return makeStatus(ort::ORT_FAIL, "out of memory");
        std::memcpy(buffer, name.c_str(), name.size() + 1);
        *out = buffer;
        return nullptr;
    }
    ort::OrtStatus* sessionGetInputName(const ort::OrtSession*, size_t, ort::OrtAllocator* allocator, char** out)
    {
        record("SessionGetInputName");
        (void)allocator;
        return sessionGetName(inputName, out);
    }
    ort::OrtStatus* sessionGetOutputName(const ort::OrtSession*, size_t, ort::OrtAllocator* allocator, char** out)
    {
        record("SessionGetOutputName");
        (void)allocator;
        return sessionGetName(outputName, out);
    }

    ort::OrtStatus* createTensor(const ort::OrtMemoryInfo* info, void* data, size_t bytes, const int64_t* shape,
                                 size_t shapeCount, ort::ONNXTensorElementDataType type, ort::OrtValue** out)
    {
        record("CreateTensorWithDataAsOrtValue");
        if (failCreateTensor)
            return makeStatus(ort::ORT_FAIL, "CreateTensorWithDataAsOrtValue rejected by the mock");
        if (!data || bytes == 0 || !shape || shapeCount != 4)
            return makeStatus(ort::ORT_INVALID_ARGUMENT, "invalid tensor description");
        if (type != ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            return makeStatus(ort::ORT_INVALID_ARGUMENT, "tensor must be float32");
        ++tensorCalls;
        tensorBytes.push_back(bytes);
        tensorShapes.emplace_back(shape, shape + shapeCount);
        auto* value = new FakeValue{data, reinterpret_cast<FakeMemoryInfo*>(const_cast<ort::OrtMemoryInfo*>(info))};
        ++liveValue;
        *out = reinterpret_cast<ort::OrtValue*>(value);
        return nullptr;
    }
    ort::OrtStatus* getTensorMutableData(ort::OrtValue* value, void** out)
    {
        record("GetTensorMutableData");
        *out = reinterpret_cast<FakeValue*>(value)->data;
        return nullptr;
    }
    void releaseValue(ort::OrtValue* value)
    {
        record("ReleaseValue");
        --liveValue;
        delete reinterpret_cast<FakeValue*>(value);
    }

    ort::OrtStatus* createMemoryInfo(const char* name, ort::OrtAllocatorType allocatorType, int id,
                                     ort::OrtMemType memType, ort::OrtMemoryInfo** out)
    {
        record("CreateMemoryInfo");
        ++memoryInfoCalls;
        memoryInfoName = name ? name : "";
        memoryInfoId = id;
        memoryInfoAllocatorType = static_cast<int>(allocatorType);
        memoryInfoMemType = static_cast<int>(memType);
        if (failCreateMemoryInfo)
            return makeStatus(ort::ORT_INVALID_ARGUMENT, "unknown memory info name");
        auto* info = new FakeMemoryInfo{memoryInfoName, memoryInfoAllocatorType, memoryInfoId, memoryInfoMemType,
                                        ort::OrtMemoryInfoDeviceType_GPU};
        ++liveMemoryInfo;
        *out = reinterpret_cast<ort::OrtMemoryInfo*>(info);
        return nullptr;
    }
    ort::OrtStatus* compareMemoryInfo(const ort::OrtMemoryInfo* lhs, const ort::OrtMemoryInfo* rhs, int* out)
    {
        record("CompareMemoryInfo");
        const auto* a = reinterpret_cast<const FakeMemoryInfo*>(lhs);
        const auto* b = reinterpret_cast<const FakeMemoryInfo*>(rhs);
        *out = (a->name == b->name && a->id == b->id && a->memType == b->memType &&
                a->allocatorType == b->allocatorType)
                   ? 1
                   : 0;
        return nullptr;
    }
    ort::OrtStatus* memoryInfoGetName(const ort::OrtMemoryInfo* info, const char** out)
    {
        *out = reinterpret_cast<const FakeMemoryInfo*>(info)->name.c_str();
        return nullptr;
    }
    ort::OrtStatus* memoryInfoGetId(const ort::OrtMemoryInfo* info, int* out)
    {
        *out = reinterpret_cast<const FakeMemoryInfo*>(info)->id;
        return nullptr;
    }
    ort::OrtStatus* memoryInfoGetMemType(const ort::OrtMemoryInfo* info, ort::OrtMemType* out)
    {
        *out = static_cast<ort::OrtMemType>(reinterpret_cast<const FakeMemoryInfo*>(info)->memType);
        return nullptr;
    }
    ort::OrtStatus* memoryInfoGetType(const ort::OrtMemoryInfo* info, ort::OrtAllocatorType* out)
    {
        *out = static_cast<ort::OrtAllocatorType>(reinterpret_cast<const FakeMemoryInfo*>(info)->allocatorType);
        return nullptr;
    }
    void memoryInfoGetDeviceType(const ort::OrtMemoryInfo* info, ort::OrtMemoryInfoDeviceType* out)
    {
        *out = static_cast<ort::OrtMemoryInfoDeviceType>(reinterpret_cast<const FakeMemoryInfo*>(info)->deviceType);
    }
    void releaseMemoryInfo(ort::OrtMemoryInfo* info)
    {
        record("ReleaseMemoryInfo");
        --liveMemoryInfo;
        delete reinterpret_cast<FakeMemoryInfo*>(info);
    }

    ort::OrtStatus* allocatorAlloc(ort::OrtAllocator*, size_t size, void** out)
    {
        record("AllocatorAlloc");
        *out = std::malloc(size ? size : 1);
        return *out ? nullptr : makeStatus(ort::ORT_FAIL, "out of memory");
    }
    ort::OrtStatus* allocatorFree(ort::OrtAllocator*, void* pointer)
    {
        record("AllocatorFree");
        std::free(pointer);
        return nullptr;
    }
    ort::OrtStatus* getDefaultAllocator(ort::OrtAllocator** out)
    {
        record("GetAllocatorWithDefaultOptions");
        static FakeAllocator hostAllocator{0};
        *out = reinterpret_cast<ort::OrtAllocator*>(&hostAllocator);
        return nullptr;
    }
    ort::OrtStatus* createAllocator(const ort::OrtSession*, const ort::OrtMemoryInfo*, ort::OrtAllocator** out)
    {
        record("CreateAllocator");
        if (failCreateAllocator)
            return makeStatus(ort::ORT_FAIL, "CreateAllocator rejected by the mock");
        ++liveAllocator;
        *out = reinterpret_cast<ort::OrtAllocator*>(new FakeAllocator{1});
        return nullptr;
    }
    void releaseAllocator(ort::OrtAllocator* allocator)
    {
        record("ReleaseAllocator");
        --liveAllocator;
        delete reinterpret_cast<FakeAllocator*>(allocator);
    }

    ort::OrtStatus* createIoBinding(ort::OrtSession*, ort::OrtIoBinding** out)
    {
        record("CreateIoBinding");
        if (failCreateIoBinding)
            return makeStatus(ort::ORT_FAIL, "CreateIoBinding rejected by the mock");
        ++liveBinding;
        *out = reinterpret_cast<ort::OrtIoBinding*>(new FakeIoBinding());
        return nullptr;
    }
    void releaseIoBinding(ort::OrtIoBinding* binding)
    {
        record("ReleaseIoBinding");
        --liveBinding;
        delete reinterpret_cast<FakeIoBinding*>(binding);
    }
    ort::OrtStatus* bindInput(ort::OrtIoBinding* binding, const char* name, const ort::OrtValue*)
    {
        record("BindInput");
        if (failBindInput)
            return makeStatus(ort::ORT_FAIL, "BindInput rejected by the mock");
        boundInputName = name ? name : "";
        ++reinterpret_cast<FakeIoBinding*>(binding)->inputs;
        return nullptr;
    }
    ort::OrtStatus* bindOutput(ort::OrtIoBinding* binding, const char* name, const ort::OrtValue*)
    {
        record("BindOutput");
        if (failBindOutput)
            return makeStatus(ort::ORT_FAIL, "BindOutput rejected by the mock: a pre-allocated output is not "
                                             "supported");
        boundOutputName = name ? name : "";
        ++reinterpret_cast<FakeIoBinding*>(binding)->boundOutputs;
        return nullptr;
    }
    ort::OrtStatus* bindOutputToDevice(ort::OrtIoBinding* binding, const char* name, const ort::OrtMemoryInfo*)
    {
        record("BindOutputToDevice");
        if (failBindOutputToDevice)
            return makeStatus(ort::ORT_FAIL, "BindOutputToDevice rejected by the mock");
        ++boundOutputToDeviceCalls;
        boundOutputName = name ? name : "";
        ++reinterpret_cast<FakeIoBinding*>(binding)->deviceOutputs;
        return nullptr;
    }
    ort::OrtStatus* runWithBinding(ort::OrtSession*, const ort::OrtRunOptions*, const ort::OrtIoBinding*)
    {
        record("RunWithBinding");
        ++runCalls;
        if (failRun)
            return makeStatus(ort::ORT_EP_FAIL, "RunWithBinding rejected by the mock");
        return nullptr;
    }
    ort::OrtStatus* getBoundOutputValues(const ort::OrtIoBinding*, ort::OrtAllocator*, ort::OrtValue*** values,
                                        size_t* count)
    {
        record("GetBoundOutputValues");
        if (failGetBoundOutputValues)
            return makeStatus(ort::ORT_FAIL, "GetBoundOutputValues rejected by the mock");
        static FakeMemoryInfo deviceInfo{"DML", ort::OrtDeviceAllocator, 0, ort::OrtMemTypeDefault,
                                         ort::OrtMemoryInfoDeviceType_GPU};
        auto** array = static_cast<ort::OrtValue**>(std::malloc(sizeof(ort::OrtValue*)));
        auto* value = new FakeValue{outputAllocation, &deviceInfo};
        ++liveValue;
        array[0] = reinterpret_cast<ort::OrtValue*>(value);
        *values = array;
        *count = 1;
        return nullptr;
    }
    ort::OrtStatus* getTensorMemoryInfo(const ort::OrtValue* value, const ort::OrtMemoryInfo** out)
    {
        record("GetTensorMemoryInfo");
        *out = reinterpret_cast<const ort::OrtMemoryInfo*>(reinterpret_cast<const FakeValue*>(value)->info);
        return nullptr;
    }
    ort::OrtStatus* getExecutionProviderApi(const char* providerName, uint32_t version, const void** out)
    {
        record("GetExecutionProviderApi");
        providerNameSeen = providerName ? providerName : "";
        providerVersionSeen = static_cast<int>(version);
        if (failGetProviderApi)
            return makeStatus(ort::ORT_FAIL, "DirectML execution provider is not available");
        *out = &g_dmlApi;
        return nullptr;
    }

    // ---------------------------------------------------------------
    // OrtDmlApi
    // ---------------------------------------------------------------
    ort::OrtStatus* appendDml(ort::OrtSessionOptions*, int deviceId)
    {
        record("SessionOptionsAppendExecutionProvider_DML");
        ++appendCalls;
        appendKind = "DML";
        appendDeviceId = deviceId;
        if (failAppendDml)
            return makeStatus(ort::ORT_FAIL, "SessionOptionsAppendExecutionProvider_DML rejected by the mock");
        return nullptr;
    }
    ort::OrtStatus* appendDml1(ort::OrtSessionOptions*, void* dmlDevice, ID3D12CommandQueue* queue)
    {
        record("SessionOptionsAppendExecutionProvider_DML1");
        ++appendCalls;
        appendKind = "DML1";
        appendDmlDevice = dmlDevice;
        appendQueue = queue;
        if (failAppendDml)
            return makeStatus(ort::ORT_FAIL, "SessionOptionsAppendExecutionProvider_DML1 rejected by the mock");
        return nullptr;
    }
    ort::OrtStatus* appendDml2(ort::OrtSessionOptions*, ort::OrtDmlDeviceOptions* options)
    {
        record("SessionOptionsAppendExecutionProvider_DML2");
        ++appendCalls;
        appendKind = "DML2";
        if (options) {
            appendPreference = static_cast<int>(options->Preference);
            appendFilter = static_cast<int>(options->Filter);
        }
        if (failAppendDml)
            return makeStatus(ort::ORT_FAIL, "SessionOptionsAppendExecutionProvider_DML2 rejected by the mock");
        return nullptr;
    }
    ort::OrtStatus* createGpuAllocation(ID3D12Resource* resource, void** out)
    {
        record("CreateGPUAllocationFromD3DResource");
        if (!resource)
            return makeStatus(ort::ORT_INVALID_ARGUMENT, "the D3D12 resource is null");
        void* allocation = reinterpret_cast<void*>(resource);
        allocations.push_back(allocation);
        if (allocations.size() == 2)
            outputAllocation = allocation;
        *out = allocation;
        return nullptr;
    }
    ort::OrtStatus* freeGpuAllocation(void* allocation)
    {
        record("FreeGPUAllocation");
        ++freeCalls;
        (void)allocation;
        return nullptr;
    }
    ort::OrtStatus* getD3D12Resource(ort::OrtAllocator*, void*, ID3D12Resource** out)
    {
        record("GetD3D12ResourceFromAllocation");
        // Matches upstream microsoft/onnxruntime dml_provider_factory.cc:
        // GetD3D12ResourceFromAllocation calls (*d3d_resource)->AddRef().
        ID3D12Resource* resource = fakeResource();
        resource->AddRef();
        *out = resource;
        return nullptr;
    }

    // ---------------------------------------------------------------
    // Helpers used by the tests
    // ---------------------------------------------------------------
    void checkBalanced(const char* what)
    {
        const int leaked = liveStatus + liveValue + liveMemoryInfo + liveBinding + liveSession + liveOptions +
                           liveEnv + liveTypeInfo + liveAllocator;
        ++g_checks;
        if (leaked != 0) {
            ++g_failures;
            std::printf("  FAIL [%s] %s: leaked status=%d value=%d memInfo=%d binding=%d session=%d options=%d "
                        "env=%d typeInfo=%d allocator=%d\n",
                        g_test, what, liveStatus, liveValue, liveMemoryInfo, liveBinding, liveSession, liveOptions,
                        liveEnv, liveTypeInfo, liveAllocator);
        }
        checkEqual("every OrtStatus released", releasedStatus, createdStatus);
        checkEqual("unexpected slot calls", unexpectedCalls, 0);
        checkEqual("fake D3D12 resource references balanced", g_fakeResource.refs, 0);
        checkEqual("fake D3D12 device references balanced", g_fakeDevice.refs, 0);
        checkEqual("fake D3D12 queue references balanced", g_fakeQueue.refs, 0);
    }

    // Unused slots: any call is a bug in the backend or in this mock.
    static void VEYRA_ORT_API_CALL mockUnused() { ++g_mock->unexpectedCalls; }
};

// ---------------------------------------------------------------------------
// ABI stubs. Every OrtApi/OrtDmlApi slot the backend can reach forwards to the
// mock; the static_assert block below is the real ABI test — if OrtDmlApi.h
// ever drifted from a slot signature this file would stop compiling.
// ---------------------------------------------------------------------------
ort::OrtStatus* VEYRA_ORT_API_CALL mockCreateStatus(ort::OrtErrorCode code, const char* message)
{
    return g_mock->createStatus(code, message);
}
ort::OrtErrorCode VEYRA_ORT_API_CALL mockGetErrorCode(const ort::OrtStatus* status)
{
    return g_mock->getErrorCode(status);
}
const char* VEYRA_ORT_API_CALL mockGetErrorMessage(const ort::OrtStatus* status)
{
    return g_mock->getErrorMessage(status);
}
void VEYRA_ORT_API_CALL mockReleaseStatus(ort::OrtStatus* status) { g_mock->releaseStatus(status); }

ort::OrtStatus* VEYRA_ORT_API_CALL mockCreateEnv(ort::OrtLoggingLevel level, const char* logid, ort::OrtEnv** out)
{
    return g_mock->createEnv(level, logid, out);
}
void VEYRA_ORT_API_CALL mockReleaseEnv(ort::OrtEnv* env) { g_mock->releaseEnv(env); }
ort::OrtStatus* VEYRA_ORT_API_CALL mockCreateSession(const ort::OrtEnv* env, const wchar_t* modelPath,
                                                     const ort::OrtSessionOptions* options, ort::OrtSession** out)
{
    return g_mock->createSession(env, modelPath, options, out);
}
void VEYRA_ORT_API_CALL mockReleaseSession(ort::OrtSession* session) { g_mock->releaseSession(session); }
void VEYRA_ORT_API_CALL mockReleaseValue(ort::OrtValue* value) { g_mock->releaseValue(value); }

ort::OrtStatus* VEYRA_ORT_API_CALL mockCreateSessionOptions(ort::OrtSessionOptions** out)
{
    return g_mock->createSessionOptions(out);
}
void VEYRA_ORT_API_CALL mockReleaseSessionOptions(ort::OrtSessionOptions* options)
{
    g_mock->releaseSessionOptions(options);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockSetExecutionMode(ort::OrtSessionOptions* options, ort::ExecutionMode mode)
{
    return g_mock->setExecutionMode(options, mode);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockDisableMemPattern(ort::OrtSessionOptions* options)
{
    return g_mock->disableMemPattern(options);
}

ort::OrtStatus* VEYRA_ORT_API_CALL mockSessionGetInputCount(const ort::OrtSession* session, size_t* out)
{
    return g_mock->sessionGetInputCount(session, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockSessionGetOutputCount(const ort::OrtSession* session, size_t* out)
{
    return g_mock->sessionGetOutputCount(session, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockSessionGetInputTypeInfo(const ort::OrtSession* session, size_t index,
                                                               ort::OrtTypeInfo** out)
{
    return g_mock->sessionGetInputTypeInfo(session, index, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockSessionGetOutputTypeInfo(const ort::OrtSession* session, size_t index,
                                                                ort::OrtTypeInfo** out)
{
    return g_mock->sessionGetOutputTypeInfo(session, index, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockSessionGetInputName(const ort::OrtSession* session, size_t index,
                                                           ort::OrtAllocator* allocator, char** out)
{
    return g_mock->sessionGetInputName(session, index, allocator, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockSessionGetOutputName(const ort::OrtSession* session, size_t index,
                                                            ort::OrtAllocator* allocator, char** out)
{
    return g_mock->sessionGetOutputName(session, index, allocator, out);
}

ort::OrtStatus* VEYRA_ORT_API_CALL mockCreateTensor(const ort::OrtMemoryInfo* info, void* data, size_t bytes,
                                                    const int64_t* shape, size_t shapeCount,
                                                    ort::ONNXTensorElementDataType type, ort::OrtValue** out)
{
    return g_mock->createTensor(info, data, bytes, shape, shapeCount, type, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockGetTensorMutableData(ort::OrtValue* value, void** out)
{
    return g_mock->getTensorMutableData(value, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockCastTypeInfoToTensorInfo(const ort::OrtTypeInfo* info,
                                                                const ort::OrtTensorTypeAndShapeInfo** out)
{
    return g_mock->castTypeInfoToTensorInfo(info, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockGetTensorElementType(const ort::OrtTensorTypeAndShapeInfo* shape,
                                                            ort::ONNXTensorElementDataType* out)
{
    return g_mock->getTensorElementType(shape, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockGetDimensionsCount(const ort::OrtTensorTypeAndShapeInfo* shape, size_t* out)
{
    return g_mock->getDimensionsCount(shape, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockGetDimensions(const ort::OrtTensorTypeAndShapeInfo* shape, int64_t* dims,
                                                     size_t count)
{
    return g_mock->getDimensions(shape, dims, count);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockGetSymbolicDimensions(const ort::OrtTensorTypeAndShapeInfo* shape,
                                                             const char** dimParams, size_t count)
{
    return g_mock->getSymbolicDimensions(shape, dimParams, count);
}
void VEYRA_ORT_API_CALL mockReleaseTypeInfo(ort::OrtTypeInfo* info) { g_mock->releaseTypeInfo(info); }

ort::OrtStatus* VEYRA_ORT_API_CALL mockCreateMemoryInfo(const char* name, ort::OrtAllocatorType allocatorType,
                                                        int id, ort::OrtMemType memType, ort::OrtMemoryInfo** out)
{
    return g_mock->createMemoryInfo(name, allocatorType, id, memType, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockCompareMemoryInfo(const ort::OrtMemoryInfo* lhs,
                                                         const ort::OrtMemoryInfo* rhs, int* out)
{
    return g_mock->compareMemoryInfo(lhs, rhs, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockMemoryInfoGetName(const ort::OrtMemoryInfo* info, const char** out)
{
    return g_mock->memoryInfoGetName(info, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockMemoryInfoGetId(const ort::OrtMemoryInfo* info, int* out)
{
    return g_mock->memoryInfoGetId(info, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockMemoryInfoGetMemType(const ort::OrtMemoryInfo* info, ort::OrtMemType* out)
{
    return g_mock->memoryInfoGetMemType(info, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockMemoryInfoGetType(const ort::OrtMemoryInfo* info,
                                                         ort::OrtAllocatorType* out)
{
    return g_mock->memoryInfoGetType(info, out);
}
void VEYRA_ORT_API_CALL mockMemoryInfoGetDeviceType(const ort::OrtMemoryInfo* info,
                                                    ort::OrtMemoryInfoDeviceType* out)
{
    g_mock->memoryInfoGetDeviceType(info, out);
}
void VEYRA_ORT_API_CALL mockReleaseMemoryInfo(ort::OrtMemoryInfo* info) { g_mock->releaseMemoryInfo(info); }

ort::OrtStatus* VEYRA_ORT_API_CALL mockAllocatorAlloc(ort::OrtAllocator* allocator, size_t size, void** out)
{
    return g_mock->allocatorAlloc(allocator, size, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockAllocatorFree(ort::OrtAllocator* allocator, void* pointer)
{
    return g_mock->allocatorFree(allocator, pointer);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockGetDefaultAllocator(ort::OrtAllocator** out)
{
    return g_mock->getDefaultAllocator(out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockCreateAllocator(const ort::OrtSession* session,
                                                       const ort::OrtMemoryInfo* info, ort::OrtAllocator** out)
{
    return g_mock->createAllocator(session, info, out);
}
void VEYRA_ORT_API_CALL mockReleaseAllocator(ort::OrtAllocator* allocator) { g_mock->releaseAllocator(allocator); }

ort::OrtStatus* VEYRA_ORT_API_CALL mockRunWithBinding(ort::OrtSession* session, const ort::OrtRunOptions* options,
                                                      const ort::OrtIoBinding* binding)
{
    return g_mock->runWithBinding(session, options, binding);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockCreateIoBinding(ort::OrtSession* session, ort::OrtIoBinding** out)
{
    return g_mock->createIoBinding(session, out);
}
void VEYRA_ORT_API_CALL mockReleaseIoBinding(ort::OrtIoBinding* binding) { g_mock->releaseIoBinding(binding); }
ort::OrtStatus* VEYRA_ORT_API_CALL mockBindInput(ort::OrtIoBinding* binding, const char* name,
                                                 const ort::OrtValue* value)
{
    return g_mock->bindInput(binding, name, value);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockBindOutput(ort::OrtIoBinding* binding, const char* name,
                                                  const ort::OrtValue* value)
{
    return g_mock->bindOutput(binding, name, value);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockBindOutputToDevice(ort::OrtIoBinding* binding, const char* name,
                                                          const ort::OrtMemoryInfo* info)
{
    return g_mock->bindOutputToDevice(binding, name, info);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockGetBoundOutputValues(const ort::OrtIoBinding* binding,
                                                            ort::OrtAllocator* allocator, ort::OrtValue*** values,
                                                            size_t* count)
{
    return g_mock->getBoundOutputValues(binding, allocator, values, count);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockGetTensorMemoryInfo(const ort::OrtValue* value,
                                                           const ort::OrtMemoryInfo** out)
{
    return g_mock->getTensorMemoryInfo(value, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockGetExecutionProviderApi(const char* name, uint32_t version,
                                                               const void** out)
{
    return g_mock->getExecutionProviderApi(name, version, out);
}

ort::OrtStatus* VEYRA_ORT_API_CALL mockAppendDml(ort::OrtSessionOptions* options, int deviceId)
{
    return g_mock->appendDml(options, deviceId);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockAppendDml1(ort::OrtSessionOptions* options, void* dmlDevice,
                                                  ID3D12CommandQueue* queue)
{
    return g_mock->appendDml1(options, dmlDevice, queue);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockAppendDml2(ort::OrtSessionOptions* options,
                                                  ort::OrtDmlDeviceOptions* deviceOptions)
{
    return g_mock->appendDml2(options, deviceOptions);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockCreateGpuAllocation(ID3D12Resource* resource, void** out)
{
    return g_mock->createGpuAllocation(resource, out);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockFreeGpuAllocation(void* allocation)
{
    return g_mock->freeGpuAllocation(allocation);
}
ort::OrtStatus* VEYRA_ORT_API_CALL mockGetD3D12Resource(ort::OrtAllocator* allocator, void* allocation,
                                                        ID3D12Resource** out)
{
    return g_mock->getD3D12Resource(allocator, allocation, out);
}

void Mock::install()
{
    auto set = [this](size_t slot, void* fn) { table[slot] = fn; };
    set(ort::kSlot_CreateStatus, fnToVoid(&mockCreateStatus));
    set(ort::kSlot_GetErrorCode, fnToVoid(&mockGetErrorCode));
    set(ort::kSlot_GetErrorMessage, fnToVoid(&mockGetErrorMessage));
    set(ort::kSlot_ReleaseStatus, fnToVoid(&mockReleaseStatus));
    set(ort::kSlot_CreateEnv, fnToVoid(&mockCreateEnv));
    set(ort::kSlot_ReleaseEnv, fnToVoid(&mockReleaseEnv));
    set(ort::kSlot_CreateSession, fnToVoid(&mockCreateSession));
    set(ort::kSlot_ReleaseSession, fnToVoid(&mockReleaseSession));
    set(ort::kSlot_ReleaseValue, fnToVoid(&mockReleaseValue));
    set(ort::kSlot_CreateSessionOptions, fnToVoid(&mockCreateSessionOptions));
    set(ort::kSlot_ReleaseSessionOptions, fnToVoid(&mockReleaseSessionOptions));
    set(ort::kSlot_SetSessionExecutionMode, fnToVoid(&mockSetExecutionMode));
    set(ort::kSlot_DisableMemPattern, fnToVoid(&mockDisableMemPattern));
    set(ort::kSlot_SessionGetInputCount, fnToVoid(&mockSessionGetInputCount));
    set(ort::kSlot_SessionGetOutputCount, fnToVoid(&mockSessionGetOutputCount));
    set(ort::kSlot_SessionGetInputTypeInfo, fnToVoid(&mockSessionGetInputTypeInfo));
    set(ort::kSlot_SessionGetOutputTypeInfo, fnToVoid(&mockSessionGetOutputTypeInfo));
    set(ort::kSlot_SessionGetInputName, fnToVoid(&mockSessionGetInputName));
    set(ort::kSlot_SessionGetOutputName, fnToVoid(&mockSessionGetOutputName));
    set(ort::kSlot_CreateTensorWithDataAsOrtValue, fnToVoid(&mockCreateTensor));
    set(ort::kSlot_GetTensorMutableData, fnToVoid(&mockGetTensorMutableData));
    set(ort::kSlot_CastTypeInfoToTensorInfo, fnToVoid(&mockCastTypeInfoToTensorInfo));
    set(ort::kSlot_GetTensorElementType, fnToVoid(&mockGetTensorElementType));
    set(ort::kSlot_GetDimensionsCount, fnToVoid(&mockGetDimensionsCount));
    set(ort::kSlot_GetDimensions, fnToVoid(&mockGetDimensions));
    set(ort::kSlot_GetSymbolicDimensions, fnToVoid(&mockGetSymbolicDimensions));
    set(ort::kSlot_ReleaseTypeInfo, fnToVoid(&mockReleaseTypeInfo));
    set(ort::kSlot_CreateMemoryInfo, fnToVoid(&mockCreateMemoryInfo));
    set(ort::kSlot_CompareMemoryInfo, fnToVoid(&mockCompareMemoryInfo));
    set(ort::kSlot_MemoryInfoGetName, fnToVoid(&mockMemoryInfoGetName));
    set(ort::kSlot_MemoryInfoGetId, fnToVoid(&mockMemoryInfoGetId));
    set(ort::kSlot_MemoryInfoGetMemType, fnToVoid(&mockMemoryInfoGetMemType));
    set(ort::kSlot_MemoryInfoGetType, fnToVoid(&mockMemoryInfoGetType));
    set(ort::kSlot_MemoryInfoGetDeviceType, fnToVoid(&mockMemoryInfoGetDeviceType));
    set(ort::kSlot_ReleaseMemoryInfo, fnToVoid(&mockReleaseMemoryInfo));
    set(ort::kSlot_AllocatorAlloc, fnToVoid(&mockAllocatorAlloc));
    set(ort::kSlot_AllocatorFree, fnToVoid(&mockAllocatorFree));
    set(ort::kSlot_GetAllocatorWithDefaultOptions, fnToVoid(&mockGetDefaultAllocator));
    set(ort::kSlot_CreateAllocator, fnToVoid(&mockCreateAllocator));
    set(ort::kSlot_ReleaseAllocator, fnToVoid(&mockReleaseAllocator));
    set(ort::kSlot_RunWithBinding, fnToVoid(&mockRunWithBinding));
    set(ort::kSlot_CreateIoBinding, fnToVoid(&mockCreateIoBinding));
    set(ort::kSlot_ReleaseIoBinding, fnToVoid(&mockReleaseIoBinding));
    set(ort::kSlot_BindInput, fnToVoid(&mockBindInput));
    set(ort::kSlot_BindOutput, fnToVoid(&mockBindOutput));
    set(ort::kSlot_BindOutputToDevice, fnToVoid(&mockBindOutputToDevice));
    set(ort::kSlot_GetBoundOutputValues, fnToVoid(&mockGetBoundOutputValues));
    set(ort::kSlot_GetTensorMemoryInfo, fnToVoid(&mockGetTensorMemoryInfo));
    set(ort::kSlot_GetExecutionProviderApi, fnToVoid(&mockGetExecutionProviderApi));

    g_dmlApi.SessionOptionsAppendExecutionProvider_DML = &mockAppendDml;
    g_dmlApi.SessionOptionsAppendExecutionProvider_DML1 = omitDml1 ? nullptr : &mockAppendDml1;
    g_dmlApi.SessionOptionsAppendExecutionProvider_DML2 = omitDml2 ? nullptr : &mockAppendDml2;
    g_dmlApi.CreateGPUAllocationFromD3DResource = stripDmlBridge ? nullptr : &mockCreateGpuAllocation;
    g_dmlApi.FreeGPUAllocation = stripDmlBridge ? nullptr : &mockFreeGpuAllocation;
    g_dmlApi.GetD3D12ResourceFromAllocation = stripDmlBridge ? nullptr : &mockGetD3D12Resource;

}

// The ABI gate: each stub must have exactly the signature the header declares.
static_assert(std::is_same_v<decltype(&mockCreateStatus), ort::FnCreateStatus>, "CreateStatus ABI");
static_assert(std::is_same_v<decltype(&mockGetErrorCode), ort::FnGetErrorCode>, "GetErrorCode ABI");
static_assert(std::is_same_v<decltype(&mockGetErrorMessage), ort::FnGetErrorMessage>, "GetErrorMessage ABI");
static_assert(std::is_same_v<decltype(&mockReleaseStatus), ort::FnReleaseStatus>, "ReleaseStatus ABI");
static_assert(std::is_same_v<decltype(&mockCreateEnv), ort::FnCreateEnv>, "CreateEnv ABI");
static_assert(std::is_same_v<decltype(&mockReleaseEnv), ort::FnReleaseEnv>, "ReleaseEnv ABI");
static_assert(std::is_same_v<decltype(&mockCreateSession), ort::FnCreateSession>, "CreateSession ABI");
static_assert(std::is_same_v<decltype(&mockReleaseSession), ort::FnReleaseSession>, "ReleaseSession ABI");
static_assert(std::is_same_v<decltype(&mockReleaseValue), ort::FnReleaseValue>, "ReleaseValue ABI");
static_assert(std::is_same_v<decltype(&mockCreateSessionOptions), ort::FnCreateSessionOptions>,
              "CreateSessionOptions ABI");
static_assert(std::is_same_v<decltype(&mockReleaseSessionOptions), ort::FnReleaseSessionOptions>,
              "ReleaseSessionOptions ABI");
static_assert(std::is_same_v<decltype(&mockSetExecutionMode), ort::FnSetSessionExecutionMode>,
              "SetSessionExecutionMode ABI");
static_assert(std::is_same_v<decltype(&mockDisableMemPattern), ort::FnDisableMemPattern>, "DisableMemPattern ABI");
static_assert(std::is_same_v<decltype(&mockSessionGetInputCount), ort::FnSessionGetInputCount>,
              "SessionGetInputCount ABI");
static_assert(std::is_same_v<decltype(&mockSessionGetOutputCount), ort::FnSessionGetOutputCount>,
              "SessionGetOutputCount ABI");
static_assert(std::is_same_v<decltype(&mockSessionGetInputTypeInfo), ort::FnSessionGetInputTypeInfo>,
              "SessionGetInputTypeInfo ABI");
static_assert(std::is_same_v<decltype(&mockSessionGetOutputTypeInfo), ort::FnSessionGetOutputTypeInfo>,
              "SessionGetOutputTypeInfo ABI");
static_assert(std::is_same_v<decltype(&mockSessionGetInputName), ort::FnSessionGetInputName>,
              "SessionGetInputName ABI");
static_assert(std::is_same_v<decltype(&mockSessionGetOutputName), ort::FnSessionGetOutputName>,
              "SessionGetOutputName ABI");
static_assert(std::is_same_v<decltype(&mockCreateTensor), ort::FnCreateTensorWithDataAsOrtValue>,
              "CreateTensorWithDataAsOrtValue ABI");
static_assert(std::is_same_v<decltype(&mockGetTensorMutableData), ort::FnGetTensorMutableData>,
              "GetTensorMutableData ABI");
static_assert(std::is_same_v<decltype(&mockCastTypeInfoToTensorInfo), ort::FnCastTypeInfoToTensorInfo>,
              "CastTypeInfoToTensorInfo ABI");
static_assert(std::is_same_v<decltype(&mockGetTensorElementType), ort::FnGetTensorElementType>,
              "GetTensorElementType ABI");
static_assert(std::is_same_v<decltype(&mockGetDimensionsCount), ort::FnGetDimensionsCount>, "GetDimensionsCount ABI");
static_assert(std::is_same_v<decltype(&mockGetDimensions), ort::FnGetDimensions>, "GetDimensions ABI");
static_assert(std::is_same_v<decltype(&mockGetSymbolicDimensions), ort::FnGetSymbolicDimensions>,
              "GetSymbolicDimensions ABI");
static_assert(std::is_same_v<decltype(&mockReleaseTypeInfo), ort::FnReleaseTypeInfo>, "ReleaseTypeInfo ABI");
static_assert(std::is_same_v<decltype(&mockCreateMemoryInfo), ort::FnCreateMemoryInfo>, "CreateMemoryInfo ABI");
static_assert(std::is_same_v<decltype(&mockCompareMemoryInfo), ort::FnCompareMemoryInfo>, "CompareMemoryInfo ABI");
static_assert(std::is_same_v<decltype(&mockMemoryInfoGetName), ort::FnMemoryInfoGetName>, "MemoryInfoGetName ABI");
static_assert(std::is_same_v<decltype(&mockMemoryInfoGetId), ort::FnMemoryInfoGetId>, "MemoryInfoGetId ABI");
static_assert(std::is_same_v<decltype(&mockMemoryInfoGetMemType), ort::FnMemoryInfoGetMemType>,
              "MemoryInfoGetMemType ABI");
static_assert(std::is_same_v<decltype(&mockMemoryInfoGetType), ort::FnMemoryInfoGetType>, "MemoryInfoGetType ABI");
static_assert(std::is_same_v<decltype(&mockMemoryInfoGetDeviceType), ort::FnMemoryInfoGetDeviceType>,
              "MemoryInfoGetDeviceType ABI");
static_assert(std::is_same_v<decltype(&mockReleaseMemoryInfo), ort::FnReleaseMemoryInfo>, "ReleaseMemoryInfo ABI");
static_assert(std::is_same_v<decltype(&mockAllocatorAlloc), ort::FnAllocatorAlloc>, "AllocatorAlloc ABI");
static_assert(std::is_same_v<decltype(&mockAllocatorFree), ort::FnAllocatorFree>, "AllocatorFree ABI");
static_assert(std::is_same_v<decltype(&mockGetDefaultAllocator), ort::FnGetAllocatorWithDefaultOptions>,
              "GetAllocatorWithDefaultOptions ABI");
static_assert(std::is_same_v<decltype(&mockCreateAllocator), ort::FnCreateAllocator>, "CreateAllocator ABI");
static_assert(std::is_same_v<decltype(&mockReleaseAllocator), ort::FnReleaseAllocator>, "ReleaseAllocator ABI");
static_assert(std::is_same_v<decltype(&mockRunWithBinding), ort::FnRunWithBinding>, "RunWithBinding ABI");
static_assert(std::is_same_v<decltype(&mockCreateIoBinding), ort::FnCreateIoBinding>, "CreateIoBinding ABI");
static_assert(std::is_same_v<decltype(&mockReleaseIoBinding), ort::FnReleaseIoBinding>, "ReleaseIoBinding ABI");
static_assert(std::is_same_v<decltype(&mockBindInput), ort::FnBindInput>, "BindInput ABI");
static_assert(std::is_same_v<decltype(&mockBindOutput), ort::FnBindOutput>, "BindOutput ABI");
static_assert(std::is_same_v<decltype(&mockBindOutputToDevice), ort::FnBindOutputToDevice>,
              "BindOutputToDevice ABI");
static_assert(std::is_same_v<decltype(&mockGetBoundOutputValues), ort::FnGetBoundOutputValues>,
              "GetBoundOutputValues ABI");
static_assert(std::is_same_v<decltype(&mockGetTensorMemoryInfo), ort::FnGetTensorMemoryInfo>,
              "GetTensorMemoryInfo ABI");
static_assert(std::is_same_v<decltype(&mockGetExecutionProviderApi), ort::FnGetExecutionProviderApi>,
              "GetExecutionProviderApi ABI");

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
Mock g_mockInstance;

const void* VEYRA_ORT_API_CALL testGetApi(uint32_t version) { return g_mock->apiForVersion(version); }
const char* VEYRA_ORT_API_CALL testGetVersionString() { return "1.20.1-veyra-test"; }

const ort::OrtApiBase g_apiBase{&testGetApi, &testGetVersionString};

std::filesystem::path writeModelFile()
{
    const auto path = std::filesystem::temp_directory_path() / "veyra_ort_dml_session_test.onnx";
    std::FILE* file = std::fopen(path.string().c_str(), "wb");
    if (!file)
        return {};
    const size_t written =
        std::fwrite(ort::kMinimalIdentityOnnxModel, 1, ort::kMinimalIdentityOnnxModelSize, file);
    std::fclose(file);
    if (written != ort::kMinimalIdentityOnnxModelSize)
        return {};
    return path;
}

OrtDmlSrBackend::Config baseConfig(const std::filesystem::path& model)
{
    OrtDmlSrBackend::Config config;
    config.modelPath = model;
    config.testApiBase = &g_apiBase;
    config.testDmlApi = &g_dmlApi;
    config.skipD3dResourcesForTest = true;
    config.channels = 3;
    return config;
}

void runTest(const char* name, void (*test)())
{
    g_test = name;
    std::printf("-- %s\n", name);
    test();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
void testAbiTable()
{
    g_mock->reset();
    checkEqual("sizeof(OrtApiBase)", (long long)sizeof(ort::OrtApiBase), 2LL * (long long)sizeof(void*));
    checkEqual("sizeof(OrtDmlApi)", (long long)sizeof(ort::OrtDmlApi), 6LL * (long long)sizeof(void*));
    checkEqual("sizeof(OrtDmlDeviceOptions)", (long long)sizeof(ort::OrtDmlDeviceOptions),
               2LL * (long long)sizeof(int32_t));
    checkEqual("OrtApi v20", (long long)ort::kOrtApiVersion, 20);
    checkEqual("OrtApi slot count", (long long)ort::kOrtApiSlotCount, 285);
    checkEqual("slot CreateStatus", (long long)ort::kSlot_CreateStatus, 0);
    checkEqual("slot GetErrorMessage", (long long)ort::kSlot_GetErrorMessage, 2);
    checkEqual("slot CreateEnv", (long long)ort::kSlot_CreateEnv, 3);
    checkEqual("slot CreateSession", (long long)ort::kSlot_CreateSession, 7);
    checkEqual("slot CreateSessionOptions", (long long)ort::kSlot_CreateSessionOptions, 10);
    checkEqual("slot DisableMemPattern", (long long)ort::kSlot_DisableMemPattern, 17);
    checkEqual("slot CreateTensorWithDataAsOrtValue", (long long)ort::kSlot_CreateTensorWithDataAsOrtValue, 49);
    checkEqual("slot CreateMemoryInfo", (long long)ort::kSlot_CreateMemoryInfo, 68);
    checkEqual("slot ReleaseEnv", (long long)ort::kSlot_ReleaseEnv, 92);
    checkEqual("slot ReleaseSession", (long long)ort::kSlot_ReleaseSession, 95);
    checkEqual("slot ReleaseValue", (long long)ort::kSlot_ReleaseValue, 96);
    checkEqual("slot CreateAllocator", (long long)ort::kSlot_CreateAllocator, 131);
    checkEqual("slot RunWithBinding", (long long)ort::kSlot_RunWithBinding, 133);
    checkEqual("slot CreateIoBinding", (long long)ort::kSlot_CreateIoBinding, 134);
    checkEqual("slot BindOutputToDevice", (long long)ort::kSlot_BindOutputToDevice, 138);
    checkEqual("slot GetBoundOutputValues", (long long)ort::kSlot_GetBoundOutputValues, 140);
    checkEqual("slot GetTensorMemoryInfo", (long long)ort::kSlot_GetTensorMemoryInfo, 194);
    checkEqual("slot GetExecutionProviderApi", (long long)ort::kSlot_GetExecutionProviderApi, 195);
    checkEqual("slot MemoryInfoGetDeviceType", (long long)ort::kSlot_MemoryInfoGetDeviceType, 225);

    checkEqual("ORT_SEQUENTIAL", (long long)ort::ORT_SEQUENTIAL, 0);
    checkEqual("ORT_LOGGING_LEVEL_WARNING", (long long)ort::ORT_LOGGING_LEVEL_WARNING, 2);
    checkEqual("float32 element type", (long long)ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 1);
    checkEqual("OrtDeviceAllocator", (long long)ort::OrtDeviceAllocator, 0);
    checkEqual("OrtMemTypeDefault", (long long)ort::OrtMemTypeDefault, 0);
    checkEqual("GPU memory info type", (long long)ort::OrtMemoryInfoDeviceType_GPU, 1);
    checkEqual("DmlPreferenceHighPerformance", (long long)ort::DmlPreferenceHighPerformance, 1);
    checkEqual("DmlFilterGpu", (long long)ort::DmlFilterGpu, 1);
    checkEqual("ORT_EP_FAIL", (long long)ort::ORT_EP_FAIL, 11);

    const ort::OrtApiView view(g_mock->table.data());
    check("vtable view is valid", view.valid());
    check("vtable view resolves a slot", view.slot<ort::FnCreateStatus>(ort::kSlot_CreateStatus) != nullptr);
    const ort::OrtApiView empty;
    check("empty view is invalid", !empty.valid());
    check("empty view yields null slots", empty.slot<ort::FnCreateStatus>(0) == nullptr);

    ort::OrtStatus* status = g_mock->makeStatus(ort::ORT_FAIL, "boom");
    const std::string text = ort::consumeStatus(view, status);
    checkText("consumeStatus keeps the message", text, "boom");
    checkEqual("consumeStatus appends the numeric code", text.find("ORT code") != std::string::npos, 1);
    checkEqual("consumeStatus released the status", g_mock->liveStatus, 0);
    check("consumeStatus(nullptr) is empty", ort::consumeStatus(view, nullptr).empty());
    g_mock->checkBalanced("abi table");
}

void testModelFixture()
{
    g_mock->reset();
    const std::vector<uint8_t> bytes(ort::kMinimalIdentityOnnxModel,
                                     ort::kMinimalIdentityOnnxModel + ort::kMinimalIdentityOnnxModelSize);
    checkEqual("fixture byte count", (long long)bytes.size(), 115);
    veyra::beauty::OnnxModel model;
    std::string error;
    check("fixture parses as ONNX", veyra::beauty::parseOnnx(bytes, model, error));
    checkEqual("fixture opset", model.opset, 11);
    checkEqual("fixture node count", (long long)model.nodes.size(), 1);
    checkText("fixture op type", model.nodes[0].op, "Identity");
    checkEqual("fixture node inputs", (long long)model.nodes[0].inputs.size(), 1);
    checkText("fixture node input name", model.nodes[0].inputs[0], "x");
    checkText("fixture node output name", model.nodes[0].outputs[0], "y");
    checkEqual("fixture input count", (long long)model.inputs.size(), 1);
    checkText("fixture input name", model.inputs[0].name, "x");
    checkEqual("fixture output count", (long long)model.outputs.size(), 1);
    checkText("fixture output name", model.outputs[0].name, "y");
    checkEqual("fixture input rank", (long long)model.inputs[0].dims.size(), 4);
    checkEqual("fixture dim 0", model.inputs[0].dims[0], 1);
    checkEqual("fixture dim 1", model.inputs[0].dims[1], 1);
    checkText("fixture producer", model.producer, "veyra-ort-dml");

    const auto path = writeModelFile();
    check("fixture file written", !path.empty());
    check("fixture file exists", std::filesystem::is_regular_file(path));
    checkEqual("fixture file size", (long long)std::filesystem::file_size(path), 115);
    g_mock->checkBalanced("model fixture");
}

void testOpenPlumbing()
{
    g_mock->reset();
    const auto model = writeModelFile();
    check("model file written", std::filesystem::is_regular_file(model));

    OrtDmlSrBackend backend;
    check("fresh backend is not ready", !backend.ready());
    check("fresh backend is not prepared", !backend.prepared());
    check("module name is onnxruntime.dll", std::wstring(OrtDmlSrBackend::moduleName()) == L"onnxruntime.dll");
    checkEqual("channels default", (long long)backend.channels(), 3);
    checkEqual("runs default", (long long)backend.stats().runs, 0);

    auto config = baseConfig(model);
    // Exercise the real provider lookup (GetExecutionProviderApi) instead of
    // the injected table, so the provider name and version are covered here.
    config.testDmlApi = nullptr;
    check("open succeeds", backend.open(config));
    check("backend is ready", backend.ready());
    check("backend is not prepared", !backend.prepared());
    check("error is empty after a successful open", backend.error().empty());
    check("warning is empty after a successful open", backend.warning().empty());
    checkText("runtime version", backend.runtimeVersion(), "1.20.1-veyra-test");
    checkText("input name", backend.inputName(), "input");
    checkText("output name", backend.outputName(), "output");
    checkText("warning-free session", backend.warning(), "");
    checkEqual("channels", (long long)backend.channels(), 3);

    checkEqual("env log level", g_mock->logLevel, (long long)ort::ORT_LOGGING_LEVEL_WARNING);
    checkText("env log id", g_mock->logId, "veyra-ort-dml");
    checkEqual("execution mode", g_mock->executionMode, 0);
    checkEqual("memory pattern disabled", (long long)g_mock->memPattern, 0);
    checkText("execution provider appended", g_mock->appendKind, "DML");
    checkEqual("device id forwarded to the provider", g_mock->appendDeviceId, 0);
    checkEqual("one provider append", g_mock->appendCalls, 1);
    checkText("provider name queried", g_mock->providerNameSeen, "DML");
    checkEqual("provider queried with the C API version", g_mock->providerVersionSeen, 20);
    checkEqual("provider api queries", g_mock->count("GetExecutionProviderApi"), 1);
    check("model path handed to CreateSession", g_mock->modelPathSeen == model.wstring());

    check("session created after CreateEnv", g_mock->happenedBefore("CreateEnv", "CreateSession"));
    check("options created before the session", g_mock->happenedBefore("CreateSessionOptions", "CreateSession"));
    check("memory pattern disabled before the session",
          g_mock->happenedBefore("DisableMemPattern", "CreateSession"));
    check("memory pattern disabled after the options exist",
          g_mock->happenedBefore("CreateSessionOptions", "DisableMemPattern"));
    check("sequential mode set after the memory pattern",
          g_mock->happenedBefore("DisableMemPattern", "SetSessionExecutionMode"));
    check("provider appended after the execution mode",
          g_mock->happenedBefore("SetSessionExecutionMode", "SessionOptionsAppendExecutionProvider_DML"));
    check("provider appended before the session is created",
          g_mock->happenedBefore("SessionOptionsAppendExecutionProvider_DML", "CreateSession"));
    checkEqual("one live session", g_mock->liveSession, 1);
    checkEqual("one live env", g_mock->liveEnv, 1);
    checkEqual("one live options object", g_mock->liveOptions, 1);

    backend.shutdown();
    check("shutdown releases the session", !backend.ready());
    check("shutdown keeps the error text empty", backend.error().empty());
    g_mock->checkBalanced("open/shutdown");
}

void testPrepareAndRun()
{
    g_mock->reset();
    const auto model = writeModelFile();
    OrtDmlSrBackend backend;
    auto config = baseConfig(model);
    check("open", backend.open(config));
    check("prepare", backend.prepare(64, 64, 64, 64));
    check("prepared", backend.prepared());
    checkEqual("input width", (long long)backend.inputWidth(), 64);
    checkEqual("input height", (long long)backend.inputHeight(), 64);
    checkEqual("output width", (long long)backend.outputWidth(), 64);
    checkEqual("output height", (long long)backend.outputHeight(), 64);

    checkEqual("two tensors created", g_mock->tensorCalls, 2);
    checkEqual("input tensor bytes", (long long)g_mock->tensorBytes[0], 64LL * 64LL * 3LL * 4LL);
    checkEqual("output tensor bytes", (long long)g_mock->tensorBytes[1], 64LL * 64LL * 3LL * 4LL);
    checkEqual("input tensor rank", (long long)g_mock->tensorShapes[0].size(), 4);
    checkEqual("input batch", g_mock->tensorShapes[0][0], 1);
    checkEqual("input channels", g_mock->tensorShapes[0][1], 3);
    checkEqual("input rows", g_mock->tensorShapes[0][2], 64);
    checkEqual("input cols", g_mock->tensorShapes[0][3], 64);
    checkEqual("output tensor rows", g_mock->tensorShapes[1][2], 64);
    checkText("DirectML memory info name", g_mock->memoryInfoName, "DML");
    checkEqual("memory info device id", g_mock->memoryInfoId, 0);
    checkEqual("memory info allocator", g_mock->memoryInfoAllocatorType, 0);
    checkEqual("memory info mem type", g_mock->memoryInfoMemType, 0);
    checkEqual("two GPU allocations", (long long)g_mock->allocations.size(), 2);
    checkText("bound input", g_mock->boundInputName, "input");
    checkText("bound output", g_mock->boundOutputName, "output");
    check("input bound before output", g_mock->happenedBefore("BindInput", "BindOutput"));
    check("no fallback", !backend.usesFallback());
    check("memory info created before the tensors",
          g_mock->happenedBefore("CreateMemoryInfo", "CreateTensorWithDataAsOrtValue"));
    check("tensors created before the binding",
          g_mock->happenedBefore("CreateTensorWithDataAsOrtValue", "CreateIoBinding"));
    check("host-only tensors in test mode", backend.inputBuffer() == nullptr && backend.outputBuffer() == nullptr);
    check("no output resource before run", backend.outputResource() == nullptr);

    check("run", backend.run());
    checkEqual("run count", (long long)backend.stats().runs, 1);
    check("stats agree on the primary path", !backend.stats().usedFallback);
    check("run again", backend.run());
    checkEqual("run count after the second run", (long long)backend.stats().runs, 2);
    checkEqual("runtime ran the graph twice", g_mock->runCalls, 2);
    check("still no output resource in test mode", backend.outputResource() == nullptr);

    check("preprocess without a list is rejected", !backend.recordPreprocess(nullptr, nullptr));
    checkText("preprocess list message", backend.error(), "recordPreprocess() without a command list");
    check("preprocess without a source is rejected", !backend.recordPreprocess(fakeList(), nullptr));
    checkText("preprocess source message", backend.error(), "recordPreprocess() without a source texture");
    check("preprocess without a pipeline is rejected", !backend.recordPreprocess(fakeList(), fakeResource()));
    checkText("preprocess pipeline message", backend.error(), "the preprocess pipeline is not available");
    check("postprocess without a list is rejected", !backend.recordPostprocess(nullptr, nullptr, nullptr));
    checkText("postprocess list message", backend.error(), "recordPostprocess() without a command list");
    check("postprocess without an output is rejected", !backend.recordPostprocess(fakeList(), nullptr, nullptr));
    checkText("postprocess output message", backend.error(), "recordPostprocess() without an output texture");
    check("postprocess without a pipeline is rejected", !backend.recordPostprocess(fakeList(), fakeResource(), nullptr));
    checkText("postprocess pipeline message", backend.error(), "the postprocess pipeline is not available");

    backend.shutdown();
    checkEqual("both allocations freed", g_mock->freeCalls, 2);
    check("binding released before the values", g_mock->happenedBefore("ReleaseIoBinding", "ReleaseValue"));
    check("values released before the memory info", g_mock->happenedBefore("ReleaseValue", "ReleaseMemoryInfo"));
    check("memory info released before the session", g_mock->happenedBefore("ReleaseMemoryInfo", "ReleaseSession"));
    check("session released before the options", g_mock->happenedBefore("ReleaseSession", "ReleaseSessionOptions"));
    check("options released before the env", g_mock->happenedBefore("ReleaseSessionOptions", "ReleaseEnv"));
    check("session released before the allocations", g_mock->happenedBefore("ReleaseSession", "FreeGPUAllocation"));
    check("session state cleared", !backend.prepared());
    g_mock->checkBalanced("prepare/run/shutdown");
}

void testDml1Path()
{
    g_mock->reset();
    const auto model = writeModelFile();
    OrtDmlSrBackend backend;
    auto config = baseConfig(model);
    config.testDmlDevice = reinterpret_cast<void*>(static_cast<uintptr_t>(0x2000));
    check("open with a shared DML device and queue", backend.open(config, fakeDevice(), fakeQueue()));
    checkText("DML1 is used when a device and queue are available", g_mock->appendKind, "DML1");
    checkEqual("DML device forwarded", (long long)reinterpret_cast<uintptr_t>(g_mock->appendDmlDevice), 0x2000);
    check("command queue forwarded", g_mock->appendQueue == fakeQueue());
    check("no fallback warning", backend.warning().empty());
    checkEqual("device held once", g_fakeDevice.refs, 1);
    checkEqual("queue held once", g_fakeQueue.refs, 1);
    backend.shutdown();
    checkEqual("device released", g_fakeDevice.refs, 0);
    checkEqual("queue released", g_fakeQueue.refs, 0);
    g_mock->checkBalanced("dml1");
}

void testDml2Path()
{
    g_mock->reset();
    const auto model = writeModelFile();
    OrtDmlSrBackend backend;
    auto config = baseConfig(model);
    config.useDml2 = true;
    config.createDmlDevice = false;
    check("open with DML2", backend.open(config));
    checkText("DML2 append", g_mock->appendKind, "DML2");
    checkEqual("high performance preference", g_mock->appendPreference, 1);
    checkEqual("GPU filter", g_mock->appendFilter, 1);
    backend.shutdown();
    g_mock->checkBalanced("dml2");

    g_mock->reset();
    g_mock->omitDml2 = true;
    g_mock->install();
    OrtDmlSrBackend missing;
    check("open fails when DML2 is missing", !missing.open(config));
    checkText("missing DML2 message", missing.error(), "no DML2 entry point");
    g_mock->checkBalanced("dml2 missing");
}

void testDmlLegacyPath()
{
    g_mock->reset();
    const auto model = writeModelFile();
    OrtDmlSrBackend backend;
    auto config = baseConfig(model);
    config.deviceId = 3;
    config.dmlMemoryInfoName = "DML2";
    check("open with a device id", backend.open(config));
    checkText("legacy DML append", g_mock->appendKind, "DML");
    checkEqual("device id forwarded", g_mock->appendDeviceId, 3);
    check("prepare", backend.prepare(16, 16, 16, 16));
    checkText("custom DirectML memory info name", g_mock->memoryInfoName, "DML2");
    checkEqual("custom memory info id", g_mock->memoryInfoId, 3);
    backend.shutdown();
    g_mock->checkBalanced("dml legacy");
}

void testDml1Missing()
{
    g_mock->reset();
    const auto model = writeModelFile();
    OrtDmlSrBackend backend;
    auto config = baseConfig(model);
    config.testDmlDevice = reinterpret_cast<void*>(static_cast<uintptr_t>(0x2000));
    g_mock->omitDml1 = true;
    g_mock->install();
    check("open fails when DML1 is missing", !backend.open(config, fakeDevice(), fakeQueue()));
    checkText("missing DML1 message", backend.error(), "no DML1 entry point");
    g_mock->checkBalanced("dml1 missing");
}

void testRuntimeRejection()
{
    g_mock->reset();
    const auto model = writeModelFile();
    OrtDmlSrBackend backend;
    auto config = baseConfig(model);
    g_mock->acceptApiVersion = false;
    check("open fails without OrtApi v20", !backend.open(config));
    checkText("api version message", backend.error(), "does not provide C API version 20");
    check("not ready after the failure", !backend.ready());
    g_mock->checkBalanced("api version");
}

void testProviderApiFailure()
{
    g_mock->reset();
    const auto model = writeModelFile();
    OrtDmlSrBackend backend;
    auto config = baseConfig(model);
    // The injected table would bypass GetExecutionProviderApi, so ask the
    // backend to look the provider up in the mocked API table instead.
    config.testDmlApi = nullptr;
    g_mock->failGetProviderApi = true;
    check("open fails without the DirectML provider", !backend.open(config));
    checkText("provider message", backend.error(), "DirectML execution provider is unavailable");
    g_mock->checkBalanced("provider api");

    g_mock->reset();
    g_mock->stripDmlBridge = true;
    g_mock->install();
    OrtDmlSrBackend incomplete;
    check("open fails with an incomplete DirectML table", !incomplete.open(config));
    checkText("incomplete table message", incomplete.error(), "DirectML API table is incomplete");
    g_mock->checkBalanced("dml bridge");
}

void testBindOutputFallback()
{
    g_mock->reset();
    const auto model = writeModelFile();
    OrtDmlSrBackend backend;
    auto config = baseConfig(model);
    g_mock->failBindOutput = true;
    check("open", backend.open(config));
    check("prepare falls back", backend.prepare(32, 32, 32, 32));
    check("fallback recorded", backend.usesFallback());
    checkEqual("one device-side output binding", g_mock->boundOutputToDeviceCalls, 1);
    checkEqual("one DirectML allocator", g_mock->count("CreateAllocator"), 1);
    checkText("warning explains the fallback", backend.warning(), "refused the pre-allocated output binding");
    checkText("warning promises zero readback", backend.warning(), "zero readback");
    check("no output resource before run", backend.outputResource() == nullptr);
    check("run", backend.run());
    checkEqual("bound outputs queried once", g_mock->count("GetBoundOutputValues"), 1);
    check("output resource resolved", backend.outputResource() == fakeResource());
    checkEqual("resolved resource held once", g_fakeResource.refs, 1);
    checkEqual("fallback OrtValue kept alive for postprocess (input+output+fallback)", g_mock->liveValue, 3);
    check("run again", backend.run());
    checkEqual("the resource is not re-referenced", g_fakeResource.refs, 1);
    checkEqual("previous fallback OrtValue replaced without leaking", g_mock->liveValue, 3);
    checkEqual("bound values queried once per run", g_mock->count("GetBoundOutputValues"), 2);
    checkEqual("runs", (long long)backend.stats().runs, 2);
    backend.shutdown();
    checkEqual("resolved resource released", g_fakeResource.refs, 0);
    checkEqual("both allocations freed", g_mock->freeCalls, 2);
    g_mock->checkBalanced("bind output fallback");
}

void testBoundOutputFailures()
{
    // The runtime refuses the output and the device binding as well.
    {
        g_mock->reset();
        const auto model = writeModelFile();
        OrtDmlSrBackend backend;
        auto config = baseConfig(model);
        check("open", backend.open(config));
        g_mock->failBindOutput = true;
        g_mock->failBindOutputToDevice = true;
        check("prepare fails when both bindings fail", !backend.prepare(16, 16, 16, 16));
        checkText("double binding failure message", backend.error(), "BindOutputToDevice failed too");
        g_mock->checkBalanced("bind output and device failure");
    }
    // The runtime refuses the output and the DirectML allocator cannot be created.
    {
        g_mock->reset();
        const auto model = writeModelFile();
        OrtDmlSrBackend backend;
        auto config = baseConfig(model);
        check("open", backend.open(config));
        g_mock->failBindOutput = true;
        g_mock->failCreateAllocator = true;
        check("prepare fails without an allocator", !backend.prepare(16, 16, 16, 16));
        checkText("allocator failure message", backend.error(), "CreateAllocator for the DirectML output failed");
        g_mock->checkBalanced("allocator failure");
    }
    // Reading the bound outputs fails after a successful run.
    {
        g_mock->reset();
        const auto model = writeModelFile();
        OrtDmlSrBackend backend;
        auto config = baseConfig(model);
        check("open", backend.open(config));
        g_mock->failBindOutput = true;
        check("prepare", backend.prepare(16, 16, 16, 16));
        g_mock->failGetBoundOutputValues = true;
        check("run fails when the outputs cannot be read", !backend.run());
        checkText("bound output failure message", backend.error(), "GetBoundOutputValues failed");
        g_mock->checkBalanced("bound output read failure");
    }
    // The graph run itself fails.
    {
        g_mock->reset();
        const auto model = writeModelFile();
        OrtDmlSrBackend backend;
        auto config = baseConfig(model);
        check("open", backend.open(config));
        check("prepare", backend.prepare(16, 16, 16, 16));
        g_mock->failRun = true;
        check("run fails", !backend.run());
        checkText("run failure message", backend.error(), "RunWithBinding failed");
        checkEqual("a failed run is not counted", (long long)backend.stats().runs, 0);
        g_mock->checkBalanced("run failure");
    }
}

// ---------------------------------------------------------------------------
// Failure matrices
// ---------------------------------------------------------------------------
bool openFails(const char* what, const char* needle, void (*arm)(Mock&),
               void (*tweak)(OrtDmlSrBackend::Config&) = nullptr)
{
    g_mock->reset();
    const auto model = writeModelFile();
    OrtDmlSrBackend backend;
    auto config = baseConfig(model);
    if (tweak)
        tweak(config);
    arm(*g_mock);
    const std::string openLabel = std::string("open is rejected for ") + what;
    check(openLabel.c_str(), !backend.open(config));
    check("not ready after a rejected open", !backend.ready());
    const std::string textLabel = std::string("reason for ") + what;
    checkText(textLabel.c_str(), backend.error(), needle);
    g_mock->checkBalanced(what);
    return true;
}

void testOpenFailures()
{
    struct Case {
        const char* what;
        void (*arm)(Mock&);
        const char* needle;
    };
    const Case cases[] = {
        {"CreateEnv", [](Mock& m) { m.failCreateEnv = true; }, "CreateEnv failed"},
        {"CreateSessionOptions", [](Mock& m) { m.failCreateSessionOptions = true; }, "CreateSessionOptions failed"},
        {"DisableMemPattern", [](Mock& m) { m.failDisableMemPattern = true; }, "DisableMemPattern failed"},
        {"SetSessionExecutionMode", [](Mock& m) { m.failSetExecutionMode = true; },
         "SetSessionExecutionMode failed"},
        {"SessionOptionsAppendExecutionProvider_DML", [](Mock& m) { m.failAppendDml = true; },
         "SessionOptionsAppendExecutionProvider_DML failed"},
        {"CreateSession", [](Mock& m) { m.failCreateSession = true; }, "CreateSession failed"},
    };
    for (const auto& testCase : cases)
        openFails(testCase.what, testCase.needle, testCase.arm);
}

void testValidation()
{
    openFails("hdrWorking", "HDR working space is not supported", [](Mock&) {},
              [](OrtDmlSrBackend::Config& c) { c.hdrWorking = true; });
    openFails("channels=2", "channels must be 1 (luma) or 3 (RGB), got 2", [](Mock&) {},
              [](OrtDmlSrBackend::Config& c) { c.channels = 2; });
    openFails("empty modelPath", "modelPath is empty", [](Mock&) {},
              [](OrtDmlSrBackend::Config& c) { c.modelPath.clear(); });
    openFails("deviceId=-1", "deviceId must be >= 0, got -1", [](Mock&) {},
              [](OrtDmlSrBackend::Config& c) { c.deviceId = -1; });
    openFails("empty dmlMemoryInfoName", "dmlMemoryInfoName is empty", [](Mock&) {},
              [](OrtDmlSrBackend::Config& c) { c.dmlMemoryInfoName.clear(); });
    openFails("missing model file", "model file not found", [](Mock&) {},
              [](OrtDmlSrBackend::Config& c) {
                  c.modelPath = std::filesystem::temp_directory_path() / "veyra-missing-model.onnx";
              });
    openFails("relative runtimeDllPath", "runtimeDllPath must be absolute", [](Mock&) {},
              [](OrtDmlSrBackend::Config& c) {
                  c.testApiBase = nullptr;
                  c.runtimeDllPath = "onnxruntime.dll";
              });
    openFails("missing runtimeDllPath", "runtime DLL not found", [](Mock&) {},
              [](OrtDmlSrBackend::Config& c) {
                  c.testApiBase = nullptr;
                  c.runtimeDllPath = std::filesystem::temp_directory_path() / "veyra-missing-onnxruntime.dll";
              });
}

void testIntrospectionRejection()
{
    openFails("non-float tensor", "must be float32",
              [](Mock& m) { m.elementType = ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; });
    openFails("rank-3 input", "must be NCHW (rank 4)", [](Mock& m) { m.inputDims = {1, 3, 4}; });
    openFails("rank-5 output", "must be NCHW (rank 4)", [](Mock& m) { m.outputDims = {1, 3, 4, 4, 4}; });
}

bool prepareFails(const char* what, const char* needle, void (*arm)(Mock&))
{
    g_mock->reset();
    const auto model = writeModelFile();
    OrtDmlSrBackend backend;
    auto config = baseConfig(model);
    const std::string openLabel = std::string("open before ") + what;
    check(openLabel.c_str(), backend.open(config));
    arm(*g_mock);
    const std::string prepareLabel = std::string("prepare is rejected for ") + what;
    check(prepareLabel.c_str(), !backend.prepare(32, 32, 32, 32));
    const std::string textLabel = std::string("reason for ") + what;
    checkText(textLabel.c_str(), backend.error(), needle);
    check("a rejected prepare leaves no binding", !backend.prepared());
    backend.shutdown();
    g_mock->checkBalanced(what);
    return true;
}

void testPrepareFailures()
{
    prepareFails("CreateMemoryInfo", "CreateMemoryInfo(\"DML\") failed",
                 [](Mock& m) { m.failCreateMemoryInfo = true; });
    prepareFails("CreateTensorWithDataAsOrtValue", "creating the input tensor failed",
                 [](Mock& m) { m.failCreateTensor = true; });
    prepareFails("CreateIoBinding", "CreateIoBinding failed", [](Mock& m) { m.failCreateIoBinding = true; });
    prepareFails("BindInput", "BindInput(\"input\") failed", [](Mock& m) { m.failBindInput = true; });
}

void testPrepareValidation()
{
    {
        g_mock->reset();
        OrtDmlSrBackend backend;
        check("prepare before open is rejected", !backend.prepare(16, 16, 16, 16));
        checkText("prepare before open message", backend.error(), "prepare() before a successful open()");
        check("run before prepare is rejected", !backend.run());
        checkText("run before prepare message", backend.error(), "run() before prepare()");
        g_mock->checkBalanced("unopened backend");
    }
    {
        g_mock->reset();
        const auto model = writeModelFile();
        OrtDmlSrBackend backend;
        auto config = baseConfig(model);
        check("open", backend.open(config));
        check("zero-sized prepare is rejected", !backend.prepare(0, 64, 64, 64));
        checkText("zero size message", backend.error(), "tensor dimensions must be non-zero");
        // A rejected prepare tears the session down (fail(), not reject()), so
        // reopen before the next bounds case.
        check("reopen after the failed prepare", backend.open(config));
        check("oversized prepare is rejected", !backend.prepare(64, 64, 16385, 64));
        checkText("oversize message", backend.error(), "unsupported size");
        backend.shutdown();
        g_mock->checkBalanced("prepare bounds");
    }
    {
        g_mock->reset();
        const auto model = writeModelFile();
        OrtDmlSrBackend backend;
        auto config = baseConfig(model);
        g_mock->inputDims = {1, 3, 10, 10};
        check("open", backend.open(config));
        check("a mismatched input extent is rejected", !backend.prepare(64, 64, 64, 64));
        checkText("input mismatch names the tensor", backend.error(), "model input `input` is");
        checkText("input mismatch states the request", backend.error(), "cannot serve 1x3x64x64");
        backend.shutdown();
        g_mock->checkBalanced("input mismatch");
    }
    {
        g_mock->reset();
        const auto model = writeModelFile();
        OrtDmlSrBackend backend;
        auto config = baseConfig(model);
        g_mock->outputDims = {1, 3, 10, 10};
        check("open", backend.open(config));
        check("a mismatched output extent is rejected", !backend.prepare(64, 64, 64, 64));
        checkText("output mismatch names the tensor", backend.error(), "model output `output` is");
        backend.shutdown();
        g_mock->checkBalanced("output mismatch");
    }
    {
        g_mock->reset();
        const auto model = writeModelFile();
        OrtDmlSrBackend backend;
        auto config = baseConfig(model);
        config.skipD3dResourcesForTest = false;
        config.createDmlDevice = false;
        check("open without a D3D12 device", backend.open(config));
        check("prepare needs the graph device", !backend.prepare(64, 64, 64, 64));
        checkText("device requirement message", backend.error(), "needs the graph's ID3D12Device");
        backend.shutdown();
        g_mock->checkBalanced("device requirement");
    }
}

void testLumaChannels()
{
    g_mock->reset();
    const auto model = writeModelFile();
    OrtDmlSrBackend backend;
    auto config = baseConfig(model);
    config.channels = 1;
    g_mock->inputDims = {1, 1, -1, -1};
    g_mock->outputDims = {1, 1, -1, -1};
    check("open with a luma model", backend.open(config));
    checkEqual("luma channels", (long long)backend.channels(), 1);
    check("prepare", backend.prepare(128, 128, 128, 128));
    checkEqual("luma input bytes", (long long)g_mock->tensorBytes[0], 128LL * 128LL * 4LL);
    checkEqual("luma output bytes", (long long)g_mock->tensorBytes[1], 128LL * 128LL * 4LL);
    checkEqual("luma channel count in the shape", g_mock->tensorShapes[0][1], 1);
    checkEqual("luma batch in the shape", g_mock->tensorShapes[0][0], 1);
    check("run", backend.run());
    backend.shutdown();
    g_mock->checkBalanced("luma");

    g_mock->reset();
    OrtDmlSrBackend small;
    auto smallConfig = baseConfig(model);
    smallConfig.channels = 1;
    g_mock->inputDims = {1, 1, -1, -1};
    g_mock->outputDims = {1, 1, -1, -1};
    check("open for the small luma case", small.open(smallConfig));
    check("prepare 64x64", small.prepare(64, 64, 64, 64));
    checkEqual("64x64 luma bytes", (long long)g_mock->tensorBytes[0], 64LL * 64LL * 4LL);
    small.shutdown();
    g_mock->checkBalanced("small luma");
}

void testReopen()
{
    g_mock->reset();
    const auto model = writeModelFile();
    OrtDmlSrBackend backend;
    auto config = baseConfig(model);
    check("open", backend.open(config));
    check("prepare", backend.prepare(16, 16, 16, 16));
    check("run", backend.run());
    checkEqual("runs", (long long)backend.stats().runs, 1);
    check("reopen", backend.open(config));
    checkEqual("runs reset by open()", (long long)backend.stats().runs, 0);
    check("not prepared after a reopen", !backend.prepared());
    check("prepare again", backend.prepare(16, 16, 16, 16));
    check("run again", backend.run());
    checkEqual("runs after reopening", (long long)backend.stats().runs, 1);
    backend.shutdown();
    backend.shutdown();
    check("double shutdown is harmless", !backend.ready());
    g_mock->checkBalanced("reopen");
}

void testDestructor()
{
    g_mock->reset();
    const auto model = writeModelFile();
    {
        OrtDmlSrBackend scoped;
        auto config = baseConfig(model);
        check("open", scoped.open(config));
        check("prepare", scoped.prepare(16, 16, 16, 16));
        check("run", scoped.run());
    }
    g_mock->checkBalanced("destructor");
}

} // namespace

int main()
{
    g_mock = &g_mockInstance;

    std::printf("OrtDmlSessionTests: ONNX Runtime C API v%u (%d slots), DirectML EP v1\n",
                static_cast<unsigned>(ort::kOrtApiVersion), static_cast<int>(ort::kOrtApiSlotCount));
    runTest("abi table", testAbiTable);
    runTest("model fixture", testModelFixture);
    runTest("open plumbing", testOpenPlumbing);
    runTest("prepare and run", testPrepareAndRun);
    runTest("dml1 path", testDml1Path);
    runTest("dml2 path", testDml2Path);
    runTest("dml legacy path", testDmlLegacyPath);
    runTest("dml1 missing", testDml1Missing);
    runTest("runtime rejection", testRuntimeRejection);
    runTest("provider api failure", testProviderApiFailure);
    runTest("bind output fallback", testBindOutputFallback);
    runTest("bound output failures", testBoundOutputFailures);
    runTest("open failures", testOpenFailures);
    runTest("validation", testValidation);
    runTest("introspection rejection", testIntrospectionRejection);
    runTest("prepare failures", testPrepareFailures);
    runTest("prepare validation", testPrepareValidation);
    runTest("luma channels", testLumaChannels);
    runTest("reopen", testReopen);
    runTest("destructor", testDestructor);

    std::printf("OrtDmlSessionTests: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
