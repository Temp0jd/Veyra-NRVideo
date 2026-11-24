// SPDX-License-Identifier: GPL-3.0-only
// Versioned C ABI of MochizukiNrRuntime.dll (Mochi). Pinned transcription of the upstream
// MochiNrApi.h (mochinr.cpp) / MochizukiNrControls.h pair; third_party/mochi/ records the
// upstream commit. The table layout is the same 16-slot lmxxf v1 order, so a runtime exported
// as LmxxfNrGetApi is accepted through the same struct.
// Rules taken from the upstream header: no STL, no exceptions, no CRT objects cross the
// boundary; every struct starts with struct_size and a reader uses only the fields that fit.
#pragma once
#include <cstdint>
#include <cstddef>

#define MOCHI_NR_ABI_VERSION 1u
// Two uint32 fields + sixteen function pointers: the lmxxf v1 table size.
#define MOCHI_NR_API_V1_SIZE 136u

// CreateInfo.flags. ZERO_OUTPUT_FALLBACK: when SET, a failed network frame is replaced by a
// zeroed (black) surface instead of the frame's original colour. Veyra keeps this bit CLEAR so
// a failing frame can never publish black. ACO: run dlssnr-amd\aco\ machine code where the
// driver offers VK_KHR_pipeline_binary (mochizuki only; decided once per process by the first
// session, and lmxxf-compatible builds may reject it).
#define MOCHI_NR_CREATE_FLAG_ZERO_OUTPUT_FALLBACK (1u << 0)
#define MOCHI_NR_CREATE_FLAG_ACO (1u << 1)

// FrameInfo.flags: only the two Veyra drives are listed. Veyra applies residual, protection and
// temporal controls after the full network output, exactly as on the lmxxf path, so the
// runtime's own TEMPORAL/SMOOTH/STYLE/VIT/INT4/PASS3/SKIN/CASCADE/ANTI_HALO/AUTO_WHITE stages
// stay off and the network is never rebuilt for a per-frame control.
#define MOCHI_NR_FRAME_FLAG_STRENGTH (1u << 0)
#define MOCHI_NR_FRAME_FLAG_ORIGINAL_MODEL (1u << 13)

// JobState: NONE, PREPARED, PRODUCER_SUBMITTED, NR_ENQUEUED, NR_COMPLETE, CONSUMER_COMPLETE, RETIRED.
#define MOCHI_NR_JOB_HANDLE_MASK 0xFFFFFFFFFFFFFFFFull

extern "C" {
enum MochiNrStatus {
    MOCHI_NR_OK = 0,
    MOCHI_NR_UNSUPPORTED_ABI = 1,
    MOCHI_NR_INVALID_ARGUMENT = 2,
    MOCHI_NR_NOT_IMPLEMENTED = 3,
    MOCHI_NR_UNAVAILABLE = 4,
    MOCHI_NR_FAILED = 5
};

typedef struct MochiNrCapabilities {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t max_input_width;
    uint32_t max_input_height;
    uint32_t history_supported; /* first product version: 0 */
    uint32_t overlap_supported; /* first product version: 0 */
    uint32_t graph_supported;   /* first product version: 0 */
    uint32_t hip_ready;         /* 1 once host+hsaco are loaded */
    uint32_t gfx1201_target;    /* 1 = this binary is for gfx1201 */
} MochiNrCapabilities;

typedef struct MochiNrCreateInfo {
    uint32_t struct_size;
    void* device;               /* ID3D12Device* */
    void* queue;                /* ID3D12CommandQueue* */
    const wchar_t* assets_directory;
    uint32_t flags;             /* MOCHI_NR_CREATE_FLAG_*; unknown bits are rejected */
} MochiNrCreateInfo;

typedef struct MochiNrFrameInfo {
    uint32_t struct_size;
    uint64_t session_id;
    uint64_t frame_id;
    uint64_t list_generation;
    void* command_list;         /* ID3D12GraphicsCommandList* */
    uint32_t color_width;
    uint32_t color_height;
    void* color;                /* ID3D12Resource*: linear RGBA16F */
    uint32_t color_state;       /* D3D12_RESOURCE_STATES at RecordInputs */
    uint32_t flags;             /* MOCHI_NR_FRAME_FLAG_* */
    float transfer_strength;    /* 0..2, default 1.0 */
    float color_strength;       /* lmxxf 0..2, mochizuki 0..4 */
    uint32_t debug_view;        /* 0..4 */
    float model_scale;          /* 0.25..1.0 */
    uint32_t passes;            /* 1..3 (mochizuki 1..30), 0 = 1 */
    void* motion;               /* ID3D12Resource*; required for TEMPORAL only */
    uint32_t motion_state;
    uint32_t motion_width;
    uint32_t motion_height;
    float motion_scale_x;
    float motion_scale_y;
    uint32_t reset;
    float smooth_threshold;
    float smooth_strength;
    uint32_t style;             /* NGX Style; needs MOCHI_NR_FRAME_FLAG_STYLE */
    void* depth;                /* ID3D12Resource*; D32, D32S8, R32 or D16 */
    uint32_t depth_state;
    uint32_t depth_inverted;
} MochiNrFrameInfo;

typedef struct MochiNrJob {
    uint32_t struct_size;       /* a set without FIRST_PASS sends offsetof(first_pass_output) */
    void* handle;
    void* private_output;       /* ID3D12Resource*; null until PrepareFrame succeeds */
    void* first_pass_output;
} MochiNrJob;

typedef struct MochiNrApi {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t (*QueryCapabilities)(MochiNrCapabilities* out);
    int32_t (*Create)(const MochiNrCreateInfo* info, void** context);
    int32_t (*Destroy)(void* context);
    int32_t (*PrepareSession)(void* context);
    int32_t (*PrepareFrame)(void* context, const MochiNrFrameInfo* info, MochiNrJob* job);
    int32_t (*RecordInputs)(void* context, void* job, void* command_list);
    int32_t (*EnqueueHip)(void* context, void* job, void* command_queue);
    int32_t (*RecordOutputs)(void* context, void* job, void* command_list);
    int32_t (*ExecuteAfterProducer)(void* context, void* job, void* command_queue);
    int32_t (*CancelUnsubmitted)(void* context, void* job);
    int32_t (*Poll)(void* context, void* job, uint32_t* state);
    int32_t (*Retire)(void* context, void* job);
    int32_t (*ResetHistory)(void* context);
    int32_t (*Drain)(void* context);
    int32_t (*GetStatus)(void* context, char* buf, uint32_t buf_chars);
    int32_t (*GetLastError)(char* buf, uint32_t buf_chars);
} MochiNrApi;

} // extern "C"

// Sizes the host relies on. A mismatch is a compile error, not a runtime surprise.
static_assert(sizeof(MochiNrCapabilities) == 36, "MochiNrCapabilities ABI");
static_assert(sizeof(MochiNrCreateInfo) == 40, "MochiNrCreateInfo ABI");
static_assert(sizeof(MochiNrFrameInfo) == 152, "MochiNrFrameInfo ABI");
static_assert(sizeof(MochiNrJob) == 32, "MochiNrJob ABI");
static_assert(sizeof(MochiNrApi) == 136, "MochiNrApi ABI");
static_assert(offsetof(MochiNrFrameInfo, passes) == 80, "MochiNrFrameInfo V1 prefix ends at passes");
static_assert(offsetof(MochiNrFrameInfo, depth) == 136, "MochiNrFrameInfo depth offset");
