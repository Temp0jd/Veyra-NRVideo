#include "veyra/pipeline/ArtCnnBackend.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <unordered_map>
#include <unordered_set>

#include "veyra/beauty/OnnxLite.h"

#if VEYRA_ARTCNN_HAS_D3D12
#include "veyra/Log.h"
#include "veyra/RuntimePaths.h"
#endif

namespace veyra::pipeline {
namespace {

using beauty::OnnxModel;
using beauty::OnnxNode;
using beauty::OnnxTensor;

constexpr uint32_t kMaxChannels = 64;

// ---------------------------------------------------------------- model reader

// A conv layer as found by the graph walk, before packing.
struct PendingConv {
    const OnnxTensor* weight = nullptr;
    const OnnxTensor* bias = nullptr;
    uint32_t inChannels = 0;
    uint32_t outChannels = 0;
    bool relu = false;
};

std::string nodeLabel(const OnnxNode& node)
{
    if (!node.name.empty()) return node.op + " (" + node.name + ")";
    return node.op.empty() ? std::string("anonymous node") : node.op;
}

bool allOnes(const std::vector<int64_t>& values)
{
    for (const int64_t v : values)
        if (v != 1) return false;
    return true;
}

// Validates one Conv node and hands back its weight/bias tensors. The network
// is 3x3, stride 1, dilation 1, group 1 with a symmetric one-pixel zero pad
// (ONNX SAME_UPPER for an odd kernel), so every other configuration is refused
// rather than reinterpreted.
bool checkConv(const OnnxModel& model, const OnnxNode& node, PendingConv& conv, std::string& error)
{
    const std::string label = nodeLabel(node);
    if (node.inputs.size() != 3) {
        error = label + ": ArtCNN convolutions need data, weight and bias inputs";
        return false;
    }
    const auto intsOr = [&node](const char* key) -> const std::vector<int64_t>* {
        const auto it = node.ints.find(key);
        return it == node.ints.end() ? nullptr : &it->second;
    };
    if (const auto* strides = intsOr("strides"); strides && !allOnes(*strides)) {
        error = label + ": only stride 1 is supported";
        return false;
    }
    if (const auto* dilations = intsOr("dilations"); dilations && !allOnes(*dilations)) {
        error = label + ": only dilation 1 is supported";
        return false;
    }
    if (node.intOr("group", 1) != 1) {
        error = label + ": grouped convolution is not supported";
        return false;
    }
    if (const auto* kernel = intsOr("kernel_shape");
        kernel && (kernel->size() != 2 || (*kernel)[0] != 3 || (*kernel)[1] != 3)) {
        error = label + ": only a 3x3 kernel is supported";
        return false;
    }
    if (const auto* pads = intsOr("pads"); pads && !allOnes(*pads)) {
        error = label + ": only a one-pixel symmetric pad is supported";
        return false;
    }
    if (const auto it = node.strings.find("auto_pad"); it != node.strings.end()) {
        if (!it->second.empty() && it->second != "NOTSET" && it->second != "SAME_UPPER") {
            error = label + ": auto_pad " + it->second + " is not supported (SAME_UPPER only)";
            return false;
        }
    }

    const auto weight = model.initializers.find(node.inputs[1]);
    if (weight == model.initializers.end()) {
        error = label + ": weight initializer " + node.inputs[1] + " is missing";
        return false;
    }
    const auto bias = model.initializers.find(node.inputs[2]);
    if (bias == model.initializers.end()) {
        error = label + ": bias initializer " + node.inputs[2] + " is missing";
        return false;
    }
    const OnnxTensor& w = weight->second;
    const OnnxTensor& b = bias->second;
    if (w.dataType != 1 || b.dataType != 1) {
        error = label + ": weights and biases must be float32";
        return false;
    }
    if (w.dims.size() != 4 || w.dims[2] != 3 || w.dims[3] != 3) {
        error = label + ": weight tensor must be [out, in, 3, 3]";
        return false;
    }
    const uint32_t outChannels = uint32_t(w.dims[0]);
    const uint32_t inChannels = uint32_t(w.dims[1]);
    if (outChannels == 0 || inChannels == 0) {
        error = label + ": empty weight tensor";
        return false;
    }
    if (w.floats.size() != size_t(outChannels) * inChannels * 9) {
        error = label + ": weight tensor size does not match [out, in, 3, 3]";
        return false;
    }
    if (b.dims.size() != 1 || b.dims[0] != int64_t(outChannels) ||
        b.floats.size() != size_t(outChannels)) {
        error = label + ": bias tensor must be [out]";
        return false;
    }
    conv.weight = &w;
    conv.bias = &b;
    conv.inChannels = inChannels;
    conv.outChannels = outChannels;
    return true;
}

} // namespace

bool parseAndPackArtCnnModel(const std::vector<uint8_t>& onnxBytes, ArtCnnPackedModel& out,
                             std::string& error)
{
    out = {};
    OnnxModel model;
    if (!beauty::parseOnnx(onnxBytes, model, error)) return false;

    if (model.inputs.size() != 1) {
        error = "ArtCNN model must have exactly one graph input";
        return false;
    }
    if (model.outputs.size() != 1) {
        error = "ArtCNN model must have exactly one graph output";
        return false;
    }
    const auto& inputDims = model.inputs[0].dims;
    if (inputDims.size() == 4) {
        if (inputDims[1] != 1 && inputDims[1] != -1) {
            error = "ArtCNN input must be a single-channel image [1, 1, H, W]";
            return false;
        }
        if (inputDims[0] != 1 && inputDims[0] != -1) {
            error = "ArtCNN input must be a single-channel image [1, 1, H, W]";
            return false;
        }
    } else if (!inputDims.empty()) {
        error = "ArtCNN input must be [1, 1, H, W]";
        return false;
    }

    std::unordered_map<std::string, std::vector<int>> consumers;
    for (size_t i = 0; i < model.nodes.size(); ++i)
        for (const auto& name : model.nodes[i].inputs) consumers[name].push_back(int(i));

    // One consumer per tensor is what makes this a chain. The single exception is
    // the stem output: it is the long skip, so it feeds both the first interior
    // convolution and the residual add. Otherwise the only branch is that add.
    const auto resolveConsumer = [&](const std::string& name, bool isSkipSource, int& index,
                                     std::string& why) {
        const auto it = consumers.find(name);
        if (it == consumers.end() || it->second.empty()) {
            why = "tensor " + name + " is never consumed";
            return false;
        }
        if (!isSkipSource) {
            if (it->second.size() > 1) {
                why = "tensor " + name + " is consumed by more than one node";
                return false;
            }
            index = it->second.front();
            return true;
        }
        int convConsumer = -1;
        for (const int candidate : it->second) {
            const std::string& op = model.nodes[size_t(candidate)].op;
            if (op == "Conv") {
                if (convConsumer >= 0) {
                    why = "the stem output feeds more than one convolution";
                    return false;
                }
                convConsumer = candidate;
            } else if (op != "Add") {
                why = "the stem output feeds unsupported operator " + op;
                return false;
            }
        }
        if (convConsumer < 0) {
            why = "the stem output never reaches an interior convolution";
            return false;
        }
        index = convConsumer;
        return true;
    };

    std::unordered_set<int> visited;
    std::vector<PendingConv> convs;

    // conv0: graph input -> C channels, no activation.
    int index = -1;
    if (!resolveConsumer(model.inputs[0].name, false, index, error)) return false;
    const OnnxNode& stemNode = model.nodes[size_t(index)];
    if (stemNode.op != "Conv") {
        error = "ArtCNN graph must start with a convolution, found " + nodeLabel(stemNode);
        return false;
    }
    PendingConv stem;
    if (!checkConv(model, stemNode, stem, error)) return false;
    if (stem.inChannels != 1) {
        error = "ArtCNN stem must convolve the single luma channel";
        return false;
    }
    const uint32_t channels = stem.outChannels;
    if (channels < 4 || channels > kMaxChannels || channels % 4 != 0) {
        error = std::format("ArtCNN channel count {} must be a multiple of 4 in [4, {}]", channels,
            kMaxChannels);
        return false;
    }
    visited.insert(index);
    convs.push_back(stem);
    const std::string skipTensor = stemNode.outputs[0];

    // Interior C -> C convolutions. A Relu node directly after a Conv marks it;
    // the walk stops at the residual add.
    std::string current = skipTensor;
    std::string lastInteriorTensor = skipTensor;
    int interiorCount = 0;
    for (;;) {
        if (!resolveConsumer(current, current == skipTensor, index, error)) return false;
        const OnnxNode& node = model.nodes[size_t(index)];
        if (node.op == "Add") break;
        if (node.op != "Conv") {
            error = "unsupported operator " + nodeLabel(node) + " inside the ArtCNN interior";
            return false;
        }
        PendingConv conv;
        if (!checkConv(model, node, conv, error)) return false;
        if (conv.inChannels != channels) {
            error = std::format("{}: input channels {} do not match the stem's {}", nodeLabel(node),
                conv.inChannels, channels);
            return false;
        }
        // With C == 4 an interior convolution is also 4 -> 4, so only a wider
        // model can tell the output convolution apart by its channel count. For
        // C == 4 a misplaced output convolution falls through to the operator
        // check below instead, which still rejects the graph.
        if (conv.outChannels == 4 && channels != 4) {
            error = "the 4-channel output convolution must follow the residual add, not precede it";
            return false;
        }
        if (conv.outChannels != channels) {
            error = std::format("{}: interior convolutions must keep {} channels", nodeLabel(node),
                channels);
            return false;
        }
        visited.insert(index);
        int nextIndex = -1;
        if (!resolveConsumer(node.outputs[0], false, nextIndex, error)) return false;
        const OnnxNode& nextNode = model.nodes[size_t(nextIndex)];
        conv.relu = nextNode.op == "Relu";
        if (conv.relu) {
            if (nextNode.inputs.size() != 1 || nextNode.outputs.size() != 1) {
                error = nodeLabel(nextNode) + ": unsupported activation shape";
                return false;
            }
            visited.insert(nextIndex);
            lastInteriorTensor = nextNode.outputs[0];
        } else {
            lastInteriorTensor = node.outputs[0];
        }
        convs.push_back(conv);
        ++interiorCount;
        current = lastInteriorTensor;
    }
    if (interiorCount < 1) {
        error = "ArtCNN graph has no interior convolution";
        return false;
    }

    // Add(last interior, stem output), in either operand order.
    const OnnxNode& addNode = model.nodes[size_t(index)];
    if (addNode.inputs.size() != 2 || addNode.outputs.size() != 1) {
        error = "ArtCNN residual add must have two inputs and one output";
        return false;
    }
    const bool addMatches = (addNode.inputs[0] == lastInteriorTensor &&
                                addNode.inputs[1] == skipTensor) ||
        (addNode.inputs[0] == skipTensor && addNode.inputs[1] == lastInteriorTensor);
    if (!addMatches) {
        error = "ArtCNN residual add must combine the last interior convolution with the stem output";
        return false;
    }
    visited.insert(index);

    // Final C -> 4 convolution.
    if (!resolveConsumer(addNode.outputs[0], false, index, error)) return false;
    const OnnxNode& finalNode = model.nodes[size_t(index)];
    if (finalNode.op != "Conv") {
        error = "ArtCNN residual add must be followed by the 4-channel convolution, found " +
            nodeLabel(finalNode);
        return false;
    }
    PendingConv finalConv;
    if (!checkConv(model, finalNode, finalConv, error)) return false;
    if (finalConv.inChannels != channels || finalConv.outChannels != 4) {
        error = std::format("{}: the output convolution must map {} channels to 4",
            nodeLabel(finalNode), channels);
        return false;
    }
    finalConv.relu = false;
    visited.insert(index);
    convs.push_back(finalConv);

    // DepthToSpace(2, DCR) then an optional Clip(0, 1).
    if (!resolveConsumer(finalNode.outputs[0], false, index, error)) return false;
    const OnnxNode& d2sNode = model.nodes[size_t(index)];
    if (d2sNode.op != "DepthToSpace") {
        error = "ArtCNN output convolution must be followed by DepthToSpace, found " +
            nodeLabel(d2sNode);
        return false;
    }
    if (d2sNode.intOr("blocksize", 1) != 2) {
        error = "ArtCNN DepthToSpace must use blocksize 2";
        return false;
    }
    if (const auto it = d2sNode.strings.find("mode"); it != d2sNode.strings.end() && !it->second.empty() &&
        it->second != "DCR") {
        error = "ArtCNN DepthToSpace must use DCR mode, found " + it->second;
        return false;
    }
    visited.insert(index);
    current = d2sNode.outputs[0];

    int clipIndex = -1;
    if (!resolveConsumer(current, false, clipIndex, error)) return false;
    if (model.nodes[size_t(clipIndex)].op == "Clip") {
        const OnnxNode& clipNode = model.nodes[size_t(clipIndex)];
        // Optional min/max inputs must pin the range to [0, 1]; the finish pass saturates too.
        const auto boundValue = [&](size_t inputSlot, float& value) {
            if (clipNode.inputs.size() <= inputSlot || clipNode.inputs[inputSlot].empty()) return false;
            const auto it = model.initializers.find(clipNode.inputs[inputSlot]);
            if (it == model.initializers.end() || it->second.floats.size() != 1) return false;
            value = it->second.floats.front();
            return true;
        };
        float minimum = 0.0f, maximum = 1.0f;
        const bool hasMin = boundValue(1, minimum);
        const bool hasMax = boundValue(2, maximum);
        if ((hasMin && std::abs(minimum) > 1e-6f) || (hasMax && std::abs(maximum - 1.0f) > 1e-6f)) {
            error = "ArtCNN Clip must clamp to [0, 1]";
            return false;
        }
        visited.insert(clipIndex);
        current = clipNode.outputs[0];
    }
    if (current != model.outputs[0].name) {
        error = "ArtCNN graph output is not produced by the expected pipeline";
        return false;
    }
    if (visited.size() != model.nodes.size()) {
        for (size_t i = 0; i < model.nodes.size(); ++i) {
            if (!visited.contains(int(i))) {
                error = "unsupported operator " + nodeLabel(model.nodes[i]) +
                    " is outside the ArtCNN pipeline";
                return false;
            }
        }
    }

    // ------------------------------------------------------------- weight pack
    out.channels = channels;
    out.quads = channels / 4;
    out.weightsF4.reserve(size_t(convs.size()) * size_t(out.quads) * (9u * out.quads * 4u + 1u));
    uint32_t offsetF4 = 0;
    auto pushFloat4 = [&out](float x, float y, float z, float w) {
        out.weightsF4.push_back(x);
        out.weightsF4.push_back(y);
        out.weightsF4.push_back(z);
        out.weightsF4.push_back(w);
    };
    for (const PendingConv& conv : convs) {
        ArtCnnLayerSpec spec;
        spec.inChannels = conv.inChannels;
        spec.outChannels = conv.outChannels;
        spec.relu = conv.relu;
        const uint32_t outQuads = conv.outChannels / 4;
        const uint32_t inQuads = conv.inChannels / 4;
        // Bias: one float4 per output quad, (b[4q+0], .., b[4q+3]).
        spec.biasOffsetF4 = offsetF4;
        for (uint32_t q = 0; q < outQuads; ++q) {
            const uint32_t base = 4u * q;
            pushFloat4(conv.bias->floats[base + 0], conv.bias->floats[base + 1],
                conv.bias->floats[base + 2], conv.bias->floats[base + 3]);
            ++offsetF4;
        }
        // Weights.
        spec.weightOffsetF4 = offsetF4;
        const std::vector<float>& w = conv.weight->floats;
        if (conv.inChannels == 1) {
            // Stem: scalar input, so one float4 per (output quad, tap), holding the
            // four output channels of that quad.
            for (uint32_t q = 0; q < outQuads; ++q)
                for (uint32_t tap = 0; tap < 9; ++tap) {
                    pushFloat4(w[(4u * q + 0) * 9 + tap], w[(4u * q + 1) * 9 + tap],
                        w[(4u * q + 2) * 9 + tap], w[(4u * q + 3) * 9 + tap]);
                    ++offsetF4;
                }
        } else {
            // Interior: an input quad times an output quad is a 4x4 matrix, so each
            // (output quad, input quad, tap) owns four float4 rows.
            for (uint32_t q = 0; q < outQuads; ++q)
                for (uint32_t iq = 0; iq < inQuads; ++iq)
                    for (uint32_t tap = 0; tap < 9; ++tap)
                        for (uint32_t c = 0; c < 4; ++c) {
                            const uint32_t inChannel = 4u * iq + c;
                            float column[4];
                            for (uint32_t r = 0; r < 4; ++r)
                                column[r] =
                                    w[(size_t(4u * q + r) * conv.inChannels + inChannel) * 9 + tap];
                            pushFloat4(column[0], column[1], column[2], column[3]);
                            ++offsetF4;
                        }
        }
        out.paramCount += uint64_t(conv.inChannels) * conv.outChannels * 9u;
        out.layers.push_back(spec);
    }
    if (uint64_t(out.weightsF4.size()) != uint64_t(offsetF4) * 4ull) {
        error = "ArtCNN weight packing bookkeeping mismatch";
        out = {};
        return false;
    }
    return true;
}

bool loadAndPackArtCnnModel(const std::filesystem::path& modelPath, ArtCnnPackedModel& out,
                            std::string& error)
{
    std::vector<uint8_t> bytes;
    if (!beauty::readFileBytes(modelPath, bytes, error)) return false;
    return parseAndPackArtCnnModel(bytes, out, error);
}

// --------------------------------------------------------------- D3D12 runtime

#if VEYRA_ARTCNN_HAS_D3D12
namespace {

constexpr uint32_t kDwords = 16;   // root constants shared by all four passes
constexpr UINT kStagingSlots = 8;

// Descriptor slots, one compute pass per shader. The SRV table always starts at
// slot 0 of the window bound at dispatch time; the UAV table is a single slot.
namespace prepSlots {
constexpr UINT srvColor = 0, uavLuma = 1;
}
namespace stemSlots {
constexpr UINT srvWeights = 0, srvLuma = 1, uavOut = 2;
}
namespace convSlots {
constexpr UINT srvInStem = 0;      // [weights, S, S]
constexpr UINT srvInA = 3;         // [weights, A, S]
constexpr UINT srvInB = 6;         // [weights, B, S]
constexpr UINT uavA = 9, uavB = 10;
constexpr UINT count = 11;
}
namespace finishSlots {
constexpr UINT srvWeights = 0, srvIn = 1, srvColor = 2, uavOut = 3;
}

bool isFloatWorkingFormat(DXGI_FORMAT format)
{
    return format == DXGI_FORMAT_R16G16B16A16_FLOAT || format == DXGI_FORMAT_R32G32B32A32_FLOAT;
}

D3D12_RESOURCE_DESC bufferDesc(uint64_t bytes, D3D12_RESOURCE_FLAGS flags)
{
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = flags;
    return d;
}

ComPtr<ID3D12Resource> createBuffer(ID3D12Device* device, uint64_t bytes, D3D12_HEAP_TYPE heap,
                                    D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state)
{
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = heap;
    const auto desc = bufferDesc(bytes, flags);
    ComPtr<ID3D12Resource> resource;
    if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
            IID_PPV_ARGS(&resource))))
        return nullptr;
    return resource;
}

void structuredSrv(ID3D12Device* device, ID3D12Resource* resource, uint64_t bytes,
                   D3D12_CPU_DESCRIPTOR_HANDLE at)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC d{};
    d.Format = DXGI_FORMAT_UNKNOWN;
    d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Buffer.NumElements = UINT(bytes / 16);
    d.Buffer.StructureByteStride = 16;
    device->CreateShaderResourceView(resource, &d, at);
}

void structuredUav(ID3D12Device* device, ID3D12Resource* resource, uint64_t bytes,
                   D3D12_CPU_DESCRIPTOR_HANDLE at)
{
    D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
    d.Format = DXGI_FORMAT_UNKNOWN;
    d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    d.Buffer.NumElements = UINT(bytes / 16);
    d.Buffer.StructureByteStride = 16;
    device->CreateUnorderedAccessView(resource, nullptr, &d, at);
}

uint32_t groupsOf(uint64_t elements) { return uint32_t((elements + 63ull) / 64ull); }

} // namespace

struct ArtCnnBackend::Impl {
    ArtCnnPackedModel model;
    uint32_t inW = 0, inH = 0, outW = 0, outH = 0;
    bool ready = false;

    bool failed = false, failureLogged = false, weightsUploaded = false;
    ID3D12Device* device = nullptr;
    ComputePass prep, stem, conv, finish;
    // Non-shader-visible staging heap for the per-frame texture views (RTX 5070 /
    // 616.56 workaround documented in GpuPassUtils.h). The internal buffers keep
    // their own StateTracker: the graph's tracker must not have to know them.
    ComPtr<ID3D12DescriptorHeap> staging;
    UINT stagingIncrement = 0, stagingNext = 0;
    StateTracker states;
    ComPtr<ID3D12Resource> luma, stemOut, bufferA, bufferB;
    ComPtr<ID3D12Resource> weights, weightsUpload;
    uint64_t lumaBytes = 0, layerBytes = 0;
    // Descriptor re-staging only when the graph swaps its working textures; a
    // resize path drains the queue before rebuilding, so no in-flight list can
    // still reference an overwritten slot.
    ID3D12Resource* colorCached = nullptr;
    ID3D12Resource* outputCached = nullptr;
    DXGI_FORMAT colorFormatCached = DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT outputFormatCached = DXGI_FORMAT_UNKNOWN;

    void fail(const std::string& message)
    {
        failed = true;
        if (failureLogged) return;
        failureLogged = true;
        veyra::log::error("artcnn", message + " (ArtCNN passes are suppressed until re-initialisation)");
    }

    void stageSrv(ID3D12Resource* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc,
                  const ComputePass& pass, UINT slot)
    {
        const UINT index = stagingNext % kStagingSlots;
        ++stagingNext;
        D3D12_CPU_DESCRIPTOR_HANDLE from{staging->GetCPUDescriptorHandleForHeapStart().ptr +
            SIZE_T(index) * stagingIncrement};
        device->CreateShaderResourceView(resource, desc, from);
        const D3D12_CPU_DESCRIPTOR_HANDLE to = cpuHandleOf(pass, slot);
        device->CopyDescriptorsSimple(1, to, from, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

    void stageUav(ID3D12Resource* resource, const D3D12_UNORDERED_ACCESS_VIEW_DESC* desc,
                  const ComputePass& pass, UINT slot)
    {
        const UINT index = stagingNext % kStagingSlots;
        ++stagingNext;
        D3D12_CPU_DESCRIPTOR_HANDLE from{staging->GetCPUDescriptorHandleForHeapStart().ptr +
            SIZE_T(index) * stagingIncrement};
        device->CreateUnorderedAccessView(resource, nullptr, desc, from);
        const D3D12_CPU_DESCRIPTOR_HANDLE to = cpuHandleOf(pass, slot);
        device->CopyDescriptorsSimple(1, to, from, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
};
#else
// CPU-only build (unit tests): the packing API exists, the runtime does not.
struct ArtCnnBackend::Impl {
    ArtCnnPackedModel model;
    uint32_t inW = 0, inH = 0, outW = 0, outH = 0;
    bool ready = false;
};
#endif

ArtCnnBackend::ArtCnnBackend() : impl_(std::make_unique<Impl>()) {}
ArtCnnBackend::~ArtCnnBackend() = default;

bool ArtCnnBackend::created() const { return impl_ && impl_->ready; }
const ArtCnnPackedModel& ArtCnnBackend::model() const { return impl_->model; }
uint64_t ArtCnnBackend::weightBytes() const { return impl_->model.weightBytes(); }
uint64_t ArtCnnBackend::workingSetBytes() const
{
    return impl_->model.workingSetBytes(impl_->inW, impl_->inH);
}
uint64_t ArtCnnBackend::multiplyAccumulates() const
{
    return impl_->model.multiplyAccumulates(impl_->inW, impl_->inH);
}

void ArtCnnBackend::shutdown() { impl_ = std::make_unique<Impl>(); }

#if VEYRA_ARTCNN_HAS_D3D12

bool ArtCnnBackend::initialize(ID3D12Device* device, const std::filesystem::path& modelPath,
                               uint32_t inW, uint32_t inH, uint32_t outW, uint32_t outH,
                               bool hdrWorking, std::string& error)
{
    shutdown();
    Impl& m = *impl_;
    if (!device) {
        error = "ArtCNN needs a D3D12 device";
        return false;
    }
    if (hdrWorking) {
        error = "ArtCNN only supports SDR working space";
        return false;
    }
    if (!supportedRatio(inW, inH, outW, outH)) {
        error = std::format("ArtCNN needs an exact 2x output extent, got {}x{} -> {}x{}", inW, inH,
            outW, outH);
        return false;
    }
    if (!loadAndPackArtCnnModel(modelPath, m.model, error)) return false;
    if (m.model.layers.size() < 3) {
        error = "ArtCNN model must have a stem, at least one interior convolution and an output convolution";
        return false;
    }
    m.device = device;
    m.inW = inW;
    m.inH = inH;
    m.outW = outW;
    m.outH = outH;

    std::vector<uint8_t> prepCode, stemCode, convCode, finishCode;
    if (!loadShaderBytes("ArtCnnPrep.dxil", prepCode) ||
        !loadShaderBytes("ArtCnnStem.dxil", stemCode) ||
        !loadShaderBytes("ArtCnnConv.dxil", convCode) ||
        !loadShaderBytes("ArtCnnFinish.dxil", finishCode)) {
        error = "ArtCNN shaders are missing (ArtCnnPrep/Stem/Conv/Finish.dxil)";
        return false;
    }
    if (!m.prep.create(device, prepCode, 2, 1, 1, kDwords) ||
        !m.stem.create(device, stemCode, 3, 2, 1, kDwords) ||
        !m.conv.create(device, convCode, convSlots::count, 3, 1, kDwords) ||
        !m.finish.create(device, finishCode, 4, 3, 1, kDwords)) {
        error = "ArtCNN compute pipeline creation failed";
        return false;
    }
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = kStagingSlots;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m.staging)))) {
        error = "ArtCNN staging descriptor heap creation failed";
        return false;
    }
    m.stagingIncrement = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    const uint32_t quads = m.model.quads;
    m.lumaBytes = uint64_t(inW) * inH * 16ull;
    m.layerBytes = uint64_t(inW) * inH * quads * 16ull;
    m.luma = createBuffer(device, m.lumaBytes, D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    m.stemOut = createBuffer(device, m.layerBytes, D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    m.bufferA = createBuffer(device, m.layerBytes, D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    m.bufferB = createBuffer(device, m.layerBytes, D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    m.weights = createBuffer(device, m.model.weightBytes(), D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
    m.weightsUpload = makeUploadBuffer(device, m.model.weightBytes());
    if (!m.luma || !m.stemOut || !m.bufferA || !m.bufferB || !m.weights || !m.weightsUpload) {
        error = "ArtCNN buffer allocation failed";
        return false;
    }
    void* mapped = nullptr;
    if (FAILED(m.weightsUpload->Map(0, nullptr, &mapped)) || !mapped) {
        error = "ArtCNN weight upload buffer mapping failed";
        return false;
    }
    std::memcpy(mapped, m.model.weightsF4.data(), size_t(m.model.weightBytes()));
    m.weightsUpload->Unmap(0, nullptr);

    for (ID3D12Resource* resource : {m.luma.Get(), m.stemOut.Get(), m.bufferA.Get(), m.bufferB.Get(),
             m.weights.Get()})
        m.states.set(resource, D3D12_RESOURCE_STATE_COMMON);

    // Static views: weights in every pass that reads them, the internal buffers.
    for (const ComputePass* pass : {&m.stem, &m.conv, &m.finish})
        structuredSrv(device, m.weights.Get(), m.model.weightBytes(), cpuHandleOf(*pass, 0));
    structuredSrv(device, m.luma.Get(), m.lumaBytes, cpuHandleOf(m.stem, stemSlots::srvLuma));
    structuredUav(device, m.stemOut.Get(), m.layerBytes, cpuHandleOf(m.stem, stemSlots::uavOut));
    structuredSrv(device, m.stemOut.Get(), m.layerBytes, cpuHandleOf(m.conv, convSlots::srvInStem + 1));
    structuredSrv(device, m.bufferA.Get(), m.layerBytes, cpuHandleOf(m.conv, convSlots::srvInA + 1));
    structuredSrv(device, m.bufferB.Get(), m.layerBytes, cpuHandleOf(m.conv, convSlots::srvInB + 1));
    // The skip input of every interior convolution is the stem output.
    for (UINT base : {convSlots::srvInStem, convSlots::srvInA, convSlots::srvInB}) {
        structuredSrv(device, m.weights.Get(), m.model.weightBytes(), cpuHandleOf(m.conv, base));
        structuredSrv(device, m.stemOut.Get(), m.layerBytes, cpuHandleOf(m.conv, base + 2));
    }
    structuredUav(device, m.bufferA.Get(), m.layerBytes, cpuHandleOf(m.conv, convSlots::uavA));
    structuredUav(device, m.bufferB.Get(), m.layerBytes, cpuHandleOf(m.conv, convSlots::uavB));
    structuredSrv(device, m.bufferA.Get(), m.layerBytes, cpuHandleOf(m.finish, finishSlots::srvIn));
    structuredUav(device, m.luma.Get(), m.lumaBytes, cpuHandleOf(m.prep, prepSlots::uavLuma));
    // Slots the graph fills per frame start as valid null views so the bound
    // descriptor ranges are never uninitialised.
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC nullSrv{};
        nullSrv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        nullSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        nullSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        nullSrv.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(nullptr, &nullSrv, cpuHandleOf(m.prep, prepSlots::srvColor));
        device->CreateShaderResourceView(nullptr, &nullSrv, cpuHandleOf(m.finish, finishSlots::srvColor));
        D3D12_UNORDERED_ACCESS_VIEW_DESC nullUav{};
        nullUav.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        nullUav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        device->CreateUnorderedAccessView(nullptr, nullptr, &nullUav,
            cpuHandleOf(m.finish, finishSlots::uavOut));
    }

    m.ready = true;
    veyra::log::info("artcnn", std::format(
        "ArtCNN {}x{} -> {}x{} luma SR: {} channels, {} layers, weights {} KiB, working set {:.1f} MiB, "
        "{:.2f} GMAC/frame",
        inW, inH, outW, outH, m.model.channels, m.model.layers.size(),
        m.model.weightBytes() / 1024,
        double(m.model.workingSetBytes(inW, inH)) / (1024.0 * 1024.0),
        double(m.model.multiplyAccumulates(inW, inH)) / 1e9));
    return true;
}

bool ArtCnnBackend::run(ID3D12GraphicsCommandList* list, StateTracker& tracker,
                        ID3D12Resource* color, ID3D12Resource* output)
{
    Impl& m = *impl_;
    if (!list || !m.ready || m.failed) return false;
    if (!color || !output) {
        m.fail("null colour/output resource");
        return false;
    }
    const D3D12_RESOURCE_DESC colorDesc = color->GetDesc();
    const D3D12_RESOURCE_DESC outputDesc = output->GetDesc();
    if (colorDesc.Width != m.inW || colorDesc.Height != m.inH) {
        m.fail(std::format("colour extent {}x{} does not match the initialised {}x{}", colorDesc.Width,
            colorDesc.Height, m.inW, m.inH));
        return false;
    }
    if (outputDesc.Width != m.outW || outputDesc.Height != m.outH) {
        m.fail(std::format("output extent {}x{} does not match the initialised {}x{}", outputDesc.Width,
            outputDesc.Height, m.outW, m.outH));
        return false;
    }
    if (!isFloatWorkingFormat(colorDesc.Format) || !isFloatWorkingFormat(outputDesc.Format)) {
        m.fail("ArtCNN needs RGBA16F/RGBA32F working textures");
        return false;
    }
    if ((outputDesc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0) {
        m.fail("ArtCNN output texture is missing the unordered-access flag");
        return false;
    }

    // Per-frame views, written only when the graph hands over a different
    // resource (a resize drains the queue before rebuilding its textures).
    if (color != m.colorCached || colorDesc.Format != m.colorFormatCached) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = colorDesc.Format;   // RGBA16F/RGBA32F both read as float4
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        m.stageSrv(color, &srv, m.prep, prepSlots::srvColor);
        m.stageSrv(color, &srv, m.finish, finishSlots::srvColor);
        m.colorCached = color;
        m.colorFormatCached = colorDesc.Format;
    }
    if (output != m.outputCached || outputDesc.Format != m.outputFormatCached) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = outputDesc.Format;
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        m.stageUav(output, &uav, m.finish, finishSlots::uavOut);
        m.outputCached = output;
        m.outputFormatCached = outputDesc.Format;
    }

    // Weights: one copy inside this command list, no CPU fence, no extra sync.
    if (!m.weightsUploaded) {
        m.states.transition(list, m.weights.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(m.weights.Get(), 0, m.weightsUpload.Get(), 0, m.model.weightBytes());
        m.states.transition(list, m.weights.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        m.weightsUploaded = true;
    }

    const uint32_t quads = m.model.quads;
    const uint64_t pixels = uint64_t(m.inW) * m.inH;
    uint32_t c[kDwords] = {};

    // 1. Linear BT.709 working colour -> sRGB luma [0, 1] (one float per pixel).
    c[0] = m.inW; c[1] = m.inH; c[2] = m.inW; c[3] = m.inH;
    tracker.transition(list, color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    m.states.transition(list, m.luma.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m.prep.bind(list, reinterpret_cast<const float*>(c), gpuHandleOf(m.prep, prepSlots::srvColor).ptr,
        gpuHandleOf(m.prep, prepSlots::uavLuma).ptr);
    list->Dispatch(groupsOf(pixels), 1, 1);
    m.states.uavBarrier(list, m.luma.Get());

    // 2. Stem 1 -> C into the skip source S.
    const ArtCnnLayerSpec& stemLayer = m.model.layers.front();
    c[0] = m.inW; c[1] = m.inH; c[2] = m.inW; c[3] = m.inH;
    c[4] = 1; c[5] = quads; c[6] = stemLayer.weightOffsetF4; c[7] = stemLayer.biasOffsetF4; c[8] = 0;
    m.states.transition(list, m.luma.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    m.states.transition(list, m.stemOut.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m.stem.bind(list, reinterpret_cast<const float*>(c), gpuHandleOf(m.stem, stemSlots::srvWeights).ptr,
        gpuHandleOf(m.stem, stemSlots::uavOut).ptr);
    list->Dispatch(groupsOf(pixels * quads), 1, 1);
    m.states.uavBarrier(list, m.stemOut.Get());

    // 3. Interior C -> C convolutions, ping-ponging A/B. The last one fuses
    //    Add(stem output) exactly where the graph adds before the final conv, and
    //    the ping-pong start is chosen so the result always lands in A.
    const size_t layerCount = m.model.layers.size();
    const size_t interiorCount = layerCount - 2;
    bool writeA = (interiorCount % 2) == 1;
    ID3D12Resource* in = m.stemOut.Get();
    UINT inBase = convSlots::srvInStem;
    for (size_t i = 1; i + 1 < layerCount; ++i) {
        const ArtCnnLayerSpec& layer = m.model.layers[i];
        ID3D12Resource* dst = writeA ? m.bufferA.Get() : m.bufferB.Get();
        const UINT uavSlot = writeA ? convSlots::uavA : convSlots::uavB;
        uint32_t flags = 0;
        if (layer.relu) flags |= 1u;
        if (i + 2 == layerCount) flags |= 2u;   // fused long skip
        c[0] = m.inW; c[1] = m.inH; c[2] = m.inW; c[3] = m.inH;
        c[4] = quads; c[5] = quads; c[6] = layer.weightOffsetF4; c[7] = layer.biasOffsetF4;
        c[8] = flags;
        m.states.transition(list, in, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        m.states.transition(list, dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        m.conv.bind(list, reinterpret_cast<const float*>(c), gpuHandleOf(m.conv, inBase).ptr,
            gpuHandleOf(m.conv, uavSlot).ptr);
        list->Dispatch(groupsOf(pixels * quads), 1, 1);
        m.states.uavBarrier(list, dst);
        in = dst;
        inBase = writeA ? convSlots::srvInA : convSlots::srvInB;
        writeA = !writeA;
    }

    // 4. Final C -> 4, DepthToSpace(2, DCR), Clip, then chroma recombination from
    //    the original sRGB source into linear BT.709 at 2x.
    const ArtCnnLayerSpec& finalLayer = m.model.layers.back();
    c[0] = m.inW; c[1] = m.inH; c[2] = m.outW; c[3] = m.outH;
    c[4] = quads; c[5] = 1; c[6] = finalLayer.weightOffsetF4; c[7] = finalLayer.biasOffsetF4; c[8] = 0;
    m.states.transition(list, m.bufferA.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    tracker.transition(list, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m.finish.bind(list, reinterpret_cast<const float*>(c), gpuHandleOf(m.finish, finishSlots::srvWeights).ptr,
        gpuHandleOf(m.finish, finishSlots::uavOut).ptr);
    list->Dispatch(groupsOf(pixels), 1, 1);
    tracker.uavBarrier(list, output);
    return true;
}

#else

// CPU-only build: the runtime entry points are not part of the class, so a
// caller that only needs the packer compiles without D3D12 at all.

#endif

} // namespace veyra::pipeline
