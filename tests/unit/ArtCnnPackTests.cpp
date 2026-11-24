// Unit tests for the ArtCNN 2x luma super resolution backend (M5).
//
// Everything here runs without D3D12: the model reader/packer and the exact
// float4 math are pure CPU code, so the weight layout and the DepthToSpace /
// YCbCr mapping can be verified on Linux under Wine. The D3D12 half of
// ArtCnnBackend is compiled out with VEYRA_ARTCNN_NO_D3D12.
//
// Two independent forward passes are compared:
//   * a naive NCHW reference written straight from the ONNX graph, and
//   * a faithful simulation of the packed float4 kernels (ArtCnnStem.hlsl,
//     ArtCnnConv.hlsl with the fused long skip, ArtCnnFinish.hlsl).
// Both accumulate in double, so any difference beyond rounding means an
// indexing/packing bug rather than floating point noise.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "veyra/beauty/OnnxLite.h"
#include "veyra/pipeline/ArtCnnBackend.h"

using veyra::beauty::OnnxModel;
using veyra::beauty::parseOnnx;
using veyra::pipeline::ArtCnnPackedModel;
using veyra::pipeline::parseAndPackArtCnnModel;

namespace {

int gChecks = 0;
int gFailures = 0;

void check(bool condition, const std::string& what)
{
    ++gChecks;
    if (condition) return;
    ++gFailures;
    std::printf("FAIL: %s\n", what.c_str());
}

void checkNear(double actual, double expected, double tolerance, const std::string& what)
{
    ++gChecks;
    if (std::abs(actual - expected) <= tolerance) return;
    ++gFailures;
    std::printf("FAIL: %s (got %.9g, want %.9g, tol %.3g)\n", what.c_str(), actual, expected,
        tolerance);
}

void checkEqual(uint64_t actual, uint64_t expected, const std::string& what)
{
    ++gChecks;
    if (actual == expected) return;
    ++gFailures;
    std::printf("FAIL: %s (got %llu, want %llu)\n", what.c_str(),
        static_cast<unsigned long long>(actual), static_cast<unsigned long long>(expected));
}

// --------------------------------------------------------- protobuf/ONNX writer

struct Pb {
    std::vector<uint8_t> bytes;

    void varint(uint64_t value)
    {
        while (value >= 0x80) {
            bytes.push_back(uint8_t(value) | 0x80u);
            value >>= 7;
        }
        bytes.push_back(uint8_t(value));
    }
    void tag(uint32_t field, uint32_t wire) { varint((uint64_t(field) << 3) | wire); }
    void uintField(uint32_t field, uint64_t value)
    {
        tag(field, 0);
        varint(value);
    }
    void intField(uint32_t field, int64_t value) { uintField(field, uint64_t(value)); }
    void str(uint32_t field, const std::string& text)
    {
        tag(field, 2);
        varint(text.size());
        bytes.insert(bytes.end(), text.begin(), text.end());
    }
    void message(uint32_t field, const Pb& nested)
    {
        tag(field, 2);
        varint(nested.bytes.size());
        bytes.insert(bytes.end(), nested.bytes.begin(), nested.bytes.end());
    }
    void blob(uint32_t field, const void* data, size_t size)
    {
        tag(field, 2);
        varint(size);
        const auto* p = static_cast<const uint8_t*>(data);
        bytes.insert(bytes.end(), p, p + size);
    }
    void packedInts(uint32_t field, const std::vector<int64_t>& values)
    {
        Pb packed;
        for (const int64_t value : values) packed.varint(uint64_t(value));
        message(field, packed);
    }
};

Pb floatTensor(const std::string& name, const std::vector<int64_t>& dims,
    const std::vector<float>& values)
{
    Pb t;
    t.packedInts(1, dims);
    t.uintField(2, 1);   // TensorProto.DataType.FLOAT
    t.str(8, name);
    t.blob(9, values.data(), values.size() * sizeof(float));
    return t;
}

// A float16 initializer, either as raw halfs or in the (unpacked) float data
// field. OnnxLite must refuse both instead of decoding a half-empty tensor.
Pb float16Tensor(const std::string& name, const std::vector<int64_t>& dims, size_t count, bool raw)
{
    Pb t;
    t.packedInts(1, dims);
    t.uintField(2, 10);   // TensorProto.DataType.FLOAT16
    t.str(8, name);
    if (raw) {
        const std::vector<uint16_t> halfs(count, 0x3c00);
        t.blob(9, halfs.data(), halfs.size() * sizeof(uint16_t));
    } else {
        const std::vector<float> values(count, 1.0f);
        t.blob(4, values.data(), values.size() * sizeof(float));
    }
    return t;
}

Pb attributeInts(const std::string& name, const std::vector<int64_t>& values)
{
    Pb a;
    a.str(1, name);
    a.packedInts(8, values);
    return a;
}

Pb attributeString(const std::string& name, const std::string& value)
{
    Pb a;
    a.str(1, name);
    a.str(4, value);
    return a;
}

struct NodeSpec {
    std::string op;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<Pb> attributes;
};

Pb node(const NodeSpec& spec)
{
    Pb n;
    for (const auto& input : spec.inputs) n.str(1, input);
    for (const auto& output : spec.outputs) n.str(2, output);
    n.str(4, spec.op);
    for (const auto& attribute : spec.attributes) n.message(5, attribute);
    return n;
}

Pb tensorType(const std::vector<int64_t>& dims)
{
    Pb shape;
    for (const int64_t dim : dims) {
        Pb dimension;
        if (dim < 0) dimension.str(2, "dynamic");
        else dimension.intField(1, dim);
        shape.message(1, dimension);
    }
    Pb tensor;
    tensor.message(2, shape);
    Pb type;
    type.message(1, tensor);
    return type;
}

Pb valueInfo(const std::string& name, const std::vector<int64_t>& dims)
{
    Pb info;
    info.str(1, name);
    info.message(2, tensorType(dims));
    return info;
}

std::vector<uint8_t> buildModel(const std::vector<Pb>& nodes, const std::vector<Pb>& initializers,
    const std::vector<Pb>& inputs, const std::vector<Pb>& outputs)
{
    Pb graph;
    for (const auto& n : nodes) graph.message(1, n);
    graph.str(2, "artcnn_test");
    for (const auto& init : initializers) graph.message(5, init);
    for (const auto& input : inputs) graph.message(11, input);
    for (const auto& output : outputs) graph.message(12, output);

    Pb opset;
    opset.uintField(2, 19);

    Pb model;
    model.intField(1, 8);   // ir_version
    model.message(8, opset);
    model.message(7, graph);
    return model.bytes;
}

// --------------------------------------------------------------- model factory

// A convolution as the model file stores it: [out][in][3][3] plus [out].
struct RawConv {
    std::vector<float> weights;
    std::vector<float> biases;
    uint32_t inChannels = 0;
    uint32_t outChannels = 0;
};

struct SyntheticOptions {
    uint32_t channels = 16;
    int interior = 5;   // conv1 .. conv(interior); the last one carries no ReLU
    bool includeAdd = true;
    int strides = 1;
    int dilations = 1;
    int group = 1;
    int blocksize = 2;
    uint32_t finalOutChannels = 4;
    bool includeClip = true;
    bool float16Initializer = false;     // raw float16 payload
    bool float16InFloatField = false;    // float16 dtype, float_data payload
    bool emitPads = true;
    uint32_t seed = 12345u;
};

struct SyntheticModel {
    std::vector<uint8_t> bytes;
    std::vector<RawConv> convs;
};

float nextWeight(uint32_t& state)
{
    state = state * 1664525u + 1013904223u;
    return (float((state >> 8) & 0xFFFFu) / 32768.0f - 1.0f) * 0.5f;
}

RawConv makeConv(uint32_t& state, uint32_t inChannels, uint32_t outChannels)
{
    RawConv conv;
    conv.inChannels = inChannels;
    conv.outChannels = outChannels;
    conv.weights.resize(size_t(inChannels) * outChannels * 9);
    conv.biases.resize(outChannels);
    for (auto& value : conv.weights) value = nextWeight(state);
    for (auto& value : conv.biases) value = nextWeight(state);
    return conv;
}

SyntheticModel buildSynthetic(const SyntheticOptions& options)
{
    SyntheticModel model;
    uint32_t state = options.seed;
    const uint32_t channels = options.channels;

    model.convs.push_back(makeConv(state, 1, channels));
    for (int i = 0; i < options.interior; ++i) model.convs.push_back(makeConv(state, channels, channels));
    model.convs.push_back(makeConv(state, channels, options.finalOutChannels));

    std::vector<Pb> nodes;
    std::vector<Pb> initializers;

    const auto convAttributes = [&options](std::vector<Pb>& attrs) {
        attrs.push_back(attributeInts("kernel_shape", {3, 3}));
        if (options.emitPads) attrs.push_back(attributeInts("pads", {1, 1, 1, 1}));
        if (options.strides != 1) attrs.push_back(attributeInts("strides", {options.strides, options.strides}));
        if (options.dilations != 1)
            attrs.push_back(attributeInts("dilations", {options.dilations, options.dilations}));
        if (options.group != 1) attrs.push_back(attributeInts("group", {options.group}));
    };

    const auto emitConv = [&](const std::string& weightName, const std::string& biasName,
                              const std::string& input, const std::string& output,
                              const RawConv& conv, size_t convIndex) {
        const std::vector<int64_t> weightDims = {int64_t(conv.outChannels), int64_t(conv.inChannels), 3, 3};
        const std::vector<int64_t> biasDims = {int64_t(conv.outChannels)};
        if (options.float16Initializer && convIndex == 0) {
            initializers.push_back(float16Tensor(weightName, weightDims, conv.weights.size(), true));
            initializers.push_back(float16Tensor(biasName, biasDims, conv.biases.size(), true));
        } else if (options.float16InFloatField && convIndex == 0) {
            initializers.push_back(float16Tensor(weightName, weightDims, conv.weights.size(), false));
            initializers.push_back(float16Tensor(biasName, biasDims, conv.biases.size(), false));
        } else {
            initializers.push_back(floatTensor(weightName, weightDims, conv.weights));
            initializers.push_back(floatTensor(biasName, biasDims, conv.biases));
        }
        NodeSpec node;
        node.op = "Conv";
        node.inputs = {input, weightName, biasName};
        node.outputs = {output};
        convAttributes(node.attributes);
        nodes.push_back(::node(node));
    };

    emitConv("w0", "b0", "input", "conv0_out", model.convs[0], 0);
    std::string current = "conv0_out";
    const std::string skip = "conv0_out";
    for (int i = 1; i <= options.interior; ++i) {
        const std::string convOut = "conv" + std::to_string(i) + "_out";
        emitConv("w" + std::to_string(i), "b" + std::to_string(i), current, convOut,
            model.convs[size_t(i)], size_t(i));
        // Upstream leaves the last interior conv unactivated so the residual add
        // is applied to a linear tensor.
        if (i < options.interior) {
            NodeSpec relu;
            relu.op = "Relu";
            relu.inputs = {convOut};
            relu.outputs = {"conv" + std::to_string(i) + "_relu"};
            nodes.push_back(::node(relu));
            current = "conv" + std::to_string(i) + "_relu";
        } else {
            current = convOut;
        }
    }
    const std::string lastInterior = current;

    if (options.includeAdd) {
        NodeSpec add;
        add.op = "Add";
        add.inputs = {lastInterior, skip};
        add.outputs = {"add_out"};
        nodes.push_back(::node(add));
        current = "add_out";
    }

    const size_t finalIndex = model.convs.size() - 1;
    emitConv("wf", "bf", current, "conv_final_out", model.convs[finalIndex], finalIndex);
    current = "conv_final_out";

    NodeSpec d2s;
    d2s.op = "DepthToSpace";
    d2s.inputs = {current};
    d2s.outputs = {"d2s_out"};
    d2s.attributes.push_back(attributeString("mode", "DCR"));
    d2s.attributes.push_back(attributeInts("blocksize", {options.blocksize}));
    nodes.push_back(::node(d2s));
    current = "d2s_out";

    if (options.includeClip) {
        initializers.push_back(floatTensor("clip_min", {}, {0.0f}));
        initializers.push_back(floatTensor("clip_max", {}, {1.0f}));
        NodeSpec clip;
        clip.op = "Clip";
        clip.inputs = {current, "clip_min", "clip_max"};
        clip.outputs = {"output"};
        nodes.push_back(::node(clip));
    } else {
        NodeSpec identity;
        identity.op = "Clip";
        identity.inputs = {current};
        identity.outputs = {"output"};
        nodes.push_back(::node(identity));
    }

    model.bytes = buildModel(nodes, initializers, {valueInfo("input", {1, 1, -1, -1})},
        {valueInfo("output", {1, 1, -1, -1})});
    return model;
}

// Pulls conv0..conv6 out of a parsed model in graph order, using only the node
// list (no initializer naming assumptions), so the same extraction works for the
// synthetic models and for the upstream files.
bool extractConvChain(const OnnxModel& model, std::vector<RawConv>& convs, std::string& error)
{
    convs.clear();
    for (const auto& node : model.nodes) {
        if (node.op != "Conv") continue;
        if (node.inputs.size() != 3) {
            error = "conv with unexpected arity";
            return false;
        }
        const auto weight = model.initializers.find(node.inputs[1]);
        const auto bias = model.initializers.find(node.inputs[2]);
        if (weight == model.initializers.end() || bias == model.initializers.end()) {
            error = "conv initializer missing";
            return false;
        }
        const auto& w = weight->second;
        if (w.dims.size() != 4) {
            error = "conv weight is not 4D";
            return false;
        }
        RawConv conv;
        conv.outChannels = uint32_t(w.dims[0]);
        conv.inChannels = uint32_t(w.dims[1]);
        conv.weights = w.floats;
        conv.biases = bias->second.floats;
        convs.push_back(std::move(conv));
    }
    if (convs.empty()) {
        error = "no convolutions found";
        return false;
    }
    return true;
}

// ------------------------------------------------------------ reference forward

// Naive NCHW 3x3 convolution, zero pad 1, stride 1, accumulated in double.
void convReference(const std::vector<double>& source, int width, int height, const RawConv& conv,
    bool relu, std::vector<double>& destination)
{
    destination.assign(size_t(conv.outChannels) * width * height, 0.0);
    for (uint32_t o = 0; o < conv.outChannels; ++o) {
        const double biasValue = double(conv.biases[o]);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double sum = biasValue;
                for (uint32_t i = 0; i < conv.inChannels; ++i)
                    for (int ky = 0; ky < 3; ++ky) {
                        const int iy = y + ky - 1;
                        if (iy < 0 || iy >= height) continue;
                        for (int kx = 0; kx < 3; ++kx) {
                            const int ix = x + kx - 1;
                            if (ix < 0 || ix >= width) continue;
                            const double w = double(conv.weights[(size_t(o) * conv.inChannels + i) * 9 +
                                size_t(ky) * 3 + size_t(kx)]);
                            sum += w * source[(size_t(i) * height + iy) * width + ix];
                        }
                    }
                if (relu) sum = std::max(sum, 0.0);
                destination[(size_t(o) * height + y) * width + x] = sum;
            }
        }
    }
}

// Reference pipeline. Returns the clipped super-resolved luma for the 2x2
// DepthToSpace(DCR) block, indexed [outY][outX].
std::vector<double> referenceForward(const std::vector<RawConv>& convs,
    const std::vector<double>& luma, int width, int height)
{
    std::vector<double> skip;
    convReference(luma, width, height, convs.front(), false, skip);
    std::vector<double> current = skip;
    for (size_t i = 1; i + 1 < convs.size(); ++i) {
        const bool relu = i + 2 < convs.size(); // the last interior conv has no ReLU
        std::vector<double> next;
        convReference(current, width, height, convs[i], relu, next);
        current = std::move(next);
    }
    // Add(convN+1, conv0) before the final convolution.
    for (size_t i = 0; i < current.size(); ++i) current[i] += skip[i];

    std::vector<double> finalQuad;
    convReference(current, width, height, convs.back(), false, finalQuad);

    const int outWidth = width * 2;
    const int outHeight = height * 2;
    std::vector<double> out(size_t(outWidth) * outHeight, 0.0);
    for (uint32_t channel = 0; channel < 4; ++channel) {
        const int i = int(channel) / 2;   // row offset
        const int j = int(channel) % 2;   // column offset
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                const double value = finalQuad[(size_t(channel) * height + y) * width + x];
                out[size_t(2 * y + i) * outWidth + size_t(2 * x + j)] = std::clamp(value, 0.0, 1.0);
            }
    }
    return out;
}

// ------------------------------------------------------- packed kernel simulator

// Mirrors the HLSL exactly: channel-last float4 tensors, the float4 weight
// layouts, mad() per lane and the fused long skip. One float4 is stored as four
// consecutive doubles.
struct PackedTensor {
    int width = 0;
    int height = 0;
    int quads = 0;
    std::vector<double> values;

    void resize(int w, int h, int q)
    {
        width = w;
        height = h;
        quads = q;
        values.assign(size_t(w) * h * q * 4, 0.0);
    }
    double* quad(int x, int y, int q) { return values.data() + index(x, y, q) * 4; }
    const double* quad(int x, int y, int q) const { return values.data() + index(x, y, q) * 4; }
    // Explicit index form used by the shaders, kept for the parentheses check.
    size_t index(int x, int y, int q) const { return (size_t(y) * width + x) * quads + size_t(q); }
};

std::vector<double> packedForward(const ArtCnnPackedModel& model, const std::vector<double>& luma,
    int width, int height)
{
    const int quads = int(model.quads);
    const auto fetch = [&model](uint32_t float4Index, double out[4]) {
        for (int lane = 0; lane < 4; ++lane)
            out[lane] = double(model.weightsF4[size_t(float4Index) * 4 + size_t(lane)]);
    };
    const auto inRange = [&](int x, int y, int w, int h) {
        return x >= 0 && y >= 0 && x < w && y < h;
    };

    PackedTensor input;   // scalar luma, quads == 1, stored in .x
    input.resize(width, height, 1);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) input.quad(x, y, 0)[0] = luma[size_t(y) * width + x];

    // conv0: 1 -> C.
    PackedTensor stem;
    stem.resize(width, height, quads);
    {
        const auto& layer = model.layers.front();
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                for (int q = 0; q < quads; ++q) {
                    double acc[4];
                    fetch(layer.biasOffsetF4 + uint32_t(q), acc);
                    for (int tap = 0; tap < 9; ++tap) {
                        const int iy = y + tap / 3 - 1;
                        const int ix = x + tap % 3 - 1;
                        const double v = inRange(ix, iy, width, height) ? input.quad(ix, iy, 0)[0] : 0.0;
                        double w4[4];
                        fetch(layer.weightOffsetF4 + uint32_t(q * 9 + tap), w4);
                        for (int lane = 0; lane < 4; ++lane) acc[lane] = acc[lane] + w4[lane] * v;
                    }
                    double* out = stem.quad(x, y, q);
                    for (int lane = 0; lane < 4; ++lane) out[lane] = acc[lane];
                }
    }

    const size_t layerCount = model.layers.size();
    PackedTensor current = stem;
    PackedTensor scratch;
    scratch.resize(width, height, quads);
    for (size_t i = 1; i + 1 < layerCount; ++i) {
        const auto& layer = model.layers[i];
        const bool fuseSkip = i + 2 == layerCount;
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                for (int q = 0; q < quads; ++q) {
                    double acc[4];
                    fetch(layer.biasOffsetF4 + uint32_t(q), acc);
                    for (int tap = 0; tap < 9; ++tap) {
                        const int iy = y + tap / 3 - 1;
                        const int ix = x + tap % 3 - 1;
                        if (!inRange(ix, iy, width, height)) continue;
                        for (int iq = 0; iq < quads; ++iq) {
                            const double* v = current.quad(ix, iy, iq);
                            const uint32_t base = layer.weightOffsetF4 +
                                uint32_t(((q * quads + iq) * 9 + tap) * 4);
                            for (int c = 0; c < 4; ++c) {
                                double w4[4];
                                fetch(base + uint32_t(c), w4);
                                for (int lane = 0; lane < 4; ++lane) acc[lane] = acc[lane] + w4[lane] * v[c];
                            }
                        }
                    }
                    if (layer.relu)
                        for (int lane = 0; lane < 4; ++lane) acc[lane] = std::max(acc[lane], 0.0);
                    if (fuseSkip) {
                        // Skip[id.x] is the conv0 tensor at the same (pixel, quad).
                        const double* skipQuad = stem.quad(x, y, q);
                        for (int lane = 0; lane < 4; ++lane) acc[lane] += skipQuad[lane];
                    }
                    double* out = scratch.quad(x, y, q);
                    for (int lane = 0; lane < 4; ++lane) out[lane] = acc[lane];
                }
        current = scratch;
    }

    // Finish: C -> 4, saturate, DepthToSpace(2, DCR).
    const auto& finalLayer = model.layers.back();
    const int outWidth = width * 2;
    const int outHeight = height * 2;
    std::vector<double> out(size_t(outWidth) * outHeight, 0.0);
    double bias4[4];
    fetch(finalLayer.biasOffsetF4, bias4);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            double acc[4] = {bias4[0], bias4[1], bias4[2], bias4[3]};
            for (int tap = 0; tap < 9; ++tap) {
                const int iy = y + tap / 3 - 1;
                const int ix = x + tap % 3 - 1;
                if (!inRange(ix, iy, width, height)) continue;
                for (int iq = 0; iq < quads; ++iq) {
                    const double* v = current.quad(ix, iy, iq);
                    const uint32_t base = finalLayer.weightOffsetF4 + uint32_t((iq * 9 + tap) * 4);
                    for (int c = 0; c < 4; ++c) {
                        double w4[4];
                        fetch(base + uint32_t(c), w4);
                        for (int lane = 0; lane < 4; ++lane) acc[lane] = acc[lane] + w4[lane] * v[c];
                    }
                }
            }
            for (uint32_t channel = 0; channel < 4; ++channel) {
                const int i = int(channel) / 2;
                const int j = int(channel) % 2;
                out[size_t(2 * y + i) * outWidth + size_t(2 * x + j)] = std::clamp(acc[channel], 0.0, 1.0);
            }
        }
    return out;
}

// ---------------------------------------------------------- colour round trips

double srgbEncode(double c)
{
    c = std::clamp(c, 0.0, 1.0);
    return c <= 0.0031308 ? c * 12.92 : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
}

double srgbDecode(double c)
{
    c = std::clamp(c, 0.0, 1.0);
    return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}

struct Rgb {
    double r, g, b;
};

// Exactly the matrix in ArtCnn.hlsli.
Rgb srgbToYcbcr(const Rgb& srgb)
{
    const double luma = 0.2126 * srgb.r + 0.7152 * srgb.g + 0.0722 * srgb.b;
    return {luma, (srgb.b - luma) / 1.8556, (srgb.r - luma) / 1.5748};
}

Rgb ycbcrToSrgb(const Rgb& ycc)
{
    const double r = ycc.r + 1.5748 * ycc.b;
    const double b = ycc.r + 1.8556 * ycc.g;
    const double g = ycc.r - 0.46812427 * ycc.b - 0.18732427 * ycc.g;
    return {r, g, b};
}

Rgb linearToSrgb(const Rgb& linear)
{
    return {srgbEncode(linear.r), srgbEncode(linear.g), srgbEncode(linear.b)};
}

Rgb srgbToLinear(const Rgb& srgb)
{
    return {srgbDecode(srgb.r), srgbDecode(srgb.g), srgbDecode(srgb.b)};
}

// -------------------------------------------------------------- model fixtures

struct ExpectedStats {
    const char* label;
    uint32_t channels;
    uint64_t paramCount;
    uint64_t weightBytes;
    uint64_t workingSetBytes;
    uint64_t multiplyAccumulates;
};

void checkStats(const ArtCnnPackedModel& model, const ExpectedStats& expected)
{
    const std::string prefix = std::string(expected.label) + ": ";
    checkEqual(model.channels, expected.channels, prefix + "channels");
    checkEqual(model.quads, expected.channels / 4, prefix + "quads");
    checkEqual(model.layers.size(), 7, prefix + "layer count");
    checkEqual(model.paramCount, expected.paramCount, prefix + "paramCount");
    checkEqual(model.weightBytes(), expected.weightBytes, prefix + "weightBytes");
    checkEqual(model.workingSetBytes(1920, 1080), expected.workingSetBytes,
        prefix + "workingSetBytes(1920,1080)");
    checkEqual(model.multiplyAccumulates(1920, 1080), expected.multiplyAccumulates,
        prefix + "multiplyAccumulates(1920,1080)");
    checkEqual(uint64_t(model.weightsF4.size()), expected.weightBytes / 4,
        prefix + "packed float count");
}

void checkLayerStructure(const ArtCnnPackedModel& model, const std::string& label)
{
    const uint32_t channels = model.channels;
    check(model.layers.front().inChannels == 1, label + ": stem reads one channel");
    check(model.layers.front().outChannels == channels, label + ": stem writes C channels");
    check(!model.layers.front().relu, label + ": stem has no activation");
    check(model.layers.back().inChannels == channels, label + ": final conv reads C channels");
    check(model.layers.back().outChannels == 4, label + ": final conv writes 4 channels");
    check(!model.layers.back().relu, label + ": final conv has no activation");
    for (size_t i = 1; i + 1 < model.layers.size(); ++i) {
        const auto& layer = model.layers[i];
        check(layer.inChannels == channels && layer.outChannels == channels,
            label + ": interior layer keeps C channels");
        check(layer.relu == (i + 2 < model.layers.size()),
            label + ": only the last interior layer is unactivated");
    }
    // Offsets must tile weightsF4 without gaps or overlaps.
    const uint64_t total = model.weightBytes() / 16;
    uint64_t next = 0;
    for (size_t i = 0; i < model.layers.size(); ++i) {
        const auto& layer = model.layers[i];
        const uint64_t outQuads = layer.outChannels / 4;
        const uint64_t inQuads = layer.inChannels / 4;
        check(uint64_t(layer.biasOffsetF4) == next, label + ": bias offsets tile the buffer");
        next += outQuads;
        check(uint64_t(layer.weightOffsetF4) == next, label + ": weight offsets tile the buffer");
        next += layer.inChannels == 1 ? outQuads * 9 : outQuads * inQuads * 9 * 4;
    }
    check(next == total, label + ": packed buffer has no trailing slack");
}

// Runs the equivalence check for one model and reports the maximum error.
double checkEquivalence(const ArtCnnPackedModel& packed, const std::vector<RawConv>& convs,
    const std::string& label)
{
    const int width = 5;
    const int height = 4;
    std::vector<double> luma(size_t(width) * height);
    uint32_t state = 777u;
    for (auto& value : luma) value = double(nextWeight(state)) + 0.5;   // ~[0, 1] sRGB luma

    const std::vector<double> reference = referenceForward(convs, luma, width, height);
    const std::vector<double> packedOut = packedForward(packed, luma, width, height);
    check(reference.size() == packedOut.size(), label + ": output extents match");
    double maxError = 0.0;
    for (size_t i = 0; i < reference.size() && i < packedOut.size(); ++i)
        maxError = std::max(maxError, std::abs(reference[i] - packedOut[i]));
    check(reference.size() == size_t(width) * 2 * size_t(height) * 2, label + ": 5x4 -> 10x8 output");
    checkNear(maxError, 0.0, 1e-5, label + ": packed kernels match the NCHW reference");
    std::printf("  %s: max |reference - packed| = %.3e over %zu output pixels\n", label.c_str(),
        maxError, reference.size());
    return maxError;
}

void runSyntheticTests()
{
    // A synthetic C=16 model must pack and repack identically to the upstream one.
    SyntheticOptions options;
    options.channels = 16;
    SyntheticModel synthetic = buildSynthetic(options);
    ArtCnnPackedModel packed;
    std::string error;
    const bool packedOk = parseAndPackArtCnnModel(synthetic.bytes, packed, error);
    check(packedOk, "synthetic C4F16 model packs (" + error + ")");
    if (!packedOk) return;

    // Derived from the layout, not from the packer: stem bias 4 + weights 4*9,
    // five interior layers of (bias 4 + 4*4*9*4), and a final (bias 1 + 4*9*4).
    const uint64_t stemFloat4s = 4ull + 36ull;
    const uint64_t interiorFloat4s = 5ull * (4ull + 4ull * 4ull * 9ull * 4ull);
    const uint64_t finalFloat4s = 1ull + 4ull * 9ull * 4ull;
    const uint64_t paramCount = 16ull * 9ull + 5ull * (16ull * 16ull * 9ull) + 4ull * 16ull * 9ull;
    const ExpectedStats expected{"synthetic C4F16", 16, paramCount,
        (stemFloat4s + interiorFloat4s + finalFloat4s) * 16ull,
        1920ull * 1080ull * 16ull + 3ull * 1920ull * 1080ull * 4ull * 16ull,
        1920ull * 1080ull * paramCount};
    checkStats(packed, expected);
    checkLayerStructure(packed, "synthetic C4F16");
    check(packed.channels == 16 && packed.quads == 4, "synthetic model keeps C=16, quads=4");
    checkEquivalence(packed, synthetic.convs, "synthetic C4F16");

    // A small C=4 model exercises quads == 1, the smallest legal layout.
    SyntheticOptions small;
    small.channels = 4;
    small.interior = 1;
    SyntheticModel smallModel = buildSynthetic(small);
    ArtCnnPackedModel smallPacked;
    error.clear();
    const bool smallOk = parseAndPackArtCnnModel(smallModel.bytes, smallPacked, error);
    check(smallOk, "synthetic C4F4 packs (" + error + ")");
    if (smallOk) {
        checkEqual(smallPacked.quads, 1, "synthetic C4F4 quads");
        checkEqual(smallPacked.layers.size(), 3, "synthetic C4F4 layer count (stem, 1 interior, final)");
        checkEquivalence(smallPacked, smallModel.convs, "synthetic C4F4");
    }

    // An even interior count must still land the residual sum in buffer A, which
    // is what the finish pass reads.
    SyntheticOptions evenInterior;
    evenInterior.channels = 8;
    evenInterior.interior = 2;
    SyntheticModel evenModel = buildSynthetic(evenInterior);
    ArtCnnPackedModel evenPacked;
    error.clear();
    const bool evenOk = parseAndPackArtCnnModel(evenModel.bytes, evenPacked, error);
    check(evenOk, "synthetic C4F8 with 2 interior layers packs (" + error + ")");
    if (evenOk) checkEquivalence(evenPacked, evenModel.convs, "synthetic C4F8 (2 interior)");
}

void runRejectionTests()
{
    ArtCnnPackedModel packed;
    std::string error;

    // FP16 initializers of both encodings.
    for (const bool raw : {true, false}) {
        SyntheticOptions options;
        if (raw) options.float16Initializer = true;
        else options.float16InFloatField = true;
        const SyntheticModel model = buildSynthetic(options);
        OnnxModel parsed;
        error.clear();
        check(!parseOnnx(model.bytes, parsed, error), "OnnxLite rejects a float16 initializer");
        check(error.find("float16") != std::string::npos,
            "float16 rejection names the dtype: " + error);
        error.clear();
        check(!parseAndPackArtCnnModel(model.bytes, packed, error),
            "packer propagates the float16 rejection");
        check(error.find("float16") != std::string::npos,
            "packer float16 message names the dtype: " + error);
    }

    // Non-2x ratios.
    check(!veyra::pipeline::ArtCnnBackend::supportedRatio(100, 100, 100, 100), "supportedRatio rejects 1x");
    check(!veyra::pipeline::ArtCnnBackend::supportedRatio(100, 100, 150, 100), "supportedRatio rejects 1.5x");
    check(!veyra::pipeline::ArtCnnBackend::supportedRatio(100, 100, 300, 300), "supportedRatio rejects 3x");
    check(!veyra::pipeline::ArtCnnBackend::supportedRatio(0, 0, 0, 0), "supportedRatio rejects 0x0");
    check(!veyra::pipeline::ArtCnnBackend::supportedRatio(0, 10, 0, 20), "supportedRatio rejects an empty input");
    check(!veyra::pipeline::ArtCnnBackend::supportedRatio(100, 200, 200, 300), "supportedRatio rejects a mixed ratio");
    check(!veyra::pipeline::ArtCnnBackend::supportedRatio(100, 200, 150, 400), "supportedRatio rejects a half-width 2x");
    check(veyra::pipeline::ArtCnnBackend::supportedRatio(1920, 1080, 3840, 2160), "supportedRatio accepts 2x");

    const auto expectReject = [&](const SyntheticOptions& options, const std::string& what,
                                  const std::string& expectedSubstring) {
        const SyntheticModel model = buildSynthetic(options);
        packed = {};
        error.clear();
        const bool accepted = parseAndPackArtCnnModel(model.bytes, packed, error);
        check(!accepted, what);
        check(!error.empty(), what + " reports an error");
        if (!expectedSubstring.empty())
            check(error.find(expectedSubstring) != std::string::npos,
                what + " message mentions '" + expectedSubstring + "': " + error);
    };

    SyntheticOptions wrongStride;
    wrongStride.strides = 2;
    expectReject(wrongStride, "rejects stride 2", "stride");

    SyntheticOptions wrongDilation;
    wrongDilation.dilations = 2;
    expectReject(wrongDilation, "rejects dilation 2", "dilation");

    SyntheticOptions grouped;
    grouped.group = 2;
    expectReject(grouped, "rejects grouped convolution", "group");

    SyntheticOptions wrongBlocksize;
    wrongBlocksize.blocksize = 3;
    expectReject(wrongBlocksize, "rejects DepthToSpace blocksize 3", "blocksize");

    SyntheticOptions noAdd;
    noAdd.includeAdd = false;
    expectReject(noAdd, "rejects a model without the residual add", "");

    SyntheticOptions oddChannels;
    oddChannels.channels = 6;
    expectReject(oddChannels, "rejects a channel count that is not a multiple of 4", "multiple of 4");

    SyntheticOptions tooManyChannels;
    tooManyChannels.channels = 68;
    expectReject(tooManyChannels, "rejects more than 64 channels", "multiple of 4");

    SyntheticOptions wrongFinal;
    wrongFinal.finalOutChannels = 8;
    expectReject(wrongFinal, "rejects a final convolution that is not 4 channels", "");
}

void runColourTests()
{
    const double tolerance = 1e-5;
    // Linear RGB -> sRGB -> YCbCr -> sRGB -> linear must be an identity, which is
    // what makes a pass-through luma reconstruction exact.
    for (double r : {0.0, 0.02, 0.18, 0.5, 0.9, 1.0})
        for (double g : {0.0, 0.35, 0.7, 1.0})
            for (double b : {0.0, 0.05, 0.4, 1.0}) {
                const Rgb linear{r, g, b};
                const Rgb srgb = linearToSrgb(linear);
                const Rgb ycc = srgbToYcbcr(srgb);
                const Rgb back = srgbToLinear(ycbcrToSrgb(ycc));
                check(std::abs(back.r - r) <= tolerance && std::abs(back.g - g) <= tolerance &&
                        std::abs(back.b - b) <= tolerance,
                    "linear RGB round trip through sRGB YCbCr");
            }
    // Grey stays neutral (Cb == Cr == 0) and the luma equals the grey level.
    for (double level : {0.0, 0.05, 0.5, 1.0}) {
        const Rgb ycc = srgbToYcbcr({level, level, level});
        checkNear(ycc.g, 0.0, 1e-12, "grey has zero Cb");
        checkNear(ycc.b, 0.0, 1e-12, "grey has zero Cr");
        checkNear(ycc.r, level, 1e-12, "grey luma equals the grey level");
    }
    // The inverse matrix must undo the forward matrix for a chromatic colour.
    const Rgb chromatic = srgbToYcbcr({0.9, 0.2, 0.4});
    const Rgb restored = ycbcrToSrgb(chromatic);
    checkNear(restored.r, 0.9, 1e-6, "BT.709 inverse reproduces red");
    checkNear(restored.g, 0.2, 1e-6, "BT.709 inverse reproduces green");
    checkNear(restored.b, 0.4, 1e-6, "BT.709 inverse reproduces blue");
}

void runRealModelTests(const std::string& modelDirectory)
{
    struct RealModel {
        std::string fileName;
        ExpectedStats stats;
    };
    const RealModel models[] = {
        {"ArtCNN_C4F16.onnx", {"C4F16", 16, 12240, 49360, 431308800, 25380864000ull}},
        {"ArtCNN_C4F32.onnx", {"C4F32", 32, 47520, 190864, 829440000, 98537472000ull}},
    };
    for (const auto& real : models) {
        const std::string path = modelDirectory + "/" + real.fileName;
        if (!std::filesystem::exists(path)) {
            std::printf("SKIP: %s not found at %s\n", real.fileName.c_str(), path.c_str());
            ++gFailures;
            continue;
        }
        ArtCnnPackedModel packed;
        std::string error;
        const bool ok = veyra::pipeline::loadAndPackArtCnnModel(path, packed, error);
        check(ok, std::string(real.stats.label) + ": real model loads (" + error + ")");
        if (!ok) continue;
        checkStats(packed, real.stats);
        checkLayerStructure(packed, real.stats.label);

        std::vector<uint8_t> bytes;
        check(veyra::beauty::readFileBytes(path, bytes, error), "real model bytes read back");
        OnnxModel parsed;
        error.clear();
        check(parseOnnx(bytes, parsed, error), std::string(real.stats.label) + ": ONNX parses");
        std::vector<RawConv> convs;
        error.clear();
        check(extractConvChain(parsed, convs, error), std::string(real.stats.label) + ": conv chain extracted");
        checkEqual(convs.size(), 7, std::string(real.stats.label) + ": 7 convolutions in the graph");
        checkEquivalence(packed, convs, real.stats.label);
    }
}

} // namespace

int main(int argc, char** argv)
{
    // Under Wine the upstream models are reachable through the Z: drive.
    const std::string modelDirectory =
        argc > 1 ? std::string(argv[1]) : std::string("Z:/tmp/m5m7recon");
    std::printf("ArtCNN pack tests, model directory: %s\n", modelDirectory.c_str());

    runSyntheticTests();
    runRejectionTests();
    runColourTests();
    runRealModelTests(modelDirectory);

    std::printf("\n%d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
