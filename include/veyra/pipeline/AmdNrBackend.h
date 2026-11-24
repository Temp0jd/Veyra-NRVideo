// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <d3d12.h>
#include <cstdint>
#include <string>
namespace veyra::pipeline {
// Same-frame linear input/output contract shared by independent AMD providers.
class AmdNrBackend {
public:
    virtual ~AmdNrBackend()=default;
    virtual bool ready()const=0;
    virtual bool admits(uint32_t,uint32_t)const=0;
    virtual bool recordInputs(ID3D12GraphicsCommandList*,ID3D12Resource*,uint64_t,bool,float)=0;
    virtual bool enqueue()=0;
    virtual bool recordOutputs(ID3D12GraphicsCommandList*,ID3D12Resource*)=0;
    virtual bool retireSubmitted()=0;
    // True when recordInputs reads the graph's zeroed motion/depth guidance textures, so the
    // graph must publish them in NON_PIXEL_SHADER_RESOURCE before the call.
    virtual bool needsGuidance()const{return false;}
    virtual const std::string& error()const=0;
};
}
