#pragma once

#include "vk_init/presentation.hpp"
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

class ComputePipeline : public Presentation {
protected:
    vk::raii::DescriptorSetLayout computeDescriptorSetLayout = nullptr;
    vk::raii::PipelineLayout      computePipelineLayout      = nullptr;
    vk::raii::Pipeline            computePipeline            = nullptr;

public:
    void createComputePipeline();

private:
    void createComputeDescriptorSetLayout();
    void createComputePipelineLayout();
    vk::raii::ShaderModule createComputeShaderModule(const std::vector<uint32_t>& code);
    std::vector<uint32_t> loadComputeShader();
};