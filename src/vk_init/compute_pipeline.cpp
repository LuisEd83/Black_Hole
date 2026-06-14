#include "vk_init/compute_pipeline.hpp"
#include "utils.hpp"

vk::raii::ShaderModule ComputePipeline::createComputeShaderModule(const std::vector<uint32_t>& code) {
    vk::ShaderModuleCreateInfo createInfo{
        .codeSize = code.size() * sizeof(uint32_t),
        .pCode    = code.data()
    };
    return vk::raii::ShaderModule(this->device, createInfo);
}

std::vector<uint32_t> ComputePipeline::loadComputeShader() {
    return readFile("shaders/compute.spv");
}

void ComputePipeline::createComputeDescriptorSetLayout() {
    std::array<vk::DescriptorSetLayoutBinding, 4> bindings{{
        { .binding = 0, .descriptorType = vk::DescriptorType::eStorageImage,     .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute },
        { .binding = 1, .descriptorType = vk::DescriptorType::eUniformBuffer,     .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute },
        { .binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute },
        { .binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute }
    }};

    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings    = bindings.data()
    };
    this->computeDescriptorSetLayout = vk::raii::DescriptorSetLayout(this->device, layoutInfo);
}

void ComputePipeline::createComputePipelineLayout() {
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount         = 1,
        .pSetLayouts            = &*this->computeDescriptorSetLayout,
        .pushConstantRangeCount = 0
    };
    this->computePipelineLayout = vk::raii::PipelineLayout(this->device, pipelineLayoutInfo);
}

void ComputePipeline::createComputePipeline() {
    createComputeDescriptorSetLayout();
    createComputePipelineLayout();
    
    auto shaderCode   = loadComputeShader();
    auto shaderModule = createComputeShaderModule(shaderCode);

    vk::PipelineShaderStageCreateInfo stageInfo{
        .stage  = vk::ShaderStageFlagBits::eCompute,
        .module = *shaderModule,
        .pName  = "computeMain"
    };

    vk::ComputePipelineCreateInfo pipelineInfo{
        .stage  = stageInfo,
        .layout = *this->computePipelineLayout
    };

    this->computePipeline = vk::raii::Pipeline(this->device, nullptr, pipelineInfo);
}