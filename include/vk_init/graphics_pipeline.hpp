#pragma once

#include <array>
#include <cstddef>
#include <glm/glm.hpp>
#include "vk_init/compute_pipeline.hpp"
#include <vulkan/vulkan.hpp>
#include <vector>
#include <string>
#include <vulkan/vulkan_raii.hpp>

class GraphicsPipeline : public ComputePipeline {
protected:
    vk::raii::DescriptorSetLayout       graphicsDescriptorSetLayout = nullptr;
    vk::raii::PipelineLayout            graphicsPipelineLayout      = nullptr;
    vk::raii::Pipeline                  graphicsPipeline            = nullptr;

public:
    void createGraphicsPipeline();

private:
    void createGraphicsDescriptorSetLayout();
    void createGraphicsPipelineLayout();
    vk::raii::ShaderModule createShaderModule(const std::vector<uint32_t>& code);
    std::vector<uint32_t> loadShader();
    std::vector<vk::PipelineShaderStageCreateInfo> createShaderStages(vk::raii::ShaderModule& shaderModule);
    void createPipeline(std::vector<vk::PipelineShaderStageCreateInfo>& shaderStages);
};
