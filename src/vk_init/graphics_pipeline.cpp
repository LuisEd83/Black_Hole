#include "vk_init/graphics_pipeline.hpp"
#include "utils.hpp"
#include "vulkan/vulkan.hpp"

#include <cstdint>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

vk::raii::ShaderModule GraphicsPipeline::createShaderModule(const std::vector<uint32_t>& code) {
    vk::ShaderModuleCreateInfo createInfo{ .codeSize = code.size() * sizeof(uint32_t), .pCode = code.data() };
    vk::raii::ShaderModule shaderModule{ this->device, createInfo };
    return shaderModule;
}

std::vector<uint32_t> GraphicsPipeline::loadShader() {
    return readFile("shaders/graphics.spv");
}

std::vector<vk::PipelineShaderStageCreateInfo> GraphicsPipeline::createShaderStages(vk::raii::ShaderModule& shaderModule) {
    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = *shaderModule,  .pName = "vertMain" };
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = *shaderModule, .pName = "fragMain" };

    return {vertShaderStageInfo, fragShaderStageInfo};
}

void GraphicsPipeline::createGraphicsDescriptorSetLayout() {
    std::array<vk::DescriptorSetLayoutBinding, 1> bindings{{
        { .binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment }
    }};

    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings    = bindings.data()
    };
    this->graphicsDescriptorSetLayout = vk::raii::DescriptorSetLayout(this->device, layoutInfo);
}

void GraphicsPipeline::createGraphicsPipelineLayout() {
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount         = 1,
        .pSetLayouts            = &*this->graphicsDescriptorSetLayout,
        .pushConstantRangeCount = 0
    };
    this->graphicsPipelineLayout = vk::raii::PipelineLayout(this->device, pipelineLayoutInfo);
}

void GraphicsPipeline::createPipeline(std::vector<vk::PipelineShaderStageCreateInfo>& shaderStages) {

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo {
        .vertexBindingDescriptionCount = 0,
        .vertexAttributeDescriptionCount = 0
    };
    
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleStrip};

    std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

    vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data()};

    vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

    vk::PipelineRasterizationStateCreateInfo rasterizer{.depthClampEnable        = vk::False,
                                                        .rasterizerDiscardEnable = vk::False,
                                                        .polygonMode             = vk::PolygonMode::eFill,
                                                        .cullMode                = vk::CullModeFlagBits::eNone,
                                                        .frontFace               = vk::FrontFace::eCounterClockwise,
                                                        .depthBiasEnable         = vk::False,
                                                        .lineWidth               = 1.0f};


    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable         = vk::True,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp        = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp        = vk::BlendOp::eAdd,
        .colorWriteMask      = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment};

    vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};

    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
        {.stageCount          = 2,
         .pStages             = shaderStages.data(),
         .pVertexInputState   = &vertexInputInfo,
         .pInputAssemblyState = &inputAssembly,
         .pViewportState      = &viewportState,
         .pRasterizationState = &rasterizer,
         .pMultisampleState   = &multisampling,
         .pColorBlendState    = &colorBlending,
         .pDynamicState       = &dynamicState,
         .layout              = *this->graphicsPipelineLayout,
         .renderPass          = nullptr},
        {.colorAttachmentCount = 1, .pColorAttachmentFormats = &this->swapChainSurfaceFormat.format}};

    this->graphicsPipeline = vk::raii::Pipeline(this->device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
}

void GraphicsPipeline::createGraphicsPipeline() {
    auto shaderCode     = this->loadShader();
    auto shaderModule   = this->createShaderModule(shaderCode);
    auto shaderStages   = this->createShaderStages(shaderModule);
    this->createGraphicsDescriptorSetLayout();
    this->createGraphicsPipelineLayout();
    this->createPipeline(shaderStages);
}
