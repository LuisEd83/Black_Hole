#include <chrono>
#include <cmath>
#include <glm/ext/vector_double3.hpp>
#include <glm/ext/vector_float3.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "vk_main/render.hpp"
#include "vulkan/vulkan.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <vulkan/vulkan_raii.hpp>
#include <fstream>
#include <iostream>


void Render::createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo{.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                       .queueFamilyIndex = this->queueIndex};

    this->commandPool = vk::raii::CommandPool(this->device, poolInfo);
}

void Render::createCommandBuffers() {
    vk::CommandBufferAllocateInfo allocInfo {
        .commandPool = *this->commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT
    };

    this->commandBuffers = vk::raii::CommandBuffers(this->device, allocInfo);
}

void Render::transitionImageLayout(uint32_t imageIndex, vk::ImageLayout old_layout, vk::ImageLayout new_layout,
                                   vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask,
                                   vk::PipelineStageFlags2 src_stage_mask, vk::PipelineStageFlags2 dst_stage_mask) {
    vk::ImageMemoryBarrier2 barrier = {
        .srcStageMask        = src_stage_mask,
        .srcAccessMask       = src_access_mask,
        .dstStageMask        = dst_stage_mask,
        .dstAccessMask       = dst_access_mask,
        .oldLayout           = old_layout,
        .newLayout           = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = this->swapChainImages[imageIndex],
        .subresourceRange    = {
               .aspectMask     = vk::ImageAspectFlagBits::eColor,
               .baseMipLevel   = 0,
               .levelCount     = 1,
               .baseArrayLayer = 0,
               .layerCount     = 1}};
    vk::DependencyInfo dependency_info = {
        .dependencyFlags         = {},
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &barrier};

    this->commandBuffers[this->frameIndex].pipelineBarrier2(dependency_info);
}

void Render::recordCommandBuffer(uint32_t imageIndex) {
    this->commandBuffers[this->frameIndex].begin({});

        // --- Timestamp: start of compute ---
        this->commandBuffers[this->frameIndex].resetQueryPool(*this->queryPool, 0, 2);
        this->commandBuffers[this->frameIndex].writeTimestamp(
            vk::PipelineStageFlagBits::eTopOfPipe, *this->queryPool, 0);

        // --- 1. Transition storage image to GENERAL for compute write ---
        {
            vk::ImageMemoryBarrier2 barrier = {
                .srcStageMask        = vk::PipelineStageFlagBits2::eTopOfPipe,
                .srcAccessMask       = {},
                .dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask       = vk::AccessFlagBits2::eShaderWrite,
                .oldLayout           = vk::ImageLayout::eUndefined,
                .newLayout           = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                .image               = this->storageImage->getImage(),
                .subresourceRange    = {
                    .aspectMask     = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel   = 0,
                    .levelCount     = 1,
                    .baseArrayLayer = 0,
                    .layerCount     = 1
                }
            };
            vk::DependencyInfo depInfo = {
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers    = &barrier
            };
            this->commandBuffers[this->frameIndex].pipelineBarrier2(depInfo);
        }

        // --- 2. Compute dispatch ---
        this->commandBuffers[this->frameIndex].bindPipeline(
            vk::PipelineBindPoint::eCompute, *this->computePipeline);
        this->commandBuffers[this->frameIndex].bindDescriptorSets(
            vk::PipelineBindPoint::eCompute,
            this->computePipelineLayout,
            0,
            *this->computeDescriptorSets[this->frameIndex],
            nullptr);

        uint32_t groupsX = (this->swapChainExtent.width + 15) / 16;
        uint32_t groupsY = (this->swapChainExtent.height + 15) / 16;
        this->commandBuffers[this->frameIndex].dispatch(groupsX, groupsY, 1);

        // --- 3. Barrier: compute write -> fragment read ---
        {
            vk::ImageMemoryBarrier2 barrier = {
                .srcStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask       = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask        = vk::PipelineStageFlagBits2::eFragmentShader,
                .dstAccessMask       = vk::AccessFlagBits2::eShaderRead,
                .oldLayout           = vk::ImageLayout::eGeneral,
                .newLayout           = vk::ImageLayout::eShaderReadOnlyOptimal,
                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                .image               = this->storageImage->getImage(),
                .subresourceRange    = {
                    .aspectMask     = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel   = 0,
                    .levelCount     = 1,
                    .baseArrayLayer = 0,
                    .layerCount     = 1
                }
            };
            vk::DependencyInfo depInfo = {
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers    = &barrier
            };
            this->commandBuffers[this->frameIndex].pipelineBarrier2(depInfo);
        }

        // --- 4. Existing graphics pass ---
        transitionImageLayout(
            imageIndex,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput
        );

        vk::RenderingAttachmentInfo attachmentInfo = {
            .imageView   = *this->swapChainImageViews[imageIndex],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eStore,
            .clearValue  = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f)};

        vk::RenderingInfo renderingInfo = {
            .renderArea           = {.offset = {0, 0}, .extent = this->swapChainExtent},
            .layerCount           = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &attachmentInfo};

        this->commandBuffers[this->frameIndex].beginRendering(renderingInfo);
        this->commandBuffers[this->frameIndex].bindPipeline(vk::PipelineBindPoint::eGraphics, *this->graphicsPipeline);

        this->commandBuffers[this->frameIndex].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(this->swapChainExtent.width), static_cast<float>(this->swapChainExtent.height), 0.0f, 1.0f));
        this->commandBuffers[this->frameIndex].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), this->swapChainExtent));

        this->commandBuffers[this->frameIndex].bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            this->graphicsPipelineLayout,
            0,
            *this->graphicsDescriptorSets[this->frameIndex],
            nullptr);

        this->commandBuffers[this->frameIndex].draw(4, 1, 0, 0);

        this->commandBuffers[this->frameIndex].endRendering();

        transitionImageLayout(
            imageIndex,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            {},
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eBottomOfPipe
        );

        // --- Timestamp: end of frame ---
        this->commandBuffers[this->frameIndex].writeTimestamp(
            vk::PipelineStageFlagBits::eBottomOfPipe, *this->queryPool, 1);

        this->commandBuffers[this->frameIndex].end();
}

void Render::cleanupSyncObjects() {
    this->imageAvailableSemaphores.clear();
    this->renderFinishedSemaphores.clear();
    this->inFlightFences.clear();
}

void Render::createSyncObjects() {
    this->imageAvailableSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
    this->renderFinishedSemaphores.reserve(this->swapChainImages.size());
    this->inFlightFences.reserve(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < this->swapChainImages.size(); i++) {
        this->renderFinishedSemaphores.emplace_back(this->device, vk::SemaphoreCreateInfo());
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        this->imageAvailableSemaphores.emplace_back(this->device, vk::SemaphoreCreateInfo());
        this->inFlightFences.emplace_back(this->device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
    }
}

void Render::createQueryPool() {
    vk::QueryPoolCreateInfo queryPoolInfo{
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = 2,  // start and end timestamp
    };
    this->queryPool = vk::raii::QueryPool(this->device, queryPoolInfo);
}

uint32_t Render::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties = this->physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> Render::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties) {
  vk::BufferCreateInfo   bufferInfo{.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};
  vk::raii::Buffer       buffer          = vk::raii::Buffer(this->device, bufferInfo);
  vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
  vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size, .memoryTypeIndex = this->findMemoryType(memRequirements.memoryTypeBits, properties)};
  vk::raii::DeviceMemory bufferMemory = vk::raii::DeviceMemory(this->device, allocInfo);
  buffer.bindMemory(*bufferMemory, 0);
  return {std::move(buffer), std::move(bufferMemory)};
}

vk::raii::CommandBuffer Render::beginSingleTimeCommands() {
  vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
  vk::raii::CommandBuffer       commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());

  vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
  commandBuffer.begin(beginInfo);

  return std::move(commandBuffer);
}

void Render::endSingleTimeCommands(vk::raii::CommandBuffer &&commandBuffer) {
  commandBuffer.end();

  vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer};
  this->graphicsQueue.submit(submitInfo, nullptr);
  this->graphicsQueue.waitIdle();
}

void Render::copyBuffer(vk::raii::Buffer & srcBuffer, vk::raii::Buffer & dstBuffer, vk::DeviceSize size) {
    vk::raii::CommandBuffer commandCopyBuffer = beginSingleTimeCommands();
      commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy{.size = size});
      endSingleTimeCommands(std::move(commandCopyBuffer));
}

void Render::createStarmapTexture() {
    int             texWidth, texHeight, texChannels;
    stbi_uc         *pixels     = stbi_load("data/starmap.png", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    vk::DeviceSize  imageSize   = static_cast<vk::DeviceSize>(texWidth * texHeight * 4);

    if (!pixels) {
            throw std::runtime_error("Failed to load texture image!");
        }


    this->texture.emplace(this->device, this->physicalDevice, texWidth, texHeight, 1, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst);

    auto cmd = beginSingleTimeCommands();

    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    this->texture->upload(cmd, stagingBuffer, stagingBufferMemory, std::span<const stbi_uc>(pixels, static_cast<size_t>(imageSize)));

    stbi_image_free(pixels);
    endSingleTimeCommands(std::move(cmd));


}

std::vector<float> Render::loadPerlinData(const std::string& filename, bool is3d, int& outWidth, int& outHeight, int&outDepth) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open perlin data file: " + filename);
    }

    std::vector<float> values;
    float val;
    while (file >> val) {
        values.push_back(val);
    }

    file.close();

    if (values.empty()) {
        throw std::runtime_error("Perlin data file is empty: " + filename);
    }


    // Check if first two values are integer dimensions
    int maybeW = static_cast<int>(values[0]);
    int maybeH = static_cast<int>(values[1]);
    int maybeD = static_cast<int>(values[2]);
    bool hasHeader = false;
    if (maybeW > 0 && maybeH > 0 && maybeD > 0 && std::fabs(values[0] - maybeW) < 1e-6f && std::fabs(values[1] - maybeH) < 1e-6f && std::fabs(values[2] - maybeD) < 1e-6f) {
        size_t expected = static_cast<size_t>(maybeW) * maybeH * maybeD;
        if (values.size() == expected + 3) {
            outWidth = maybeW;
            outHeight = maybeH;
            outDepth = maybeD;
            values.erase(values.begin(), values.begin() + 3);
            hasHeader = true;
        } else if (values.size() == expected) {
            outWidth = maybeW;
            outHeight = maybeH;
            outDepth = maybeD;
            hasHeader = true;
        }
    }

    if (!hasHeader) {
        size_t count = values.size();
        size_t dim = is3d ? static_cast<size_t>(std::cbrt(static_cast<double>(count))) : static_cast<size_t>(std::sqrt(static_cast<double>(count)));
        if (is3d) {
            if (dim * dim * dim != count) {
                throw std::runtime_error("Perlin data count is not a perfect cube and no valid dimensions found!");
            }
            outDepth = static_cast<int>(dim);
        } else {
            if (dim * dim != count) {
                throw std::runtime_error("Perlin data count is not a perfect square and no valid dimensions found!");
            }
            outDepth = 1;
        }
        outWidth = static_cast<int>(dim);
        outHeight = static_cast<int>(dim);
    }

    return values;
}


void Render::createPerlinTexture() {

    int texWidth, texHeight, texDepth;
    std::vector<float> perlinData = loadPerlinData("data/perlin.txt", false, texWidth, texHeight, texDepth);
    vk::DeviceSize perlinSize = static_cast<vk::DeviceSize>(texWidth * texHeight * texDepth * static_cast<int>(sizeof(float)));

    this->perlinTexture.emplace(
        this->device, this->physicalDevice,
        texWidth, texHeight, texDepth,
        vk::Format::eR32Sfloat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::Filter::eLinear
    );

    auto cmd = beginSingleTimeCommands();
    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(perlinSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    this->perlinTexture->upload<float>(cmd, stagingBuffer, stagingBufferMemory, perlinData);
    endSingleTimeCommands(std::move(cmd));
}

void Render::createStorageImage() {
    this->storageImage.emplace(
        this->device, this->physicalDevice,
        this->swapChainExtent.width, this->swapChainExtent.height, 1,
        vk::Format::eR32G32B32A32Sfloat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
        vk::Filter::eNearest
    );

    auto cmd = beginSingleTimeCommands();
    this->storageImage->transitionLayout(cmd,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eGeneral);
    endSingleTimeCommands(std::move(cmd));
}

void Render::recreateStorageImage() {
    this->device.waitIdle();
    this->storageImage.reset();

    this->createStorageImage();
}

void Render::createUniformBuffers() {
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DeviceSize bufferSize = sizeof(RenderParams);
        auto [buffer, bufferMem] =
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        this->uniformBuffers.emplace_back(std::move(buffer));
        this->uniformBuffersMemory.emplace_back(std::move(bufferMem));
        this->uniformBuffersMapped.emplace_back(uniformBuffersMemory.back().mapMemory(0, bufferSize));
    }
}

void Render::processKeyboardInput() {
    if (!this->window) return;

        // Initialize camera on first frame
        // All positions are in units of RS (shader uses rs = 1.0)
        // Matches original Black_Hole CUDA camera position:
        //   pos = RS * CAMERA_FACTOR * (X_COEF, Y_COEF, Z_COEF)
        //   with CAMERA_FACTOR=12.0, X_COEF=1.0, Y_COEF=1.1, Z_COEF=0.7
        // This gives theta ≈ 64.8° (near equator, disk visible).
        if (!this->cameraInitialized) {
            double cam_dist = 12.0;  // CAMERA_FACTOR
            double x_coef = 1.0;
            double y_coef = 1.1;
            double z_coef = 0.7;

            this->cameraPos = glm::dvec3(
                cam_dist * x_coef,
                cam_dist * y_coef,
                cam_dist * z_coef
            );

            // Original target from gl.cpp: target = vec3(-RS*3.0, RS*1.5, 0.0)
            // Camera looks slightly away from origin so the black hole is not dead-center
            glm::dvec3 target = glm::dvec3(-3.0, 1.5, 0.0);
            this->cameraFwd = glm::normalize(target - this->cameraPos);
            this->cameraRight = glm::normalize(glm::cross(this->cameraFwd, glm::dvec3(0.0, 0.0, 1.0)));
            this->cameraUp = glm::normalize(glm::cross(this->cameraRight, this->cameraFwd));

            this->cameraYaw = atan2(this->cameraFwd.y, this->cameraFwd.x);
            this->cameraPitch = asin(this->cameraFwd.z);
            this->cameraInitialized = true;
        }

    // Movement speed (in units of RS, matching the shader scale)
    double moveSpeed = 0.5;         // Move half a Schwarzschild radii per frame
    double rotSpeed = 0.02;       // Rotation speed

    // Movement (WASD)
    if (glfwGetKey(this->window, GLFW_KEY_W) == GLFW_PRESS) {
        this->cameraPos += this->cameraFwd * moveSpeed;
    }
    if (glfwGetKey(this->window, GLFW_KEY_S) == GLFW_PRESS) {
        this->cameraPos -= this->cameraFwd * moveSpeed;
    }
    if (glfwGetKey(this->window, GLFW_KEY_A) == GLFW_PRESS) {
        this->cameraPos -= this->cameraRight * moveSpeed;
    }
    if (glfwGetKey(this->window, GLFW_KEY_D) == GLFW_PRESS) {
        this->cameraPos += this->cameraRight * moveSpeed;
    }
    if (glfwGetKey(this->window, GLFW_KEY_Q) == GLFW_PRESS) {
        this->cameraPos -= this->cameraUp * moveSpeed;
    }
    if (glfwGetKey(this->window, GLFW_KEY_E) == GLFW_PRESS) {
        this->cameraPos += this->cameraUp * moveSpeed;
    }

    // Rotation (Arrow keys)
    if (glfwGetKey(this->window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        this->cameraYaw += rotSpeed;
    }
    if (glfwGetKey(this->window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        this->cameraYaw -= rotSpeed;
    }
    if (glfwGetKey(this->window, GLFW_KEY_UP) == GLFW_PRESS) {
        this->cameraPitch -= rotSpeed;
    }
    if (glfwGetKey(this->window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        this->cameraPitch += rotSpeed;
    }

    // Clamp pitch
    if (this->cameraPitch > 89.0 * 3.14159265358979323846 / 180.0) {
        this->cameraPitch = 89.0 * 3.14159265358979323846 / 180.0;
    }
    if (this->cameraPitch < -89.0 * 3.14159265358979323846 / 180.0) {
        this->cameraPitch = -89.0 * 3.14159265358979323846 / 180.0;
    }

    // Recalculate forward vector from yaw and pitch
    this->cameraFwd = glm::dvec3(
        cos(this->cameraPitch) * cos(this->cameraYaw),
        cos(this->cameraPitch) * sin(this->cameraYaw),
        sin(this->cameraPitch)
    );

    // Recalculate right and up vectors
    this->cameraRight = glm::normalize(glm::cross(this->cameraFwd, glm::dvec3(0.0, 0.0, 1.0)));
    this->cameraUp = glm::normalize(glm::cross(this->cameraRight, this->cameraFwd));

    // Ensure minimum distance from black hole (in units of RS)
    double minDist = 3.0;
    double dist = glm::length(this->cameraPos);
    if (dist < minDist) {
        this->cameraPos = glm::normalize(this->cameraPos) * minDist;
    }
}

void Render::updateUniformBuffer(uint32_t currentImage) {
    processKeyboardInput();

    // Camera positions are already in units of RS (shader uses rs = 1.0)
    // The cameraPos was initialized to 30.0 (not 30*RS), so no division needed
    RenderParams renderParams{
        .width  = (int32_t) this->swapChainExtent.width,
        .height = (int32_t) this->swapChainExtent.height,
        .pos    = this->cameraPos,
        .fwd    = this->cameraFwd,
        .right  = this->cameraRight,
        .up     = this->cameraUp,
        .fov_y  = 60.0f
    };

    memcpy(this->uniformBuffersMapped[currentImage], &renderParams, sizeof(renderParams));
}

void Render::createDescriptorPool() {
    std::array<vk::DescriptorPoolSize, 3> poolSize{{
            { .type = vk::DescriptorType::eUniformBuffer,        .descriptorCount = MAX_FRAMES_IN_FLIGHT + 2 },  // +2 for export
            { .type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 3 + 4 }, // +4 for export
            { .type = vk::DescriptorType::eStorageImage,        .descriptorCount = MAX_FRAMES_IN_FLIGHT + 2 }   // +2 for export
        }};

        vk::DescriptorPoolCreateInfo poolInfo{
            .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets       = MAX_FRAMES_IN_FLIGHT * 2 + 4, // compute + graphics per frame + extra for export
            .poolSizeCount = static_cast<uint32_t>(poolSize.size()),
            .pPoolSizes    = poolSize.data()
        };

        this->descriptorPool = vk::raii::DescriptorPool(this->device, poolInfo);
}

void Render::createDescriptorSets() {
    // Free old sets before allocating new ones
    this->computeDescriptorSets.clear();
    this->graphicsDescriptorSets.clear();

    // Allocate compute sets
    std::vector<vk::DescriptorSetLayout> computeLayouts(MAX_FRAMES_IN_FLIGHT, *this->computeDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo computeAllocInfo{
        .descriptorPool     = descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(computeLayouts.size()),
        .pSetLayouts        = computeLayouts.data()
    };
    this->computeDescriptorSets = this->device.allocateDescriptorSets(computeAllocInfo);

    // Allocate graphics sets
    std::vector<vk::DescriptorSetLayout> graphicsLayouts(MAX_FRAMES_IN_FLIGHT, *this->graphicsDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo graphicsAllocInfo{
        .descriptorPool     = descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(graphicsLayouts.size()),
        .pSetLayouts        = graphicsLayouts.data()
    };
    this->graphicsDescriptorSets = this->device.allocateDescriptorSets(graphicsAllocInfo);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        // --- Compute set writes ---
        vk::DescriptorBufferInfo bufferInfo{
            .buffer = this->uniformBuffers[i],
            .offset = 0,
            .range  = sizeof(RenderParams)
        };

        vk::DescriptorImageInfo storageImageComputeInfo{
            .imageView   = this->storageImage->getImageView(),
            .imageLayout = vk::ImageLayout::eGeneral
        };

        vk::DescriptorImageInfo starmapInfo = this->texture->getDescriptorImageInfo();

        // If you haven't finished 4b (perlin loading), load it first or create a dummy VulkanImage.
        vk::DescriptorImageInfo perlinInfo = this->perlinTexture->getDescriptorImageInfo();

        std::array<vk::WriteDescriptorSet, 4> computeWrites{{
            { .dstSet = *this->computeDescriptorSets[i], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage,        .pImageInfo = &storageImageComputeInfo },
            { .dstSet = *this->computeDescriptorSets[i], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer,      .pBufferInfo = &bufferInfo },
            { .dstSet = *this->computeDescriptorSets[i], .dstBinding = 2, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &starmapInfo },
            { .dstSet = *this->computeDescriptorSets[i], .dstBinding = 3, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &perlinInfo }
        }};

        // --- Graphics set write ---
        vk::DescriptorImageInfo storageImageGraphicsInfo{
            .sampler     = this->storageImage->getSampler(),
            .imageView   = this->storageImage->getImageView(),
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        };

        std::array<vk::WriteDescriptorSet, 1> graphicsWrites{{
            { .dstSet = *this->graphicsDescriptorSets[i], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &storageImageGraphicsInfo }
        }};

        this->device.updateDescriptorSets(computeWrites, {});
        this->device.updateDescriptorSets(graphicsWrites, {});
    }
}

void Render::transitionImageLayout(vk::raii::CommandBuffer &commandBuffer, const vk::raii::Image &image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
    vk::ImageMemoryBarrier barrier{.oldLayout           = oldLayout,
                                   .newLayout           = newLayout,
                                   .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                                   .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                                   .image               = image,
                                   .subresourceRange    = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1}};

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
      barrier.srcAccessMask = {};
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

      sourceStage      = vk::PipelineStageFlagBits::eTopOfPipe;
      destinationStage = vk::PipelineStageFlagBits::eTransfer;
    }
    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

      sourceStage      = vk::PipelineStageFlagBits::eTransfer;
      destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    }
    else {
      throw std::invalid_argument("unsupported layout transition!");
    }

    commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, nullptr, barrier);
}

void copyBufferToImage(vk::raii::CommandBuffer &commandBuffer, const vk::raii::Buffer &buffer, vk::raii::Image &image, uint32_t width, uint32_t height) {
    vk::BufferImageCopy region{.bufferOffset      = 0,
                               .bufferRowLength   = 0,
                               .bufferImageHeight = 0,
                               .imageSubresource  = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                               .imageOffset       = {0, 0, 0},
                               .imageExtent       = {width, height, 1}};

    commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
}

void Render::drawFrame() {
    auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);

    if (fenceResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to wait for fence!");
    }

    auto [result, imageIndex] = this->swapChain.acquireNextImage(UINT64_MAX, *this->imageAvailableSemaphores[this->frameIndex], nullptr);

    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || this->framebufferResized) {
        this->framebufferResized = false;
        this->recreateSwapChain();
        this->recreateStorageImage();
        this->createDescriptorSets();
        this->cleanupSyncObjects();
        this->createSyncObjects();

        // Try to acquire again with the new swapchain
        auto [result2, imageIndex2] = this->swapChain.acquireNextImage(UINT64_MAX, *this->imageAvailableSemaphores[this->frameIndex], nullptr);
        if (result2 != vk::Result::eSuccess && result2 != vk::Result::eSuboptimalKHR) {
            return;
        }
        result = result2;
        imageIndex = imageIndex2;
    }

    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
      if (result == vk::Result::eTimeout || result == vk::Result::eNotReady) {
          throw std::runtime_error("swap chain image aquisition wasn't ready/timed out!");
      }
      throw std::runtime_error("failed to acquire swap chain image!");
    }

    // Only reset fence after successful acquire — ensures fence is never left
    // unsignaled if acquire fails and we return early.
    this->device.resetFences(*inFlightFences[frameIndex]);

    this->commandBuffers[frameIndex].reset();
    this->recordCommandBuffer(imageIndex);

    vk::Semaphore waitSemaphore   = *this->imageAvailableSemaphores[this->frameIndex];
    vk::Semaphore signalSemaphore = *this->renderFinishedSemaphores[imageIndex];
    vk::CommandBuffer cmdBuffer   = *this->commandBuffers[this->frameIndex];
    vk::PipelineStageFlags waitDestinationStageMask( vk::PipelineStageFlagBits::eColorAttachmentOutput );

    this->updateUniformBuffer(this->frameIndex);

    const vk::SubmitInfo   submitInfo{.waitSemaphoreCount   = 1,
                                      .pWaitSemaphores      = &waitSemaphore,
                                      .pWaitDstStageMask    = &waitDestinationStageMask,
                                      .commandBufferCount   = 1,
                                      .pCommandBuffers      = &cmdBuffer,
                                      .signalSemaphoreCount = 1,
                                      .pSignalSemaphores    = &signalSemaphore};

    this->graphicsQueue.submit(submitInfo, *this->inFlightFences[this->frameIndex]);

    vk::SwapchainKHR swapChainKHR = *this->swapChain;

    const vk::PresentInfoKHR presentInfoKHR{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores    = &*this->renderFinishedSemaphores[imageIndex],
        .swapchainCount     = 1,
        .pSwapchains        = &swapChainKHR,
        .pImageIndices      = &imageIndex};

    result = this->graphicsQueue.presentKHR(presentInfoKHR);

    if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR)) {
      this->recreateSwapChain();
      this->recreateStorageImage();
      this->createDescriptorSets();
    } else {
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("unknown error while drawing frames!");
        }
    }


    this->frameIndex = (this->frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Render::exportToImage(const std::string& filename, uint32_t width, uint32_t height) {
    // Create a temporary storage image for the specified size
    this->storageImage.emplace(
        this->device, this->physicalDevice,
        width, height, 1,
        vk::Format::eR32G32B32A32Sfloat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
        vk::Filter::eNearest
    );

    // Create a temporary uniform buffer
    vk::DeviceSize bufferSize = sizeof(RenderParams);
    auto [uniformBuffer, uniformBufferMemory] =
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* uniformBufferMapped = uniformBufferMemory.mapMemory(0, bufferSize);

    // Create a temporary descriptor set
    std::vector<vk::DescriptorSetLayout> computeLayouts(1, *this->computeDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo computeAllocInfo{
        .descriptorPool     = this->descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts        = computeLayouts.data()
    };
    auto computeDescriptorSet = this->device.allocateDescriptorSets(computeAllocInfo);

    this->cameraPos = glm::dvec3(13.0, 16.0, -7.37);

    // Original target from gl.cpp: target = vec3(-RS*3.0, RS*1.5, 0.0)
    // Camera looks slightly away from origin so the black hole is not dead-center
    this->cameraFwd = glm::dvec3(-0.6, -0.73, 0.32);
    this->cameraRight = glm::normalize(glm::cross(this->cameraFwd, glm::dvec3(0.0, 0.0, 1.0)));
    this->cameraUp = glm::normalize(glm::cross(this->cameraRight, this->cameraFwd));

    this->cameraYaw = atan2(this->cameraFwd.y, this->cameraFwd.x);
    this->cameraPitch = asin(this->cameraFwd.z);

    // Update uniform buffer
    RenderParams renderParams{
        .width  = static_cast<int32_t>(width),
        .height = static_cast<int32_t>(height),
        .pos    = this->cameraPos,
        .fwd    = this->cameraFwd,
        .right  = this->cameraRight,
        .up     = this->cameraUp,
        .fov_y  = 60.0f
    };
    memcpy(uniformBufferMapped, &renderParams, sizeof(renderParams));

    // Create descriptor set writes
    vk::DescriptorBufferInfo bufferInfo{
        .buffer = uniformBuffer,
        .offset = 0,
        .range  = sizeof(RenderParams)
    };
    vk::DescriptorImageInfo storageImageInfo{
        .imageView   = this->storageImage->getImageView(),
        .imageLayout = vk::ImageLayout::eGeneral
    };
    vk::DescriptorImageInfo starmapInfo = this->texture->getDescriptorImageInfo();
    vk::DescriptorImageInfo perlinInfo = this->perlinTexture->getDescriptorImageInfo();

    std::array<vk::WriteDescriptorSet, 4> computeWrites{{
        { .dstSet = *computeDescriptorSet[0], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage,        .pImageInfo = &storageImageInfo },
        { .dstSet = *computeDescriptorSet[0], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer,      .pBufferInfo = &bufferInfo },
        { .dstSet = *computeDescriptorSet[0], .dstBinding = 2, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &starmapInfo },
        { .dstSet = *computeDescriptorSet[0], .dstBinding = 3, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &perlinInfo }
    }};
    this->device.updateDescriptorSets(computeWrites, {});

    // Create a buffer to copy the image to
    vk::DeviceSize imageSize = width * height * 4 * sizeof(float);
    auto [readBuffer, readBufferMemory] =
        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    // Record command buffer
    auto cmd = beginSingleTimeCommands();

    // Transition storage image to GENERAL
    this->storageImage->transitionLayout(cmd,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eGeneral);

    // Bind compute pipeline and dispatch
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *this->computePipeline);
    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        this->computePipelineLayout,
        0,
        *computeDescriptorSet[0],
        nullptr);

    uint32_t groupsX = (width + 15) / 16;
    uint32_t groupsY = (height + 15) / 16;
    cmd.dispatch(groupsX, groupsY, 1);

    // Barrier: compute write -> transfer read
    vk::ImageMemoryBarrier2 barrier = {
        .srcStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask       = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask        = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask       = vk::AccessFlagBits2::eTransferRead,
        .oldLayout           = vk::ImageLayout::eGeneral,
        .newLayout           = vk::ImageLayout::eTransferSrcOptimal,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image               = this->storageImage->getImage(),
        .subresourceRange    = {
            .aspectMask     = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1
        }
    };
    vk::DependencyInfo depInfo = {
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &barrier
    };
    cmd.pipelineBarrier2(depInfo);

    // Copy image to buffer
    vk::BufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = {width, height, 1}
    };
    cmd.copyImageToBuffer(this->storageImage->getImage(), vk::ImageLayout::eTransferSrcOptimal, readBuffer, region);

    endSingleTimeCommands(std::move(cmd));

    // Read back the image data
    void* data = readBufferMemory.mapMemory(0, imageSize);
    float* floatData = static_cast<float*>(data);

    // Convert float32 to uint8 and save
    std::vector<uint8_t> pixels(width * height * 3);
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t srcIdx = (y * width + x) * 4;
            uint32_t dstIdx = ((height - 1 - y) * width + x) * 3; // Flip Y for PNG
            pixels[dstIdx + 0] = static_cast<uint8_t>(std::clamp(floatData[srcIdx + 0] * 255.0f, 0.0f, 255.0f));
            pixels[dstIdx + 1] = static_cast<uint8_t>(std::clamp(floatData[srcIdx + 1] * 255.0f, 0.0f, 255.0f));
            pixels[dstIdx + 2] = static_cast<uint8_t>(std::clamp(floatData[srcIdx + 2] * 255.0f, 0.0f, 255.0f));
        }
    }

    stbi_write_png(filename.c_str(), width, height, 3, pixels.data(), width * 3);

    readBufferMemory.unmapMemory();
    uniformBufferMemory.unmapMemory();

    std::cout << "Exported frame to " << filename << " (" << width << "x" << height << ")" << std::endl;
}
