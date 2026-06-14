#pragma once

#include <chrono>
#include "vk_init/graphics_pipeline.hpp"
#include "vulkan/vulkan.hpp"
#include <cstdint>
#include <glm/gtc/matrix_transform.hpp>
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <vulkan/vulkan.hpp>
#include <optional>
#include <string>
#include <vector>
#include "vk_utils/vkimage.hpp"
#include <vulkan/vulkan_raii.hpp>

constexpr int MAX_FRAMES_IN_FLIGHT = 2;
constexpr double RS = 12683881740.0;

struct RenderParams {
    int32_t width;
    int32_t height;
    alignas(32) glm::dvec3 pos;   // double3 in UBO requires 32-byte alignment
    alignas(32) glm::dvec3 fwd;
    alignas(32) glm::dvec3 right;
    alignas(32) glm::dvec3 up;
    float fov_y;
};

class Render : public GraphicsPipeline {
protected:
    vk::raii::CommandPool                   commandPool         = nullptr;

    std::optional<VulkanImage>              texture;
    std::optional<VulkanImage>              storageImage;
    std::optional<VulkanImage>              perlinTexture;
    
    std::vector<vk::raii::Buffer>           uniformBuffers;
    std::vector<vk::raii::DeviceMemory>     uniformBuffersMemory;
    std::vector<void *>                     uniformBuffersMapped;

    vk::raii::DescriptorPool                descriptorPool      = nullptr;
    std::vector<vk::raii::DescriptorSet>    computeDescriptorSets;
    std::vector<vk::raii::DescriptorSet>    graphicsDescriptorSets;
    
    std::vector<vk::raii::CommandBuffer>    commandBuffers;

    std::vector<vk::raii::Semaphore>        imageAvailableSemaphores;
    std::vector<vk::raii::Semaphore>        renderFinishedSemaphores;

    std::vector<vk::raii::Fence>            inFlightFences;
    
    // GPU timestamp queries for performance profiling
    vk::raii::QueryPool                     queryPool             = nullptr;
    std::array<uint64_t, 2>                 queryResults          = {};
    float                                   lastComputeTimeMs     = 0.0f;
    float                                   lastTotalTimeMs       = 0.0f;
    
    // Camera state for keyboard controls
    glm::dvec3                              cameraPos             = glm::dvec3(0.0);
    glm::dvec3                              cameraFwd             = glm::dvec3(0.0);
    glm::dvec3                              cameraRight           = glm::dvec3(0.0);
    glm::dvec3                              cameraUp              = glm::dvec3(0.0);
    double                                  cameraYaw             = 0.0;
    double                                  cameraPitch           = 0.0;
    bool                                    cameraInitialized     = false;
    
    uint32_t                                frameIndex          = 0;

public:
    void createCommandPool();
    void createCommandBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void cleanupSyncObjects();
    void createSyncObjects();
    void createQueryPool();
    void createPerlinTexture();
    void createTexture();
    void createStorageImage();
    void createUniformBuffers();
    void recordCommandBuffer(uint32_t imageIndex);
    void recreateStorageImage();
    void drawFrame();
    void exportToImage(const std::string& filename, uint32_t width, uint32_t height);
    

private:
    std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(   vk::DeviceSize size, vk::BufferUsageFlags usage,
                                                                        vk::MemoryPropertyFlags properties);

    void                    copyBuffer(vk::raii::Buffer & srcBuffer, vk::raii::Buffer & dstBuffer, vk::DeviceSize size);

    vk::raii::CommandBuffer beginSingleTimeCommands();
    void                    endSingleTimeCommands(vk::raii::CommandBuffer &&commandBuffer);
    
    void                    transitionImageLayout(uint32_t imageIndex, vk::ImageLayout old_layout, vk::ImageLayout new_layout, 
                                        vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask,
                                        vk::PipelineStageFlags2 src_stage_mask, vk::PipelineStageFlags2 dst_stage_mask);
    
    void                    transitionImageLayout(vk::raii::CommandBuffer &commandBuffer, const vk::raii::Image &image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);

    void                    updateUniformBuffer(uint32_t currentImage);
    void                    processKeyboardInput();

    uint32_t                findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);

    std::vector<float>      loadPerlinData(const std::string& filename, bool is3d, int& outWidth, int& outHeight, int&outDepth);

};
