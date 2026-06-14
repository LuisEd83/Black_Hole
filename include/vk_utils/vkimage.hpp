#pragma once

#include "vulkan/vulkan.hpp"
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <vulkan/vulkan_raii.hpp>

class VulkanImage {
public:
    VulkanImage() = default;

    VulkanImage(
        vk::raii::Device&           device,
        vk::raii::PhysicalDevice&   physicalDevice,
        uint32_t                    width,
        uint32_t                    height,
        uint32_t                    depth,
        vk::Format                  format,
        vk::ImageTiling             tiling,
        vk::ImageUsageFlags         usage,
        vk::Filter                  samplerFilter   = vk::Filter::eLinear,
        vk::SamplerAddressMode      addressMode     = vk::SamplerAddressMode::eRepeat
    ) : m_device(&device),
        m_width(width),
        m_height(height),
        m_depth(depth),
        m_format(format),
        m_usage(usage)
    {
        createImage();
        allocateMemory(physicalDevice);
        createImageView();
        createSampler(samplerFilter, addressMode);
    }

    template<typename T>
    requires std::is_trivially_copyable_v<T>
    void upload(
        vk::raii::CommandBuffer&    commandBuffer,
        vk::raii::Buffer&           stagingBuffer,
        vk::raii::DeviceMemory&     stagingBufferMemory,
        std::span<const T>          data
    ) {
        vk::DeviceSize expectedSize  = this->m_width * this->m_height * this->m_depth * getBytesPerDataUnit(this->m_format);
        vk::DeviceSize actualSize    = data.size() * sizeof(T);

        if (actualSize != expectedSize) {
            throw std::runtime_error("Data size mismatch");
        }

        void* bufmem = stagingBufferMemory.mapMemory(0, actualSize);
        memcpy(bufmem, data.data(), actualSize);
        stagingBufferMemory.unmapMemory();

        transitionLayout(
            commandBuffer,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal
        );

        vk::BufferImageCopy region{
            .bufferOffset      = 0,
            .bufferRowLength   = 0,
            .bufferImageHeight = 0,
            .imageSubresource  = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
            .imageOffset       = {0, 0, 0},
            .imageExtent       = {this->m_width, this->m_height, this->m_depth}
        };

        commandBuffer.copyBufferToImage(*stagingBuffer, **this->m_image, vk::ImageLayout::eTransferDstOptimal, region);

        transitionLayout(
            commandBuffer,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal
        );
    }

    void transitionLayout(
        vk::raii::CommandBuffer&    commandBuffer,
        vk::ImageLayout             oldLayout,
        vk::ImageLayout             newLayout
    ) {
        vk::ImageMemoryBarrier barrier{
            .oldLayout           = oldLayout,
            .newLayout           = newLayout,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image               = **this->m_image,
            .subresourceRange    = {
                .aspectMask     = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel   = 0,
                .levelCount     = 1,
                .baseArrayLayer = 0,
                .layerCount     = 1
            }
        };

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined &&
            newLayout == vk::ImageLayout::eTransferDstOptimal)
        {
            barrier.srcAccessMask = vk::AccessFlagBits::eNone;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage      = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        }
        else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
                 newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
        {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            sourceStage      = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else if (oldLayout == vk::ImageLayout::eUndefined &&
                 newLayout == vk::ImageLayout::eGeneral)
        {
            barrier.srcAccessMask = vk::AccessFlagBits::eNone;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderWrite;

            sourceStage      = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eComputeShader;
        }

        commandBuffer.pipelineBarrier(
            sourceStage, destinationStage,
            vk::DependencyFlagBits::eByRegion,
            {}, {}, barrier
        );
    }


    vk::Image           getImage()      const { return **this->m_image; }
    vk::ImageView       getImageView()  const { return **this->m_imageView; }
    vk::Sampler         getSampler()    const { return **this->m_sampler; }

    vk::DescriptorImageInfo getDescriptorImageInfo(
        vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal
    ) const {
        return vk::DescriptorImageInfo{
            .sampler     = **this->m_sampler,
            .imageView   = **this->m_imageView,
            .imageLayout = layout
        };
    }

    uint32_t    getWidth()  const { return this->m_width; }
    uint32_t    getHeight() const { return this->m_height; }
    vk::Format  getFormat() const { return this->m_format; }

    bool        isValid()   const { return this->m_image.has_value(); }

private:

    void createImage() {
        vk::ImageCreateInfo imageInfo{
            .imageType     = this->m_depth > 1 ? vk::ImageType::e3D : vk::ImageType::e2D,
            .format        = this->m_format,
            .extent        = {.width = this->m_width, .height = this->m_height, .depth = this->m_depth},
            .mipLevels     = 1,
            .arrayLayers   = 1,
            .samples       = vk::SampleCountFlagBits::e1,
            .tiling        = vk::ImageTiling::eOptimal,
            .usage         = this->m_usage,
            .sharingMode   = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined
        };

        this->m_image.emplace(*this->m_device, imageInfo);

        vk::MemoryRequirements memRequirements = (*this->m_image).getMemoryRequirements();
        this->m_size = memRequirements.size;
    }

    void createImageView() {
        vk::ImageViewCreateInfo viewInfo{
            .image            = **this->m_image,
            .viewType         = this->m_depth > 1 ? vk::ImageViewType::e3D : vk::ImageViewType::e2D,
            .format           = this->m_format,
            .subresourceRange = {
                .aspectMask     = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel   = 0,
                .levelCount     = 1,
                .baseArrayLayer = 0,
                .layerCount     = 1
            }
        };

        this->m_imageView.emplace(*this->m_device, viewInfo);
    }

    void createSampler(vk::Filter filter, vk::SamplerAddressMode addressMode) {
        vk::SamplerCreateInfo samplerInfo{
            .magFilter    = filter,
            .minFilter    = filter,
            .addressModeU = addressMode,
            .addressModeV = addressMode,
            .addressModeW = addressMode
        };

        this->m_sampler.emplace(*this->m_device, samplerInfo);
    }

    void allocateMemory(vk::raii::PhysicalDevice& physicalDevice) {
        vk::MemoryRequirements memRequirements = (*this->m_image).getMemoryRequirements();

        vk::MemoryAllocateInfo allocInfo{
            .allocationSize  = memRequirements.size,
            .memoryTypeIndex = findMemoryType(
                physicalDevice,
                memRequirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal
            )
        };

        this->m_memory.emplace(*this->m_device, allocInfo);
        (*this->m_image).bindMemory(**this->m_memory, 0);
    }


    static uint32_t findMemoryType(
        vk::raii::PhysicalDevice&   physicalDevice,
        uint32_t                    typeFilter,
        vk::MemoryPropertyFlags     properties
    ) {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type");
    }

    static vk::DeviceSize getBytesPerDataUnit(vk::Format format) {
        switch (format) {
            case vk::Format::eR8G8B8A8Unorm:
            case vk::Format::eR8G8B8A8Srgb:
            case vk::Format::eB8G8R8A8Unorm:
            case vk::Format::eB8G8R8A8Srgb:
                return 4;
            case vk::Format::eR8Unorm:
                return 1;

            case vk::Format::eR16G16B16A16Sfloat:
                return 8;

            case vk::Format::eR32Sfloat:
                return 4;

            case vk::Format::eR32G32B32A32Sfloat:
            case vk::Format::eR32G32B32Sfloat:
                return 16;

            case vk::Format::eD32Sfloat:
                return 4;

            default:
                throw std::runtime_error("Unsupported format");
        }
    }

    // Non-owning pointer — the device's lifetime is managed by the caller (Setup/Render)
    vk::raii::Device*                       m_device    = nullptr;

    std::optional<vk::raii::Image>          m_image;
    std::optional<vk::raii::DeviceMemory>   m_memory;
    std::optional<vk::raii::ImageView>      m_imageView;
    std::optional<vk::raii::Sampler>        m_sampler;

    uint32_t                m_width     = 0;
    uint32_t                m_height    = 0;
    uint32_t                m_depth     = 1;
    vk::Format              m_format    = vk::Format::eUndefined;
    vk::DeviceSize          m_size      = 0;
    vk::ImageUsageFlags     m_usage     = {};
};
