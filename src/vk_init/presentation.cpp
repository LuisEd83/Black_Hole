#include "vk_init/presentation.hpp"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <limits>

void Presentation::createSurface() {
    // NOTE: This is the only raw C Vulkan API interop in the codebase.
    // GLFW is a C library and returns a VkSurfaceKHR handle. We must
    // wrap it in vk::raii::SurfaceKHR immediately. If the RAII constructor
    // throws, we destroy the raw handle manually to prevent a leak.
    VkSurfaceKHR surf;
    VkResult result = glfwCreateWindowSurface(*this->inst, this->window, nullptr, &surf);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }

    try {
        this->surface = vk::raii::SurfaceKHR(this->inst, surf);
    } catch (...) {
        vkDestroySurfaceKHR(*this->inst, surf, nullptr);
        throw;
    }
}

vk::PresentModeKHR Presentation::chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const &availablePresentModes)
{
    if (!std::ranges::any_of(availablePresentModes, [](auto presentMode) { return presentMode == vk::PresentModeKHR::eFifo; })) {
        throw std::runtime_error("FIFO present mode not available — required by Vulkan spec");
    }
    // Use FIFO (vsync) to cap frame rate and reduce GPU load
    // Mailbox is available but uncaps frame rate, causing max GPU usage
    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D Presentation::chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        return capabilities.currentExtent;
    }
    int width, height;
    glfwGetFramebufferSize(this->window, &width, &height);

    return {
        std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
    };
}

uint32_t Presentation::chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities)
{
    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount))
    {
        minImageCount = surfaceCapabilities.maxImageCount;
    }
    return minImageCount;
}

vk::SurfaceFormatKHR Presentation::chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const &availableFormats)
{
    if (availableFormats.empty()) {
        throw std::runtime_error("No swapchain surface formats available");
    }

    // Prefer B8G8R8A8_SRGB with SRGB_NONLINEAR color space for correct gamma
    for (const auto &format : availableFormats) {
        if (format.format == vk::Format::eB8G8R8A8Srgb &&
            format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return format;
        }
    }

    // Fall back to the first available format
    return availableFormats[0];
}

void Presentation::createSwapChain() {
    vk::SurfaceCapabilitiesKHR surfaceCapabilities = this->physicalDevice.getSurfaceCapabilitiesKHR( *this->surface );
    this->swapChainExtent                          = chooseSwapExtent(surfaceCapabilities);
    uint32_t minImageCount                         = chooseSwapMinImageCount(surfaceCapabilities);

    std::vector<vk::SurfaceFormatKHR> availableFormats = this->physicalDevice.getSurfaceFormatsKHR(*this->surface);
    this->swapChainSurfaceFormat                       = chooseSwapSurfaceFormat(availableFormats);

    std::vector<vk::PresentModeKHR> availablePresentModes = this->physicalDevice.getSurfacePresentModesKHR( this->surface );

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{.surface          = *this->surface,
                                                   .minImageCount    = minImageCount,
                                                   .imageFormat      = this->swapChainSurfaceFormat.format,
                                                   .imageColorSpace  = this->swapChainSurfaceFormat.colorSpace,
                                                   .imageExtent      = this->swapChainExtent,
                                                   .imageArrayLayers = 1,
                                                   .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
                                                   .imageSharingMode = vk::SharingMode::eExclusive,
                                                   .preTransform     = surfaceCapabilities.currentTransform,
                                                   .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
                                                   .presentMode      = chooseSwapPresentMode(availablePresentModes),
                                                   .clipped          = true};

    this->swapChain       = vk::raii::SwapchainKHR( this->device, swapChainCreateInfo );
    this->swapChainImages = this->swapChain.getImages();
}

void Presentation::createImageViews() {
    if (!this->swapChainImageViews.empty()) {
        throw std::runtime_error("swapChainImageViews must be empty before creation");
    }

    swapChainImageViews.reserve(swapChainImages.size());

    for ( auto &image: swapChainImages ) {
        swapChainImageViews.emplace_back(this->createImageView(image, swapChainSurfaceFormat.format));
    }
}

void Presentation::cleanupSwapChain() {
    this->swapChainImageViews.clear();
    this->swapChain = nullptr;
}

void Presentation::recreateSwapChain() {

    int width = 0, height = 0;
    glfwGetFramebufferSize(this->window, &width, &height);

    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    this->device.waitIdle();

    this->cleanupSwapChain();
    this->createSwapChain();
    this->createImageViews();
}
