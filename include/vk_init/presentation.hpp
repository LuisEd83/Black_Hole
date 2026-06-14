#pragma once

#include "vk_init/setup.hpp"
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

class Presentation : public Setup {
protected:
    vk::raii::SwapchainKHR              swapChain = nullptr;
    std::vector<vk::Image>              swapChainImages;
    vk::SurfaceFormatKHR                swapChainSurfaceFormat;
    vk::Extent2D                        swapChainExtent;
    std::vector<vk::raii::ImageView>    swapChainImageViews;

public:
    void createSurface();
    void createSwapChain();
    void createImageViews();
    void recreateSwapChain();
    void cleanupSwapChain();

private:
    static vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const &availablePresentModes);
    vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &capabilities);
    static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities);
    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const &availableFormats);
};
