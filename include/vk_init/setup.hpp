#pragma once

#include "utils.hpp"
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>
#include <cstdint>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

class VulkanTestAppBase {
protected:
    // NOTE: RAII members are destroyed in REVERSE order of declaration.
    // The dependency chain is: child -> parent -> instance -> context
    // Do NOT reorder these members without understanding the dependency graph.
    // Context must be last (fundamental Vulkan resource).
    vk::raii::Context                   ctx;
    // Instance depends on context.
    vk::raii::Instance                  inst = nullptr;
    // Debug messenger depends on instance.
    vk::raii::DebugUtilsMessengerEXT    debugMessenger = nullptr;
    // Physical device depends on instance.
    vk::raii::PhysicalDevice            physicalDevice = nullptr;
    // Device depends on physical device. All resources below depend on device.
    vk::raii::Device                    device = nullptr;
    // Queue depends on device.
    vk::raii::Queue                     graphicsQueue = nullptr;
    // Surface depends on instance.
    vk::raii::SurfaceKHR                surface = nullptr;
    uint32_t                            queueIndex;
    GLFWwindow*                         window = nullptr;
    bool                                framebufferResized = false;
};

// Centralized device capability requirements.
// Used by deviceHasMinimumRequirements() to filter GPUs and by createLogicalDevice()
// to enable features on the selected device. This ensures both functions agree.
struct DeviceCapabilities {
    // Minimum Vulkan API version
    static constexpr uint32_t minimumApiVersion = vk::ApiVersion13;

    // Required device extensions
    static const std::vector<const char *> requiredExtensions;

    // Hardware requirements (not features, just capabilities)
    static constexpr bool requireGeometryShader = true;
    static constexpr bool requireGraphicsQueue  = true;
    static constexpr bool requireVulkan1_3      = true;
};

// Feature requirements. These must be enabled in the feature chain
// when creating the logical device.
struct RequiredFeatures {
    bool shaderDrawParameters;
    bool synchronization2;
    bool dynamicRendering;
    bool extendedDynamicState;
    bool samplerAnisotropy;
    bool shaderFloat64;
};

inline constexpr RequiredFeatures requiredFeatures{
    .shaderDrawParameters   = true,
    .synchronization2       = true,
    .dynamicRendering       = true,
    .extendedDynamicState   = true,
    .samplerAnisotropy      = true,
    .shaderFloat64          = true
};

class Setup : public VulkanTestAppBase {
public:
    void initWindow();
    void createInstance();
    void setupDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void pickPhysicalDeviceHeadless();
    void createLogicalDeviceHeadless();
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    vk::raii::ImageView createImageView(vk::Image const &image, vk::Format format);
private:
    static const std::vector<char const *> validationLayers;
    static constexpr bool enableValidationLayers =
#ifdef NDEBUG
        false;
#else
        true;
#endif

    bool    deviceHasMinimumRequirements(vk::raii::PhysicalDevice physicalDevice);
    static  uint32_t deviceScore(vk::raii::PhysicalDevice const &physicalDevice);
    static  std::vector<const char *> getRequiredInstanceExtensions();
};
