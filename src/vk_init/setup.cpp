#include "vk_init/setup.hpp"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

const std::vector<char const *> Setup::validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

const std::vector<const char *> DeviceCapabilities::requiredExtensions = {
    vk::KHRSwapchainExtensionName};

std::vector<const char *> Setup::getRequiredInstanceExtensions() {
  uint32_t glfwExtensionCount = 0;
  auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

  std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

  if (enableValidationLayers) {
    extensions.push_back(vk::EXTDebugUtilsExtensionName);
  }

  return extensions;
}

void Setup::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = static_cast<Setup*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

void Setup::initWindow() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

  this->window = glfwCreateWindow(WIDTH, HEIGHT, "Black Hole Sim - Vulkan", nullptr, nullptr);

  glfwSetWindowUserPointer(this->window, this);
  glfwSetFramebufferSizeCallback(this->window, framebufferResizeCallback);
}

void Setup::createInstance() {
  constexpr vk::ApplicationInfo appInfo{
      .pApplicationName     = "Black Hole Sim - Vulkan Edition",
      .applicationVersion   = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName          = "No Engine",
      .engineVersion        = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion           = vk::ApiVersion14};

  auto requiredExtensions = getRequiredInstanceExtensions();

  auto extensionProperties = this->ctx.enumerateInstanceExtensionProperties();
  auto unsupportedPropertyIt = std::ranges::find_if(
      requiredExtensions,
      [&extensionProperties](auto const &requiredExtension) {
        return std::ranges::none_of(
            extensionProperties,
            [requiredExtension](auto const &extensionProperty) {
              return strcmp(extensionProperty.extensionName,
                            requiredExtension) == 0;
            });
      });
  if (unsupportedPropertyIt != requiredExtensions.end()) {
    throw std::runtime_error("Required extension not supported: " +
                             std::string(*unsupportedPropertyIt));
  }

  std::vector<char const *> requiredLayers;
  if (enableValidationLayers) {
    requiredLayers.assign(validationLayers.begin(), validationLayers.end());
  }

  auto layerProperties = this->ctx.enumerateInstanceLayerProperties();
  auto unsupportedLayerIt = std::ranges::find_if(
      requiredLayers, [&layerProperties](auto const &requiredLayer) {
        return std::ranges::none_of(
            layerProperties, [requiredLayer](auto const &layerProperty) {
              return strcmp(layerProperty.layerName, requiredLayer) == 0;
            });
      });
  if (unsupportedLayerIt != requiredLayers.end()) {
    throw std::runtime_error("Required layer not supported: " +
                             std::string(*unsupportedLayerIt));
  }

  vk::InstanceCreateInfo createInfo{
      .pApplicationInfo         = &appInfo,
      .enabledLayerCount        = static_cast<uint32_t>(requiredLayers.size()),
      .ppEnabledLayerNames      = requiredLayers.data(),
      .enabledExtensionCount    = static_cast<uint32_t>(requiredExtensions.size()),
      .ppEnabledExtensionNames  = requiredExtensions.data()};

  this->inst = vk::raii::Instance(this->ctx, createInfo);
}

void Setup::setupDebugMessenger() {
  if (!enableValidationLayers)
    return;

  vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);

  vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
      vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
      vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
      vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

  vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
      .messageSeverity = severityFlags,
      .messageType = messageTypeFlags,
      .pfnUserCallback = &debugCallback};
  this->debugMessenger =
      inst.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
}

bool Setup::deviceHasMinimumRequirements(vk::raii::PhysicalDevice physicalDevice) {
  // Check minimum API version
  bool supportsVulkan1_3 =
      physicalDevice.getProperties().apiVersion >= DeviceCapabilities::minimumApiVersion;

  // Check required hardware capabilities
  bool hasGeometryShader = !DeviceCapabilities::requireGeometryShader ||
                           physicalDevice.getFeatures().geometryShader;
  auto queueFamilies = physicalDevice.getQueueFamilyProperties();
  bool supportsGraphics = !DeviceCapabilities::requireGraphicsQueue ||
      std::ranges::any_of(queueFamilies, [](auto const &qfp) {
        return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
      });

  // Check required extensions
  auto availableDeviceExtensions =
      physicalDevice.enumerateDeviceExtensionProperties();
  bool supportsAllRequiredExtensions = std::ranges::all_of(
      DeviceCapabilities::requiredExtensions,
      [&availableDeviceExtensions](auto const &requiredExtension) {
        return std::ranges::any_of(
            availableDeviceExtensions,
            [requiredExtension](auto const &availableExtension) {
              return strcmp(availableExtension.extensionName,
                            requiredExtension) == 0;
            });
      });

  // Check required features (must match RequiredFeatures struct)
  auto features = physicalDevice.template getFeatures2<
      vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
      vk::PhysicalDeviceVulkan13Features,
      vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
  bool supportsRequiredFeatures =
      features.template get<vk::PhysicalDeviceFeatures2>()
          .features.samplerAnisotropy == requiredFeatures.samplerAnisotropy &&
      features.template get<vk::PhysicalDeviceFeatures2>()
          .features.shaderFloat64 == requiredFeatures.shaderFloat64 &&        
      features.template get<vk::PhysicalDeviceVulkan11Features>()
          .shaderDrawParameters == requiredFeatures.shaderDrawParameters &&
      features.template get<vk::PhysicalDeviceVulkan13Features>()
          .synchronization2 == requiredFeatures.synchronization2 &&
      features.template get<vk::PhysicalDeviceVulkan13Features>()
          .dynamicRendering == requiredFeatures.dynamicRendering &&
      features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
          .extendedDynamicState == requiredFeatures.extendedDynamicState;

  return supportsVulkan1_3 && hasGeometryShader && supportsGraphics &&
         supportsAllRequiredExtensions && supportsRequiredFeatures;
}

uint32_t Setup::deviceScore(vk::raii::PhysicalDevice const &physicalDevice) {
    auto deviceProperties = physicalDevice.getProperties();
    auto deviceFeatures = physicalDevice.getFeatures();

    auto vram = [physicalDevice]() {
      auto memoryProperties = physicalDevice.getMemoryProperties();
      uint64_t vram = 0;
      for (uint32_t i = 0; i < memoryProperties.memoryHeapCount; i++) {
        if (memoryProperties.memoryHeaps[i].flags &
            vk::MemoryHeapFlagBits::eDeviceLocal) {
          vram += memoryProperties.memoryHeaps[i].size;
        }
      }
      return vram;
    }();

    uint32_t score = 0;

    switch (deviceProperties.deviceType) {
    case vk::PhysicalDeviceType::eDiscreteGpu:
      score += 1000;
      break;
    case vk::PhysicalDeviceType::eIntegratedGpu:
      score += 100;
      break;
    case vk::PhysicalDeviceType::eVirtualGpu:
      score += 50;
      break;
    default:
      break;
    }

    score += deviceProperties.limits.maxImageDimension2D;
    score += static_cast<uint32_t>(vram / (1024 * 1024));

    return score;
}

void Setup::pickPhysicalDevice() {
  auto physicalDevices = inst.enumeratePhysicalDevices();

  if (physicalDevices.empty()) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }

  std::multimap<int, vk::raii::PhysicalDevice> candidates;

  for (auto &physicalDevice : physicalDevices) {
    if (!deviceHasMinimumRequirements(physicalDevice))
      continue;
    candidates.insert(
        std::make_pair(deviceScore(physicalDevice), physicalDevice));
  }

  if (!candidates.empty() && candidates.rbegin()->first > 0) {
    std::cout << "Selected GPU: "
                << candidates.rbegin()->second.getProperties().deviceName
                << " (score: " << candidates.rbegin()->first << ")" << std::endl;
    this->physicalDevice = candidates.rbegin()->second;
  } else {
    throw std::runtime_error("failed to find a suitable GPU!");
  }
}

void Setup::createLogicalDevice() {
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    uint32_t queueIndex = ~0;
    for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++) {
      if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
          physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface)) {
        queueIndex = qfpIndex;
        break;
      }
    }

    if (queueIndex == ~0) {
      throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
    }

    this->queueIndex = queueIndex;

    auto graphicsQueueFamilyProperty = std::ranges::find_if(queueFamilyProperties, [](auto const &qfp) { return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0); });
    auto graphicsIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));
    float queuePriority = 1.0f;

    vk::DeviceQueueCreateInfo deviceQueueCreateInfo {
            .queueFamilyIndex = graphicsIndex,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority};

    // Enable the same features that deviceHasMinimumRequirements() checks for.
    // This must match the RequiredFeatures struct in setup.hpp.
    vk::StructureChain<vk::PhysicalDeviceFeatures2,
                       vk::PhysicalDeviceVulkan11Features,
                       vk::PhysicalDeviceVulkan13Features,
                       vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
                       featureChain = {
                           {.features = {.samplerAnisotropy = requiredFeatures.samplerAnisotropy, .shaderFloat64 = requiredFeatures.shaderFloat64}},
                           {.shaderDrawParameters = requiredFeatures.shaderDrawParameters},
                           {.synchronization2 = requiredFeatures.synchronization2,
                            .dynamicRendering = requiredFeatures.dynamicRendering},
                           {.extendedDynamicState = requiredFeatures.extendedDynamicState}
                       };

    vk::DeviceCreateInfo deviceCreateInfo{
        .pNext                      = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount       = 1,
        .pQueueCreateInfos          = &deviceQueueCreateInfo,
        .enabledExtensionCount      = static_cast<uint32_t>(DeviceCapabilities::requiredExtensions.size()),
        .ppEnabledExtensionNames    = DeviceCapabilities::requiredExtensions.data()
    };

    this->device = vk::raii::Device(this->physicalDevice, deviceCreateInfo);
    this->graphicsQueue = vk::raii::Queue(this->device, graphicsIndex, 0);
}

vk::raii::ImageView Setup::createImageView(vk::Image const &image, vk::Format format) {
    vk::ImageViewCreateInfo viewInfo{
          .image            = image,
          .viewType         = vk::ImageViewType::e2D,
          .format           = format,
          .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
      return vk::raii::ImageView(this->device, viewInfo);
}

void Setup::pickPhysicalDeviceHeadless() {
    // Same as pickPhysicalDevice but without surface support check
    std::multimap<uint32_t, vk::raii::PhysicalDevice> candidates;
    for (const auto& physicalDevice : this->inst.enumeratePhysicalDevices()) {
        if (deviceHasMinimumRequirements(physicalDevice)) {
            candidates.insert(std::make_pair(deviceScore(physicalDevice), physicalDevice));
        }
    }

    if (!candidates.empty() && candidates.rbegin()->first > 0) {
        std::cout << "Selected GPU: " << candidates.rbegin()->second.getProperties().deviceName
                  << " (score: " << candidates.rbegin()->first << ")" << std::endl;
        this->physicalDevice = candidates.rbegin()->second;
    } else {
        throw std::runtime_error("failed to find a suitable GPU for headless rendering!");
    }
}

void Setup::createLogicalDeviceHeadless() {
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    // Find a queue that supports compute (and graphics for transfer)
    uint32_t queueIndex = ~0;
    for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++) {
        if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eCompute) &&
            (queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics)) {
            queueIndex = qfpIndex;
            break;
        }
    }

    if (queueIndex == ~0) {
        throw std::runtime_error("Could not find a queue with compute support for headless rendering");
    }

    this->queueIndex = queueIndex;

    auto graphicsQueueFamilyProperty = std::ranges::find_if(queueFamilyProperties, [](auto const &qfp) {
        return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
    });
    auto graphicsIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));
    float queuePriority = 1.0f;

    vk::DeviceQueueCreateInfo deviceQueueCreateInfo {
        .queueFamilyIndex = graphicsIndex,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority
    };

    // Build feature chain dynamically
    vk::PhysicalDeviceFeatures2 physicalDeviceFeatures2 = physicalDevice.getFeatures2();
    vk::PhysicalDeviceVulkan13Features* vulkan13Features = nullptr;
    vk::PhysicalDeviceVulkan12Features* vulkan12Features = nullptr;
    void* pNextChain = nullptr;

    if (physicalDeviceFeatures2.features.samplerAnisotropy) {
        physicalDeviceFeatures2.features.samplerAnisotropy = VK_TRUE;
    }
    physicalDeviceFeatures2.features.shaderFloat64 = VK_TRUE;

    auto* featureChain = &physicalDeviceFeatures2;
    pNextChain = physicalDeviceFeatures2.pNext;

    // Check if Vulkan 1.3 features are available
    if (physicalDevice.getProperties().apiVersion >= vk::ApiVersion13) {
        vulkan13Features = reinterpret_cast<vk::PhysicalDeviceVulkan13Features*>(pNextChain);
        if (vulkan13Features && vulkan13Features->synchronization2 && vulkan13Features->dynamicRendering) {
            vulkan13Features->synchronization2 = VK_TRUE;
            vulkan13Features->dynamicRendering = VK_TRUE;
        }
    }

    // Check if Vulkan 1.2 features are available
    if (physicalDevice.getProperties().apiVersion >= vk::ApiVersion12) {
        vulkan12Features = reinterpret_cast<vk::PhysicalDeviceVulkan12Features*>(pNextChain);
        // shaderFloat64 is a Vulkan 1.0 feature, not a Vulkan 1.2 feature
        // It's already set in physicalDeviceFeatures2.features above
    }

    vk::DeviceCreateInfo deviceCreateInfo {
        .pNext                      = featureChain,
        .queueCreateInfoCount       = 1,
        .pQueueCreateInfos          = &deviceQueueCreateInfo,
        .enabledLayerCount          = static_cast<uint32_t>(Setup::validationLayers.size()),
        .ppEnabledLayerNames        = Setup::validationLayers.data(),
        .enabledExtensionCount      = static_cast<uint32_t>(DeviceCapabilities::requiredExtensions.size()),
        .ppEnabledExtensionNames    = DeviceCapabilities::requiredExtensions.data()
    };

    this->device = vk::raii::Device(this->physicalDevice, deviceCreateInfo);
    this->graphicsQueue = vk::raii::Queue(this->device, graphicsIndex, 0);
}
