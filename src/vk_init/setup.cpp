#include "vk_init/setup.hpp"
#include "vk_main/render.hpp"
#include "utils.hpp"
#include "vulkan/vulkan.hpp"

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
#include <filesystem>
#include <iostream>
#include <vulkan/vulkan.hpp>

namespace fs = std::filesystem;

const std::vector<char const *> Setup::validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

const std::vector<const char *> DeviceCapabilities::requiredExtensions = {
    vk::KHRSwapchainExtensionName};

static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT       severity,
                                                      vk::DebugUtilsMessageTypeFlagsEXT              type,
                                                      const vk::DebugUtilsMessengerCallbackDataEXT * pCallbackData,
                                                      void *                                         pUserData)
{
	std::string sev;
	switch (severity) {
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
			sev = "ERROR";
			break;
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
			sev = "WARNING";
			break;
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
			sev = "INFO";
			break;
		default: sev = "";
	}
  std::cerr << "(" << sev << ") " << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;

  return vk::False;
}

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

std::string get_embedded_desktop_string(const fs::path& exe_dir) {
    return "[Desktop Entry]\n"
           "Type=Application\n"
           "Name=Black Hole Sim\n"
           "Exec=" + (exe_dir / "black_hole_sim").string() +"\n"
           "Icon=com.blackhole.blackhole_app\n"
           "Terminal=false\n";
}

void initialize_wayland_assets() {
    const char* home_env = std::getenv("HOME");
    if (!home_env) return;
    
    fs::path local_share = fs::path(home_env) / ".local" / "share";
    std::string app_id = "com.blackhole.blackhole_app";

    try {
    	fs::path exe_dir = get_executable_directory();
     	fs::path project_root = exe_dir.parent_path();
        std::vector<int> target_sizes = {16, 32, 256, 512};
        
        for (int size : target_sizes) {
            std::string size_str = std::to_string(size);
            
            // 1. Map your project's local source icon path
            fs::path source_icon = project_root / "icons" / ("icon_" + size_str + "x" + size_str + ".png");
            
            // Skip this size if the file doesn't exist in your project folder
            if (!fs::exists(source_icon)) continue;

            // 2. Create the target Freedesktop directory structure
            fs::path target_dir = local_share / "icons" / "hicolor" / (size_str + "x" + size_str) / "apps";
            fs::create_directories(target_dir);

            // 3. Copy the file directly over (overwrite existing cached versions)
            fs::path target_icon = target_dir / (app_id + ".png");
            fs::copy_file(source_icon, target_icon, fs::copy_options::overwrite_existing);
        }

        // 4. Create and write the .desktop launcher configuration
        fs::path desktop_dir = local_share / "applications";
        fs::create_directories(desktop_dir);
        
        fs::path desktop_file = desktop_dir / (app_id + ".desktop");
        std::ofstream out_desktop(desktop_file);
        out_desktop << get_embedded_desktop_string(exe_dir);
        out_desktop.close();

        // 5. Force the compositor / shell to refresh system indexing tables
        std::system("gtk-update-icon-cache -f -t ~/.local/share/icons/hicolor > /dev/null 2>&1");
        std::system("update-desktop-database ~/.local/share/applications > /dev/null 2>&1");

    } catch (const std::exception& e) {
        std::cerr << "Failed to copy Wayland assets: " << e.what() << std::endl;
    }
}

void Setup::setX11WindowIcon() {
	
	std::vector<int> sizes = {16, 32, 128, 512};
    std::vector<GLFWimage> icons;
    
    for (int size : sizes) {
    	std::string filename = "icons/icon_" + std::to_string(size) + "x" + std::to_string(size) + ".png";
          
        GLFWimage icon;
        icon.width = 0;
        icon.height = 0;
        icon.pixels = stbi_load(filename.c_str(), &icon.width, &icon.height, nullptr, STBI_rgb_alpha);
        
        if (icon.pixels != nullptr) {
            icons.push_back(icon);
        } else {
            fprintf(stderr, "Warning: Failed to load icon %s: %s\n", filename.c_str(), stbi_failure_reason());
        }
    }
    
      if (!icons.empty()) {
          glfwSetWindowIcon(this->window, static_cast<int>(icons.size()), icons.data());
      }
    
      for (auto& icon : icons) {
          stbi_image_free(icon.pixels);
      }
}

void Setup::initWindow() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  
  if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND) {
  		std::cout << "Platform: Wayland" << std::endl;
        // Provision files before the window mapping sequence starts
        initialize_wayland_assets();
        
        // This MUST exactly match the base filename of your desktop file
        glfwWindowHintString(GLFW_WAYLAND_APP_ID, "com.blackhole.blackhole_app");
  }

  this->window = glfwCreateWindow(WIDTH, HEIGHT, "Black Hole Sim - Vulkan", nullptr, nullptr);

  glfwSetWindowUserPointer(this->window, this);
  glfwSetFramebufferSizeCallback(this->window, framebufferResizeCallback);

  if (glfwGetPlatform() == GLFW_PLATFORM_X11) {
  	std::cout << "Platform: X11" << std::endl;
 	this->setX11WindowIcon();
  }

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

bool Setup::deviceHasMinimumRequirements(vk::raii::PhysicalDevice physicalDev) {
  // Check minimum API version
  bool supportsVulkan1_3 =
      physicalDev.getProperties().apiVersion >= DeviceCapabilities::minimumApiVersion;

  // Check required hardware capabilities
  bool hasGeometryShader = !DeviceCapabilities::requireGeometryShader ||
                           physicalDev.getFeatures().geometryShader;
  auto queueFamilies = physicalDev.getQueueFamilyProperties();
  bool supportsGraphics = !DeviceCapabilities::requireGraphicsQueue ||
      std::ranges::any_of(queueFamilies, [](auto const &qfp) {
        return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
      });

  // Check required extensions
  auto availableDeviceExtensions =
      physicalDev.enumerateDeviceExtensionProperties();
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
  auto features = physicalDev.template getFeatures2<
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

uint32_t Setup::deviceScore(vk::raii::PhysicalDevice const &physicalDev) {
    auto deviceProperties = physicalDev.getProperties();

    auto vram = [physicalDev]() {
      auto memoryProperties = physicalDev.getMemoryProperties();
      uint64_t vid_mem = 0;
      for (uint32_t i = 0; i < memoryProperties.memoryHeapCount; i++) {
        if (memoryProperties.memoryHeaps[i].flags &
            vk::MemoryHeapFlagBits::eDeviceLocal) {
          vid_mem += memoryProperties.memoryHeaps[i].size;
        }
      }
      return vid_mem;
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

  for (auto &physicalDev : physicalDevices) {
    if (!deviceHasMinimumRequirements(physicalDev))
      continue;
    candidates.insert(
        std::make_pair(deviceScore(physicalDev), physicalDev));
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

    uint32_t queueIdx = UINT32_MAX;
    for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++) {
      if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
          physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface)) {
        queueIdx = qfpIndex;
        break;
      }
    }

    if (queueIdx == UINT32_MAX) {
      throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
    }

    this->queueIndex = queueIdx;

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
    for (const auto& physicalDev : this->inst.enumeratePhysicalDevices()) {
        if (deviceHasMinimumRequirements(physicalDev)) {
            candidates.insert(std::make_pair(deviceScore(physicalDev), physicalDev));
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
    uint32_t queueIdx = UINT32_MAX;
    for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++) {
        if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eCompute) &&
            (queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics)) {
            queueIdx = qfpIndex;
            break;
        }
    }

    if (queueIdx == UINT32_MAX) {
        throw std::runtime_error("Could not find a queue with compute support for headless rendering");
    }

    this->queueIndex = queueIdx;

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

    // Enable the same features as createLogicalDevice() — explicit chain, not dynamic discovery.
        // The reinterpret_cast + getFeatures2 pNext approach doesn't properly link the chain.
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
    
        vk::DeviceCreateInfo deviceCreateInfo {
            .pNext                      = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
            .queueCreateInfoCount       = 1,
            .pQueueCreateInfos          = &deviceQueueCreateInfo,
            .enabledExtensionCount      = static_cast<uint32_t>(DeviceCapabilities::requiredExtensions.size()),
            .ppEnabledExtensionNames    = DeviceCapabilities::requiredExtensions.data()
        };
    
    this->device = vk::raii::Device(this->physicalDevice, deviceCreateInfo);
    this->graphicsQueue = vk::raii::Queue(this->device, graphicsIndex, 0);
}
