#pragma once

#include "vk_main/render.hpp"
#include <GLFW/glfw3.h>
#include <iomanip>

class VulkanTestApp : public Render {
public:
    void run() {
        this->initWindow();
        this->initVulkan();
        this->mainLoop();
        this->cleanup();
    }

    void exportFrame(const std::string& filename, uint32_t width, uint32_t height) {
        this->initWindow();
        this->initVulkanHeadless();
        this->exportToImage(filename, width, height);
        glfwDestroyWindow(this->window);
    }

    void exportDisk(int n_rings = 200) {
        this->initVulkanHeadless();
        this->exportDiskData(n_rings);
    }

private:
    void initVulkan() {
        this->createInstance();
        this->setupDebugMessenger();
        this->createSurface();
        this->pickPhysicalDevice();
        this->createLogicalDevice();
        this->createSwapChain();
        this->createImageViews();
        this->createComputePipeline();
        this->createGraphicsPipeline();
        this->createCommandPool();
        this->createStarmapTexture();
        this->createPerlinTexture();
        this->createStorageImage();
        this->createUniformBuffers();
        this->createDescriptorPool();
        this->createDescriptorSets();
        this->createCommandBuffers();
        this->createSyncObjects();
        this->createQueryPool();
    }

    void initVulkanHeadless() {
	   	this->createInstance();
	    this->setupDebugMessenger();
	    this->pickPhysicalDeviceHeadless();
	    this->createLogicalDeviceHeadless();
	    this->createComputePipeline();
	    this->createCommandPool();
	    this->createStarmapTexture();
	    this->createPerlinTexture();
	    this->createDescriptorPool();
    };

    void mainLoop() {
        while (!glfwWindowShouldClose(this->window)) {
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }

            glfwPollEvents();
            this->drawFrame();
            // Read GPU timestamps using C API
            std::array<uint64_t, 2> timestamps;
            vkGetQueryPoolResults(
                *this->device, *this->queryPool, 0, 2,
                timestamps.size() * sizeof(uint64_t), timestamps.data(), sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            
        
            // timestampPeriod is in nanoseconds per timestamp unit
            float timestampPeriod = this->physicalDevice.getProperties().limits.timestampPeriod;
            float computeTimeNs = static_cast<float>(timestamps[1] - timestamps[0]) * timestampPeriod;
            this->lastComputeTimeMs = computeTimeNs / 1'000'000.0f;
        
            static auto startTime = std::chrono::high_resolution_clock::now();
            auto currentTime    = std::chrono::high_resolution_clock::now();
            float time                      = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
            static float fps = 0;
        
            if (time >= 1) {
                fps = 1000.0f/this->lastComputeTimeMs;
                startTime = currentTime;
            }

            // Update title every 10 frames so it doesn't flicker
                static int frameCount = 0;
                if (++frameCount % 10 == 0) {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2);
                    oss << "Pos: (" << this->cameraPos.x << ", " << this->cameraPos.y << ", " << this->cameraPos.z << ") "
                        << "Fwd: (" << this->cameraFwd.x << ", " << this->cameraFwd.y << ", " << this->cameraFwd.z << ") "
                        << "(" << fps << " FPS)"<< std::endl;
                    glfwSetWindowTitle(this->window, oss.str().c_str());
                }
        }

        this->device.waitIdle();
    }

    void mainLoopExport(const std::string& filename, uint32_t width, uint32_t height) {
        // Render one frame
        this->drawFrame();
        this->device.waitIdle();

        // Save the storage image
        this->exportToImage(filename, width, height);
    }

    void cleanup() {
        this->cleanupSwapChain();

        glfwDestroyWindow(this->window);
    }
};
