#pragma once

#include "vk_main/render.hpp"
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
        this->initVulkan();
        this->mainLoopExport(filename, width, height);
        this->cleanup();
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
        this->createTexture();
        this->createPerlinTexture();
        this->createStorageImage();
        this->createUniformBuffers();
        this->createDescriptorPool();
        this->createDescriptorSets();
        this->createCommandBuffers();
        this->createSyncObjects();
        this->createQueryPool();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(this->window)) {
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }

            glfwPollEvents();
            this->drawFrame();

            // Update title every 10 frames so it doesn't flicker
                static int frameCount = 0;
                if (++frameCount % 10 == 0) {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2);
                    oss << "Pos: (" << this->cameraPos.x << ", " << this->cameraPos.y << ", " << this->cameraPos.z << ") "
                        << "Fwd: (" << this->cameraFwd.x << ", " << this->cameraFwd.y << ", " << this->cameraFwd.z << ")";
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
