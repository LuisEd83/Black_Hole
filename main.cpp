#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include "include/test_app.hpp"

int main(int argc, char* argv[]) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--export") {
            std::string filename = (argc > 2) ? argv[2] : "icons/output.png";
            uint32_t width = (argc > 3) ? std::stoi(argv[3]) : 1920;
            uint32_t height = (argc > 4) ? std::stoi(argv[4]) : 1080;
            
            VulkanTestApp app;
            app.exportFrame(filename, width, height);
        }else if (argc > 1 && std::string(argv[1]) == "--disk") {
            int n_rings = (argc > 2) ? std::stoi(argv[2]) : 200;
            VulkanTestApp app;
            app.exportDisk(n_rings);
        } else {
            VulkanTestApp app;
            app.run();
        }

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
