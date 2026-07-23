#include "../headers/starmap.hpp"
#include "../headers/lodepng.h"
#include <iostream>
#include <vector>
#include <cstdint>

cudaTextureObject_t starmap = 0;


#if BH_CPU_BACKEND
// ── macOS: store as CpuTexture ───────────────────────────────────────────────
#include "cpu_texture.hpp"

static CpuTexture* starmap_cpu = nullptr;

bool starmapLoad(const char* path) {
    std::vector<unsigned char> image;
    unsigned w, h;
    unsigned err = lodepng::decode(image, w, h, path);
    if (err) {
        std::cerr << "\n    → [IMG]: Erro ao carregar " << path
                  << ": " << lodepng_error_text(err) << "\n";
        return false;
    }
    std::cout << "\n    → [IMG]: Starmap carregado. Dimensões: "
              << w << "x" << h << " (" << image.size()/1024/1024 << " MB)";

    starmap_cpu         = new CpuTexture();
    starmap_cpu->width  = (int)w;
    starmap_cpu->height = (int)h;
    starmap_cpu->wrap_u = true;
    starmap_cpu->wrap_v = false;
    starmap_cpu->data   = new float[(size_t)w * h * 4];

    for (size_t i = 0; i < (size_t)w * h; ++i) {
        starmap_cpu->data[i*4+0] = image[i*4+0] / 255.f;
        starmap_cpu->data[i*4+1] = image[i*4+1] / 255.f;
        starmap_cpu->data[i*4+2] = image[i*4+2] / 255.f;
        starmap_cpu->data[i*4+3] = image[i*4+3] / 255.f;
    }

    starmap = (cudaTextureObject_t)(uintptr_t)starmap_cpu;
    std::cout << "\n    → [IMG]: Starmap CPU, textura criada.\n";
    return true;
}

void starmapFree() {
    delete starmap_cpu;
    starmap_cpu = nullptr;
    starmap     = 0;
}


#else
// ── Linux / Windows: CUDA texture ────────────────────────────────────────────
#include <cuda_runtime.h>

static cudaArray_t starmap_array = nullptr;

bool starmapLoad(const char* path) {
    std::vector<unsigned char> image;
    unsigned width, height;
    unsigned err = lodepng::decode(image, width, height, path);
    if (err) {
        std::cerr << "\n    → [IMG]: Erro ao carregar " << path
                  << ": " << lodepng_error_text(err) << "\n";
        return false;
    }
    std::cout << "\n    → [IMG]: Starmap carregado. Dimensões: "
              << width << "x" << height
              << " (" << image.size()/1024/1024 << " MB)";

    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaMallocArray(&starmap_array, &channelDesc, width, height);

    size_t size = width * 4 * sizeof(unsigned char);
    cudaMemcpy2DToArray(starmap_array, 0, 0,
                        image.data(), size, size, height,
                        cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc = {};
    resDesc.resType          = cudaResourceTypeArray;
    resDesc.res.array.array  = starmap_array;

    cudaTextureDesc texDesc     = {};
    texDesc.addressMode[0]      = cudaAddressModeWrap;
    texDesc.addressMode[1]      = cudaAddressModeClamp;
    texDesc.filterMode          = cudaFilterModeLinear;
    texDesc.readMode            = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords    = 1;

    cudaCreateTextureObject(&starmap, &resDesc, &texDesc, nullptr);

    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        std::cerr << "\n    → [IMG]: Erro Starmap/CUDA: "
                  << cudaGetErrorString(err2) << "\n";
        return false;
    }

    std::cout << "\n    → [IMG]: Starmap/CUDA, textura criada (obj=" << starmap << ")\n";
    return true;
}

void starmapFree() {
    if (starmap)       cudaDestroyTextureObject(starmap);
    if (starmap_array) cudaFreeArray(starmap_array);
    starmap       = 0;
    starmap_array = nullptr;
}

#endif
