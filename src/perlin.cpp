#include "../headers/perlin.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>

cudaTextureObject_t perlin = 0;

cudaTextureObject_t perlinGet() {
    return perlin;
}


#if BH_CPU_BACKEND
// ── macOS: CPU-side 2-D texture with bilinear sampling ───────────────────────
#include "cpu_texture.hpp"

static CpuTexture* perlin_cpu = nullptr;

static bool parseDimensions(const std::vector<float>& values,
                            int& width, int& height,
                            size_t& offset) {
    if (values.size() < 3) return false;

    int maybeW = static_cast<int>(std::round(values[0]));
    int maybeH = static_cast<int>(std::round(values[1]));
    int maybeD = static_cast<int>(std::round(values[2]));

    if (maybeW <= 0 || maybeH <= 0 || maybeD <= 0)
        return false;

    if (std::fabs(values[0] - static_cast<float>(maybeW)) >= 1e-6f ||
        std::fabs(values[1] - static_cast<float>(maybeH)) >= 1e-6f ||
        std::fabs(values[2] - static_cast<float>(maybeD)) >= 1e-6f)
        return false;

    size_t expected2d = static_cast<size_t>(maybeW) * static_cast<size_t>(maybeH);
    size_t expected3d = expected2d * static_cast<size_t>(maybeD);

    if (values.size() == expected3d + 3 ||
        values.size() == expected2d + 3) {
        width  = maybeW;
        height = maybeH;
        offset = 3;
        return true;
    }

    if (values.size() == expected3d ||
        values.size() == expected2d) {
        width  = maybeW;
        height = maybeH;
        offset = 0;
        return true;
    }

    return false;
}

static bool derive2DDimensions(const std::vector<float>& values,
                               int& width, int& height,
                               size_t& offset) {
    size_t count = values.size();
    size_t dim = static_cast<size_t>(std::sqrt(static_cast<double>(count)));
    if (dim * dim != count) {
        std::cerr << "\n    → [IMG]: Quantidade de valores (" << count
                  << ") não é quadrado perfeito e não há cabeçalho válido\n";
        return false;
    }
    width  = static_cast<int>(dim);
    height = static_cast<int>(dim);
    offset = 0;
    return true;
}

bool perlinLoad(const char* path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "\n    → [IMG]: Erro ao abrir: " << path << "\n";
        return false;
    }

    std::vector<float> values;
    float val;
    while (file >> val) {
        values.push_back(val);
    }

    if (values.empty()) {
        std::cerr << "\n    → [IMG]: Arquivo vazio: " << path << "\n";
        return false;
    }

    int width, height;
    size_t offset;
    if (!parseDimensions(values, width, height, offset) &&
        !derive2DDimensions(values, width, height, offset)) {
        return false;
    }

    size_t slice = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (offset + slice > values.size()) {
        std::cerr << "\n    → [IMG]: Dados insuficientes para dimensões "
                  << width << "x" << height << "\n";
        return false;
    }

    perlin_cpu         = new CpuTexture();
    perlin_cpu->width  = width;
    perlin_cpu->height = height;
    perlin_cpu->wrap_u = false;
    perlin_cpu->wrap_v = true;
    perlin_cpu->data   = new float[slice * 4];

    for (size_t i = 0; i < slice; ++i) {
        float v = values[offset + i];
        perlin_cpu->data[i*4+0] = v;
        perlin_cpu->data[i*4+1] = 0.f;
        perlin_cpu->data[i*4+2] = 0.f;
        perlin_cpu->data[i*4+3] = 1.f;
    }

    std::cout << "\n    → [IMG]: Perlin carregado. Dimensões: "
              << width << "x" << height;

    perlin = reinterpret_cast<cudaTextureObject_t>(perlin_cpu);
    std::cout << "\n    → [IMG]: Perlin CPU, textura 2D criada.\n";
    return true;
}

void perlinFree() {
    delete perlin_cpu;
    perlin_cpu = nullptr;
    perlin     = 0;
}


#else
// ── Linux / Windows: CUDA 2-D float4 texture ───────────────────────────────────
#include <cuda_runtime.h>

static cudaArray_t perlin_array = nullptr;

static bool parseDimensions(const std::vector<float>& values,
                            int& width, int& height,
                            size_t& offset) {
    if (values.size() < 3) return false;

    int maybeW = static_cast<int>(std::round(values[0]));
    int maybeH = static_cast<int>(std::round(values[1]));
    int maybeD = static_cast<int>(std::round(values[2]));

    if (maybeW <= 0 || maybeH <= 0 || maybeD <= 0)
        return false;

    if (std::fabs(values[0] - static_cast<float>(maybeW)) >= 1e-6f ||
        std::fabs(values[1] - static_cast<float>(maybeH)) >= 1e-6f ||
        std::fabs(values[2] - static_cast<float>(maybeD)) >= 1e-6f)
        return false;

    size_t expected2d = static_cast<size_t>(maybeW) * static_cast<size_t>(maybeH);
    size_t expected3d = expected2d * static_cast<size_t>(maybeD);

    if (values.size() == expected3d + 3 ||
        values.size() == expected2d + 3) {
        width  = maybeW;
        height = maybeH;
        offset = 3;
        return true;
    }

    if (values.size() == expected3d ||
        values.size() == expected2d) {
        width  = maybeW;
        height = maybeH;
        offset = 0;
        return true;
    }

    return false;
}

static bool derive2DDimensions(const std::vector<float>& values,
                               int& width, int& height,
                               size_t& offset) {
    size_t count = values.size();
    size_t dim = static_cast<size_t>(std::sqrt(static_cast<double>(count)));
    if (dim * dim != count) {
        std::cerr << "\n    → [IMG]: Quantidade de valores (" << count
                  << ") não é quadrado perfeito e não há cabeçalho válido\n";
        return false;
    }
    width  = static_cast<int>(dim);
    height = static_cast<int>(dim);
    offset = 0;
    return true;
}

bool perlinLoad(const char* path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "\n    → [IMG]: Erro ao abrir: " << path << "\n";
        return false;
    }

    std::vector<float> values;
    float val;
    while (file >> val) {
        values.push_back(val);
    }

    if (values.empty()) {
        std::cerr << "\n    → [IMG]: Arquivo vazio: " << path << "\n";
        return false;
    }

    int width, height;
    size_t offset;
    if (!parseDimensions(values, width, height, offset) &&
        !derive2DDimensions(values, width, height, offset)) {
        return false;
    }

    size_t slice = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (offset + slice > values.size()) {
        std::cerr << "\n    → [IMG]: Dados insuficientes para dimensões "
                  << width << "x" << height << "\n";
        return false;
    }

    std::cout << "\n    → [IMG]: Perlin carregado. Dimensões: "
              << width << "x" << height;

    std::vector<float4> texData(slice);
    for (size_t i = 0; i < slice; ++i) {
        float v = values[offset + i];
        texData[i] = make_float4(v, 0.0f, 0.0f, 1.0f);
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&perlin_array, &channelDesc,
                    static_cast<size_t>(width), static_cast<size_t>(height));

    size_t pitch = static_cast<size_t>(width) * sizeof(float4);
    cudaMemcpy2DToArray(perlin_array, 0, 0, texData.data(),
                        pitch,
                        pitch, static_cast<size_t>(height),
                        cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc = {};
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = perlin_array;

    cudaTextureDesc texDesc  = {};
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaCreateTextureObject(&perlin, &resDesc, &texDesc, nullptr);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "\n    → [IMG]: Erro Perlin/CUDA: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }

    std::cout << "\n    → [IMG]: Perlin/CUDA, textura 2D criada (obj=" << perlin << ")\n";
    return true;
}

void perlinFree() {
    if (perlin)       cudaDestroyTextureObject(perlin);
    if (perlin_array) cudaFreeArray(perlin_array);
    perlin       = 0;
    perlin_array = nullptr;
}

#endif
