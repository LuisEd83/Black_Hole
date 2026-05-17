#include "../headers/perlin.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>

cudaTextureObject_t perlin = 0;


#if BH_CPU_BACKEND
// ── macOS: store first NR×NPHI slice as CpuTexture ──────────────────────────
// The kernel uses tex2D<float4>(perlin, u, v) so only the 2-D projection
// matters; NZ layers beyond the first are discarded.
#include "cpu_texture.hpp"

static CpuTexture* perlin_cpu = nullptr;

bool perlinLoad(const char* path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "\n    → [IMG]: Erro ao abrir: " << path << "\n";
        return false;
    }

    int NR, NPHI, NZ;
    {
        std::string header;
        std::getline(file, header);
        std::istringstream ss(header);
        ss >> NR >> NPHI >> NZ;
    }

    size_t slice = (size_t)NR * NPHI;

    perlin_cpu         = new CpuTexture();
    perlin_cpu->width  = NR;
    perlin_cpu->height = NPHI;
    perlin_cpu->wrap_u = false;
    perlin_cpu->wrap_v = true;
    perlin_cpu->data   = new float[slice * 4];

    std::string line;
    size_t i = 0;
    while (std::getline(file, line) && i < slice) {
        if (line.empty()) continue;
        try {
            float val = std::stof(line);
            perlin_cpu->data[i*4+0] = val;
            perlin_cpu->data[i*4+1] = 0.f;
            perlin_cpu->data[i*4+2] = 0.f;
            perlin_cpu->data[i*4+3] = 1.f;
            ++i;
        } catch (...) {
            std::cerr << "\n    → [IMG]: Erro no parsing da linha '" << line << "'\n";
            return false;
        }
    }

    std::cout << "\n    → [IMG]: Perlin carregado. Dimensões: "
              << NR << "x" << NPHI << "x" << NZ;

    perlin = (cudaTextureObject_t)(uintptr_t)perlin_cpu;
    std::cout << "\n    → [IMG]: Perlin CPU, textura criada.\n";
    return true;
}

void perlinFree() {
    delete perlin_cpu;
    perlin_cpu = nullptr;
    perlin     = 0;
}


#else
// ── Linux / Windows: CUDA 3-D texture ────────────────────────────────────────
#include <cuda_runtime.h>

static cudaArray_t perlin_array = nullptr;

bool perlinLoad(const char* path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "\n    → [IMG]: Erro ao abrir: " << path << "\n";
        return false;
    }

    int NR, NPHI, NZ;
    {
        std::string header;
        std::getline(file, header);
        std::istringstream ss(header);
        ss >> NR >> NPHI >> NZ;
    }

    size_t total = (size_t)NR * NPHI * NZ;

    std::vector<float> perlin_noise;
    perlin_noise.reserve(total);

    std::string line;
    while (std::getline(file, line) && perlin_noise.size() < total) {
        if (line.empty()) continue;
        try {
            perlin_noise.push_back(std::stof(line));
        } catch (...) {
            std::cerr << "\n    → [IMG]: Erro no parsing da linha '" << line << "'\n";
            return false;
        }
    }

    std::cout << "\n    → [IMG]: Perlin carregado. Dimensões: "
              << NR << "x" << NPHI << "x" << NZ
              << " (" << perlin_noise.size()/1024/1024 << " MB)";

    if (perlin_noise.size() != total)
        std::cerr << "\n    → [IMG]: Esperava " << total
                  << ", recebeu " << perlin_noise.size() << "\n";

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extensao = make_cudaExtent(NR, NPHI, NZ);
    cudaMalloc3DArray(&perlin_array, &channelDesc, extensao);

    cudaMemcpy3DParms copyParams     = {};
    copyParams.srcPtr  = make_cudaPitchedPtr(perlin_noise.data(),
                                              NR*sizeof(float), NR, NPHI);
    copyParams.dstArray = perlin_array;
    copyParams.extent   = extensao;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "\n    → [IMG]: Erro Perlin/CUDA: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }

    cudaResourceDesc resDesc = {};
    resDesc.resType          = cudaResourceTypeArray;
    resDesc.res.array.array  = perlin_array;

    cudaTextureDesc texDesc      = {};
    texDesc.addressMode[0]       = cudaAddressModeClamp;
    texDesc.addressMode[1]       = cudaAddressModeWrap;
    texDesc.addressMode[2]       = cudaAddressModeClamp;
    texDesc.filterMode           = cudaFilterModeLinear;
    texDesc.readMode             = cudaReadModeElementType;
    texDesc.normalizedCoords     = 1;

    cudaCreateTextureObject(&perlin, &resDesc, &texDesc, nullptr);

    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        std::cerr << "\n    → [IMG]: Erro Perlin/CUDA: "
                  << cudaGetErrorString(err2) << "\n";
        return false;
    }

    std::cout << "\n    → [IMG]: Perlin/CUDA, textura criada (obj=" << perlin << ")\n";
    return true;
}

void perlinFree() {
    if (perlin)       cudaDestroyTextureObject(perlin);
    if (perlin_array) cudaFreeArray(perlin_array);
    perlin       = 0;
    perlin_array = nullptr;
}

#endif
