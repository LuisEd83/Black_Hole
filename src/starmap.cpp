// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Carrega o starmap.png e cria uma cudaTextureObject_t para o kernel amostrar.

#include "starmap.hpp"
#include "lodepng.h"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>

// objeto de textura — visível globalmente no host para passar ao kernel
cudaTextureObject_t starmap = 0;

// array CUDA — precisa ser mantido vivo enquanto a textura existir
static cudaArray_t starmap_array = nullptr;


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


bool starmapLoad(const char* path){

    std::vector<unsigned char>image;
    unsigned width, height;

    unsigned err = lodepng::decode(image, width, height, path);
    if(err){
        std::cerr << "\n    → [IMG]: Erro ao carregar " << path << ": " << lodepng_error_text(err) << "\n";

        return false;
    }


    std::cout << "\n    → [IMG]: Starmap carregado. Dimensões: " << width << "x" << height << " (" << image.size()/1024/1024 << " MB)";
   

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // cria cudaArray com formato RGBA8


    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaMallocArray(&starmap_array, &channelDesc, width, height);
    

    // copia pixels para o cudaArray
    size_t size = width * 4 * sizeof(unsigned char);
    cudaMemcpy2DToArray(starmap_array, 0, 0,
                        image.data(),
                        size, size, height,
                        cudaMemcpyHostToDevice);

    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // configura a textura


    cudaResourceDesc resDesc = {};
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = starmap_array;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0]  = cudaAddressModeWrap;    // U (longitude) — wrapa em 360°
    texDesc.addressMode[1]  = cudaAddressModeClamp;   // V (latitude)  — clamp nos polos
    texDesc.filterMode      = cudaFilterModeLinear;   // interpolação bilinear
    texDesc.readMode        = cudaReadModeNormalizedFloat;  // retorna [0,1] em vez de [0,255]
    texDesc.normalizedCoords = 1;                     // UV em [0,1]
    

    cudaCreateTextureObject(&starmap, &resDesc, &texDesc, nullptr);


    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        std::cerr << "\n    → [IMG]: Erro Starmap/CUDA: " << cudaGetErrorString(err2) << "\n";
        return false;
    }


    std::cout << "\n    → [IMG]: Starmap/CUDA, textura criada (obj=" << starmap << ")\n";
    return true;
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


void starmapFree(){

    if(starmap)  cudaDestroyTextureObject(starmap);
    if(starmap_array)   cudaFreeArray(starmap_array);

    starmap = 0;
    starmap_array  = nullptr;
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
