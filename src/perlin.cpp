// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


#include "perlin.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>

cudaTextureObject_t perlin = 0;

static cudaArray_t perlin_array = nullptr;


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

bool perlinLoad(const char* path){
    
    std::ifstream file(path);
    
    if(!file.is_open()){
        
        std::cerr << "\n    → [IMG]: Erro ao abrir: " << path << "\n";
        
        return false;
    }
        
     
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    int NR, NPHI, NZ;
    {
        std::string header;
        std::getline(file, header);
        std::istringstream ss(header);
        ss >> NR >> NPHI >> NZ;
    }

    std::cout << "\n    → [IMG]: Dimensões Perlin: " << NR << "x" << NPHI << "x" << NZ;
 
    size_t total = (size_t)NR * NPHI * NZ;
        
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // lendo linha-a-linha

    std::vector<float>perlin_noise;
    perlin_noise.reserve(total);

    std::string line;
    
    while(std::getline(file, line) && perlin_noise.size() < total){

        if(line.empty()) continue;

        try{

            perlin_noise.push_back(std::stof(line));

        } catch (...) {
        
            std::cerr << "\n    → [IMG]: Erro no parsing da linha '" << line << "'\n";
            return false;
        }
    }

    if(perlin_noise.size() != total){
        
        std::cerr << "\n    → [IMG]: Esperava " << total << ", recebeu " << perlin_noise.size() << "\n";

    }



    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // cria cudaArray 

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    
    cudaExtent extensao = make_cudaExtent(NR, NPHI, NZ);
    cudaMalloc3DArray(&perlin_array, &channelDesc,extensao);
    
    cudaMemcpy3DParms copyParams = {};    
        
    copyParams.srcPtr = make_cudaPitchedPtr(perlin_noise.data(),NR * sizeof(float), NR, NPHI); 
    copyParams.dstArray = perlin_array;
    copyParams.extent = extensao;
    copyParams.kind = cudaMemcpyHostToDevice;
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "\n    → [IMG]: Erro Perlin/CUDA: " << cudaGetErrorString(err) << "\n";
        return false;
    }


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // config de textura


    cudaResourceDesc resDesc = {};
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = perlin_array;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0]  = cudaAddressModeClamp;
    texDesc.addressMode[1]  = cudaAddressModeWrap;  
    texDesc.addressMode[2]  = cudaAddressModeClamp;  
    texDesc.filterMode      = cudaFilterModeLinear;
    texDesc.readMode        = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaCreateTextureObject(&perlin, &resDesc, &texDesc, nullptr);


    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        std::cerr << "\n    → [IMG]: Erro Perlin/CUDA: " << cudaGetErrorString(err2) << "\n";
        return false;
    }

    std::cout << "\n    → [IMG]: Perlin/CUDA, textura criada (obj=" << perlin << ")" << "\n";

    return true;
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


void perlinFree(){

    if(perlin) cudaDestroyTextureObject(perlin);
    if(perlin_array) cudaFreeArray(perlin_array);
    
    perlin = 0;
    perlin_array = nullptr;
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
