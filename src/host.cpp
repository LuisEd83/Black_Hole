// -------------------------------------------------------------------------------------------------
// geodesic_host.cpp
//
// Código host puro — compilado pelo GCC, sem NVCC.
// Responsável por: alocar memória na GPU, lançar o kernel, copiar resultado de volta.
//
// A separação existe porque <vector>, <iostream> e GLM conflitam com o CCCL
// (CUDA C++ Core Libraries) quando incluídos num arquivo .cu com CUDA 13+ e GCC 15.


#include "../cuda/geodesic.cuh"

#include <glm/glm.hpp>
#include <vector>
#include <iostream>
#include <cstdio>


// -------------------------------------------------------------------------------------------------
// raytraceCUDA — chamada pelo gl_engine.cpp a cada frame em que a câmera muda
//
//  pixels  → buffer RGB já alocado pelo caller (WIDTH * HEIGHT * 3 bytes)
//  pos     → posição da câmera em coordenadas cartesianas
//  fwd     → vetor forward normalizado
//  right   → vetor right normalizado
//  up      → vetor up normalizado
//  fov_y   → campo de visão vertical em graus

void raytraceCUDA( unsigned char* pixels,
                   int WIDTH, int HEIGHT,
                   glm::vec3 pos, glm::vec3 fwd, glm::vec3 right, glm::vec3 up,
                   float fov_y ) {

    size_t nbytes = WIDTH * HEIGHT * 3;


    // -------------------------------------------------------------------------------------------------
    // aloca buffer de pixels na GPU


    unsigned char* d_pixels;
    cudaMalloc(&d_pixels, nbytes);
    cudaMemset(d_pixels, 0, nbytes);    // começa tudo preto


    // -------------------------------------------------------------------------------------------------
    // parsing de glm::vec3 → double3 do CUDA

    double3 c_pos   = { pos.x,   pos.y,   pos.z   };
    double3 c_fwd   = { fwd.x,   fwd.y,   fwd.z   };
    double3 c_right = { right.x, right.y, right.z };
    double3 c_up    = { up.x,    up.y,    up.z    };


    // -------------------------------------------------------------------------------------------------
    // dimensões do grid: blocos 16×16, grid cobre a resolução inteira


    dim3 blockSize(16, 16);
    dim3 numBlocks( (WIDTH  + 15) / 16,
                    (HEIGHT + 15) / 16 );


    // -------------------------------------------------------------------------------------------------
    // lança o kernel


    std::cout << "[geodesic] Lançando kernel " 
              << numBlocks.x << "×" << numBlocks.y << " blocos\n";

    raytraceKernel<<<numBlocks, blockSize>>>(
        d_pixels, WIDTH, HEIGHT,
        c_pos, c_fwd, c_right, c_up,
        fov_y, RS
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[geodesic] Erro no kernel: " << cudaGetErrorString(err) << "\n";
    }

    cudaDeviceSynchronize();


    // -------------------------------------------------------------------------------------------------
    // copia resultado de volta para o buffer host

    cudaMemcpy(pixels, d_pixels, nbytes, cudaMemcpyDeviceToHost);
    


    cudaFree(d_pixels);

}
