// Código host puro — compilado pelo GCC, sem NVCC.
// Responsável por: alocar memória na GPU, lançar o kernel, copiar resultado de volta.


#include "starmap.hpp"
#include "perlin.hpp"
#include "../cuda/geodesic.cuh"

#include <iostream>
#include <glm/glm.hpp>

#if BH_CPU_BACKEND
#include "cpu_raytrace.hpp"
#endif


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/* raytraceCUDA — chamada pelo engine.cpp a cada frame em que a câmera muda

        pixels  → buffer RGB já alocado pelo caller (WIDTH * HEIGHT * 3 bytes),
                  ou cudaArray_t quando is_gl=true (CUDA path)
        pos     → posição da câmera em coordenadas cartesianas
        fwd     → vetor forward normalizado
        right   → vetor right normalizado
        up      → vetor up normalizado
        fov_y   → campo de visão vertical em graus
*/


inline const double rs_local = 2.0 * G * BH_MASS / (c * c);

#if !BH_CPU_BACKEND
void launchRaytrace( bool is_gl, void* pixels, int WIDTH, int HEIGHT,
                     double3 pos, double3 fwd, double3 right, double3 up,
                     float fov_y, double rs,
                     cudaTextureObject_t starmap, cudaTextureObject_t perlin);
#endif


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


void raytraceCUDA( bool is_gl, void* pixels,
                   int WIDTH, int HEIGHT,
                   glm::vec3 pos, glm::vec3 fwd, glm::vec3 right, glm::vec3 up,
                   float fov_y){

    double3 c_pos   = { pos.x,   pos.y,   pos.z   };
    double3 c_fwd   = { fwd.x,   fwd.y,   fwd.z   };
    double3 c_right = { right.x, right.y, right.z };
    double3 c_up    = { up.x,    up.y,    up.z    };

#if BH_CPU_BACKEND
    launchRaytraceCPU(pixels, WIDTH, HEIGHT,
                      c_pos, c_fwd, c_right, c_up,
                      fov_y, rs_local, starmap, perlin);
#else
    launchRaytrace(is_gl, pixels, WIDTH, HEIGHT,
                   c_pos, c_fwd, c_right, c_up,
                   fov_y, rs_local, starmap, perlin);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "[CUDA] Erro: " << cudaGetErrorString(err) << "\n";
#endif
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
