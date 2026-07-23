// Código host puro — compilado pelo GCC, sem NVCC.
// Responsável por: alocar memória na GPU, lançar o kernel, copiar resultado de volta.


#include "../headers/starmap.hpp"
#include "../headers/perlin.hpp"
#include "../headers/geodesic.cuh"

#include <iostream>
#include <glm/glm.hpp>

#if BH_CPU_BACKEND
#include "cpu_raytrace.hpp"
#endif


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/* raytraceCUDA — chamada pelo gl_engine.cpp a cada frame em que a câmera muda

        pixels  → buffer RGB já alocado pelo caller (WIDTH * HEIGHT * 3 bytes)
        pos     → posição da câmera em coordenadas cartesianas
        fwd     → vetor forward normalizado
        right   → vetor right normalizado
        up      → vetor up normalizado
        fov_y   → campo de visão vertical em graus
*/ 

using namespace std;
using namespace glm;

#if !BH_CPU_BACKEND
void launchGL(  cudaSurfaceObject_t surface, 
                RenderParams rnd);

void launchPNG( unsigned char* pixels,
                RenderParams rnd);
#endif


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


void raytraceCUDA(  unsigned char* pixels,
                    cudaSurfaceObject_t surface,
                    int WIDTH, 
                    int HEIGHT,
                    dvec3 pos, 
                    dvec3 fwd, 
                    dvec3 right, 
                    dvec3 up,
                    double fov_y){

        
    double3 c_pos   = { pos.x,   pos.y,   pos.z   };
    double3 c_fwd   = { fwd.x,   fwd.y,   fwd.z   };
    double3 c_right = { right.x, right.y, right.z };
    double3 c_up    = { up.x,    up.y,    up.z    };
    
    RenderParams rnd =   {      WIDTH, 
                                HEIGHT, 
                                c_pos, 
                                c_fwd, 
                                c_right,
                                c_up,
                                fov_y,
                                starmap,    
                                perlin
                            }; 



    #if BH_CPU_BACKEND
        launchRaytraceCPU(  pixels, WIDTH, HEIGHT,
                            c_pos, c_fwd, c_right, c_up,
                            fov_y, rs_local, starmap, perlin);
    #else
    if(BH::is_gl){
        launchGL(surface, rnd);

    } else {
        launchPNG(pixels, rnd);

    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        cerr << "[CUDA] Erro: " << cudaGetErrorString(err) << "\n";
    
    #endif
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
