// Código host puro — compilado pelo GCC, sem NVCC.
// Responsável por: alocar memória na GPU, lançar o kernel, copiar resultado de volta.


#include "starmap.hpp"
#include "perlin.hpp"
#include "../cuda/headers/geodesic.cuh"

#include <iostream>
#include <glm/glm.hpp>
#include <texture_types.h>


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/* raytraceCUDA — chamada pelo gl_engine.cpp a cada frame em que a câmera muda

        pixels  → buffer RGB já alocado pelo caller (WIDTH * HEIGHT * 3 bytes)
        pos     → posição da câmera em coordenadas cartesianas
        fwd     → vetor forward normalizado
        right   → vetor right normalizado
        up      → vetor up normalizado
        fov_y   → campo de visão vertical em graus
*/ 

extern const double RS;

using namespace std;
using namespace glm;

void launchGL(  cudaSurfaceObject_t surface, 
                int WIDTH, 
                int HEIGHT,
                double3 pos, 
                double3 fwd, 
                double3 right, 
                double3 up,
                float fov_y, 
                double rs, 
                cudaTextureObject_t starmap, 
                cudaTextureObject_t perlin);


void launchPNG( unsigned char* pixels,
                int WIDTH, 
                int HEIGHT,
                double3 pos, 
                double3 fwd, 
                double3 right, 
                double3 up,
                float fov_y, 
                double rs, 
                cudaTextureObject_t starmap, 
                cudaTextureObject_t perlin);


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


void raytraceCUDA(  unsigned char* pixels,
                    cudaSurfaceObject_t surface,
                    int WIDTH, 
                    int HEIGHT,
                    vec3 pos, 
                    vec3 fwd, 
                    vec3 right, 
                    vec3 up,
                    float fov_y){

    
    double3 c_pos   = { pos.x,   pos.y,   pos.z   };
    double3 c_fwd   = { fwd.x,   fwd.y,   fwd.z   };
    double3 c_right = { right.x, right.y, right.z };
    double3 c_up    = { up.x,    up.y,    up.z    };
        
    if(BH::is_gl){
        cout << "Acessando HOST GL\n";
        cout << "Starmap: " << starmap << ", Perlin: " << perlin << "\n";
        launchGL(surface, WIDTH, HEIGHT, c_pos, c_fwd, c_right, c_up, fov_y, RS, starmap, perlin);

    } else {
        cout << "Acessando HOST PNG\n";
        cout << "Starmap: " << starmap << ", Perlin: " << perlin << "\n";
        launchPNG(pixels, WIDTH, HEIGHT, c_pos, c_fwd, c_right, c_up, fov_y, RS, starmap, perlin);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        cerr << "[CUDA] Erro: " << cudaGetErrorString(err) << "\n";

}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
