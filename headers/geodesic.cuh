#pragma once

#include <cuda_runtime.h>

#include "temp_and_time.hpp"
#include "constants.hpp"
#include "platform.hpp"
#include "comms.cuh"
#include "distribution.hpp"
#include "feedbacks.cuh"

#include <cmath>

using namespace BH;


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// struct Rays:
// estado completo de um raio em coordenadas esféricas de Schwarzschild


struct Rays {

    //  coordenadas esféricas:
    //  distância radial ao centro do buraco negro:  r     (metros)
    //  ângulo polar relativo ao polo norte:         theta (radianos, [0, π])
    //  ângulo azimutal relativo ao equador:         phi   (radianos, [0, 2π])
    double r, theta, phi;

    //  primeira derivada em relação ao parâmetro afim λ:
    //  ∂r/∂λ, ∂θ/∂λ, ∂φ/∂λ
    double dr, dtheta, dphi;

    //  constantes de movimento (conservadas ao longo da geodésica):
    //  E → energia por unidade de massa
    //  L → momento angular por unidade de massa
    double E, L;

};


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// declaração do kernel — definido em geodesic.cu, lançado em geodesic_host.cpp


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
                cudaTextureObject_t perlin
            );


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
                cudaTextureObject_t perlin
            );


__device__ void pixelProcess(   int x, 
                                int y,
                                unsigned char &R, 
                                unsigned char &G, 
                                unsigned char &B, 
                                int WIDTH, int HEIGHT,
                                double3 pos,
                                double3 fwd,
                                double3 right,
                                double3 up,
                                float fov_y,
                                double rs,
                                cudaTextureObject_t starmap,
                                cudaTextureObject_t perlin,
                                RayResult& result
                            );


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
