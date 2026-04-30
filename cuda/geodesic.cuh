#pragma once



// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// geodesic.cuh
// Header compartilhado entre geodesic.cu (NVCC) e geodesic_host.cpp (GCC).

#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// constantes físicas


static const double c       = 299792458.0;  // velocidade da luz (m/s)
static const double G       = 6.67430e-11; // constante de gravitação universal, m³/(kg·s²)
static const double BH_MASS = 8.54e36;     // massa de Sagittarius A*, kg

// Schwarzschild radius: r_s = 2GM/c²
// raio do horizonte de eventos — abaixo disso, a luz não escapa
static const double RS = 2.0 * G * BH_MASS / (c * c);


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


__global__ void raytraceKernel( bool is_gl, unsigned char* pixels,
                                int WIDTH, int HEIGHT,
                                double3 cam_position,
                                double3 cam_fwd,
                                double3 cam_right,
                                double3 cam_up,
                                float fov_y,
                                double rs,
                                cudaTextureObject_t starmap,
                                cudaTextureObject_t perlin
                               );

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// mais declarações de funções:


void activateSetFlags();
unsigned int* getStateCountsPtr();


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


extern __device__ cudaTextureObject_t starmap;
extern __device__ cudaTextureObject_t perlin;


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
