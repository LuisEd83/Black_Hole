#include "geodesic.cuh"
#include <cmath>


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/* geodesicRHS — calcula as derivadas da geodésica de Schwarzschild

  equação:  d²xᵘ/dλ² + Γᵘ_αβ (dxᵅ/dλ)(dx^β/dλ) = 0
      → dados os estados atuais, computa as derivadas relativas a esse estado.

  Γ é o símbolo de Christoffel da métrica de Schwarzschild.
  Na expansão em 3 dimensões, resultam 6 EDOs:
     d/dλ [ r, θ, φ, dr, dθ, dφ ] = rhs[0..5]
*/ 


__device__ void geodesicRHS(const Rays& s, double rhs[6], double rs) {

    // f = 1 - r_s/r  — fator de Schwarzschild.
    // Quando r >> r_s, f ≈ 1 (espaço plano). Em r = r_s, f = 0 (horizonte).
    double f     = 1.0 - rs / s.r;
    double dt    = s.E / f;

    double sin_t = sin(s.theta);
    double cos_t = cos(s.theta);


    // RHS[0,1,2] — triviais, vêm direto de Rays
    rhs[0] = s.dr;
    rhs[1] = s.dtheta;
    rhs[2] = s.dphi;


    // RHS[3] — aceleração radial
    rhs[3] =  - (rs / (2.0 * s.r * s.r)) * f * dt * dt
              + (rs / (2.0 * s.r * s.r * f)) * s.dr * s.dr
              + s.r * (s.dtheta * s.dtheta + sin_t * sin_t * s.dphi * s.dphi);


    // RHS[4] — aceleração angular polar (θ)
    rhs[4] =  - (2.0 / s.r) * s.dr * s.dtheta
              + sin_t * cos_t * s.dphi * s.dphi;


    // RHS[5] — aceleração angular azimutal (φ)
    rhs[5] =  - (2.0 / s.r) * s.dr * s.dphi
              - 2.0 * (cos_t / sin_t) * s.dtheta * s.dphi;

}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/*
 rk4Step — integrador Runge-Kutta de 4ª ordem

  Resolve as 6 EDOs acima em um passo de tamanho dl.
  Faz uma média ponderada de 4 avaliações da derivada em pontos distintos,
  o que reduz o erro de truncamento para O(dl⁵) por passo.

  k1 → derivada no ponto inicial
  k2 → derivada no ponto médio estimado por k1
  k3 → derivada no ponto médio estimado por k2
  k4 → derivada no ponto final estimado por k3
  média final: state += (dl/6) * (k1 + 2k2 + 2k3 + k4)

*/

__device__ void rk4Step(Rays& s, double dl, double rs){

    double y0[6] = { s.r, s.theta, s.phi, s.dr, s.dtheta, s.dphi };
    double k1[6], k2[6], k3[6], k4[6], tmp[6];
    Rays t;


    // k1: variação no ponto inicial
    geodesicRHS(s, k1, rs);

        
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // k2: variação num ponto médio estimado por k1
    t = s;
    for (int i = 0; i < 6; i++) tmp[i] = y0[i] + k1[i] * (dl / 2.0);
    t.r = tmp[0]; t.theta = tmp[1]; t.phi = tmp[2];
    t.dr = tmp[3]; t.dtheta = tmp[4]; t.dphi = tmp[5];
    geodesicRHS(t, k2, rs);
    

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // k3: variação num ponto médio estimado por k2
    t = s;
    for (int i = 0; i < 6; i++) tmp[i] = y0[i] + k2[i] * (dl / 2.0);
    t.r = tmp[0]; t.theta = tmp[1]; t.phi = tmp[2];
    t.dr = tmp[3]; t.dtheta = tmp[4]; t.dphi = tmp[5];
    geodesicRHS(t, k3, rs);


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // k4: variação no ponto final estimado por k3
    t = s;
    for (int i = 0; i < 6; i++) tmp[i] = y0[i] + k3[i] * dl;
    t.r = tmp[0]; t.theta = tmp[1]; t.phi = tmp[2];
    t.dr = tmp[3]; t.dtheta = tmp[4]; t.dphi = tmp[5];
    geodesicRHS(t, k4, rs);

    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // média final: (k1 + 2k2 + 2k3 + k4) / 6
    s.r      += (dl / 6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
    s.theta  += (dl / 6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
    s.phi    += (dl / 6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]);
    s.dr     += (dl / 6.0) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]);
    s.dtheta += (dl / 6.0) * (k1[4] + 2*k2[4] + 2*k3[4] + k4[4]);
    s.dphi   += (dl / 6.0) * (k1[5] + 2*k2[5] + 2*k3[5] + k4[5]);

}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// raytraceKernel — kernel principal, um thread por pixel

__global__ void raytraceKernel( unsigned char* pixels,
                                int WIDTH, int HEIGHT,
                                double3 cam_position,
                                double3 cam_fwd,
                                double3 cam_right,
                                double3 cam_up,
                                float fov_y,
                                double rs ){


    // alocando thread a um pixel singular
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;


    // u,v em [-1,1] — FOV e aspect ratio controlam a escala
    float aspect     = float(WIDTH) / float(HEIGHT);
    float tanHalfFov = tan(fov_y * 0.5f * (float)M_PI / 180.0f);
    float u          = (2.0f * (x + 0.5f) / WIDTH  - 1.0f) * aspect * tanHalfFov;
    float v          = (1.0f - 2.0f * (y + 0.5f) / HEIGHT) * tanHalfFov;


    // direção do raio: u*right + v*up + forward, normalizada
    double dx = u*cam_right.x + v*cam_up.x + cam_fwd.x;
    double dy = u*cam_right.y + v*cam_up.y + cam_fwd.y;
    double dz = u*cam_right.z + v*cam_up.z + cam_fwd.z;
    double dlen = sqrt(dx*dx + dy*dy + dz*dz);
    dx /= dlen; dy /= dlen; dz /= dlen;

    

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // transformando para coordenadas esféricas e inicializando Rays

    double ox = cam_position.x;
    double oy = cam_position.y;
    double oz = cam_position.z;

    double r0     = sqrt(ox*ox + oy*oy + oz*oz);
    double theta0 = acos(oz / r0);
    double phi0   = atan2(oy, ox);

    // coordenadas cartesianas → esféricas via Jacobiano da transformação esférica → cartesiana
    double st = sin(theta0), ct = cos(theta0), sp = sin(phi0), cp = cos(phi0);

    Rays s;
    s.r      = r0;
    s.theta  = theta0;
    s.phi    = phi0;
    s.dr     =  st*cp*dx + st*sp*dy + ct*dz;
    s.dtheta = (ct*cp*dx + ct*sp*dy - st*dz) / r0;
    s.dphi   = (-sp*dx   + cp*dy)            / (r0 * st);

    double f0   = 1.0 - rs / r0;
    double vmag = sqrt(  s.dr*s.dr/f0
                       + r0*r0 * s.dtheta*s.dtheta
                       + r0*r0 * st*st * s.dphi*s.dphi );

    s.L = r0 * r0 * st * s.dphi;
    s.E = f0 * vmag;


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // integração — loop principal do ray tracer
    // TODO: adicionar loop rk4Step + testes de interseção (horizonte, disco, skybox)

}
