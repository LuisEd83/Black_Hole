// -------------------------------------------------------------------------------------------------
// bibliotecas e mini-resumo de OpenGL:

/*

    •   GLFW: faz o processo de background, criação de janelas, contexto na OS de
        que há uma janela OpenGL rodando, i/o mouse e teclado.
    
    •   GLEW: carrega todas as funções primitivas da API OpenGL 1.1.
    
    •   glm/glm.hpp & glm/gtc/matrix_transform.hpp: headers matemáticos que dão
        ao programador a liberdade de escrever com tipos da própria OpenGL (vec3, mat4...).
        Nesse contexto, é usada para transformação de matriz em relação ao movimento da câmera.
    
    •   OpenGL: uma API que se relaciona com o driver de qualquer GPU.
        
        → Trabalha com a criação de triângulos para renderização de objetos.
        → Esses triângulos tem vértices, em que cada triângulo terá operações computadas para esses vértices.
        → A pipeline desse programa será: dados → CUDA Kernel → OpenGL → janela.

*/

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>     // → é utilizada para cout/cerr.
#include <vector>       // → é utilizada para a criação de vetores dinâmicos para os buffers de cada pixel.
#include <cmath>        // → usa sin,cos,sqrt,atan → coords polares.
#include <chrono>       // → timer de alta precisão para computar FPS.

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace glm;
using namespace std;
using Clock = std::chrono::high_resolution_clock;


// -------------------------------------------------------------------------------------------------
// constantes & struct Rays


static const double c = 299792458.0;    // velocidade da luz (m/s)
static const double G  = 6.67430e-11;   // constante de gravitação universal, m³/(kg * s²)
static const double BH_MASS = 8.54e36;  // massa de Saggitarius A*, kg. Esse buraco negro é o de nossa análise.

// Schwarzschild radius: r_s, raio do horizonte de eventos.ali, a luz não escapa mais.
// r_s = 2GM/c²
static const double RS = 2.0 * G * BH_MASS / (c * c); 


struct Rays {
    
    //  indicativo de coordenadas polares. respectivamente:
    //  distância radial do centro do buraco-negro:  r(m),
    //  ângulo polar relativo ao polo-norte:         theta(radianos[0, π]), 
    //  ângulo azimutal em relação ao equador:       phi(radianos[0,2π])
    double r, theta, phi;       
    
    //  primeira derivada dessas variáveis. sendo λ a "partícula do raio", temos, respectivamente:
    //
    //  taxa de variação do raio → velocidade:   ∂r/∂λ
    //  taxa de variação do ângulo polar:        ∂θ/∂λ
    //  taxa de variação do ângulo azimutal:     ∂φ/∂λ
    double dr, dtheta, dphi;   

    //  constantes de conservação. respectivamente:
    //  energia por unidade de massa. há conservação da massa.
    //  momento angular por unidade de massa.
    double E, L;                 

};

 
// -------------------------------------------------------------------------------------------------
// geodesic __device__:

/*

    temos a Geodésica RHS como a equação:

        • d²xᵘ/dλ² + Γᵘ_αβ (dxᵅ/dλ)(dx^β/dλ) = 0
        → traduzindo: dados estados atuais, compute as derivadas relativas a esse estado. 
    
        → Γ é o símbolo de Christoffel da métrica de Schwarzschild.
        → Na expansão de 3 dimensões, resultam 6 EDOs:
            d/dλ [ r, θ, φ, dr, dθ, dφ ] = rhs[0..5]
*/


__device__ void geodesicRHS(const Rays& s, double rhs[6], double rs){
    
    /*
        esse kernel calcula todas as derivadas das equações.
    */

    // f = 1 - r_s/r  — fator de Schwarzschild.
    // Quando r >> r_s, f ≈ 1 (flat space). At r = r_s, f = 0 (horizon).
    double f  = 1.0 - rs / s.r;
    
    double dt = s.E / f;

    double sin_t = sin(s.theta);
    double cos_t = cos(s.theta);
        

    // -------------------------------------------------------------------------------------------------
    
    // RHS[0,1,2] ------------------------
    // esses dados são mais triviais, vêm direto de Rays. 
    rhs[0] = s.dr;  
    rhs[1] = s.dtheta;
    rhs[2] = s.dphi;
    

    // RHS[3] ----------------------------
    // aceleração radial:
    rhs[3] =    - (rs / (2.0 * s.r * s.r)) * f * dt
                + (rs / (2.0 * s.r * s.r * f)) * s.dr * s.dr
                + s.r * (s.dtheta * s.dtheta + sin_t * sin_t * s.dphi * s.dphi);  


    // RHS[4] ----------------------------
    // aceleração angular (polar)
    rhs[4] =    - (2.0/s.r) * s.dr * s.dtheta
                + sin_t * cos_t * s.dphi * s.dphi;
    

    // RHS[5] ----------------------------
    // aceleração angular (azimutal)
    rhs[5] =    - (2.0/s.r) * s.dr * s.dphi
                - 2.0 * (sin_t/cos_t) * s.dtheta * s.dphi;
    

}


// -------------------------------------------------------------------------------------------------
// rk4Step - integrador de 4a ordem de Runge-Kutta.

/*
    Basicamente, um solucionador de EDOs. Porém, esse tem uma função não trivial:
    
    →   Estamos fazendo: state += derivada * dl. Ou seja, uma passagem direta do estado duma variável anterior
        para uma nova pelo produto com a derivada. Entretanto, a variação mínima de uma variável pode mudar substancialmente
        o todo. Logo, é feita uma média dos resultados de 4 iterações com pontos análogos. 
    
    →   Teremos k[1,4], e logo depois a média.

*/

__device__ void rk4Step(Rays& s, double dl, double rs){

    double y0[6] = { s.r, s.theta, s.phi, s.dr, s.dtheta, s.dphi };
    double k1[6], k2[6], k3[6], k4[6], tmp[6];
    

    // k1: ----------------------------
    // variação no ponto inicial:
    geodesicRHS(s, k1, rs);
    

    // k2: ----------------------------
    // variação num ponto médio por k1:
    Rays t = s;

    for(int i = 0; i < 6; i++){
        tmp[i] = y0[i] + k1[i] * (dl/2.0);  // computa variações
    }
    
    t.r = tmp[0];
    t.theta = tmp[1];
    t.phi = tmp[2];
    t.dr = tmp[3];
    t.dtheta = tmp[4];
    t.dphi = tmp[5];
        
    geodesicRHS(s, k2, rs);
    

    // k3: ----------------------------
    // variação num ponto médio por k2:
    
    for(int i = 0; i < 6; i++){
        tmp[i] = y0[i] + k2[i] * (dl/2.0); 
    }

    t.r = tmp[0];
    t.theta = tmp[1];
    t.phi = tmp[2];
    t.dr = tmp[3];
    t.dtheta = tmp[4];
    t.dphi = tmp[5];
        
    geodesicRHS(s, k3, rs);
   

    // k4: ----------------------------
    // variação dum ponto final por k3:

    for(int i = 0; i < 6; i++){
        tmp[i] = y0[i] + k3[i] * dl;
    }

    t.r = tmp[0];
    t.theta = tmp[1];
    t.phi = tmp[2];
    t.dr = tmp[3];
    t.dtheta = tmp[4];
    t.dphi = tmp[5];
        
    geodesicRHS(s, k4, rs);
   

    
    // -------------------------------------------------------------------------------------------------
    // média final:
    // wds: (k1 + 2k2 + 2k3 + k4) / 6
     
    s.r     += (dl/6.9) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
    s.theta += (dl/6.9) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
    s.phi   += (dl/6.9) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
    s.dr    += (dl/6.9) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]);
    s.dtheta+= (dl/6.9) * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]);
    s.dphi  += (dl/6.9) * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5]);

}


// -------------------------------------------------------------------------------------------------
// CUDA Kernel:


__global__ void raytraceKernel( unsigned char* pixels,
                                int WIDTH, int HEIGHT,
                                
                                double3 cam_position,
                                double3 cam_fwd,
                                double3 cam_right,
                                double3 cam_up,

                                float fov_y,
                                double rs){
    
    // alocando thread à pixel singular
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(x >= WIDTH || y >= HEIGHT) return;

    
    // usando de raios → → → | câmera, temos os pixels.
    // u,v estão num intervalo [-1,1], sendo o FOV e o aspect ratio quem controla em runtime.
    
    float aspect    = float(WIDTH) / float(HEIGHT);
    float tanHalfFov = tan(fov_y * 0.5f * M_PI / 180.0f);
    float u         = (2.0f * (x + 5.0f) / WIDTH - 1.0f) * aspect * tanHalfFov;
    float v         = (1.0f -2.0f * (y + 5.0f) / HEIGHT) * tanHalfFov;
    

    // direção do raio = u*right + v*up + forward. normaliza-se após. 
    double dx = u*cam_right.x + v*cam_up.x + cam_fwd.x;
    double dy = u*cam_right.y + v*cam_up.y + cam_fwd.y;
    double dz = u*cam_right.z + v*cam_up.z + cam_fwd.z;
    double dlen = sqrt(dx*dx + dy*dy + dz*dz);
    dx /= dlen; 
    dy /= dlen; 
    dz /= dlen;
    
    
    // -------------------------------------------------------------------------------------------------
    // transformando para coordenadas polares e mandando dados para Rays
    

    double ox=cam_position.x; 
    double oy=cam_position.y; 
    double oz=cam_position.z;

    double r0     = sqrt(ox*ox + oy*oy + oz*oz);
    double theta0 = acos(oz / r0);
    double phi0   = atan2(oy, ox);

    // Convert the Cartesian direction (dx,dy,dz) into spherical velocity
    // components using the Jacobian of the spherical→Cartesian transform.
    double st=sin(theta0), ct=cos(theta0), sp=sin(phi0), cp=cos(phi0);

    Rays s;
    s.r     = r0;
    s.theta = theta0;
    s.phi   = phi0;
    s.dr     =  st*cp*dx + st*sp*dy + ct*dz;
    s.dtheta = (ct*cp*dx + ct*sp*dy - st*dz) / r0;
    s.dphi   = (-sp*dx  + cp*dy)              / (r0*st);

    double f0   = 1.0 - rs/r0;
    double vmag = sqrt(s.dr*s.dr/f0 + r0*r0*s.dtheta*s.dtheta + r0*r0*st*st*s.dphi*s.dphi);

    s.L = r0 * r0 * st * s.dphi;
    s.E = f0 * vmag;
    
    
    // -------------------------------------------------------------------------------------------------
    //  

}



void raytraceCUDA(vector<unsigned char>& pixels,
                  int WIDTH, 
                  int HEIGHT,
                  vec3 pos, vec3 fwd, vec3 right, vec3 up,
                  float fov_y
                  ){

    
    size_t nbytes = WIDTH * HEIGHT * 3;
    pixels.resize(nbytes);

    unsigned char* d_pixels;
    cudaMalloc(&d_pixels, nbytes);

    //parsing de glm::vec3 → CUDA's double3
    double3 c_pos   = {pos.x,   pos.y,   pos.z};
    double3 c_fwd   = {fwd.x,   fwd.y,   fwd.z};
    double3 c_right = {right.x, right.y, right.z};
    double3 c_up    = {up.x,    up.y,    up.z};


    
    // -------------------------------------------------------------------------------------------------
    // cálculo do tamanho do kernel:
    

    dim3 blockSize(16,16);
    dim3 numBlocks((WIDTH + 15)/16, (HEIGHT + 15)/16);

    
    // -------------------------------------------------------------------------------------------------
    // kernel launch:
   

    printf("Lançando Kernel.\n");

    raytraceKernel<<<numBlocks, blockSize>>>(
        d_pixels, WIDTH, HEIGHT,
        c_pos, c_fwd, c_right, c_up,
        fov_y, RS
    );
        
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){ printf("Houve erro no launch do kernel: %s", cudaGetErrorString(err)); }

    cudaDeviceSynchronize();
    
    
    // -------------------------------------------------------------------------------------------------
    // volta dos dados


    cudaMemcpy(pixels.data(), d_pixels, nbytes, cudaMemcpyDeviceToHost);
    

    // -------------------------------------------------------------------------------------------------
    //  limpeza de memória alocada:


    cudaFree(d_pixels);

}
