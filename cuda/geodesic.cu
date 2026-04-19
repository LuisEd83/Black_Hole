#include "geodesic.cuh"
#include <iostream>
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
                                double rs
                                ){


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
    double dx = u * cam_right.x + v * cam_up.x + cam_fwd.x;
    double dy = u * cam_right.y + v * cam_up.y + cam_fwd.y;
    double dz = u * cam_right.z + v * cam_up.z + cam_fwd.z;
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
        
    //if (theta0 < 0.05) theta0 = 0.05; 
    if (theta0 > M_PI - 1e-6) theta0 = M_PI - 1e-6;

    // coordenadas cartesianas → esféricas via Jacobiano da transformação esférica → cartesiana
    double st = sin(theta0), ct = cos(theta0), sp = sin(phi0), cp = cos(phi0);
    if (st < 1e-12) st = 1e-12;   // só protege a divisão, não altera a física

    Rays s;
    s.r      = r0;
    s.theta  = theta0;
    s.phi    = phi0;
    s.dr     =  st * cp * dx + st * sp * dy + ct * dz;
    s.dtheta = (ct * cp * dx + ct * sp * dy - st * dz) / r0;
    s.dphi   = (-sp * dx + cp * dy)            / (r0 * st);

    double f0   = 1.0 - rs / r0;
    double vmag = sqrt(  s.dr*s.dr/f0
                       + r0 * r0 * s.dtheta * s.dtheta
                       + r0 * r0 * st * st * s.dphi * s.dphi );

    s.L = r0 * r0 * st * s.dphi;
    s.E = f0 * vmag;
    

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // integração — loop principal do ray tracer
        

    const int MAX_STEPS = 2000;
    const double step = rs * 1.0;
    const double escape_radius = rs * 200.0;
    const double disk_r1 = rs * 3.0;
    const double disk_r2 = rs * 12.0;

    unsigned char R = 0, G = 0, B = 0;

    double y_prev = s.r * cos(s.theta);
                
        

        // ─────────────────────────────────────────────────────────────────────────────────────────────────
        /*
        */
        double* dbg_inside;

        if (s.r <= rs * 0.1) {
            atomicAdd(dbg_inside, 1.0);
        }
        // ─────────────────────────────────────────────────────────────────────────────────────────────────



    if (s.r >= escape_radius) {
        pixels[(y * WIDTH + x) * 3 + 0] = 255;  // branco puro = bug de escape imediato
        pixels[(y * WIDTH + x) * 3 + 1] = 255;
        pixels[(y * WIDTH + x) * 3 + 2] = 255;
        return;
    }

    if (s.r <= rs) {
        pixels[(y * WIDTH + x) * 3 + 0] = 0;    // preto = bug de horizonte imediato
        pixels[(y * WIDTH + x) * 3 + 1] = 0;
        pixels[(y * WIDTH + x) * 3 + 2] = 0;
        return;
    }

    if (s.r != s.r) {
        R = 0; G = 128; B = 0;
    }



        // ─────────────────────────────────────────────────────────────────────────────────────────────────
        /*
        */
        if (x == WIDTH/2 && y == HEIGHT/2) {
            printf("centro: dr=%.4f  dtheta=%.4e  dphi=%.4e  L=%.4e\n",
               s.dr, s.dtheta, s.dphi, s.L);
        }
        // ─────────────────────────────────────────────────────────────────────────────────────────────────

     
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    

    for(int i = 0; i < MAX_STEPS; i++){
       

        // ─────────────────────────────────────────────────────────────────────────────────────────────────
        /*
        if (x == WIDTH/2 && y == HEIGHT/2 && i < 5) {
            printf("step %d: r=%.3e  theta=%.4f  dr=%.3e  E=%.3e dr=%.3e  dtheta=%.3e  dphi=%.3e\n\n",
                   i, s.r, s.theta, s.dr, s.E, s.dr, s.dtheta, s.dphi);
        }
        
        if (x == WIDTH/2 && y == HEIGHT/2 && i == 0)
            printf("kernel rs=%.3e  s.r=%.3e  ratio=%.4f\n", rs, s.r, s.r/rs);
            
        if (x == WIDTH/2 && y == HEIGHT/2 && i >= 50 && i <= 65) {
            printf("step %d: r/rs=%.4f  f=%.4f\n",
                   i, s.r/rs, 1.0 - rs/s.r);
        }
        
        if (x == WIDTH/2 && y == HEIGHT/2 && i == 58)
            printf("step 58: s.r=%.6e  rs=%.6e  diferenca=%.6e\n",
                   s.r, rs, s.r - rs);
        */
        // ─────────────────────────────────────────────────────────────────────────────────────────────────


        if(s.r <= rs){
            R = G = B = 0;
            break;
        }

        if(s.r > escape_radius){
            float t = (float)(s.theta / M_PI);

            R = (unsigned char)(10 * (1.0f - t));
            G = (unsigned char)(10 * (1.0f - t));
            B = (unsigned char)(40 + 60 * (1.0f - t));

            break;
        }


        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    
            rk4Step(s, step, rs);

                
            if(s.r <= rs){
                R = G = B = 0;
                break;
            }
                    
                            
                // ─────────────────────────────────────────────────────────────────────────────────────────────────
                /*
                if (x == WIDTH/2 && y == HEIGHT/2)
                    printf("pos-step %d: s.r=%.3e  rs=%.3e  teste=%d\n",
                       i, s.r, rs, (int)(s.r <= rs));
                */    
                // ─────────────────────────────────────────────────────────────────────────────────────────────────
            

            double y_next = s.r * cos(s.theta);
       

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
            
            
            // ─────────────────────────────────────────────────────────────────────────────────────────────────
            /*
            if (s.r <= rs) {
                if (x == WIDTH/2 && y == HEIGHT/2)
                    printf("HORIZONTE DETECTADO no step %d: r/rs=%.4f\n", i, s.r/rs);
                R = 0; G = 0; B = 0;

                break;
            }
            */
            // ─────────────────────────────────────────────────────────────────────────────────────────────────
    

        if(y_prev * y_next < 0.0){

            double r_current = s.r;
                
            if(r_current >= disk_r1 && r_current <= disk_r2){

                float t = (float)((r_current - disk_r1) / (disk_r2 - disk_r1));

                R = (unsigned char)(255);
                G = (unsigned char)(180 * (1.0f - t * 0.7f));
                B = (unsigned char)(50 * (1.0f - t));
                
                break;
            }
        }

        y_prev = y_next;
    }
   

        // ─────────────────────────────────────────────────────────────────────────────────────────────────
        /*  
        if(x == WIDTH/2 && y == HEIGHT/2)
            printf("MAX_STEPS atingido: r/rs=%.4f\n", s.r/rs);
        */    
        // ─────────────────────────────────────────────────────────────────────────────────────────────────



    if(R == 0 && G == 0 && B == 0) R = 40;

    int idx = (y * WIDTH + x) * 3;

    pixels[idx+0] = R;
    pixels[idx+1] = G;
    pixels[idx+2] = B;

}



// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// wrapper para chamada do kernel


void launchRaytrace( unsigned char* pixels, 
                     int WIDTH, int HEIGHT,
                     double3 pos, double3 fwd, double3 right, double3 up,
                     float fov_y, double rs){

    
    size_t nbytes = WIDTH * HEIGHT * 3;

    unsigned char* d_pixels;
    cudaMalloc(&d_pixels, nbytes);
    cudaMemset(d_pixels, 0, nbytes);


    dim3 blockSize(16, 16);
    dim3 numBlocks((WIDTH + 15)/16, (HEIGHT + 15)/16);

    raytraceKernel<<<numBlocks, blockSize>>>(
        d_pixels, WIDTH, HEIGHT, pos, fwd, right, up, fov_y, rs
    );

    cudaDeviceSynchronize();
    cudaMemcpy(pixels, d_pixels, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
