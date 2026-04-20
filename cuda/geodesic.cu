#include "geodesic.cuh"
#include <cmath>
#include <iostream>


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
    if (fabs(f) < 1e-6) f = (f < 0.00) ? -1e-6 : 1e-6;

    double dt    = s.E / f;

    double sin_t = sin(s.theta);
    double cos_t = cos(s.theta);
    double sin_t_safe = (fabs(sin_t) < 1e-12) ? 1e-12 : sin_t;

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
              - 2.0 * (cos_t / sin_t_safe) * s.dtheta * s.dphi;

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
// funções de efeito:


__device__ float dopplerShift(double phi, double r_current, double3 camera_pos, double rs){
    
    /*
        
        Fórmulas:

            v/c = sqrt(rs / (2*r - rs))
            D = 1 / (γ * (1 - v/c * cos(α))),   α: v∠ r_current
                                                γ: 1/√(1 - v²/c²)      
    */
    
    double denominador = 2.0 * r_current - rs;
    if(denominador <= 0.0) return 1.0f;
    
    double beta = sqrt(rs/denominador);
    double gamma = 1.0 / sqrt(1.0 - beta * beta);

    double vx = -sin(phi);
    double vy = cos(phi);
    // vz = 0, disco no equador
    
    double dx = camera_pos.x - r_current * cos(phi);
    double dy = camera_pos.y - r_current * sin(phi);
    double dz = camera_pos.z;

    double dlen = sqrt(dx*dx + dy*dy + dz*dz);
    if(dlen < 1e-10) return 1.0f;

    dx /= dlen; dy /= dlen;
    

    double cos_alpha = vx * dx + vy * dy;

    double doppler = 1.0 / (gamma * (1.0 - beta * cos_alpha));
    
    return (float)doppler;
}


__device__ float perlinNoise(cudaTextureObject_t perlin, 
                             double r_current, double phi,
                             double disk_r1, double disk_r2){

    
    if(perlin == 0) return 1.0f;

    float u = (float)((r_current - disk_r1) / (disk_r2 - disk_r1));
    float v = (float)(phi / (2.0 * M_PI)) + 0.5f;

    float4 noise = tex2D<float4>(perlin, u ,v);
    
    return noise.x;
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
                                double rs,
                                cudaTextureObject_t starmap,
                                cudaTextureObject_t perlin
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
    double theta0 = acos(fmax(-1.0, fmin(1.0, oz / r0)));
    double phi0   = atan2(oy, ox);
        
    //if (theta0 > M_PI - 1e-6) theta0 = M_PI - 1e-6;

    // coordenadas cartesianas → esféricas via Jacobiano da transformação esférica → cartesiana
    double st = sin(theta0), ct = cos(theta0), sp = sin(phi0), cp = cos(phi0);
    double st_safe = (fabs(st) < 1e-12) ? 1e-12 : st;


    Rays s;
    s.r      = r0;
    s.theta  = theta0;
    s.phi    = phi0;
    s.dr     =  st * cp * dx + st * sp * dy + ct * dz;
    s.dtheta = (ct * cp * dx + ct * sp * dy - st * dz) / r0;
    s.dphi   = (-sp * dx + cp * dy) / (r0 * st_safe);

    double f0   = 1.0 - rs / r0;
    double vmag = sqrt(s.dr*s.dr/f0
                       + r0 * r0 * s.dtheta * s.dtheta
                       + r0 * r0 * st_safe * st_safe * s.dphi * s.dphi );

    s.L = r0 * r0 * st_safe * s.dphi;
    s.E = f0 * vmag;
    


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // integração — loop principal do ray tracer
        

    const int MAX_STEPS = 2000;
    const double step = rs * 1.0;
    const double escape_radius = rs * 200.0;
    const double disk_r1 = rs * 3.0;
    const double disk_r2 = rs * 10.0;

    unsigned char R = 0, G = 0, B = 0;

    double y_prev = s.r * cos(s.theta);

    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    

    for(int i = 0; i < MAX_STEPS; i++){
      
        if(s.r <= rs || s.r <= 0.0){
            R = G = B = 0;
            break;
        }
    
        if(s.r > escape_radius){
            
                // ─────────────────────────────────────────────────────────────────────────────────────────────────
                /*
                if(s.r/rs > 210)
                    printf("[debug] - ESCAPE no step %d: r/rs=%.3f\n", i, s.r/rs);
                */
                // ─────────────────────────────────────────────────────────────────────────────────────────────────

            // converte (theta, phi) do ponto de fuga para UV do mapa equiretangular
            // theta: [0, π]    → V: [0, 1]   (polo norte = 0, polo sul = 1)
            // phi:   [-π, π]   → U: [0, 1]   (wrapa em 360°)
            float u_tex = (float)(s.phi / (2.0 * M_PI)) + 0.5f;   // [-π,π] → [0,1]
            float v_tex = (float)(s.theta / M_PI);                  // [0,π]  → [0,1]

            float4 color = tex2D<float4>(starmap, u_tex, v_tex);
            
            R = (unsigned char)(fminf(color.x * 255.0f, 255.0f));
            G = (unsigned char)(fminf(color.y * 255.0f, 255.0f));
            B = (unsigned char)(fminf(color.z * 255.0f, 255.0f));

            break;
        }


        double adaptive_step = step;
        if(s.r < rs * 5.0){
            adaptive_step = step * (s.r / (rs * 5.0));  // escala linear: a 1rs → step/5
            if(adaptive_step < step * 0.005) adaptive_step = step * 0.005;  // mínimo
        }


        // ─────────────────────────────────────────────────────────────────────────────────────────────────
        /*
        if(x == WIDTH/2 && y == HEIGHT/2 && i < 5)
            printf("[debug] - step %d: r=%.3e  theta=%.4f  dr=%.3e  E=%.3e dr=%.3e  dtheta=%.3e  dphi=%.3e\n\n",
                   i, s.r, s.theta, s.dr, s.E, s.dr, s.dtheta, s.dphi);
        
        if(x == WIDTH/2 && y == HEIGHT/2 && i == 0)
            printf("[debug] - kernel rs=%.3e  s.r=%.3e  ratio=%.4f\n", 
                    rs, s.r, s.r/rs);
        
        if(x == WIDTH/2 && y == HEIGHT/2 && i == 58)
            printf("[debug] - step 58: s.r=%.6e  rs=%.6e  diferenca=%.6e\n",
                   s.r, rs, s.r - rs);

        if(x == WIDTH/2 && y == HEIGHT/2 && i == 0)
            printf("[debug] - WIDTH=%d HEIGHT=%d x=%d y=%d\n", 
                    WIDTH, HEIGHT, x, y);

        if(x == WIDTH/2 && y == HEIGHT/2 && i < 3)
            printf("loop i=%d: s.r=%.3e  r/rs=%.3f\n", 
                    i, s.r, s.r/rs);

        if(x == WIDTH/2 && y == HEIGHT/2 && i >= 50 && i <= 65)
            printf("[debug] - step %d: r/rs=%.4f  f=%.4f\n",
                   i, s.r/rs, 1.0 - rs/s.r);
        */
        // ─────────────────────────────────────────────────────────────────────────────────────────────────


        

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    
            rk4Step(s, adaptive_step, rs);
                
            if(s.r <= rs || s.r <= 0.0 || s.r != s.r){
                R = G = B = 0;
                break;
            }
                    
                            
                // ─────────────────────────────────────────────────────────────────────────────────────────────────
                /*
                if(x == WIDTH/2 && y == HEIGHT/2)
                    printf("[debug] - pos-step %d: s.r=%.3e  rs=%.3e  teste=%d\n",
                       i, s.r, rs, (int)(s.r <= rs));
                */    
                // ─────────────────────────────────────────────────────────────────────────────────────────────────
            

            double y_next = s.r * cos(s.theta);
       

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
            
            
            // ─────────────────────────────────────────────────────────────────────────────────────────────────
            /*
            if(s.r <= rs || s.r <= 0.0) {
                if (x == WIDTH/2 && y == HEIGHT/2)
                    printf("[debug] - HORIZONTE DETECTADO no step %d: r/rs=%.4f\n", i, s.r/rs);
                R = 0; G = 0; B = 0;

                break;
            }
            */
            // ─────────────────────────────────────────────────────────────────────────────────────────────────
    

        if(y_prev * y_next < 0.0){

                // ─────────────────────────────────────────────────────────────────────────────────────────────────
                /*
                if(s.r/rs > 210)
                    printf("[debug] - DISCO no step %d: r/rs=%.3f\n", i, s.r/rs);
                */
                // ─────────────────────────────────────────────────────────────────────────────────────────────────
            /*
            */

            double r_current = s.r;
            double phi = s.phi;
            //double frac  = fabs(y_prev) / (fabs(y_prev) + fabs(y_next));
            //double r_cross = s.r;   // aproximação — poderia interpolar mais fino

            //double disk_height = r_cross * 0.1;
            //double y_cross = y_prev + frac * (y_next - y_prev);  // deve ser ~0


            if(r_current >= disk_r1 && r_current <= disk_r2){

                float t = (float)((r_current - disk_r1) / (disk_r2 - disk_r1));
    

                // aplicando perlin
                float perlin_noise = perlinNoise(perlin, r_current, phi, disk_r1, disk_r2);
                float base_brightness = 0.4f + 0.6f * perlin_noise;


                // aplicando doppler
                // D > 1: lado que se aproxima  → + brilhante, + azul
                // D < 1: lado que se afasta    → + escuro, + vermelho   
                
                float doppler = 1.0f;

                //float doppler = dopplerShift(phi, r_current, cam_position, rs);
                float intensity = base_brightness * (float)pow((double)doppler, 4.0);
                
                
                float r, g, b;
                
                if(doppler >= 1.0f){
                    
                    float blend = fminf((doppler - 1.0f) / 1.5f, 1.0f);

                    // lado se aproximando: [laranja, branco]
                    r = 1.0f;
                    g = (180.0f/255.0f) * (1.0f - t*0.7f) + blend * (1.0f - (180.0f/255.0f) * (1.0f - t*0.7f));
                    b = (50.0f/255.0f)  * (1.0f - t)      + blend * (1.0f - (50.0f/255.0f) * (1.0f - t));

                } else {

                    // lado se afastando: laranja → vermelho escuro
                    float blend = fminf((1.0f - doppler) / 0.8f, 1.0f);   // D=1→0%, D=0.2→100%
                    r = 1.0f - blend * 0.3f;                    // vermelho escurece levemente
                    g = (180.0f / 255.0f) * (1.0f - t * 0.7f) * (1.0f - blend);
                    b = (50.0f / 255.0f)  * (1.0f - t)      * (1.0f - blend);
                }
                

                R = (unsigned char)(fminf(r * intensity * 255.0f, 255.0f));
                G = (unsigned char)(fminf(g * intensity * 255.0f, 255.0f));
                B = (unsigned char)(fminf(b * intensity * 255.0f, 255.0f));

                break;
            }
        }

        y_prev = y_next;
    }
    

        // ─────────────────────────────────────────────────────────────────────────────────────────────────
        /*  
        if(x == WIDTH/2 && y == HEIGHT/2)
            printf("[debug] - MAX_STEPS atingido: r/rs=%.4f\n", s.r/rs);
        */    
        // ─────────────────────────────────────────────────────────────────────────────────────────────────


    if(R == 0 && G == 0 && B == 0){
        if (s.r < rs * 3.0) {
            R = G = B = 0;   // muito perto — considerado capturado

        } else {

            float u_tex = (float)(s.phi / (2.0 * M_PI)) + 0.5f;
            float v_tex = (float)(s.theta / M_PI);
            float4 color = tex2D<float4>(starmap, u_tex, v_tex);
            R = (unsigned char)(fminf(color.x * 255.0f, 255.0f));
            G = (unsigned char)(fminf(color.y * 255.0f, 255.0f));
            B = (unsigned char)(fminf(color.z * 255.0f, 255.0f));
        
        }
    }

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
                     float fov_y, double rs, cudaTextureObject_t starmap, cudaTextureObject_t perlin){


    size_t nbytes = WIDTH * HEIGHT * 3;

    unsigned char* d_pixels;
    cudaMalloc(&d_pixels, nbytes);
    cudaMemset(d_pixels, 0, nbytes);


    dim3 blockSize(16, 16);
    dim3 numBlocks((WIDTH + 15)/16, (HEIGHT + 15)/16);

    raytraceKernel<<<numBlocks, blockSize>>>(
        d_pixels, WIDTH, HEIGHT, pos, fwd, right, up, fov_y, rs, starmap, perlin
    );

    cudaDeviceSynchronize();
    cudaMemcpy(pixels, d_pixels, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
