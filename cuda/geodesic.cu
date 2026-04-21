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
/* rk4Step — integrador Runge-Kutta de 4ª ordem

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
            
        O disco gira em torno do buraco negro. O lado esquerdo (para um observador padrão)
        se aproxima da câmera — luz comprimida, mais azul, mais brilhante.
        O lado direito se afasta — luz esticada, mais vermelho, mais escura.
    
        Fórmulas:

            v/c = sqrt(rs / (2*r - rs))
            D = 1 / (γ * (1 - v/c * cos(α))),   α: v∠ r_current
                                                γ: 1/√(1 - v²/c²)      

            A intensidade percebida escala com D^4 (3 de aberração + 1 de energia do fóton).
            A cor percebida tem o comprimento de onda dividido por D:

                D > 1: lado que se aproxima  → + brilhante, + azul
                D < 1: lado que se afasta    → + escuro, + vermelho   
               
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


__device__ float redShift(double r_current, double rs){
        
    /*
        Um fóton emitido num raio r chega ao observador com a
        sua frequência reduzida por um fator z:

            z = 1 / sqrt(1 - rs/r).

            → para r = rs,  z → ∞, a luz tem frequência reduzida por completo.
            → para r = 2rs, z → sqrt(2), 41% de redução da luz.

            Logo, quanto maior a distância, menor o efeito redshift.

        Isso tem efeito na cor vista do disco, regiões mais próximas são
        mais vermelhas porque perdem mais frequência. 
    */

    double z_grav = 1.0 / sqrt(1.0 - rs / r_current);
    float freq_shift = (float)(1.0 / z_grav);

    return freq_shift;

}


__device__ float diskEmissivity(double r_current, double z_cartesiano,
                                double disk_r1, double disk_r2,
                                double height_scale){
    
    /*
    
        Antes estávamos usando uma detecção binária (if(y_prev * y_next < 0.0)). Esse efeito causa
        a perda de vários pontos que, na realidade assumem cores gradientes, mas são "colapsadas" pela
        detecção. Com o disco volumétrico, conseguimos pegar esse gradiente pela acumulação da emissividade da luz.

        Disco fino:
            → detecta quando y muda de sinal (cruzamento do plano z=0)
            → atribui uma cor e para (break)
            → resultado: disco infinitamente fino, bordas pixeladas

       Disco volumétrico:
            → a cada step, verifica se o raio está DENTRO do volume do disco
            → acumula emissividade * densidade * ds ao longo do caminho
            → o raio NÃO para — continua integrando e somando contribuições
            → resultado: disco com espessura, bordas suaves, múltiplas camadas visíveis
    
        Fórmulas:

            • raio equatorial:  disk_r1 ≤ r_eq ≤ disk_r2
            • altura vertical:  |z_cartesiano| ≤ H(r)   onde H(r) = r * escala_altura
            • emissividade(r, z) = densidade(r, z) * temperatura(r)
                onde:
                    densidade(r, z) = exp(-z² / (2 * H²))          gaussiana vertical
                    temperatura(r)  = (r / disk_r1)^(-3/4)         lei de potência radial
            ↓
            • cor += emissividade * cor_base(r) * Doppler^4 * redshift * step

    */


    if(r_current < disk_r1 || r_current > disk_r2) return 0.0f;
    
    double H = r_current * height_scale;
    double gaussian = exp(-(z_cartesiano * z_cartesiano) / (2.0 * H * H));


    // perfil radial de temperatura — disco interno mais quente
    // T ∝ r^(-3/4) é a lei de Stefan-Boltzmann para disco de acreção
    double t_normalized = (r_current - disk_r1) / (disk_r2 - disk_r1);  // [0,1]
    double temp_profile = pow(1.0 - t_normalized * 0.8, 0.75);       // mais brilhante interno
    
    return (float)(gaussian * temp_profile);

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

    
    /*
        step        —   tamanho de cada passo de integração em metros (rs * N). Menor = mais preciso, 
                        bordas do disco mais suaves, photon ring mais visível, render mais lento. 
                        Maior = mais rápido, mais aliasing, raios podem pular o horizonte. O passo adaptativo 
                        já reduz automaticamente perto do horizonte.

        MAX_STEPS       —   quantas iterações cada raio pode fazer. Determina se raios que orbitam múltiplas vezes 
                            (photon ring) conseguem completar. Mais alto = photon ring aparece, render mais lento. 
                            O tempo de render escala quase linearmente com MAX_STEPS para pixels que atingem o limite.

        disk_r1         —   borda interna do disco em unidades de RS. Fisicamente deve ser ≥ 3 RS (ISCO — última órbita estável). 
                            Menor que isso é ficção. Aumentar esconde a região mais brilhante próxima ao horizonte.

        disk_r2         —   borda externa do disco. Controla o tamanho visual do disco. Maior = disco ocupa mais da tela. Se cam_dist < disk_r2 
                            a câmera está dentro do disco — resultado é tela laranja.

        escape_radius   —   distância que o kernel considera "infinito". Deve ser maior que cam_dist. Se muito pequeno, raios que deveriam chegar ao starmap são cortados antes. 
                            RS * 200 é seguro para qualquer cam_dist até RS * 100.


    */

    const int MAX_STEPS = 5000;
    const double step = rs * 0.50;

    const double escape_radius = rs * 300.0;

    const double disk_r1 = rs * 3.0;
    const double disk_r2 = rs * 12.0;
    const double disk_height_scale = 0.10; // maior, aumenta espessura
    const double disk_opacity = 0.95;
    const double emission_scale = 2.0;
    
    const double adaptive_factor = 3.0f;
    
    float accum_r = 0.0f;       // canal vermelho acumulado
    float accum_g = 0.0f;       // canal verde acumulado
    float accum_b = 0.0f;       // canal azul acumulado
    float accum_alpha = 0.0f;   // opacidade acumulada [0,1]
                                // quando accum_alpha ≥ 1.0 → disco completamente opaco → break


    unsigned char R = 0, G = 0, B = 0;

    double y_prev = s.r * cos(s.theta);

  


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    
    enum class RayResult { 
        NONE, 
        HORIZON, 
        ESCAPE, 
        DISK, 
        FALLBACK 
    };
    
    RayResult result = RayResult::NONE;


    for(int i = 0; i < MAX_STEPS; i++){
      
        if(s.r <= rs || s.r <= 0.0){
            R = G = B = 0;

            result = RayResult::HORIZON;
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
    
            result = RayResult::ESCAPE;
            break;
        }


        double adaptive_step = step;
        if(s.r < rs * 5.0){
            adaptive_step = step * (s.r / (rs * adaptive_factor));  // escala linear: a 1rs → step/5
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
            
            double r_prev_step = s.r;
    
            rk4Step(s, adaptive_step, rs);
                
            if(s.r <= rs || s.r <= 0.0 || s.r != s.r){
                R = G = B = 0;

                result = RayResult::HORIZON;
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
    
        /* classificador binário.
        if(y_prev * y_next < 0.0){

                // ─────────────────────────────────────────────────────────────────────────────────────────────────
                if(s.r/rs > 210)
                    printf("[debug] - DISCO no step %d: r/rs=%.3f\n", i, s.r/rs);
                // ─────────────────────────────────────────────────────────────────────────────────────────────────

            //double r_current = s.r;

            double frac  = fabs(y_prev) / (fabs(y_prev) + fabs(y_next));
            double r_current = r_prev_step + frac * (s.r - r_prev_step);
            double phi = s.phi;

            if(r_current >= disk_r1 && r_current <= disk_r2){

                float t = (float)((r_current - disk_r1) / (disk_r2 - disk_r1));
    

                // aplicando perlin
                float perlin_noise = perlinNoise(perlin, r_current, phi, disk_r1, disk_r2);
                float base_brightness = 0.4f + 0.6f * perlin_noise;


                // aplicando doppler
                float doppler = dopplerShift(phi, r_current, cam_position, rs);
    

                // aplicando redshift
                float redshift = redShift(r_current, rs);

                    
                // combinando doppler e redshift:
                float combined = doppler * redshift;
                
                //float intensity = base_brightness * (float)powf(combined, 4.0);
                float intensity = base_brightness * powf(combined, 3.0f);
                intensity = fminf(intensity, 5.0f);


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
            
                result = RayResult::DISK;
                break;
            }
        }
        */
        
        
        
        double x_cartesiano = s.r * sin(s.theta) * cos(s.phi);
        double y_cartesiano = s.r * sin(s.theta) * sin(s.phi);
        double z_cartesiano = s.r * cos(s.theta);
        
        double r_current = sqrt(x_cartesiano*x_cartesiano +  y_cartesiano*y_cartesiano);
        float emissividade = diskEmissivity(r_current, z_cartesiano, disk_r1, disk_r2, disk_height_scale);
            
        // detecção por emissividade de disco volumétrico
        if(emissividade > 0.001f){
            
            float doppler = dopplerShift(r_current, s.phi, cam_position, rs);
            float z_grav = (float)(1.0 / sqrt(fmax(1.0 - rs/s.r, 1e-6)));
            float freq = doppler / z_grav;

            float t = (float)((r_current - disk_r1) / (disk_r2 - disk_r1));
            float disk_r = 1.0f;
            float disk_g = 180.0f/255.0f * (1.0 - t * 0.7f);
            float disk_b = 50.0f/255.0f * (1.0 - t);

            float intensity = emissividade * emission_scale * powf(fmax(freq, 0.01f), 3.0f);
            
            float step_normalized = (float)(adaptive_step / (disk_r2 - disk_r1));
            float contribution = intensity * step_normalized * disk_opacity;

            float remaining = 1.0f - accum_alpha;
            accum_r     += disk_r * contribution * remaining;
            accum_g     += disk_g * contribution * remaining;
            accum_b     += disk_b * contribution * remaining;

            accum_alpha += contribution * remaining;

            // disco completamente opaco → para de integrar
            if (accum_alpha >= 1.0f){
                accum_alpha = 1.0f;
                break;
            }
            
            result = RayResult::DISK;
        }

        y_prev = y_next;
    }


    
    if (result == RayResult::NONE) result = RayResult::FALLBACK;

        // ─────────────────────────────────────────────────────────────────────────────────────────────────
        /*  
        if(x == WIDTH/2 && y == HEIGHT/2)
            printf("[debug] - MAX_STEPS atingido: r/rs=%.4f\n", s.r/rs);
        */    
        // ─────────────────────────────────────────────────────────────────────────────────────────────────
        if (result == RayResult::HORIZON) {
            R = G = B = 0;

        }else if(result == RayResult::ESCAPE || result == RayResult::FALLBACK){

            if(result == RayResult::FALLBACK && s.r < rs * 4.0) {
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

    //if(result == RayResult::DISK && accum_alpha > 0.001f) {
    if(accum_alpha > 0.001f){

        // normaliza pela opacidade acumulada
        //float inv_a = 1.0f / fmaxf(accum_alpha, 0.001f);

        // fundo: cor do skybox ou horizonte (já calculada em R,G,B)
        // compositing: disco na frente, fundo atrás
        float bg_weight = 1.0f - fminf(accum_alpha, 1.0f);

        R = (unsigned char)(fminf((accum_r + R/255.0f * bg_weight) * 255.0f, 255.0f));
        G = (unsigned char)(fminf((accum_g + G/255.0f * bg_weight) * 255.0f, 255.0f));
        B = (unsigned char)(fminf((accum_b + B/255.0f * bg_weight) * 255.0f, 255.0f));
        
        //printf("%f\n",fminf((accum_r + R/255.0f * bg_weight) * 255.0f, 255.0f));
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


    static bool flagSet = false;
    if (!flagSet) {
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        flagSet = true;
    }

    size_t nbytes = WIDTH * HEIGHT * 3;

    unsigned char* d_pixels;
    cudaMalloc(&d_pixels, nbytes);
    cudaMemset(d_pixels, 0, nbytes);


    dim3 blockSize(16, 16);
    dim3 numBlocks((WIDTH + 15)/16, (HEIGHT + 15)/16);

    raytraceKernel<<<numBlocks, blockSize>>>(
        d_pixels, WIDTH, HEIGHT, pos, fwd, right, up, fov_y, rs, starmap, perlin
    );

    //cudaDeviceSynchronize();

    cudaMemcpy(pixels, d_pixels, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
