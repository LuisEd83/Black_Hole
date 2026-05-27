#include "../headers/constants.hpp"
#include "../headers/distribution.hpp"

#include "../headers/comms.cuh"
#include "../headers/geodesic.cuh"
#include "../headers/feedbacks.cuh"

#include <cuda_runtime.h>
#include <cmath>


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// geodesicRHS
//@{
/* 
    geodesicRHS — calcula as derivadas da geodésica de Schwarzschild

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
//@}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// rk4Step
//@{
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


__device__  void rk4Step(Rays& s, double dl, double rs){

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
//@}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// funções de efeito:
//@{

__device__ float dopplerShift(double r_current, double phi, double3 camera_pos, double rs){
   
    /* 
        info
            
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
    
    double beta = sqrtf(rs/denominador);
    double gamma = 1.0 / sqrtf(1.0 - beta * beta);

    double vx = -sin(phi);
    double vy = cos(phi);
    // vz = 0, disco no equador
    
    double dx = camera_pos.x - r_current * cos(phi);
    double dy = camera_pos.y - r_current * sin(phi);
    double dz = camera_pos.z;

    double dlen = sqrtf(dx*dx + dy*dy + dz*dz);
    if(dlen < 1e-10) return 1.0f;

    dx /= dlen; dy /= dlen;
    
    double cos_alpha = vx * dx + vy * dy;


    double doppler = 1.0 / (gamma * (1.0 - beta * cos_alpha));
    
    return (float)doppler;
}


__device__  float perlinNoise(cudaTextureObject_t perlin, 
                             double r_current, double phi,
                             double disk_r1, double disk_r2){

    //printf("%llu",perlin); 
    if(perlin == 0) return 1.0f;

    float u = (float)((r_current - disk_r1) / (disk_r2 - disk_r1));
    float v = (float)(phi / (2.0 * PI)) + 0.5f;

    float4 noise = tex2D<float4>(perlin, u ,v);
    
    return noise.x;
}


__device__  float redShift(double r_current, double rs){
        
    /*
        info

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


__device__  float diskEmissivity(double r_current, double z_cartesiano,
                                double disk_r1, double disk_r2,
                                double height_scale){
    
    /*
        info

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
    //double temp_profile = pow(1.0 - t_normalized * 0.8, 0.75);       // mais brilhante interno
    double base = 1.0 - t_normalized * 0.8;
    double temp_profile = sqrt(base) * sqrt(sqrt(base));

    return (float)(gaussian * temp_profile);

}


__device__  void temperatureToColor(float t_normalized, float& disk_r, float& disk_g, float& disk_b){
    
    /*
        info

        física: um disco de acreção tem temperatura T(r) ∝ r^{-3/4}. 
        A região interna (r ≈ 3rs) atinge ~10^7 K — emite em raios-X, 
        cor percebida azul-branco. A região externa (r ≈ 12rs) está em 
        ~10^5 K — laranja-vermelho.
        
        isso nos dá:

            T_normalized: 0.0 = região externa (fria), 1.0 = região interna (quente)
            baseado em blackbody aproximado para range 3000K-30000K

                · temperatura quente → azul-branco
                · temperatura fria   → laranja-vermelho
                · ponto médio        → amarelo-laranja (cor do sol ~5800K)
    */

    if(t_normalized > 0.8f){
        
        // muito quente: branco com leve azul
        float t = (t_normalized - 0.8f) / 0.2f;

        disk_r = 1.0f;
        disk_g = 1.0f;
        disk_b = 0.8f + 0.2f * t;
    
    } else if(t_normalized > 0.5f){
        
        // quente: amarelo-branco
        float t = (t_normalized - 0.5f) / 0.3f;
        
        disk_r = 1.0f;
        disk_g = 0.7f + 0.3f * t;
        disk_b = 0.1f + 0.7f * t;
    
    } else if(t_normalized > 0.2f){

        // médio: laranja-amarelo
        float t = (t_normalized - 0.2f) / 0.3f;

        disk_r = 1.0f;
        disk_g = 0.3f  + 0.4f * t;
        disk_b = 0.0f + 0.1f * t;

    } else {
        
        // frio: vermelho-laranja escuro
        float t = t_normalized / 0.2f;
        
        disk_r = 0.6f + 0.4f * t;
        disk_g = 0.05f + 0.25f * t;
        disk_b = 0.0f;
    }

}
//@}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// raytraceKernel — kernel principal, um thread por pixel
//@{


__device__  void pixelProcess(  int x, 
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
                            ){

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // posição da câmera em cartesianos
    //@{

    /*
        alocando u,v, as coordenadas da tela.

            → relaciona x,y à aspect_ratio e u,v.
            → fov_y controla quantos graus temos visão. 
    */
 

    float aspect     = float(WIDTH) / float(HEIGHT);
    float tanHalfFov = tan(fov_y * 0.5f * (float)PI / 180.0f);
    float u          = (2.0f * (x + 0.5f) / WIDTH  - 1.0f) * aspect * tanHalfFov;
    float v          = (1.0f - 2.0f * (y + 0.5f) / HEIGHT) * tanHalfFov;
    
    /*
        determinando a direção do raio:
            
            ◦ cálculo de onde apontam na tela: combinação linear,
            ◦ se relacionam à [fwd, right, up]:
                → fwd: centro da imagem (profundidade),
                → right: deslocamento horizontal (u),
                → up: deslocamento vertical (v).
            ◦ normalização,
            ◦ usa-se double para prezar pela precisão maior que o float.
    */

    // direção do raio: u*right + v*up + forward, normalizada
    double dx = u * right.x + v * up.x + fwd.x;
    double dy = u * right.y + v * up.y + fwd.y;
    double dz = u * right.z + v * up.z + fwd.z;
    double dlen = sqrt(dx*dx + dy*dy + dz*dz);
    dx /= dlen; dy /= dlen; dz /= dlen;


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // transformando para coordenadas esféricas e inicializando Rays

    /*
        cartesianas → esféricas:

            ◦ por mais incrível que pareça, pelo fato de que Schwarzschild é
              modelado em esféricas, as equações são mais fáceis do que em cartesianas.
            
            ◦ relembrando:
                → r     = distância radial ao centro do buraco negro
                → theta = ângulo polar (0 = polo norte, π/2 = equador, π = polo sul)
                → phi   = ângulo azimutal (rotação em torno do eixo z)
    */

    double ox = pos.x;
    double oy = pos.y;
    double oz = pos.z;

    double r0     = sqrt(ox*ox + oy*oy + oz*oz);
    double theta0 = acos(fmax(-1.0, fmin(1.0, oz / r0)));
    double phi0   = atan2(oy, ox);
        
    if (theta0 > PI - 1e-6) theta0 = PI - 1e-6;

    /* 
        • coordenadas cartesianas → esféricas via Jacobiano da transformação esférica → cartesiana
            → x = r * sin(θ) * cos(φ)
            → y = r * sin(θ) * sin(φ)
            → z = r * cos(θ)

            ◦   Derivando cada componente em relação a (r, θ, φ) e invertendo, obtemos
                como (dx, dy, dz) se traduz em (dr, dθ, dφ).
            
            ◦ st_safe protege dphi contra divisão por zero quando θ = 0 ou π (polos).
    */

    double st = sin(theta0), ct = cos(theta0), sp = sin(phi0), cp = cos(phi0);
    double st_safe = (fabs(st) < 1e-12) ? 1e-12 : st;
    
    
    /*
        E = energia por unidade de massa (conservada porque a métrica não depende de t)
        L = momento angular por unidade de massa (conservada porque não depende de φ)
        ↓
        são ctes, usadas a cada step do RK4 para calcular dt/dλ = E/f sem precisar rastrear o tempo coordenado t
        

        f0 = fator de Schwarzschild na posição inicial da câmera.
        vmag é a magnitude da velocidade 3D no espaço de Schwarzschild.
    */

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
        

    //@}


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // parâmetros importantes para a simulação.
    //@{
    /*
        info

        step        —   tamanho de cada passo de integração em metros (rs * N). Menor = mais preciso, 
                        bordas do disco mais suaves, photon ring mais visível, render mais lento. 
                        Maior = mais rápido, mais aliasing, raios podem pular o horizonte. O passo adaptativo 
                        já reduz automaticamente perto do horizonte.

        MAX_STEPS       —   quantas iterações cada raio pode fazer. Determina se raios que orbitam múltiplas vezes 
                            (photon ring) conseguem completar. Mais alto = photon ring aparece, render mais lento. 
                            O tempo de render escala quase linearmente com MAX_STEPS para pixels que atingem o limite.

        disk_r1         —   borda interna do disco em unidades de RS. Fisicamente deve ser ≥ 3 RS (ISCO — última órbita estável). 
                            Menor que isso é ficção. Aumentar esconde a região mais brilhante próxima ao horizonte.

        disk_r2         —   borda externa do disco. Controla o tamanho visual do disco. Maior = disco ocupa mais da tela. Se dist < disk_r2 
                            a câmera está dentro do disco — resultado é tela laranja.

        escape_radius   —   distância que o kernel considera "infinito". Deve ser maior que dist. Se muito pequeno, raios que deveriam chegar ao starmap são cortados antes. 
                            RS * 200 é seguro para qualquer dist até RS * 100.

        as variáveis abaixo são cruciais para o funcionamento do buraco negro.
        elas estão descritas em sequência de aparição no loop.
    */
    
    const double step = rs * STEP_FACTOR;
    const double escape_radius = rs * ESCAPE_FACTOR;

    const double disk_r1 = rs * 3.0;
    const double disk_r2 = rs * 12.0;

    float accum_r = 0.0f;       // canal vermelho acumulado
    float accum_g = 0.0f;       // canal verde acumulado
    float accum_b = 0.0f;       // canal azul acumulado
    float accum_alpha = 0.0f;   // opacidade acumulada [0,1]
                                // quando accum_alpha ≥ 1.0 → disco completamente opaco → break
    

    double y_prev = s.r * cos(s.theta);
    double y_next = y_prev;

    //@}


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // loop em si
    //@{

    
    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // escape por parâmetro de impacto:
    //@{
    /*
        
        •   uma forma de Early Exit que analisa a relação entre o impacto da energia
            de um raio para determinar se ele deve ser integrado. A lógica é: se um raio
            tiver energia e momento angular, quando relacionados por 'b', maior que um threshold
            predeterminado como LIMITE que um raio integrado tem, ele deve ser cortado.


        IMPACT_CUTOFF:  [2,4]: vai de agressivo para moderado, cortando efeitos, sombra mais sólida.
                        [4,6]: conservador, preserva Photon Ring e o disco é renderizado normalmente.
                        [6,8]: ainda mais conservador.
                        [10+]: o efeito do algoritmo some, quase tudo integra.
    
    */
    {
        double b = (s.E > 1e-10) ? fabs(s.L / s.E) : 1e30;
        const double b_crit = 2.598 * rs;  // 3√3/2 * rs

        if(b > b_crit * IMPACT_CUTOFF && s.r > rs * 3.0 && s.dr > 0.0){

            // raio escapa com certeza — amostra starmap direto
            float u_tex = (float)(s.phi / (2.0 * PI)) + 0.5f;
            float v_tex = (float)(s.theta / PI);
            float4 color = tex2D<float4>(starmap, u_tex, v_tex);

            R = (unsigned char)(fminf(color.x * 255.0f, 255.0f));
            G = (unsigned char)(fminf(color.y * 255.0f, 255.0f));
            B = (unsigned char)(fminf(color.z * 255.0f, 255.0f));

            result = RayResult::ESCAPE;
            
            return;
        }
    }
    //@}


    for(int i = 0; i < MAX_STEPS; i++){
    
            
    // no início do loop, só para o pixel central
            if(x == WIDTH/2 && y == HEIGHT/2 && i < 3){
                double xc = s.r * sin(s.theta) * cos(s.phi);
                double yc = s.r * sin(s.theta) * sin(s.phi);
                double zc = s.r * cos(s.theta);
                double rc = sqrt(xc*xc + yc*yc);
                float emiss = diskEmissivity(rc, zc, disk_r1, disk_r2, DISK_HEIGHT_SCALE);
                printf("\n\n\n\n\n\nstep %d: r/rs=%.3f  rc/rs=%.3f  zc/rs=%.3f  emiss=%.6f\n",
                       i, s.r/rs, rc/rs, zc/rs, emiss);
            }




        // ─────────────────────────────────────────────────────────────────────────────────────────────────
        // horizonte pré-step
        if(s.r <= rs || s.r <= 0.0 || s.r != s.r){
            R = G = B = 0;
            
            result = RayResult::HORIZON;
            break;
        }
    

        
        // ─────────────────────────────────────────────────────────────────────────────────────────────────
        // passo adaptativo 
        //@{
        /*
            •   perto do horizonte (r < 5*limits), a curvatura do espaço-tempo cresce rapidamente.
                um step fixo grande causaria que o raio "pulasse" o horizonte — de r=1.05*rs
                para r=0.5*rs num único passo, sem o teste de captura disparar.
            
                o adaptive_step reduz linearmente com r: a 1*rs o passo é 1/5 do base_step.
                o mínimo de adaptive_clamp * step evita que o loop fique infinitamente lento
                para raios que raspam o horizonte.
        */

        double adaptive_step = step;

        if(s.r < ADAPTIVE_FACTOR * rs){
            adaptive_step = step * (s.r / (rs * ADAPTIVE_FACTOR));// escala linear: a 1rs → step/5

            if(adaptive_step < step * ADAPTIVE_CLAMP) 
                adaptive_step = step * ADAPTIVE_CLAMP;  // mínimo
        }
    
        //@}
        
            
      
        // ─────────────────────────────────────────────────────────────────────────────────────────────────
        // integrador rk4
        rk4Step(s, adaptive_step, rs);
                
        
        // ─────────────────────────────────────────────────────────────────────────────────────────────────
        // escape pré-step 
        //@{   
        /*
            quando o raio escapa (r > escape_radius), o ângulo final (θ, φ) representa
            a direção de onde a luz vem — que é a direção em que a câmera "vê" esse pixel.

            converte (theta, phi) do ponto de fuga para UV do mapa equiretangular
                → theta: [0, π]    → V: [0, 1]   (polo norte = 0, polo sul = 1)
                → phi:   [-π, π]   → U: [0, 1]   (wrapa em 360°)
         
            O lensing gravitacional dobra os raios, então estrelas que "deveriam" estar
            numa direção aparecem deslocadas — é exatamente o efeito visual de lensing.
        */
        
        //if(s.r > escape_radius || (s.dr > 0.0 && s.r > rs * 10.0 && i > 10)){
        if(s.r > escape_radius){
                

            if(s.phi >  PI) s.phi -= 2.0 * PI;
            if(s.phi < -PI) s.phi += 2.0 * PI;
            
            float u_tex = (float)(s.phi / (2.0 * PI)) + 0.5f;   // [-π,π] → [0,1]
            float v_tex = (float)(s.theta / PI);                  // [0,π]  → [0,1]
            
            float4 color = tex2D<float4>(starmap, u_tex, v_tex);

            R = (unsigned char)(fminf(color.x * 255.0f, 255.0f));
            G = (unsigned char)(fminf(color.y * 255.0f, 255.0f));
            B = (unsigned char)(fminf(color.z * 255.0f, 255.0f));
             
            result = RayResult::ESCAPE;

            break;
        }
        //@}
             

        // ─────────────────────────────────────────────────────────────────────────────────────────────────
        // horizonte pós-step
        if(s.r <= rs || s.r <= 0.0 || s.r != s.r){
            R = G = B = 0;
                
            result = RayResult::HORIZON;
                
            break;
        }
        y_next = s.r * cos(s.theta);
     

        // ─────────────────────────────────────────────────────────────────────────────────────────────────
        // early exit por MAX_STEPS parcial 
        //@{
        /* 
            • se o raio já deu muitos steps, está longe do BH e se afastando → escapa
                
            
            ◦ MAX_STEPS_DIV:    [2,4]:  Agressivo, corta a partir de metade à um quarto das iterações.
                                        Talvez corte o Photon Ring.
                                [4,6]:  Moderado.
                                [6,8]:  Conservador, preserva Photon Ring.
                                [10+]:  O algoritmo perde efeito.
        */

        if(i > MAX_STEPS / MAX_STEPS_DIV && s.r > disk_r2 * 1.5 && s.dr > 0.0){
                
            float u_tex = (float)(s.phi / (2.0 * PI)) + 0.5f;
            float v_tex = (float)(s.theta / PI);
            float4 color = tex2D<float4>(starmap, u_tex, v_tex);

            R = (unsigned char)(fminf(color.x * 255.0f, 255.0f));
            G = (unsigned char)(fminf(color.y * 255.0f, 255.0f));
            B = (unsigned char)(fminf(color.z * 255.0f, 255.0f));

            result = RayResult::ESCAPE;

            break;
        }

        //@}
    

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // case: disco
        //@{
       
        // detecção por emissividade de disco volumétrico
        double x_cartesiano = s.r * sin(s.theta) * cos(s.phi);
        double y_cartesiano = s.r * sin(s.theta) * sin(s.phi);
        double z_cartesiano = s.r * cos(s.theta);
        
        double r_current = sqrt(x_cartesiano*x_cartesiano +  y_cartesiano*y_cartesiano);
        float emissividade = diskEmissivity(r_current, z_cartesiano, disk_r1, disk_r2, DISK_HEIGHT_SCALE);
           

        if(emissividade > EMISSIVITY_RATE){
            

            /*
                Aqui usamos a composição de 3 efeitos físicos:

                    → Perlin noise:     modula o brilho local — gás turbulento, não uniforme
                    → Doppler shift:    assimetria orbital — lado se aproximando brilha mais
                    → Redshift grav:    raios que saem de r pequeno perdem energia

             
                • O beaming concentra a emissão na direção do movimento:
                    
                    lado se aproximando → feixe comprimido → mais brilhante
                    lado se afastando   → feixe dilatado   → mais escuro
                

                • Separamos os dois para controle visual independente:
                    
                    freq_effect  = doppler * rs_freq   → muda a cor
                    beam_effect  = doppler^3            → muda o brilho


                O expoente 3 vem de:
                    → 1 fator de energia do fóton
                    → fatores de aberração relativística (compressão do feixe)
                    
                    ◦ Fisicamente seria D^4 mas D^3 dá resultado visualmente mais equilibrado.

            */
            
            float doppler = dopplerShift(r_current, s.phi, pos, rs);
            float rs_freq = redShift(r_current, rs);
            
            float perlin_noise = perlinNoise(perlin, r_current, s.phi, disk_r1, disk_r2);
            float base_brightness = 0.4f + 0.6f * perlin_noise;

            float freq_effect = doppler * rs_freq;
            float beam_effect = powf(fmaxf(doppler, 0.01f), 3.0f);
            
            float temp = (float)((r_current - disk_r1) / (disk_r2 - disk_r1));
            float temp_norm = powf(1.0f - temp, 0.75f);
           
            float disk_r, disk_g, disk_b;
            temperatureToColor(temp_norm, disk_r, disk_g, disk_b);
       

            // modula a cor base pelo shift de frequência
            // freq > 1 → empurra para branco/azul (se aproximando)
            // freq < 1 → empurra para vermelho (se afastando
            if(freq_effect > 1.0f){

                float blend = fminf((1.0f - freq_effect) / 1.5f, 1.0f);
                
                disk_g = disk_g + blend * (1.0f - disk_g);
                disk_b = disk_b + blend * (1.0f - disk_b);

            } else {

                float blend = fminf((1.0f - freq_effect) / 0.8f, 1.0f);
                
                disk_r = disk_r + blend * 0.3f;
                disk_g = disk_g + blend * (1.0f - disk_g);
                disk_b = disk_b + blend * (1.0f - disk_b);

            }

            float intensity = emissividade * base_brightness * EMISSION_SCALE * beam_effect;
            
            float step_normalized = (float)(adaptive_step / (disk_r2 - disk_r1));
            float contribution = intensity * step_normalized * DISK_OPACITY;
            float remaining = 1.0f - accum_alpha;

            accum_r     += disk_r * contribution * remaining;
            accum_g     += disk_g * contribution * remaining;
            accum_b     += disk_b * contribution * remaining;

            accum_alpha += contribution * remaining;

            result = RayResult::DISK;
            
            // disco completamente opaco → para de integrar
            if (accum_alpha >= 1.0f){
                accum_alpha = 1.0f;
                
                result = RayResult::DISK;
                break;
            }
            
            //@}

        }

        y_prev = y_next;
    }
    //@}
    
    
    //  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // fallback:
    //@{ 
    /*
        Após o loop, R=G=B=0 significa que nenhum dos casos anteriores disparou.
        Dois subcasos:
            → s.r < rs * closeness  → raio ficou preso perto do horizonte → preto (capturado)
            → s.r ≥ rs * closeness  → raio ficou orbitando longe → amostra starmap na direção atual

            • A escrita final:
                ◦ int idx = (y * WIDTH + x) * 3;
                ◦ pixels[idx+0] = R;   // byte 0 do pixel (x,y) = canal R
                ◦ pixels[idx+1] = G;   // byte 1 = canal G
                ◦ pixels[idx+2] = B;   // byte 2 = canal B

            • engine.cpp faz upload disso direto para a textura OpenGL.
    */
    
    if (result == RayResult::NONE){ 
        result = RayResult::FALLBACK;
    }
       
    if(result == RayResult::HORIZON){   
        R = G = B = 0;

    } else if (result == RayResult::ESCAPE || result == RayResult::FALLBACK){
            
        if(result == RayResult::FALLBACK && s.r < rs * CLOSENESS){
            R = G = B = 0;
            
        } else {
                
            float u_tex = (float)(s.phi / (2.0 * PI)) + 0.5f;
            float v_tex = (float)(s.theta / PI);

            float4 color = tex2D<float4>(starmap, u_tex, v_tex);

            R = (unsigned char)(fminf(color.x * 255.0f, 255.0f));
            G = (unsigned char)(fminf(color.y * 255.0f, 255.0f));
            B = (unsigned char)(fminf(color.z * 255.0f, 255.0f));
                
        }
    }

        if(x == WIDTH/2 && y == HEIGHT/2){
            printf("\n\n\n\n\n\n\n\n\nfinal: result=%d  accum_alpha=%.6f  R=%d G=%d B=%d\n",
               (int)result, accum_alpha, R, G, B);
        }



    //if(result == RayResult::DISK
    if(accum_alpha > 0.0001f){

        float bg_weight = 1.0f - fminf(accum_alpha, 1.0f);

        R = (unsigned char)(fminf((accum_r + R/255.0f * bg_weight) * 255.0f, 255.0f));
        G = (unsigned char)(fminf((accum_g + G/255.0f * bg_weight) * 255.0f, 255.0f));
        B = (unsigned char)(fminf((accum_b + B/255.0f * bg_weight) * 255.0f, 255.0f));
        
    }
    //@} 


}
//@}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


__global__ void raytraceKernelPNG(  unsigned char* pixels,
                                    int WIDTH,
                                    int HEIGHT,
                                    double3 pos,
                                    double3 fwd, 
                                    double3 right,
                                    double3 up,
                                    float fov_y,
                                    double rs,  
                                    cudaTextureObject_t starmap,
                                    cudaTextureObject_t perlin){


    RayResult result = RayResult::NONE;
    int x = 0, y = 0;
    unsigned char R = 0, G = 0, B = 0;
    
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;
    

    pixelProcess(   x, y, 
                    R, G, B,
                    WIDTH, HEIGHT, 
                    pos, fwd, right, up,
                    fov_y, rs, 
                    starmap, perlin,
                    result
                 );

    int idx = (y * WIDTH + x) * 3;

    pixels[idx+0] = R;
    pixels[idx+1] = G;
    pixels[idx+2] = B;
        
    SH_RECORD(d_state_counts, result);

}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━




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
                cudaTextureObject_t perlin){
    

    size_t nbytes = WIDTH * HEIGHT * 3;
    dim3 blockSize(16, 16);
    dim3 numBlocks((WIDTH + 15)/16, (HEIGHT + 15)/16);
        
    
    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    //@{
    /*
    */
    static bool is_sim = false;
    static double ms_per_frame = -1.0;
    static auto t_last = Clock::now();

    if(ms_per_frame > 0.0){
        auto now = Clock::now();

        double delta = std::chrono::duration<double, std::milli>(now - t_last).count();
        // suaviza com média móvel exponencial — evita saltos por GC ou stall de driver
        ms_per_frame = ms_per_frame * 0.9 + delta * 0.1;
    }
    t_last = Clock::now();
    //@}
    // ─────────────────────────────────────────────────────────────────────────────────────────────────
   

    printf("\n");
    unsigned char* d_pixels;
    cudaMalloc(&d_pixels, nbytes); ck("d_pixel Malloc");
    cudaMemset(d_pixels, 0, nbytes); ck("d_pixel Memset");


    unsigned int* d_counts_ptr = nullptr;
    cudaGetSymbolAddress((void**)&d_counts_ptr, d_state_counts);
    cudaMemset(d_counts_ptr, 0, SH_NUM_STATES * sizeof(unsigned int));
    
    
    cudaStream_t kernel_stream;
    cudaStreamCreate(&kernel_stream); ck("Stream Creation");

    raytraceKernelPNG<<<numBlocks, blockSize, 0, kernel_stream>>>(   
            d_pixels, 
            WIDTH, 
            HEIGHT, 
            pos, fwd, right, up, 
            fov_y, 
            rs, 
            starmap, 
            perlin);


    //ck("Post-Kernel");
    

    cudaStreamSynchronize(kernel_stream); 
    ck("StreamSync");
    
    cudaMemcpy(pixels, d_pixels, nbytes, cudaMemcpyDeviceToHost);
    ck("cudaMemcpy");
    


    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    //@{
    /*
    printf("ms_per_frame [2]: %f", ms_per_frame);
    */
    if (ms_per_frame < 0.0) {
        auto now = Clock::now();
        ms_per_frame = std::chrono::duration<double, std::milli>(now - t_last).count();
    }

    if (ms_per_frame > 0.0 && is_sim){
        updateCorrectionFactor(ms_per_frame);
    }
    is_sim = true;
    //@}
    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    

    cudaStreamDestroy(kernel_stream); ck("Destroy Stream");  
    cudaFree(d_pixels); ck("Free d_pixels");
    printf("\n");

}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
