#include "../headers/geodesic.cuh"


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// funções de efeito:


__device__ float dopplerShift(double phi, 
                              double r_current, 
                              double3 camera_pos, 
                              double rs){
    
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

    if(denominador <= 0.0) 
        return 1.0f;
    
    double beta = sqrtf(rs/denominador);
    double gamma = 1.0 / sqrtf(1.0 - beta * beta);

    double vx = -sin(phi);
    double vy = cos(phi);
    // vz = 0, disco no equador
    
    double dx = camera_pos.x - r_current * cos(phi);
    double dy = camera_pos.y - r_current * sin(phi);
    double dz = camera_pos.z;

    double dlen = sqrtf(dx*dx + dy*dy + dz*dz);
    if(dlen < 1e-10) 
        return 1.0f;

    dx /= dlen; dy /= dlen;
    
    double cos_alpha = vx * dx + vy * dy;

    double doppler = 1.0 / (gamma * (1.0 - beta * cos_alpha));
    
    return (float)doppler;
}


// ─────────────────────────────────────────────────────────────────────────────────────────────────


__device__  float perlinNoise(  cudaTextureObject_t perlin, 
                                double r_current, 
                                double phi,
                                double disk_r1, 
                                double disk_r2){

    if(perlin == 0) 
        return 1.0f;

    float u = (float)((r_current - disk_r1) / (disk_r2 - disk_r1));
    float v = (float)(phi / (2.0 * PI)) + 0.5f;

    float4 noise = tex2D<float4>(perlin, u ,v);
    
    return noise.x;
}


__device__  float redShift(double r_current, 
                           double rs){

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


// ─────────────────────────────────────────────────────────────────────────────────────────────────


__device__  float diskEmissivity(   double r_current,
                                    double z_cartesiano,
                                    double disk_r1, 
                                    double disk_r2,
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


    if(r_current < disk_r1 || r_current > disk_r2) 
        return 0.0f;
    
    double H = r_current * height_scale;
    double gaussian = expf(-(z_cartesiano * z_cartesiano) / (2.0 * H * H));


    // perfil radial de temperatura — disco interno mais quente
    // T ∝ r^(-3/4) é a lei de Stefan-Boltzmann para disco de acreção
    double t_normalized = (r_current - disk_r1) / (disk_r2 - disk_r1);  // [0,1]

    //double temp_profile = pow(1.0 - t_normalized * 0.8, 0.75);       // mais brilhante interno
    
    double base = 1.0 - t_normalized * 0.8;
    double temp_profile = sqrtf(base) * sqrtf(sqrtf(base));

    return (float)(gaussian * temp_profile);
}


// ─────────────────────────────────────────────────────────────────────────────────────────────────


__device__  void temperatureToColor(float t_normalized, 
                                    float& disk_r, 
                                    float& disk_g, 
                                    float& disk_b){
    
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


// ─────────────────────────────────────────────────────────────────────────────────────────────────
