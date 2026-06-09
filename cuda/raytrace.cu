#include "../headers/geodesic.cuh"
#include "internals/effects.cuh"
#include "internals/geod.cuh"


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// kernels de raytrace


__device__ void pixelProcess(   int x, 
                                int y,
                                unsigned char &R, 
                                unsigned char &G, 
                                unsigned char &B,
                                const RenderParams& rnd,
                                const PipelineParams& ppl,
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
    
    int WIDTH = rnd.WIDTH;
    int HEIGHT = rnd.HEIGHT;
    double3 pos  = rnd.pos;
    double3 right  = rnd.right;
    double3 fwd  = rnd.fwd;
    double3 up  = rnd.up;
    float fov_y = rnd.fov_y;
    cudaSurfaceObject_t starmap = rnd.starmap;
    cudaSurfaceObject_t perlin = rnd.perlin;
    double rs = RS;


    float aspect     = float(WIDTH) / float(HEIGHT);
    float tanHalfFov = tanf(fov_y * 0.5f * (float)PI / 180.0f);
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
    double dlen = sqrtf(dx*dx + dy*dy + dz*dz);
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

    double r0     = sqrtf(ox*ox + oy*oy + oz*oz);
    double theta0 = acosf(fmaxf(-1.0, fmin(1.0, oz / r0)));
    double phi0   = atan2f(oy, ox);
        
    if(theta0 > PI - 1e-6) 
        theta0 = PI - 1e-6;

    /* 
        • coordenadas cartesianas → esféricas via Jacobiano da transformação esférica → cartesiana
            → x = r * sin(θ) * cos(φ)
            → y = r * sin(θ) * sin(φ)
            → z = r * cos(θ)

            ◦   Derivando cada componente em relação a (r, θ, φ) e invertendo, obtemos
                como (dx, dy, dz) se traduz em (dr, dθ, dφ).
            
            ◦ st_safe protege dphi contra divisão por zero quando θ = 0 ou π (polos).
    */ 

    double st = sinf(theta0), ct = cosf(theta0), sp = sinf(phi0), cp = cos(phi0);
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
            
    double phi_last_good = s.phi;
    if(!isfinite(s.r) || !isfinite(s.phi) || !isfinite(s.theta)){
        result = RayResult::FALLBACK;
        return;
    }

    /*
        float dx_f = st * cp;
        float dy_f = st * sp;
        float dz_f = ct;
    */
    //@}


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // parâmetros importantes para a simulação.
    //@{
    

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
 

        // testar o tempo disso aqui
        if(s.r > disk_r2 * 1.2 && s.dr > 0.0 && i > 50){
            // escapa
            break;
        }
              

        // ─────────────────────────────────────────────────────────────────────────────────────────────────
        // horizonte pré-step 
        if(s.r <= rs || s.r <= 0.0){
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


        if(!isfinite(s.phi) || !isfinite(s.r)){
            s.phi = phi_last_good;
            result = RayResult::FALLBACK;
            break;
        }

        double adaptive_step = step;
        if(s.r < ADAPTIVE_FACTOR * rs){
            adaptive_step = step * (s.r / (rs * ADAPTIVE_FACTOR));  // escala linear: a 1rs → step/5

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

            converte (theta, phi) do phi=2.7902 theta=2.5569 u=0.9441 v=0.8139 result=4

                ponto de fuga para UV do mapa equiretangular
                → theta: [0, π]    → V: [0, 1]   (polo norte = 0, polo sul = 1)
                → phi:   [-π, π]   → U: [0, 1]   (wrapa em 360°)
         
            O lensing gravitacional dobra os raios, então estrelas que "deveriam" estar
            numa direção aparecem deslocadas — é exatamente o efeito visual de lensing.
        */

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
    

        
        // ─────────────────────────────────────────────────────────────────────────────────────────────────
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

    if(!isfinite(s.phi) || !isfinite(s.r)){
        result = RayResult::FALLBACK;
        s.phi = fmodf((float)phi_last_good, 2.0f * (float)PI);
    }
    if(fabs(s.phi) > 1e6 || fabs(s.r) > escape_radius){
        result = RayResult::FALLBACK;
        s.phi = phi_last_good;
}
    
    // ─────────────────────────────────────────────────────────────────────────────────────────────────
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

            float u_tex = fmodf((float)(s.phi / (2.0 * PI)) + 1000.5f, 1.0f);
            float v_tex = (float)(s.theta / PI);
            float4 color = tex2D<float4>(starmap, u_tex, v_tex);

            R = (unsigned char)(fminf(color.x * 255.0f, 255.0f));
            G = (unsigned char)(fminf(color.y * 255.0f, 255.0f));
            B = (unsigned char)(fminf(color.z * 255.0f, 255.0f));
            
        }
    }
    
    //if(result == RayResult::DISK && accum_alpha > 0.001f) {
    if(accum_alpha > 0.0001f){

        float bg_weight = 1.0f - fminf(accum_alpha, 1.0f);
        float out_r = accum_r + R/255.0f * bg_weight;
        float out_g = accum_g + G/255.0f * bg_weight;
        float out_b = accum_b + B/255.0f * bg_weight;

        // if the blended result is too dark, blend in starmap at ray's final direction
        float luminance = 0.299f*out_r + 0.587f*out_g + 0.114f*out_b;

        if(luminance < 0.01f && result == RayResult::DISK){

            float u_tex = (float)(s.phi / (2.0 * PI)) + 0.5f;
            float v_tex = (float)(s.theta / PI);
            float4 color = tex2D<float4>(starmap, u_tex, v_tex);
            float t = 0.6f; // blend factor, tweak this

            out_r = out_r * (1.0f - t) + color.x * t;
            out_g = out_g * (1.0f - t) + color.y * t;
            out_b = out_b * (1.0f - t) + color.z * t;
        }

        R = (unsigned char)(fminf(out_r * 255.0f, 255.0f));
        G = (unsigned char)(fminf(out_g * 255.0f, 255.0f));
        B = (unsigned char)(fminf(out_b * 255.0f, 255.0f));

    }
    //@} 
    
}


// ─────────────────────────────────────────────────────────────────────────────────────────────────
