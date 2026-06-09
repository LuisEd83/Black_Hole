#pragma once

#include <string>
#include <cuda_runtime.h>
#include <cstring>

using namespace std;


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//@{

    /*
        Resoluções que estão sendo utilizadas:
            
            0800 x 600  → [Minimal]
            1280 x 720  → [HD]
            1920 x 1080 → [FHD]
            2560 x 1440 → [QHD]
            3840 x 2160 → [UHD]
            4096 x 2048 → [4K]
    */

// lambdas C++: função anônima atribuída a variável local.

inline auto ck = [](const char* tag){
    auto err = cudaGetLastError();    
    fprintf(stderr, "[CUDA] %s: %s\n", tag, cudaGetErrorString(err));
};


inline auto checkpoint = [](const char* tag, const char* description){
    
    if(strcmp(tag, "") == 0){
        fprintf(stderr, "%s", description);
    
    } else {
        fprintf(stderr, "[%s]: %s", tag, description);

    }
};


//@}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//@{


namespace BH {
    

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // de natureza da Física:


    constexpr double PI = 3.14159265358979323846;

    constexpr double c       = 299792458.0; // velocidade da luz (m/s)
    constexpr double G       = 6.67430e-11; // constante de gravitação universal, m³/(kg·s²)
    constexpr double BH_MASS = 8.54e36;     // massa de Sagittarius A*, kg

    // Schwarzschild radius: r_s = 2GM/c²
    // raio do horizonte de eventos — abaixo disso, a luz não escapa
    static const double RS = 2.0 * G * BH_MASS / (c * c);

    enum class RayResult { 
        NONE,
        HORIZON, 
        ESCAPE, 
        DISK, 
        FALLBACK 
    };


    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // de natureza do controle, boosters da simulação OpenGL:


    const string res = "Minimal";
    const bool is_sim = true;
    const bool is_gl = true;
    const bool is_persis = true;
    const bool TAA = 0;
    const bool CHECKERBOARD = 0;
    const bool UPSCALE = 0;
    
    
    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // structs de parâmetros:


    struct RenderParams{
    
        int WIDTH;
        int HEIGHT;
        double3 pos;
        double3 fwd;
        double3 right;
        double3 up;
        float fov_y;
        cudaSurfaceObject_t starmap;
        cudaSurfaceObject_t perlin;

    };


    struct PipelineParams{

        cudaSurfaceObject_t surface_prev;
        int* d_counter;
        int frame_parity;

    };   
    
    
    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // de natureza da simulação:

    /*
        CAMERA_FACTOR:  Distância da câmera ao buraco negro em unidades de rs.
                        [5.0, 20.0]. +: câmera mais longe, menos lensing extremo.

        FOV_Y:      Campo de visão vertical em graus.
                    [20.0, 90.0]. +: mais área visível, menos zoom no anel.

        X_COEF:     Escala do eixo x do ray direction no espaço da câmera.
                    [0.5, 2.0]. +: estica horizontalmente, comprime verticamente.

        Y_COEF:     Escala do eixo y do ray direction no espaço da câmera.
                    [0.5, 2.0]. +: estica verticalmente. Razão Y/X controla aspect ratio efetivo.

        Z_COEF:     Profundidade do plano de projeção da câmera.
                    [0.05, 1.0]. +: raios mais paralelos entre si (menos perspectiva).

        MAX_STEPS:      Número máximo de passos RK4 por raio.
                        [1000, 50000]. +: raios orbitais completam mais voltas antes de corte.
                        Reduzir degrada o photon ring primeiro.

        STEP_FACTOR:    Tamanho do passo como fração de rs: step = rs * STEP_FACTOR.
                        [0.05, 2.0]. +: passo maior, integração mais rápida, menos precisa perto do horizonte.
                        Reduzir abaixo de 0.1 aumenta tempo sem ganho visível fora de rs * 1.5.

        ESCAPE_FACTOR:  Raio de escape em unidades de rs: escape_radius = rs * ESCAPE_FACTOR.
                        [50.0, 500.0]. +: raios percorrem mais antes de serem declarados livres,
                        capturando lensing de baixo ângulo mais distante.
    */


    constexpr double CAMERA_FACTOR = 12.0;
    constexpr double FOV_Y = 60.0;

    constexpr double X_COEF = 1.0;
    constexpr double Y_COEF = 1.1;
    constexpr double Z_COEF = 0.2;
    
    constexpr float x_multiplier = 2.0f;
    constexpr float y_multiplier = 0.0f;
    constexpr float z_multiplier = 0.0f;
    
    constexpr float x_bias = 2.5f;
    constexpr float y_bias = 0.0f;
    constexpr float z_bias = 0.0f;

    
    constexpr int MAX_STEPS = 10000;
    constexpr double STEP_FACTOR = 0.1;
    
    constexpr double ESCAPE_FACTOR = 200.0;
            

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // de natureza avançada de simulação:

    /*
        IMPACT_CUTOFF:  Multiplicador de b_crit para corte antecipado de raios que escapam sem orbitar.
                        [3.0, 15.0]. +: corta mais raios, menos lensing fraco em raios rasantes.

        MAX_STEPS_DIV:  Divisor de MAX_STEPS para early exit: ativa após MAX_STEPS / MAX_STEPS_DIV steps.
                        [1.5, 5.0]. +: early exit mais cedo, photon ring pode ser cortado prematuramente.

        DISK_HEIGHT_SCALE:  Espessura do disco: half-height = r * DISK_HEIGHT_SCALE.
                            [0.01, 0.2]. +: disco mais espesso, anel mais largo e brilhante.

        DISK_OPACITY:   Coeficiente de extinção do volume do disco por step.
                        [0.5, 10.0]. +: disco mais opaco, fundo desaparece atrás do anel.

        EMISSION_SCALE: Fator de escala da emissividade local sobre a cor final.
                        [0.5, 10.0]. +: disco mais brilhante, pode saturar para branco.

        ADAPTIVE_CLAMP: Passo mínimo do step adaptativo como fração do step base.
                        [0.001, 0.1]. +: passo mínimo maior, menos preciso perto do horizonte.

        CLOSENESS:      Raio de fallback em unidades de rs para raios sem resultado definido.
                        [2.0, 10.0]. +: região maior, mais raios capturados antes do horizonte.

        ADAPTIVE_FACTOR:    Raio em unidades de rs abaixo do qual o step adaptativo é ativado.
                            [2.0, 10.0]. +: região adaptativa maior, mais steps finos, maior custo.

        EMISSIVITY_RATE:    Limiar mínimo de emissividade para ativar acumulação de volume e step adaptativo.
                            [0.001, 0.1]. +: menos pixels entram no caminho caro, bordas do disco mais abruptas.
    */


    constexpr double IMPACT_CUTOFF  = 7.5;
    constexpr double MAX_STEPS_DIV  = 2.5;
    
    constexpr double DISK_HEIGHT_SCALE = 0.05;
    constexpr double DISK_OPACITY = 3.0;
    constexpr double EMISSION_SCALE = 2.5;

    constexpr double ADAPTIVE_CLAMP = 0.01;
    constexpr double CLOSENESS = 5.0;
    
    constexpr double ADAPTIVE_FACTOR = 5.0;
    constexpr double EMISSIVITY_RATE = 0.01;


}
//@}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
