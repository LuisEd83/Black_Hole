#pragma once

#include <string>
#include <cuda_runtime.h>
#include "../headers/lodepng.h"   

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



inline auto ck = [](const char* tag){
    auto err = cudaGetLastError();    
    fprintf(stderr, "[CUDA] %s: %s\n", tag, cudaGetErrorString(err));
};



inline void test_png(vector<unsigned char>png_pixels, int WIDTH, int HEIGHT){

    unsigned error = lodepng::encode("test.png", png_pixels, WIDTH, HEIGHT);
        if (error)
             fprintf(stderr, "\n◦ Erro de imagem: %s", lodepng_error_text(error));
        else
            fprintf(stderr, "\n◦ Imagem salva em 'test.png' (%llu MB)\n", png_pixels.size()/1024/1024);
    }
            

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
    // de natureza do controle:


    const string res = "Minimal";
    const bool is_sim = true;
    const bool is_gl = true;
    const bool is_persis = true;

    
    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // de natureza da simulação:


    constexpr double CAMERA_FACTOR = 10.0;
    constexpr double FOV_Y = 80.0;

    constexpr double X_COEF = 1.0;
    constexpr double Y_COEF = 1.2;
    constexpr double Z_COEF = 0.16;
        
    constexpr int MAX_STEPS = 10000;
    constexpr double STEP_FACTOR = 0.5;
    
    constexpr double ESCAPE_FACTOR = 200.0;
 

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // de natureza avançada de simulação:


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
