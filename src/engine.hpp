#pragma once
#ifndef engine_h

#include "../cuda/headers/geodesic.cuh"


#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "constants.hpp"
#include <string>

using namespace std;
using namespace glm;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/* 
    gerencia janela GLFW, contexto do OpenGL, e o loop de renderização.
    Interface com o lado CUDA via raytraceCUDA().
    
    raytraceCUDA() → pixels[] → glTexSubImage2D → fullscreen quad → janela

*/

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// configuração da janela e câmera


struct SimConfig{

    string title = "";

    int WIDTH = 0;
    int HEIGHT = 0;
 
    float fov_y = 0.f;

    // posição e orientação inicial da câmera (coordenadas cartesianas)
    vec3 pos = vec3(0.0f, 0.0f, 50.0f);
    vec3 target = vec3(0.0f);
    vec3 world_up = vec3(0.0f, 0.0f, 1.0f);

    bool use_direct_vectors = false;
    vec3 fwd = vec3(0.0f);
    vec3 right = vec3(0.0f);;
    vec3 up = vec3(0.0f);;


    const double graus = 0;
    const double elevation_angle = 0;
    
    float elevation = glm::radians((float)graus);
    float azimuth  = glm::radians((float)elevation_angle); 
       
    float cam_dist = RS * BH::factor;   // distância ao target
 
    bool dragging = false;
    double last_mouse_x = 0.0;
    double last_mouse_y = 0.0;
    bool will_rerender = true;    // true → kernel precisa ser relançado
    
    unsigned long long starmap = 0;
    unsigned long long perlin = 0;


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // função que retorna os valores atuais a partir dos ângulos


    void getVectors(vec3& out_pos, vec3& out_fwd, vec3& out_right, vec3& out_up){
        
        if(use_direct_vectors){
            out_pos   = pos;
            out_fwd   = fwd;
            out_right = right;
            out_up    = up;

            return;
        }

        out_pos = target + vec3(
            cam_dist * BH::X_COEF * cos(azimuth),
            cam_dist * BH::Y_COEF * sin(elevation),
            cam_dist * BH::Z_COEF * cos(elevation)
        );
      
        out_fwd   = normalize(target - out_pos);
        out_right = normalize(cross(out_fwd, world_up));
        out_up    = normalize(cross(out_right, out_fwd));
    } 

};


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Ponto de entrada do engine. Bloqueia até a janela ser fechada.


void engineRun(SimConfig& cfg);


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#endif
