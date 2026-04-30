#pragma once
#ifndef engine_h

#include "../cuda/geodesic.cuh"
#include "constants.hpp"

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/* 
    gerencia janela GLFW, contexto do OpenGL, e o loop de renderização.
    Interface com o lado CUDA via raytraceCUDA().
    
    raytraceCUDA() → pixels[] → glTexSubImage2D → fullscreen quad → janela

*/


#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "constants.hpp"
#include <string>

using namespace std;
using namespace glm;


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// configuração da janela e câmera


struct EngineConfig{

    int WIDTH = 0;
    int HEIGHT = 0;
    string title = "BlackHole Sim";
    float fov_y = 60.0f;    // campo de visão vertical, graus
 
    // posição e orientação inicial da câmera (coordenadas cartesianas)
    vec3 cam_pos    = vec3(0.0f, 0.0f, 0.0f);
    vec3 cam_target = vec3(0.0f, 0.0f, 0.0f);
    vec3 cam_up     = vec3(0.0f, 0.0f, 1.0f);

    bool use_direct_vectors = false;
    vec3 cam_fwd = vec3(0.0f);
    vec3 cam_right = vec3(0.0f);
    vec3 cam_up_vec = vec3(0.0f);

};


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// struct da camera


struct CameraState{
    
    vec3 target = vec3(0.f, 0.f, 0.f);                // ponto no qual a câmera dista 'orbital_radius'
    vec3 world_up = vec3(0.f, 0.f, 1.f);    
        
    const double graus = 10.0f;
    const double elevation_angle = 5.0f;

    float elevation = glm::radians((float)graus);
    float azimuth  = glm::radians((float)elevation_angle); 
   
    float orbital_radius = RS * BH::factor;   // distância ao target
 
    bool dragging = false;
    double last_mouse_x = 0.0;
    double last_mouse_y = 0.0;
    bool will_rerender = true;    // true → kernel precisa ser relançado
    
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // função que retorna os valores atuais a partir dos ângulos


    void getVectors(vec3& out_pos, vec3& out_fwd, vec3& out_right, vec3& out_up){

        out_pos = glm::vec3(
            float(orbital_radius) * BH::X_COEF,
            float(orbital_radius)  * BH::Y_COEF,
            float(orbital_radius)  * BH::Z_COEF
        );
      
        out_fwd   = normalize(target - out_pos);
        out_right = normalize(cross(out_fwd, world_up));
        out_up    = normalize(cross(out_right, out_fwd));
    } 

};


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Ponto de entrada do engine. Bloqueia até a janela ser fechada.

void engineRun(const EngineConfig& cfg);


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#endif
