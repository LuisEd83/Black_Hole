#pragma once
#ifndef engine_h


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/* 
    gerencia janela GLFW, contexto do OpenGL, e o loop de renderização.
    Interface com o lado CUDA via raytraceCUDA().
    
    raytraceCUDA() → pixels[] → glTexSubImage2D → fullscreen quad → janela

*/


#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
 
#include <string>

using namespace std;
using namespace glm;


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// configuração da janela e câmera


struct EngineConfig{

    int WIDTH       = 800;
    int HEIGHT      = 600;
    string title    = "BlackHole Sim";
    float fov_y     = 60.0f;    // campo de visão vertical, graus
 
    // posição e orientação inicial da câmera (coordenadas cartesianas)
    vec3  cam_pos     = vec3(0.0f, 0.0f, 50.0f);
    vec3  cam_target  = vec3(0.0f, 0.0f,  0.0f);
    vec3  cam_up      = vec3(0.0f, 1.0f,  0.0f);
};


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// struct da camera


struct CameraState{
    
    vec3 target = vec3(0.f);                // ponto no qual a câmera dista 'orbital_radius'
    vec3 world_up = vec3(0.f, 1.f, 0.f);    
 
    float azimuth_angle   = 0.f;    // radianos — rotação horizontal
    float elevation_angle = 0.f;    // radianos — rotação vertical
    float orbital_radius  = 50.f;   // distância ao target
 
    bool dragging        = false;
    double last_mouse_x  = 0.0;
    double last_mouse_y  = 0.0;
    bool problem         = true;    // true → kernel precisa ser relançado
    
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // função que retorna os valores atuais a partir dos ângulos


    void getVectors(vec3 out_pos, vec3 out_fwd, vec3 out_right,vec3 out_up){


        float cos_az = cos(azimuth_angle), sen_az = sin(azimuth_angle);
        float cos_el = cos(elevation_angle), sen_el = sin(elevation_angle);
        
        out_pos = normalize(target - out_pos);
        out_pos = normalize(cross(out_fwd, out_pos));
        out_pos = normalize(cross(out_right, out_pos));
    } 

};


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Ponto de entrada do engine. Bloqueia até a janela ser fechada.

void engineRun(const EngineConfig& cfg);


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#endif
