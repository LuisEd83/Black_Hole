#pragma once

#include "constants.hpp"
#include "geodesic.cuh"

#include <string>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

using namespace glm;


/* 
    gerencia janela GLFW, contexto do OpenGL, e o loop de renderização.
    Interface com o lado CUDA via raytraceCUDA().

*/


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// configuração da janela e câmera


struct SimConfig{

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // de natureza do OpenGL:

 
    string title = "";

    double last_mouse_x = 0.0f;
    double last_mouse_y = 0.0f;
    bool dragging = false;
    bool will_rerender = true;    // true → kernel precisa ser relançado
    bool use_direct_vectors = false;
    
    
    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // de natureza da simulação:


    int WIDTH = 0.0f;
    int HEIGHT = 0.0f;

    double cam_dist = RS * BH::CAMERA_FACTOR;   // distância ao target
    float fov_y = 0.0f;
    
    double x_c = BH::X_COEF;
    double y_c = BH::Y_COEF;
    double z_c = BH::Z_COEF;


    vec3 pos = vec3(0.0f, 0.0f, 0.0f);
    vec3 target = vec3(0.0f, 0.0f, 0.0f);
    vec3 world_up = vec3(0.0f, 0.0f, 1.0f);

    vec3 fwd = vec3(0.0f);
    vec3 right = vec3(0.0f);
    vec3 up = vec3(0.0f);
      

    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // função que retorna os valores atuais a partir dos ângulos


    void getVectors(vec3& out_pos, vec3& out_fwd, vec3& out_right, vec3& out_up){
        
        if(use_direct_vectors){

            out_pos   = pos;
            out_fwd   = fwd;
            out_right = right;
            out_up    = up;

            return;
        }

        out_pos = target + vec3(cam_dist * x_c,
                                cam_dist * y_c,
                                cam_dist * z_c);
      
        out_fwd   = normalize(target - out_pos);
        out_right = normalize(cross(out_fwd, world_up));
        out_up    = normalize(cross(out_right, out_fwd));
    } 

};


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Ponto de entrada do engine. Bloqueia até a janela ser fechada.

void engineRun(SimConfig& cfg);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
