#include "cuda/geodesic.cuh"
#include "src/engine.hpp"
#include "src/constants.hpp"
#include "src/starmap.hpp"
#include "src/perlin.hpp"

#include <glm/glm.hpp>
#include <iostream>


enum class Resolution{
    Minimal,
    HD,
    HDplus,
    FHD,
    QHD,
    UHD,
    _4K,
    Unknown
};

Resolution fromString(const std::string& s){

    if (s == "Minimal") return Resolution::Minimal;
    if (s == "HD")      return Resolution::HD;
    if (s == "HD+")     return Resolution::HDplus;
    if (s == "FHD")     return Resolution::FHD;
    if (s == "QHD")     return Resolution::QHD;
    if (s == "UHD")     return Resolution::UHD;
    if (s == "4K")      return Resolution::_4K;

    return Resolution::Unknown;
}

using namespace BH;

int main() {

    activateSetFlags();

    // ── recursos de suporte (mesmo que o main atual) ──────────────────
    if (!starmapLoad("data/starmap.png")) {
        std::cerr << "Falha ao carregar starmap\n";
        return 1;
    }
    if (!perlinLoad("data/perlin.txt")) {
        std::cerr << "Falha ao carregar perlin\n";
        return 1;
    }
   
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    

    const double cam_dist      = RS * BH::factor;
    const double graus         = 10.0f;
    const double elevation_angle = 5.0f;
    float fov_y                = 60.0f;

    float elevation = glm::radians((float)graus);
    float azimuth   = glm::radians((float)elevation_angle);
        
    glm::vec3 pos = glm::vec3(
        float(cam_dist) * BH::X_COEF,
        float(cam_dist) * BH::Y_COEF,
        float(cam_dist) * BH::Z_COEF
    );

    glm::vec3 target    = glm::vec3(float(-RS * 3.0), float(RS * 1.5), 0.0f);
    glm::vec3 world_up = glm::vec3(0.0f, 0.0f, 1.0f);

    glm::vec3 fwd   = glm::normalize(target - pos);
    glm::vec3 right = glm::normalize(glm::cross(fwd, world_up));
    glm::vec3 up    = glm::normalize(glm::cross(right, fwd));
    



    // ── config da engine ──────────────────────────────────────────────
    EngineConfig cfg;
    cfg.WIDTH  = 800;
    cfg.HEIGHT = 600;

    
    switch(fromString(BH::res)){
        case Resolution::Minimal:   cfg.WIDTH = 800;  cfg.HEIGHT = 600;  break;
        case Resolution::HD:        cfg.WIDTH = 1280; cfg.HEIGHT = 720;  break;
        case Resolution::HDplus:    cfg.WIDTH = 1600; cfg.HEIGHT = 900;  break;
        case Resolution::FHD:       cfg.WIDTH = 1920; cfg.HEIGHT = 1080; break;
        case Resolution::QHD:       cfg.WIDTH = 2560; cfg.HEIGHT = 1440; break;
        case Resolution::UHD:       cfg.WIDTH = 3840; cfg.HEIGHT = 2160; break;
        case Resolution::_4K:       cfg.WIDTH = 4096; cfg.HEIGHT = 2048; break;

        default:
            std::cerr << "Resolução desconhecida: " << BH::res << "\n";
            return 1;
    }


    cfg.title  = "BlackHoleSim";
    cfg.fov_y  = 60.0f;

    cfg.cam_pos    = pos;
    cfg.cam_target = target;
    cfg.cam_up     = world_up;
    cfg.use_direct_vectors = true;
    cfg.cam_fwd    = fwd;
    cfg.cam_right  = right;
    cfg.cam_up_vec = up;


    // ── abre janela e entra no loop ───────────────────────────────────
    engineRun(cfg);
    
    starmapFree();
    perlinFree();

    return 0;
}
