#include "cuda/headers/geodesic.cuh"

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

using namespace BH;
using namespace glm;
using namespace std;

Resolution fromString(const string& s){

    if (s == "Minimal") return Resolution::Minimal;
    if (s == "HD")      return Resolution::HD;
    if (s == "HD+")     return Resolution::HDplus;
    if (s == "FHD")     return Resolution::FHD;
    if (s == "QHD")     return Resolution::QHD;
    if (s == "UHD")     return Resolution::UHD;
    if (s == "4K")      return Resolution::_4K;

    return Resolution::Unknown;
}

int main() {

    activateSetFlags();

    // ── recursos de suporte (mesmo que o main atual) ──────────────────
    if (!starmapLoad("data/starmap.png")) {
        cerr << "Falha ao carregar starmap\n";
        return 1;
    }
    if (!perlinLoad("data/perlin.txt")) {
        cerr << "Falha ao carregar perlin\n";
        return 1;
    }
   
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    

    const double cam_dist      = RS * factor;
    const double graus         = 10.0f;
    const double elevation_angle = 5.0f;
    float fov_y                = 60.0f;

    float elevation = radians((float)graus);
    float azimuth   = radians((float)elevation_angle);
        
    vec3 pos = vec3(
        float(cam_dist) * X_COEF,
        float(cam_dist) * Y_COEF,
        float(cam_dist) * Z_COEF
    );

    vec3 target    = vec3(float(-RS * 3.0), float(RS * 1.5), 0.0f);
    vec3 world_up = vec3(0.0f, 0.0f, 1.0f);

    vec3 fwd   = glm::normalize(target - pos);
    vec3 right = glm::normalize(glm::cross(fwd, world_up));
    vec3 up    = glm::normalize(glm::cross(right, fwd));

    // ── config da engine ──────────────────────────────────────────────
    SimConfig cfg;
    cfg.WIDTH  = 800;
    cfg.HEIGHT = 600;

    
    switch(fromString(res)){
        case Resolution::Minimal:   cfg.WIDTH = 800;  cfg.HEIGHT = 600;  break;
        case Resolution::HD:        cfg.WIDTH = 1280; cfg.HEIGHT = 720;  break;
        case Resolution::HDplus:    cfg.WIDTH = 1600; cfg.HEIGHT = 900;  break;
        case Resolution::FHD:       cfg.WIDTH = 1920; cfg.HEIGHT = 1080; break;
        case Resolution::QHD:       cfg.WIDTH = 2560; cfg.HEIGHT = 1440; break;
        case Resolution::UHD:       cfg.WIDTH = 3840; cfg.HEIGHT = 2160; break;
        case Resolution::_4K:       cfg.WIDTH = 4096; cfg.HEIGHT = 2048; break;

        default:
            cerr << "Resolução desconhecida: " << BH::res << "\n";
            return 1;
    }

    cfg.title = "BlackHoleSim";
    cfg.fov_y = 60.0f;
    cfg.pos = pos;
    cfg.target = target;
    cfg.world_up = world_up;
    cfg.use_direct_vectors = true;
    cfg.fwd = fwd;
    cfg.right = right;
    cfg.up = up;

    //cfg.starmap = starmapGet(); 
    //cfg.perlin = perlinGet(); 
    //cout << cfg.starmap << " ," << cfg.starmap << "\n";
    
    cfg.getVectors(pos, fwd, right, up);
    
    /*
        printf("=== DEBUG CAMERA ===\n");
        printf("pos   = (%.3e, %.3e, %.3e)\n", pos.x, pos.y, pos.z);
        printf("fwd   = (%.3f, %.3f, %.3f)  len=%.3f\n", fwd.x, fwd.y, fwd.z, glm::length(fwd));
        printf("right = (%.3f, %.3f, %.3f)  len=%.3f\n", right.x, right.y, right.z, glm::length(right));
        printf("up    = (%.3f, %.3f, %.3f)  len=%.3f\n", up.x, up.y, up.z, glm::length(up));
        printf("use_direct = %d\n", cfg.use_direct_vectors);
        printf("RS = %.3e  cam_dist = %.3e\n", RS, (double)cfg.cam_dist);
        printf("fov_y = %.2f  WIDTH=%d HEIGHT=%d\n", cfg.fov_y, cfg.WIDTH, cfg.HEIGHT);
        printf("target = (%.3e, %.3e, %.3e)\n", cfg.target.x, cfg.target.y, cfg.target.z);
        printf("====================\n");
    
    */

    // ── abre janela e entra no loop ───────────────────────────────────
    engineRun(cfg);
    
    starmapFree();
    perlinFree();

    return 0;
}
