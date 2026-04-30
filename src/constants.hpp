#pragma once

#include <string>
  
/*
        Resoluções que estão sendo utilizadas:
            
            0800 x 600, [Minimal]
            1280 x 720 [HD],
            1920 x 1080 [FHD],
            2560 x 1440 [QHD],
            3840 x 2160 [UHD].
            4096x2048 [4K]
*/



/*
void getRes(std::string res){

    Resolution result = fromString(res);

    switch (result){
        case Resolution::Minimal:       BH::WIDTH = 800;  BH::HEIGHT = 600;  break;
        case Resolution::HD:            BH::WIDTH = 1280; BH::HEIGHT = 720;  break;
        case Resolution::HDplus:        BH::WIDTH = 1600; BH::HEIGHT = 900;  break;
        case Resolution::FHD:           BH::WIDTH = 1920; BH::HEIGHT = 1080; break;
        case Resolution::QHD:           BH::WIDTH = 2560; BH::HEIGHT = 1440; break;
        case Resolution::UHD:           BH::WIDTH = 3840; BH::HEIGHT = 2160; break;
        case Resolution::_4K:           BH::WIDTH = 4096; BH::HEIGHT = 2048; break;

        default:
            std::cerr << "Resolução desconhecida: " << BH::result << "\n";
            return 1;
    }

}
*/


namespace BH {
    
    const bool is_sim = true;
    const bool is_gl = true;

    const std::string res = "Minimal";

    constexpr int    MAX_STEPS       = 5000;
    constexpr double STEP_FACTOR     = 0.5;
    
    constexpr double IMPACT_CUTOFF  = 7.5;
    constexpr double MAX_STEPS_DIV  = 2.5;

    constexpr double ADAPTIVE_FACTOR = 5.0;
    constexpr double EMISSIVITY_RATE = 0.001;

    constexpr double factor = 8.0f;
    
    constexpr double X_COEF = 1.2f;
    constexpr double Y_COEF = 0.6f;
    constexpr double Z_COEF = 0.12f;



    /*
    constexpr double ADAPTIVE_CLAMP  = 0.001;
    constexpr double ESCAPE_FACTOR   = 300.0;
    constexpr double DISK_R1_FACTOR  = 3.0;
    constexpr double DISK_R2_FACTOR  = 12.0;
    constexpr double DISK_HEIGHT     = 0.15;
    constexpr double DISK_OPACITY    = 2.0;
    constexpr double EMISSION_SCALE  = 3.0;
    constexpr double CLOSENESS       = 5.0;
    */
}
