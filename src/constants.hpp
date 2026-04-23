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
  
namespace BH {
    const bool is_sim = true;
    const std::string res = "UHD";
    constexpr int    MAX_STEPS       = 15000;
    constexpr double STEP_FACTOR     = 0.01;
    constexpr double ADAPTIVE_FACTOR = 5.0;
    
    constexpr double factor = 5.0f;
    constexpr double tame = 0.125f;
    
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
