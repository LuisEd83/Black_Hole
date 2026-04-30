#include "temp_and_time.hpp"


using Clock = std::chrono::high_resolution_clock;

std::string printNow(){

    auto now = Clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&t);
        
    char buf[16];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", &tm);
    return buf;

}


double elapsedMs(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}


std::string estimatedEnd(Clock::time_point t0, double duration_ms){

    auto t_end = t0 + std::chrono::milliseconds((long long)duration_ms);
    std::time_t t = Clock::to_time_t(t_end);
    std::tm tm = *std::localtime(&t);
    
    char buf[16];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", &tm);
    return buf;
}


float getCpuTemp(){
#if defined(_WIN32)
        // Windows: WMI is complex, easiest is OpenHardwareMonitor WMI bridge
        // Fallback: not natively available without driver
        return -1.0f;

#elif defined(__APPLE__)
        // macOS: use powermetrics (requires sudo) or IOKit
        FILE* f = popen("sudo powermetrics --samplers smc -n 1 2>/dev/null | grep 'CPU die' | awk '{print $4}'", "r");
        if (!f) return -1.0f;
        float temp = -1.0f;
        fscanf(f, "%f", &temp);
        pclose(f);
        return temp;

#else
        // Linux: iterate thermal zones, pick the one labeled x86_pkg or cpu
        for (int i = 0; i < 16; i++) {
            std::string type_path = "/sys/class/thermal/thermal_zone" + std::to_string(i) + "/type";
            std::string temp_path = "/sys/class/thermal/thermal_zone" + std::to_string(i) + "/temp";
            std::ifstream type_f(type_path);
            if (!type_f) break;
            std::string type;
            type_f >> type;
            if (type.find("x86_pkg") != std::string::npos ||
                type.find("cpu")     != std::string::npos) {
                std::ifstream temp_f(temp_path);
                int milli = 0;
                temp_f >> milli;
                return milli / 1000.0f;
            }
        }
        // fallback: zone0
        std::ifstream f("/sys/class/thermal/thermal_zone0/temp");
        int milli = 0;
        f >> milli;
        return milli / 1000.0f;

#endif
}


float getGpuTemp(){
#if defined(_WIN32)
    // Windows: query via nvidia-smi
    FILE* f = _popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>nul", "r");
    if (!f) return -1.0f;
    float temp = -1.0f;
    fscanf(f, "%f", &temp);
    _pclose(f);
    return temp;

#elif defined(__APPLE__)
    // macOS: nvidia-smi if available, else -1
    FILE* f = popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null", "r");
    if (!f) return -1.0f;
    float temp = -1.0f;
    fscanf(f, "%f", &temp);
    pclose(f);
    return temp;

#else
    // Search /sys/class/hwmon/ directly — more reliable than going through DRM
    for (int hwmon = 0; hwmon < 16; hwmon++) {
        std::string base = "/sys/class/hwmon/hwmon" + std::to_string(hwmon) + "/";

        // Check it's actually a GPU (amdgpu, nvidia, nouveau, radeon)
        std::ifstream nameFile(base + "name");
        if (!nameFile) continue;
        std::string name;
        nameFile >> name;
        if (name != "amdgpu" && name != "nvidia" && name != "nouveau" && name != "radeon")
            continue;

        for (int t = 1; t <= 3; t++) {
            std::ifstream f(base + "temp" + std::to_string(t) + "_input");
            if (!f) continue;
            int milli = 0;
            f >> milli;
            if (milli > 0) return milli / 1000.0f;
        }
    }
    // Fallback: nvidia-smi
    FILE* f = popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null", "r");
    if (f) {
        float temp = -1.0f;
        fscanf(f, "%f", &temp);
        pclose(f);

        if(temp > 0) return temp;
    }

    return -1;

#endif 
}
