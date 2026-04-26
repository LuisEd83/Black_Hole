#include "cuda/geodesic.cuh"
#include "cuda/feedbacks.cuh"
#include "cuda/comms.cuh"

#include "src/state_heatmap.hpp"
#include "src/constants.hpp"
#include "src/starmap.hpp"
#include "src/perlin.hpp"
#include "src/lodepng.h"

#include <glm/glm.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>

#include <ctime>
#include <sstream>
#include <iomanip>


/*
    comando de compilação (unix):

    cmake -B build -S . && cmake --build build --clean-first && ./build/BlackHoleCUDA
*/


// declaração de raytraceCUDA — definida em geodesic_host.cpp
void raytraceCUDA( unsigned char* pixels,
                   int WIDTH, int HEIGHT,
                   glm::vec3 pos, glm::vec3 fwd, glm::vec3 right, glm::vec3 up,
                   float fov_y);


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// output de temperatura/time:

using Clock = std::chrono::high_resolution_clock;

double elapsedMs(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}


std::string estimatedEnd(Clock::time_point t0, double duration_ms) {
    auto t_end = t0 + std::chrono::milliseconds((long long)duration_ms);
    std::time_t t = Clock::to_time_t(t_end);
    std::tm tm = *std::localtime(&t);
    char buf[16];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", &tm);
    return buf;
}


float getCpuTemp() {
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

float getGpuTemp() {
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
    // Linux: sysfs hwmon, works for NVIDIA and AMD
    for (int card = 0; card < 8; card++) {
        for (int hwmon = 0; hwmon < 8; hwmon++) {
            std::string path = "/sys/class/drm/card" + std::to_string(card) +
                               "/device/hwmon/hwmon"  + std::to_string(hwmon) +
                               "/temp1_input";
            std::ifstream f(path);
            if (!f) continue;
            int milli = 0;
            f >> milli;
            if (milli > 0) return milli / 1000.0f;
        }
    }
    return -1.0f;
#endif
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// parsing rápido de resolução
//@{


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

//@}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


int main() {
    
    activateSetFlags();

    // gera nome do arquivo com timestamp
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm* tm_info = std::localtime(&t);

    std::ostringstream filename;
    filename << "output_"
             << std::setfill('0') << std::setw(2) << tm_info->tm_hour
             << std::setw(2) << tm_info->tm_min
             << ".png";


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // parâmetros de teste
    //@{
    
    /*

        factor —    distância da câmera em unidades de RS. Maior = buraco negro menor na tela, mais contexto ao redor. Menor = buraco negro domina o frame, mais distorção nas bordas. Abaixo de ~5 RS a câmera entra na região onde o lensing distorce a própria imagem da câmera.
                    graus / elevation — ângulo acima do plano do disco. 0° = vista lateral, disco como linha. 90° = vista de cima, disco como anel. 20-45° = visual do Interstellar. Muda radicalmente a forma do disco na imagem.

        fov_y   —   zoom. Menor = mais zoom, buraco negro maior, bordas menos distorcidas. Maior = grande angular, mais cena visível, 
                    bordas com mais aberração. 20° é bem fechado — buraco negro ocupa mais da tela.

        pos.x   —   componente horizontal da posição da câmera. Com a fórmula atual é cam_dist * cos(elevation) — mudar isso sem mudar 
                    pos.z rotaciona a câmera em azimute, mudando de qual lado o Doppler aparece mais brilhante.

        pos.y   —   altura cartesiana. Atualmente 0.0 — se colocar um valor aqui a câmera sai do plano xz e o buraco negro aparece 
                    ligeiramente rotacionado. Normalmente deixa em zero.

        pos.z   —   componente que determina theta0 no kernel — é o que realmente controla a elevação. 
                    cam_dist * sin(elevation) — quanto maior, mais acima do disco a câmera está.

        target  —   para onde a câmera aponta. Sempre (0,0,0) para olhar para o buraco negro. 
                    Mudar isso desloca o frame — útil para composição mas fisicamente não muda a simulação.

        world_up    —   define o "cima" da câmera. (0,1,0) é o padrão. Se a câmera estiver muito próxima de 90° de elevação, 
                        mude para (1,0,0) para evitar gimbal lock.

        fwd,right,up    —   vetores de câmera calculados automaticamente a partir de pos, target e world_up. 
                            Não mude manualmente — são consequência dos outros.

        RUNS    —   quantas vezes o kernel é executado para calcular a média de tempo. Não afeta a imagem, só a precisão do benchmark. 
                    Para desenvolvimento use 1, para benchmark use 5.
        
        small_angle —
    */
    
     
    int WIDTH = 800;
    int HEIGHT = 600;

    switch(fromString(BH::res)){
        case Resolution::Minimal:   WIDTH = 800;  HEIGHT = 600;  break;
        case Resolution::HD:        WIDTH = 1280; HEIGHT = 720;  break;
        case Resolution::HDplus:    WIDTH = 1600; HEIGHT = 900;  break;
        case Resolution::FHD:       WIDTH = 1920; HEIGHT = 1080; break;
        case Resolution::QHD:       WIDTH = 2560; HEIGHT = 1440; break;
        case Resolution::UHD:       WIDTH = 3840; HEIGHT = 2160; break;
        case Resolution::_4K:       WIDTH = 4096; HEIGHT = 2048; break;

        default:
            std::cerr << "Resolução desconhecida: " << BH::res << "\n";
            return 1;
    }


    
    const int RUNS   = 1;       // quantas vezes rodar para média de tempo 
                                // bom = 5

    // câmera olhando para a origem (onde o buraco negro está)
    const double factor = BH::factor;

    const double graus = 10.0f;
    const double elevation_angle = 5.0f;
    //const double tame = 0.3f;
    float fov_y = 60.0f;
   

    const double cam_dist = RS * BH::factor;
    float elevation = glm::radians((float)graus);
    float azimuth  = glm::radians((float)elevation_angle); 
    
    glm::vec3 pos = glm::vec3(
        float(cam_dist) * cos(elevation),
        float(cam_dist) * cos(azimuth),
        float(10 * BH::tame * cam_dist) * cos(azimuth) * BH::tame

        //float(cam_dist) * cos(azimuth) * tame
    );
    
    glm::vec3 target    = glm::vec3(float(RS * 3.5), 0.0f, 0.0f);
    glm::vec3 world_up  = glm::vec3(0.0f, 0.0f, 1.0f);
    
    glm::vec3 fwd   = glm::normalize(target - pos);
    glm::vec3 right = glm::normalize(glm::cross(fwd, world_up));
    glm::vec3 up    = glm::normalize(glm::cross(right, fwd));
    
    std::vector<unsigned char>pixels(WIDTH * HEIGHT * 3, 0);
    
    //@}


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // começando a simulação
    //@{
  


    std::cout << "\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
    std::cout << "\n━━━━━━━━ Inicialização BlackHoleSim ━━━━━━━\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";

    std::cout << "\n• Parâmetros da cena \n\n";
    std::cout << "  → RS (Schwarzschild):  " << std::fixed << std::setprecision(1) <<RS*10e-11  << "·10^11m\n";
    std::cout << "  → distância câmera:    " << std::fixed << std::setprecision(1) <<(double)cam_dist * 10e-12 << "·10^12m" << " - (" << factor <<  "·RS)\n";
    std::cout << "  → elevação:            " << std::fixed << std::setprecision(2) << graus << "°\n";
    std::cout << "  → fov_y:               " << std::fixed << std::setprecision(2) << fov_y << "°\n";

    std::cout << "  → câmera pos:          (" << std::fixed << std::setprecision(1) << pos.x*10e-11 << "·10^11, " << pos.y*10e-11 << "·10^11" << ", " << pos.z*10e-10 << "·10^10" << ")\n";
    std::cout << "  → câmera fwd:          " << std::fixed << std::setprecision(2) << "(" << fwd.x  << ", " << fwd.y << ", " << fwd.z << ")" << "\n";
    std::cout << "  → resolução:           " << WIDTH << "x" << HEIGHT << " = " << WIDTH*HEIGHT << " pixels\n";
    std::cout << "  → blocos CUDA:         " << (WIDTH+15)/16 << "x" << (HEIGHT+15)/16 << " de 16x16 threads\n";
    
    //@}

     
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // infos para o usuário
    //@{


    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
    std::cout << "\n• Abrindo arquivos de suporte: \n";

    // carrega o starmap antes de qualquer kernel
    if (!starmapLoad("data/starmap.png")) {
        std::cerr << "\nFalha ao carregar starmap\n";

        return 1;
    }

    if (!perlinLoad("data/perlin.txt")) {
        std::cerr << "\nFalha ao carregar perlin\n";

        return 1;
    }
    
    //@}

    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // estimativas:
    //@{ 
    

    /*
    StateHeatmap state_warm(WIDTH/2 * HEIGHT/2, BH::is_sim);
    
    unsigned int* d_counts_warm = getStateCountsPtr();
    state_warm.start(d_counts_warm);
    */

    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
    std::cout << "\n• Estimativas da Simulação\n";
    
    warmupAndEstimate(WIDTH, HEIGHT, BH::MAX_STEPS, RS * BH::STEP_FACTOR, RS);
  
    //@}


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // warmup da gpu: etapa talvez inútil, ou boa quando curta.
    //@{

    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
    std::cout << "\n• Aquecendo GPU...\n";
     
    std::cout << "\n    → Temperatura CPU Anterior: ";
    std::flush(std::cout);
        system("echo $(($(cat /sys/class/thermal/thermal_zone0/temp)/1000))°C");

    std::cout << "    │" << "\n";

    std::cout << "    → Temperatura GPU Anterior: ";
    /*
    std::flush(std::cout);
        system("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d '\\n'");
    std::cout << "°C\n"; 
    */

    std::cout << getGpuTemp() << ", " << getCpuTemp();


    auto t_warm0 = std::chrono::high_resolution_clock::now();
    
        raytraceCUDA(pixels.data(), floor(WIDTH/2), floor(HEIGHT/2), pos, fwd, right, up, fov_y);
    
    auto t_warm1 = std::chrono::high_resolution_clock::now();
 
    double warm_ms = std::chrono::duration<double, std::milli>(t_warm1 - t_warm0).count();
    warm_ms *= 4.0;


    std::cout << "    │" << "\n";
    std::cout <<  "    ◦ Aquecimento concluído: ";

    if(warm_ms * 10e-4 / 60 >= 1){ 
        std::cout << std::fixed << std::setprecision(1) << warm_ms*10e-4/60 << "min" << std::endl;

    } else { 
        std::cout << std::fixed << std::setprecision(1) << warm_ms*10e-4 << "s" << std::endl;

    }

    std::cout << "    │" << "\n";

    std::cout << "    → Temperatura CPU Posterior: ";
    std::flush(std::cout);
        system("echo $(($(cat /sys/class/thermal/thermal_zone0/temp)/1000))°C");

    std::cout << "    │" << "\n    → Temperatura GPU Posterior: ";
    std::flush(std::cout);
        system("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d '\\n'");
    std::cout << "°C\n";

    
    cudaError_t warm_err = cudaGetLastError();
    if (warm_err != cudaSuccess)
        std::cerr << "\n    ⚠ Erro no warmup: " << cudaGetErrorString(warm_err) << "\n";
    else
        std::cout << "\n    ✓ Sem erros detectados\n";
    
    
    //state_warm.stop();
    

    //@}


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // runs de benchmark
    //@{
   

   
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
    std::cout << "\n• Benchmark (" << RUNS << " runs)" << std::endl;
    

    auto now_      = std::chrono::system_clock::now();
    std::time_t t_ = std::chrono::system_clock::to_time_t(now_);

    double total_expected_ms = warm_ms * RUNS;
    int extra_seconds = (int)(total_expected_ms / 1000.0);

    std::time_t t_done = t_ + extra_seconds;

    // copia a struct antes da segunda chamada sobrescrever
    std::tm tm_done = *std::localtime(&t_done);   // copia por valor
    std::tm tm_now  = *std::localtime(&t_);       // agora pode chamar de novo
    
    /*
    std::cout << "\n    ◦ Conclusão: ~"
              << std::setfill('0') << std::setw(2) << tm_done.tm_hour << ":"
              << std::setw(2) << tm_done.tm_min  << ":"
              << std::setw(2) << tm_done.tm_sec
              << " [Agora: "
              << tm_now.tm_hour << ":"
              << std::setw(2) << tm_now.tm_min  << ":"
              << std::setw(2) << tm_now.tm_sec
              << "]\n";
    */

    std::cout << "Conclusão: ~" << estimatedEnd(Clock::now(), total_expected_ms) << "\n";
    

    std::cout << "\n    ◦ Rodando " << RUNS << " iterações...\n";
    double total_ms = 0.0;

    for (int i = 0; i < RUNS; i++) {
        

        std::cout << "\n    ──────────────────────────────────────────\n";
        StateHeatmap state_runs(WIDTH * HEIGHT, BH::is_sim);
    
        unsigned int* d_counts_runs = getStateCountsPtr();
        state_runs.start(d_counts_runs);

            
        //auto t0 = std::chrono::high_resolution_clock::now();
        auto t0 = Clock::now();

            raytraceCUDA(pixels.data(), WIDTH, HEIGHT, pos, fwd, right, up, fov_y);
        
        //auto t1 = std::chrono::high_resolution_clock::now();
        auto t1 = Clock::now();
   

        state_runs.stop();          // stop polling thread
        std::cout << "    ├────────────────────────────────────────\n";
        //std::cout << "\n";          // push cursor below heatmap's last line
        std::cout.flush();


        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cout << "\nErro Cuda: " << cudaGetErrorString(err);

        double ms = elapsedMs(t0, t1);
        //double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;


        std::cout << "    │"  << "\n" << "    → run " << (i+1);

        if(ms * 10e-4 / 60 >= 1){ 
            std::cout << ": " << std::fixed << std::setprecision(2) << ms*10e-4/60 << "min\n";

        } else { 
            std::cout << ": " << std::fixed << std::setprecision(2) << ms*10e-4 << "s\n";
        }
    }
    
    
    //@}
    

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // resultados
    //@{

    double avg_ms      = total_ms / RUNS;
    double pixels_per_sec = (WIDTH * HEIGHT) / (avg_ms / 1000.0);

    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
    std::cout << "\n• Resultados\n\n";
    //std::cout << "  → resolução:        " << WIDTH << "×" << HEIGHT << "\n";
    std::cout << "◦ Taxa de pixel:\n";
    
    std::cout << "  → média:            ";

    if(avg_ms * 10e-4 / 60 >= 1){ 
            std::cout << std::fixed << std::setprecision(2) << avg_ms*10e-4/60 << "min\n";

        } else { 
            std::cout << std::fixed << std::setprecision(2) << avg_ms*10e-4 << "s\n";

        }

    std::cout << "  → throughput:       " << std::fixed << std::setprecision(2)  << pixels_per_sec*10e-4 << "·10^4 pixels/s\n";
    
    //@}


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // amostra de pixels — confirma que o kernel escreveu algo
    // imprime RGB de 5 pixels espalhados pela imagem
    //@{   


    std::cout << "\n◦ Amostra de pixels (RGB)\n";
    int sample_positions[] = { 0, WIDTH/4, WIDTH/2, WIDTH*3/4, WIDTH-1 };
    int row = HEIGHT / 2;   // linha do meio

    for (int sx : sample_positions) {
        int idx = (row * WIDTH + sx) * 3;
        std::cout << "  → intervalo: [" << std::setw(3) << sx << ", " << row << "]: "
                  << "R=" << int(pixels[idx+0])
                  << ", G=" << int(pixels[idx+1])
                  << ", B=" << int(pixels[idx+2]) << "\n";
    }

    //@}


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // análise da distribuição dos pixels por categoria
    //@{

    // ajeitar isso daqui
        
    int n_black = 0, n_disk = 0, n_sky = 0, n_fallback = 0, n_other = 0;

    for (int i = 0; i < WIDTH * HEIGHT; i++) {

        int idx = i * 3;
        unsigned char r = pixels[idx], g = pixels[idx+1], b = pixels[idx+2];

        if      (r == 0 && g == 0 && b == 0)    n_black++;
        else if (r == 255)                      n_disk++;
        else if (r == 40 && g == 0 && b == 0)   n_fallback++;
        else if (b > r && b > g)                n_sky++;
        else                                    n_other++;
    }
           
    double total = WIDTH * HEIGHT;

    std::cout << "\n◦ Distribuição espacial dos pixels\n";
    std::cout << "  → horizonte:  "  << std::fixed << std::setprecision(2) << 100.0*n_black/total << "%\n";
    std::cout << "  → disco:      "  << std::fixed << std::setprecision(2) << 100.0*n_disk/total  << "%\n";
    std::cout << "  → skybox:     "  << std::fixed << std::setprecision(2) << 100.0*n_sky/total   << "%\n";
    std::cout << "  → fallback:   "  << std::fixed << std::setprecision(2) << 100.0*n_fallback/total << "%\n";
    std::cout << "  → outros:     "  << std::fixed << std::setprecision(2) << 100.0*n_other/total << "%\n";
    
    //@}

        
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // salva o último frame como PNG
    //@{

    std::vector<unsigned char> png_pixels(WIDTH * HEIGHT * 4);
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        png_pixels[i*4+0] = pixels[i*3+0];  // R
        png_pixels[i*4+1] = pixels[i*3+1];  // G
        png_pixels[i*4+2] = pixels[i*3+2];  // B
        png_pixels[i*4+3] = 255;            // A
    }
    

    unsigned error = lodepng::encode("output/" + filename.str(), png_pixels, WIDTH, HEIGHT);

    if (error)
        std::cerr << "\n◦ Erro de imagem: " << lodepng_error_text(error) << "\n";
    else
        std::cout << "\n◦ Imagem salva em " << filename.str() <<  " (" << png_pixels.size()/1024/1024 << " MB)\n";
        

    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
    
    //@}


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // cleaning e fim
    
    starmapFree();
    perlinFree();

    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


    return 0;
}
