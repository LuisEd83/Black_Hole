#include "../cuda/geodesic.cuh"
#include "src/starmap.hpp"
#include "src/perlin.hpp"
#include "src/lodepng.h"

#include <glm/glm.hpp>
#include <iostream>
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


int main() {
    

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
   

    /*
        Resoluções que estão sendo utilizadas:
            
            0800 x 600,
            1280 x 720 [HD],
            1920 x 1080 [FHD],
            2560 x 1440 [QHD],
            3840 x 2160 [UHD].
            
    */
    
    /*

        factor —    distância da câmera em unidades de RS. Maior = buraco negro menor na tela, mais contexto ao redor. Menor = buraco negro domina o frame, mais distorção nas bordas. Abaixo de ~5 RS a câmera entra na região onde o lensing distorce a própria imagem da câmera.
                    graus / elevation — ângulo acima do plano do disco. 0° = vista lateral, disco como linha. 90° = vista de cima, disco como anel. 20-45° = visual do Interstellar. Muda radicalmente a forma do disco na imagem.

        fov_y —     zoom. Menor = mais zoom, buraco negro maior, bordas menos distorcidas. Maior = grande angular, mais cena visível, bordas com mais aberração. 20° é bem fechado — buraco negro ocupa mais da tela.
        pos.x —     componente horizontal da posição da câmera. Com a fórmula atual é cam_dist * cos(elevation) — mudar isso sem mudar pos.z rotaciona a câmera em azimute, mudando de qual lado o Doppler aparece mais brilhante.
        pos.y —     altura cartesiana. Atualmente 0.0 — se colocar um valor aqui a câmera sai do plano xz e o buraco negro aparece ligeiramente rotacionado. Normalmente deixa em zero.
        pos.z —     componente que determina theta0 no kernel — é o que realmente controla a elevação. cam_dist * sin(elevation) — quanto maior, mais acima do disco a câmera está.
        target  —    para onde a câmera aponta. Sempre (0,0,0) para olhar para o buraco negro. Mudar isso desloca o frame — útil para composição mas fisicamente não muda a simulação.
        world_up    —  define o "cima" da câmera. (0,1,0) é o padrão. Se a câmera estiver muito próxima de 90° de elevação, mude para (1,0,0) para evitar gimbal lock.
        fwd,right,up    — vetores de câmera calculados automaticamente a partir de pos, target e world_up. Não mude manualmente — são consequência dos outros.
        RUNS — quantas vezes o kernel é executado para calcular a média de tempo. Não afeta a imagem, só a precisão do benchmark. Para desenvolvimento use 1, para benchmark use 5.

    */



    const int WIDTH  = 800;
    const int HEIGHT = 600;
    const int RUNS   = 1;       // quantas vezes rodar para média de tempo 
    // bom = 5


    // câmera olhando para a origem (onde o buraco negro está)
    const double factor = 20.0f;
    const double graus = 5.0f;
    const double small_angle = 10.0f;
    float fov_y = 50.0f;



    const double cam_dist = RS * factor;
    float elevation = glm::radians((float)graus);
    float azimuth  = glm::radians((float)small_angle); 

    glm::vec3 pos = glm::vec3(
        float(cam_dist * cos(elevation) * cos(azimuth)), 
        float(cam_dist * sin(elevation)),
        float(cam_dist) * cos(elevation) * sin(azimuth)
    );

        
    glm::vec3 target    = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 world_up  = glm::vec3(0.0f, 0.0f, 1.0f);
    

    glm::vec3 fwd   = glm::normalize(target - pos);
    glm::vec3 right = glm::normalize(glm::cross(fwd, world_up));
    glm::vec3 up    = glm::normalize(glm::cross(right, fwd));
    
    
    std::vector<unsigned char>pixels(WIDTH * HEIGHT * 3, 0);
   


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // começando a simulação
  


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
       

    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
    std::cout << "\n• Aquecendo GPU...\n";
    
    std::cout << "\n    → Temperatura CPU Posterior: ";
    std::flush(std::cout);
        system("echo $(($(cat /sys/class/thermal/thermal_zone0/temp)/1000))°C");

    std::cout << "    |" << "\n";

    std::cout << "    → Temperatura GPU Anterior: ";
    std::flush(std::cout);
        system("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d '\\n'");
    std::cout << "°C\n"; 



    auto t_warm0 = std::chrono::high_resolution_clock::now();
    
        raytraceCUDA(pixels.data(), WIDTH, HEIGHT, pos, fwd, right, up, fov_y);
    
    auto t_warm1 = std::chrono::high_resolution_clock::now();
 
    double warm_ms = std::chrono::duration<double, std::milli>(t_warm1 - t_warm0).count();



    std::cout << "    |" << "\n";
    std::cout <<  "    ◦ Aquecimento concluído: ";

    if(warm_ms * 10e-4 / 60 >= 1){ 
        std::cout << std::fixed << std::setprecision(1) << warm_ms*10e-4/60 << "min" << std::endl;

    } else { 
        std::cout << std::fixed << std::setprecision(1) << warm_ms*10e-4 << "s" << std::endl;

    }

    std::cout << "    |" << "\n";

    std::cout << "    → Temperatura CPU Posterior: ";
    std::flush(std::cout);
        system("echo $(($(cat /sys/class/thermal/thermal_zone0/temp)/1000))°C");

    std::cout << "    |" << "\n    → Temperatura GPU Posterior: ";
    std::flush(std::cout);
        system("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d '\\n'");
    std::cout << "°C\n";

    
    cudaError_t warm_err = cudaGetLastError();
    if (warm_err != cudaSuccess)
        std::cerr << "\n    ⚠ Erro no warmup: " << cudaGetErrorString(warm_err) << "\n";
    else
        std::cout << "\n    ✓ Sem erros detectados\n";
    
    

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // infos para o usuário



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



    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // runs de benchmark
    


    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
    std::cout << "\n• Benchmark (" << RUNS << " runs)" << std::endl;

    std::cout << "\n    ◦ Rodando " << RUNS << " iterações...\n";

    double total_ms = 0.0;

    for (int i = 0; i < RUNS; i++) {
            
        auto t0 = std::chrono::high_resolution_clock::now();

        raytraceCUDA(pixels.data(), WIDTH, HEIGHT, pos, fwd, right, up, fov_y);
        
        auto t1 = std::chrono::high_resolution_clock::now();
    

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cout << "\nErro Cuda: " << cudaGetErrorString(err);


        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;


        std::cout << "    |"  << "\n" << "    → run " << (i+1);

        if(ms * 10e-4 / 60 >= 1){ 
            std::cout << ": " << std::fixed << std::setprecision(2) << ms*10e-4/60 << "min\n";

        } else { 
            std::cout << ": " << std::fixed << std::setprecision(2) << ms*10e-4 << "s\n";

        }

    }

    

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // resultados


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
    

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // amostra de pixels — confirma que o kernel escreveu algo
    // imprime RGB de 5 pixels espalhados pela imagem
   


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



    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // análise da distribuição dos pixels por categoria
    
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

    
    // salva o último frame como PNG
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
        std::cout << "\n◦ Imagem salva em " << filename.str() << " \n";
 
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


    starmapFree();
    perlinFree();


    return 0;
}
