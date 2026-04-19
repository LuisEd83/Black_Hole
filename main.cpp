#include "../cuda/geodesic.cuh"

#include <glm/glm.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

/*

    comando de compilação (unix):

    cmake -B build -S . && cmake --build build --clean-first && ./build/BlackHoleCUDA

*/



// declaração de raytraceCUDA — definida em geodesic_host.cpp
void raytraceCUDA( unsigned char* pixels,
                   int WIDTH, int HEIGHT,
                   glm::vec3 pos, glm::vec3 fwd, glm::vec3 right, glm::vec3 up,
                   float fov_y, double rs);


int main() {
    

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // parâmetros de teste
    


    const int WIDTH  = 1000;
    const int HEIGHT = 800;
    const int RUNS   = 5;       // quantas vezes rodar para média mais estável

    // câmera olhando para a origem (onde o buraco negro está)
    // posição em unidades de raio de Schwarzschild: câmera a ~30 rs de distância
    const double cam_dist = RS * 30.0;

    glm::vec3 pos       = glm::vec3(float(RS * 0.1), (RS * 0.5), float(cam_dist));
    glm::vec3 target    = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 world_up  = glm::vec3(0.0f, 1.0f, 0.0f);

    glm::vec3 fwd   = glm::normalize(target - pos);
    glm::vec3 right = glm::normalize(glm::cross(fwd, world_up));
    glm::vec3 up    = glm::normalize(glm::cross(right, fwd));
    float fov_y     = 60.0f;
    

    std::vector<unsigned char>pixels(WIDTH * HEIGHT * 3, 0);
    

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // começando o kernel
   
    

    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
    std::cout << "\n[MAIN]: inicialização BlackHoleSim...\n";
    std::cout << "\n• Parâmetros da cena \n\n";
    std::cout << "→ RS (Schwarzschild):  " << std::scientific << std::setprecision(3) << RS << " m\n";
    std::cout << "→ distância câmera:    " << std::scientific << (double)cam_dist << " m" << "(30 RS)\n";
    std::cout << "→ câmera pos:          (" << std::fixed << std::setprecision(2) << pos.x << ", " << pos.y << ", " << pos.z << ")\n";
    std::cout << "→ câmera fwd:          (" << fwd.x << ", " << fwd.y << ", " << fwd.z << ")\n";
    std::cout << "→ resolução:           " << WIDTH << "x" << HEIGHT << " = " << WIDTH*HEIGHT << " pixels\n";
    std::cout << "→ blocos CUDA:         " << (WIDTH+15)/16 << "x" << (HEIGHT+15)/16 << " de 16x16 threads\n";
    std::cout << "→ fov_y:               " << fov_y << " graus\n";
   

    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
    std::cout << "\n[MAIN]: Aquecendo GPU...\n";
    std::cout << "\n[1/2] iniciando contexto CUDA...";  // endl = flush implícito
    


    auto t_warm0 = std::chrono::high_resolution_clock::now();
    
        raytraceCUDA(pixels.data(), WIDTH, HEIGHT, pos, fwd, right, up, fov_y, RS);
    
    auto t_warm1 = std::chrono::high_resolution_clock::now();
 


    double warm_ms = std::chrono::duration<double, std::milli>(t_warm1 - t_warm0).count();
    std::cout << "\n[2/2] warmup concluido: "
              << std::fixed << std::setprecision(1) << warm_ms << " ms" << std::endl;

    cudaError_t warm_err = cudaGetLastError();
    if (warm_err != cudaSuccess)
        std::cerr << "\n[CUDA]: erro no warmup: " << cudaGetErrorString(warm_err) << "\n";
    else
        std::cout << "\n[CUDA]: sem erros detectados\n";
 


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // runs de benchmark
    


    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
    std::cout << "\n• Benchmark (" << RUNS << " runs)" << std::endl;

    std::cout << "\n[MAIN] Rodando " << RUNS << " iterações...\n";

    double total_ms = 0.0;

    for (int i = 0; i < RUNS; i++) {
            
        auto t0 = std::chrono::high_resolution_clock::now();

        raytraceCUDA(pixels.data(), WIDTH, HEIGHT, pos, fwd, right, up, fov_y, RS);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cout << "\n[ERRO CUDA: " << cudaGetErrorString(err) << "]";


        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;

        std::cout << "run " << (i+1) << ": " 
                  << std::fixed << std::setprecision(2) << ms << " ms\n";
    }

    

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // resultados



    double avg_ms      = total_ms / RUNS;
    double pixels_per_sec = (WIDTH * HEIGHT) / (avg_ms / 1000.0);

    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
    std::cout << "\n• Resultados\n\n";
    std::cout << "→ resolução:        " << WIDTH << "×" << HEIGHT << "\n";
    std::cout << "→ média:            " << std::fixed << std::setprecision(2) << avg_ms << " ms\n";
    std::cout << "→ throughput:       " << std::scientific << std::setprecision(2) 
              << pixels_per_sec << " pixels/s\n";

    

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // amostra de pixels — confirma que o kernel escreveu algo
    // imprime RGB de 5 pixels espalhados pela imagem
   


    std::cout << "\n◦ Amostra de pixels (RGB)\n";
    int sample_positions[] = { 0, WIDTH/4, WIDTH/2, WIDTH*3/4, WIDTH-1 };
    int row = HEIGHT / 2;   // linha do meio

    for (int sx : sample_positions) {
        int idx = (row * WIDTH + sx) * 3;
        std::cout << "pixel [" << std::setw(3) << sx << ", " << row << "]: "
                  << "→ R=" << int(pixels[idx+0])
                  << ", G=" << int(pixels[idx+1])
                  << ", B=" << int(pixels[idx+2]) << "\n";
    }



    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // análise da distribuição dos pixels por categoria
    
  

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
    std::cout << "→ horizonte:  " << n_black    << " (" << 100.0*n_black/total    << "%)\n";
    std::cout << "→ disco:      " << n_disk     << " (" << 100.0*n_disk/total     << "%)\n";
    std::cout << "→ skybox:     " << n_sky      << " (" << 100.0*n_sky/total      << "%)\n";
    std::cout << "→ fallback:   " << n_fallback << " (" << 100.0*n_fallback/total << "%)\n";
    std::cout << "→ outros:     " << n_other    << " (" << 100.0*n_other/total    << "%)\n";



    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


    return 0;
}
