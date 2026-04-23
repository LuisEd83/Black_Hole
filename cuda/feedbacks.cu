#include "feedbacks.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cstdio>


#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t e = (call);                                         \
        if(e != cudaSuccess){                                           \
            std::cerr << "Erro CUDA: " << cudaGetErrorString(e)         \
                      << " em " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1);                                                \
        }                                                               \
    } while(0)                                                          \


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


static constexpr const char* FACTOR_FILE = ".bh_correction_factor";
static constexpr int   FACTOR_WARMUP     = 5;   // ignora os N primeiros frames (GPU ainda aquecendo)


void updateCorrectionFactor(double real_ms, double dummy_ms){
    if(dummy_ms <= 0.0) return;

    double fator_novo = real_ms / dummy_ms;

    // lê média e contagem atuais do arquivo
    double media  = 1.25;
    int    count  = 0;

    if(FILE* f = fopen(FACTOR_FILE, "r")){
        fscanf(f, "%lf %d", &media, &count);
        fclose(f);
    }

    if(count < FACTOR_WARMUP){
        // ainda aquecendo — só incrementa contagem, não atualiza média
        count++;

    } else {

        // média móvel exponencial: peso 0.05 pro novo valor
        // converge devagar — estável após ~20 frames
        media = media * 0.95 + fator_novo * 0.05;
        count++;
    }

    if(FILE* f = fopen(FACTOR_FILE, "w")){
        fprintf(f, "%.6f %d\n", media, count);
        fclose(f);
    }

}


double getCorrectionFactor(){

    double media = 5.0;
    int    count = 0;

    if(FILE* f = fopen(FACTOR_FILE, "r")){
        fscanf(f, "%lf %d", &media, &count);
        fclose(f);
    }

    return media;
}



// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// kernel e geodesic dummies

/*
    não faz nada útil. Serve só pra forçar o driver a inicializar o contexto
    CUDA antes da gente medir qualquer coisa de verdade
    
    o geodesic dummy pega o tempo base de cálculo: talvez seja insuficiente,
    mas é útil para construir estimativa.

*/

__global__ static void dummyKernel(int *dummy){

    if(threadIdx.x == 0 && blockIdx.x == 0)
        dummy[0] = 42;
}


__global__ static void geodesicDummy(   unsigned char* pixels,
                                        int WIDTH, int HEIGHT,
                                        int maxSteps,
                                        double step,
                                        double rs){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    // Estado inicial fixo — posição arbitrária fora do horizonte
    double r     = rs * 20.0;
    double theta = 1.5707963;   // π/2, plano equatorial
    double phi   = 0.0;
    double dr    = -1.0;        // indo em direção ao buraco negro
    double dtheta= 0.0;
    double dphi  = 1.0 / (r * r);
    double E     = 1.0;

    for (int i = 0; i < maxSteps; ++i) {
        if (r <= rs) break;

        //double f  = 1.0 - rs / r;
        //double dt = E / f;
        double st = sin(theta);
        double ct = cos(theta);

        // RK4 inline (mesmo cálculo do kernel real)
        double y0[6] = {r, theta, phi, dr, dtheta, dphi};
        double k[4][6];

        auto rhs = [&](double s[6], double out[6]) {
            double _f  = 1.0 - rs/s[0];
            double _dt = E/_f;
            double _st = sin(s[1]), _ct = cos(s[1]);
            out[0] = s[3];
            out[1] = s[4];
            out[2] = s[5];
            out[3] = -(rs/(2*s[0]*s[0]))*_f*_dt*_dt
                     +(rs/(2*s[0]*s[0]*_f))*s[3]*s[3]
                     + s[0]*(_st*_st*s[5]*s[5] + s[4]*s[4]);
            out[4] = -(2.0/s[0])*s[3]*s[4] + _st*_ct*s[5]*s[5];
            out[5] = -(2.0/s[0])*s[3]*s[5] - 2.0*(_ct/_st)*s[4]*s[5];
        };

        double tmp[6];
        rhs(y0, k[0]);

        for(int j=0;j<6;j++) tmp[j]=y0[j]+k[0][j]*(step/2);
            rhs(tmp, k[1]);

        for(int j=0;j<6;j++) tmp[j]=y0[j]+k[1][j]*(step/2);
            rhs(tmp, k[2]);

        for(int j=0;j<6;j++) tmp[j]=y0[j]+k[2][j]*step;
            rhs(tmp, k[3]);

        r      += (step/6)*(k[0][0]+2*k[1][0]+2*k[2][0]+k[3][0]);
        theta  += (step/6)*(k[0][1]+2*k[1][1]+2*k[2][1]+k[3][1]);
        phi    += (step/6)*(k[0][2]+2*k[1][2]+2*k[2][2]+k[3][2]);
        dr     += (step/6)*(k[0][3]+2*k[1][3]+2*k[2][3]+k[3][3]);
        dtheta += (step/6)*(k[0][4]+2*k[1][4]+2*k[2][4]+k[3][4]);
        dphi   += (step/6)*(k[0][5]+2*k[1][5]+2*k[2][5]+k[3][5]);

        if (r > rs * 1000.0) break;
    }

    int idx = (y * WIDTH + x)*3;
    pixels[idx+0] = (unsigned char)(fmin(r/rs, 255.0));
    pixels[idx+1] = 0;
    pixels[idx+2] = 0;

}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


void warmupAndEstimate(int WIDTH, int HEIGHT, int maxSteps, double step, double rs){


    using ms = std::chrono::duration<double, std::milli>;
    auto now = []{ return std::chrono::high_resolution_clock::now(); };
    


    // ── 1. Info do device ────────────────────────────────────────────────────
    std::cout << "\n    ◦ detectando device CUDA...\n" << "      │\n";
    int deviceId;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

    std::cout << "      ├ GPU:          " << prop.name << "\n";
    std::cout << "      ├ VRAM:         " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "      └ max threads/bloco: " << prop.maxThreadsPerBlock << "\n\n";
    


    // ── 2. Latência de alocação ──────────────────────────────────────────────
    std::cout << "    ◦ medindo latência de alocação de memória...\n" << "      │\n";
    

    // Buffer de teste: tamanho real que o kernel vai usar por frame
    size_t realBufferSize = (size_t)WIDTH * HEIGHT * 3;
    unsigned char* d_test;

    auto t0 = now();

    CUDA_CHECK(cudaMalloc(&d_test, realBufferSize));
    CUDA_CHECK(cudaFree(d_test));

    double allocMs = ms(now() - t0).count();

    std::cout << "      ├ buffer real (" << WIDTH << "x" << HEIGHT << "): " << realBufferSize / 1024 << " KB\n";
    std::cout << "      └ latência alloc+free: " << std::fixed << std::setprecision(2) << allocMs << " ms\n\n";
    
    

    // ── 3. Kernel dummy — acorda o driver ────────────────────────────────────
    std::cout << "    ◦ acordando driver CUDA (dummy kernel)...\n" << "      │\n";
    std::cout << "      ├ ·≈300ms: primeira chamada CUDA inicializa o contexto\n";

    int* d_dummy;
    CUDA_CHECK(cudaMalloc(&d_dummy, sizeof(int)));

    auto t1 = now();

        dummyKernel<<<1,1>>>(d_dummy);

    CUDA_CHECK(cudaDeviceSynchronize());

    double driverInitMs = ms(now() - t1).count();

    CUDA_CHECK(cudaFree(d_dummy));
    std::cout << "      └ driver inicializado em: " << std::fixed << std::setprecision(1) << driverInitMs << " ms\n\n";
    


    // ── 4. Benchmark em resolução reduzida ───────────────────────────────────
    std::cout << "    ◦ rodando benchmark (64x64)...\n" << "      │\n";
    

    const int BW = 64, BH = 64;
    unsigned char* d_bench;
    
    CUDA_CHECK(cudaMalloc(&d_bench, BW*BH*3));
    

    dim3 block(16,16);
    dim3 grid((BW+15)/16, (BH+15)/16);
    

        // Aquece mais uma vez com resolução pequena antes de medir
        geodesicDummy<<<grid,block>>>(d_bench, BW, BH, maxSteps, step, rs);
    
    CUDA_CHECK(cudaDeviceSynchronize());


    // Medição real: média de 3 rodadas
    double totalBenchMs = 0;
    const int BENCH_RUNS = 3;

    for(int i = 0; i < BENCH_RUNS; i++){
        
        auto tb = now();

            geodesicDummy<<<grid,block>>>(d_bench, BW, BH, maxSteps, step, rs);
    
        CUDA_CHECK(cudaDeviceSynchronize());
        
        totalBenchMs += ms(now() - tb).count();
    }
    
    double benchMs = totalBenchMs / BENCH_RUNS;
    CUDA_CHECK(cudaFree(d_bench));

    // Tempo por pixel no benchmark
    double msPerPixel = benchMs / (BW * BH);

    std::cout << "      ├ tempo médio (64x64): " << std::fixed << std::setprecision(2)
         << benchMs << " ms  (" << std::setprecision(4) << msPerPixel*1000
         << " µs/pixel)\n";
    


    // ── 5. Estimativa extrapolada ─────────────────────────────────────────────
    double estKernelMs = msPerPixel * WIDTH * HEIGHT* getCorrectionFactor();
    
    std::cout << "\npixelMs" << msPerPixel << ", correction_factor: " << getCorrectionFactor(); 

    size_t testSize = 32 * 1024 * 1024; // 32 MB
    unsigned char *d_bw, *h_bw = new unsigned char[testSize];
    cudaMalloc(&d_bw, testSize);
    auto tbw0 = now();
    cudaMemcpy(h_bw, d_bw, testSize, cudaMemcpyDeviceToHost);
    double estMemcpyMs = ms(now() - tbw0).count() * ((double)realBufferSize / testSize);
    cudaFree(d_bw);
    delete[] h_bw;


    double estTotalMs     = allocMs + estKernelMs + estMemcpyMs;
    double estFps         = 1000.0 / estTotalMs;


    std::cout << "      └ estimativa p/ (" << WIDTH << "x" << HEIGHT << ")\n" << "        │\n";
    std::cout << "        ├ alocação GPU:     " << std::setw(8) << std::fixed << std::setprecision(2) << allocMs       << " ms\n";

    if(estKernelMs * 10e-4 / 60 >= 1){ 
            std::cout << "        ├ kernel:           " << std::setw(8) << std::fixed << std::setprecision(2) << estKernelMs*10e-4/60  << " min\n";
        } else { 
            std::cout << "        ├ kernel:           " << std::setw(8) << std::fixed << std::setprecision(2) << estKernelMs*10e-4   << " s\n";
    } 
    
    std::cout << "        ├ memcpy GPU→CPU:   " << std::setw(8) << std::fixed << std::setprecision(2) << estMemcpyMs   << " ms\n";
    //std::cout << "           ▶ total por frame:  " << std::setw(8) << std::fixed << std::setprecision(2) << estTotalMs    << " ms\n";
    std::cout << "        ├ FPS esperado:     " << std::setw(8) << std::fixed << std::setprecision(1) << estFps        << " fps\n";
        
    
    auto now_      = std::chrono::system_clock::now();
    std::time_t t_ = std::chrono::system_clock::to_time_t(now_);

    int extra_seconds = (int)(estTotalMs/ 1000.0);

    std::time_t t_done = t_ + extra_seconds;

    // copia a struct antes da segunda chamada sobrescrever
    std::tm tm_done = *std::localtime(&t_done);   // copia por valor
    std::tm tm_now  = *std::localtime(&t_);       // agora pode chamar de novo

    std::cout << "        └ Conclusão (benchmark): ~"
              << std::setfill('0') << std::setw(2) << tm_done.tm_hour << ":"
              << std::setw(2) << tm_done.tm_min  << ":"
              << std::setw(2) << tm_done.tm_sec
              << " [Agora: "
              << tm_now.tm_hour << ":"
              << std::setw(2) << tm_now.tm_min  << ":"
              << std::setw(2) << tm_now.tm_sec
              << "]\n";

    /*
    if (estFps < 1.0){
        std::cout << "\n    ⚠  Menos de 1 fps esperado. Considere:\n";
        std::cout << "        → Reduzir MAX_STEPS (atual: " << maxSteps << ")\n";
        std::cout << "        → Reduzir resolução (" << WIDTH << "x" << HEIGHT << ")\n";
        std::cout << "        → Aumentar step (passos maiores → menos iterações)\n\n";
    }
    */

}
