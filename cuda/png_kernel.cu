#include "../headers/geodesic.cuh"


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


__global__ void raytraceKernelPNG(  unsigned char* pixels,
                                    RenderParams& rnd){

    PipelineParams ppl = {};      

    int WIDTH = rnd.WIDTH;
    int HEIGHT = rnd.HEIGHT;
    
    RayResult result = RayResult::NONE;
    int x = 0, y = 0;
    unsigned char R = 0, G = 0, B = 0;
    
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;
    

    pixelProcess(   x, y, 
                    R, G, B,
                    rnd,
                    ppl,
                    result
                 );

    int idx = (y * WIDTH + x) * 3;

    pixels[idx+0] = R;
    pixels[idx+1] = G;
    pixels[idx+2] = B;
        
    SH_RECORD(d_state_counts, result);

}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━




void launchPNG( unsigned char* pixels,
                RenderParams rnd){
        
    
    size_t nbytes = rnd.WIDTH * rnd.HEIGHT * 3;
    dim3 blockSize(16, 16);
    dim3 numBlocks((rnd.WIDTH + 15)/16, (rnd.HEIGHT + 15)/16);
        
    
    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    //@{
    /*
    */
    static bool is_sim = false;
    static double ms_per_frame = -1.0;
    static auto t_last = Clock::now();

    if(ms_per_frame > 0.0){
        auto now = Clock::now();

        double delta = std::chrono::duration<double, std::milli>(now - t_last).count();
        // suaviza com média móvel exponencial — evita saltos por GC ou stall de driver
        ms_per_frame = ms_per_frame * 0.9 + delta * 0.1;
    }
    t_last = Clock::now();
    //@}
    // ─────────────────────────────────────────────────────────────────────────────────────────────────
   

    printf("\n");
    unsigned char* d_pixels;
    cudaMalloc(&d_pixels, nbytes); ck("d_pixel Malloc");
    cudaMemset(d_pixels, 0, nbytes); ck("d_pixel Memset");


    unsigned int* d_counts_ptr = nullptr;
    cudaGetSymbolAddress((void**)&d_counts_ptr, d_state_counts);
    cudaMemset(d_counts_ptr, 0, SH_NUM_STATES * sizeof(unsigned int));
    
    
    cudaStream_t kernel_stream;
    cudaStreamCreate(&kernel_stream); ck("Stream Creation");

    raytraceKernelPNG<<<numBlocks, blockSize, 0, kernel_stream>>>(   
            d_pixels, 
            rnd);


    //ck("Post-Kernel");
    

    cudaStreamSynchronize(kernel_stream); 
    ck("StreamSync");
    
    cudaMemcpy(pixels, d_pixels, nbytes, cudaMemcpyDeviceToHost);
    ck("cudaMemcpy");
    


    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    //@{
    /*
    printf("ms_per_frame [2]: %f", ms_per_frame);
    */
    if (ms_per_frame < 0.0) {
        auto now = Clock::now();
        ms_per_frame = std::chrono::duration<double, std::milli>(now - t_last).count();
    }

    if (ms_per_frame > 0.0 && is_sim){
        updateCorrectionFactor(ms_per_frame);
    }
    is_sim = true;
    //@}
    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    

    cudaStreamDestroy(kernel_stream); ck("Destroy Stream");  
    cudaFree(d_pixels); ck("Free d_pixels");
    printf("\n");

}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
