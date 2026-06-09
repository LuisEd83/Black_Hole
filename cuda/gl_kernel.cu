#include "../headers/geodesic.cuh"


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


__global__  void raytraceKernelGL(  cudaSurfaceObject_t surface,
                                    RenderParams rnd,
                                    PipelineParams& ppl){
        
    int x = 0, y = 0;
    RayResult result = RayResult::NONE;

    auto process = [&](int x, int y){
        if(CHECKERBOARD){
            if ((x + y + ppl.frame_parity) % 2 != 0){

                uchar4 prev;

                surf2Dread(&prev, ppl.surface_prev, x * 4, y);
                surf2Dwrite(prev, surface, x * 4, y);

                return;
            }
        } 

        unsigned char R = 0, G = 0, B = 0;
        pixelProcess(x, y, R, G, B, rnd, ppl, result);

        surf2Dwrite(make_uchar4(R, G, B, 255), surface, x * 4, y);
    };


    if(is_persis){
        while(true){
    
            int idx = atomicAdd(ppl.d_counter, 1);
            if(idx >= rnd.WIDTH * rnd.HEIGHT) 
                return;
            if (x >= rnd.WIDTH || y >= rnd.HEIGHT || x < 0 || y < 0) return;
                
            x = idx % rnd.WIDTH;
            y = idx / rnd.WIDTH;
            
            process(x,y);
        }

            
    } else {
        
        //while(1){

        x = blockIdx.x * blockDim.x + threadIdx.x;
        y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= rnd.WIDTH || y >= rnd.HEIGHT) return;

        process(x, y);
    }

}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


void launchGL(  cudaSurfaceObject_t surface,
                RenderParams rnd){
        

    PipelineParams ppl = {
                            0,
                            0,
                            0
                        };
    

    dim3 numBlocks, blockSize;

    if(is_persis){
        
        numBlocks = dim3(96);
        blockSize = dim3(256);
        
        int* d_counter = nullptr;
        cudaMalloc(&d_counter, sizeof(int));  ck("d_pixel Malloc");
        cudaMemset(d_counter, 0, sizeof(int)); ck("d_pixel Memset");
        
        raytraceKernelGL<<<numBlocks, blockSize>>>(
                surface,
                rnd,
                ppl
        );

         ck("Post-Kernel");

        //cudaStreamSynchronize(0); ck("StreamSync");
        cudaFree(d_counter); ck("Free d_counter");
    
    } else {
    
        numBlocks = dim3((rnd.WIDTH + 15)/16, (rnd.HEIGHT + 15)/16);
        blockSize = dim3(16, 16);

        raytraceKernelGL<<<numBlocks, blockSize>>>(
            surface, 
            rnd,
            ppl
        );
        
         ck("Post-Kernel");

        cudaStreamSynchronize(0);  ck("StreamSync");

    }
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
