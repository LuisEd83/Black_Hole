#include "../headers/geodesic.cuh"


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


__global__  void raytraceKernelGL(  int WIDTH,
                                    int HEIGHT,
                                    double3 pos,
                                    double3 fwd, 
                                    double3 right,
                                    double3 up,
                                    float fov_y,
                                    double rs,  
                                    cudaTextureObject_t starmap,
                                    cudaTextureObject_t perlin,
                                    cudaSurfaceObject_t surface,
                                    int* d_counter
                                    ){

    int x = 0, y = 0;
    unsigned char R = 0, G = 0, B = 0;
    RayResult result;
        
    if(is_persis){
        
        while(1){
    
            int idx = atomicAdd(d_counter, 1);
            if(idx >= WIDTH * HEIGHT) 
                return;
                
            x = idx % WIDTH;
            y = idx / WIDTH;
            
            R = G = B = 0;
            
            pixelProcess_d(   x,y,
                            R,G,B,    
                            WIDTH, HEIGHT,
                            pos, fwd, right, up,
                            fov_y, rs,  
                            starmap, perlin, result
                        );
          
            if (x >= WIDTH || y >= HEIGHT || x < 0 || y < 0) return;
                

            uchar4 pixel = make_uchar4(R, G, B, 255);
            surf2Dwrite(pixel, surface, x * 4, y);

        }
            
    } else {
        
        //while(1){

        x = blockIdx.x * blockDim.x + threadIdx.x;
        y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= WIDTH || y >= HEIGHT) return;

        R = G = B = 0;
            

        pixelProcess_d(   x, y, 
                        R, G, B,
                        WIDTH, HEIGHT, 
                        pos, fwd, right, up,
                        fov_y, rs, 
                        starmap, perlin, result
                     );
     

        if (x >= WIDTH || y >= HEIGHT || x < 0 || y < 0) return;

        uchar4 pixel = make_uchar4(R, G, B, 255);
        surf2Dwrite(pixel, surface, x * 4, y);
        
    }

}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


void launchGL(  cudaSurfaceObject_t surface,
                int WIDTH, 
                int HEIGHT,
                double3 pos, 
                double3 fwd, 
                double3 right, 
                double3 up,
                float fov_y,
                double rs,
                cudaTextureObject_t starmap,
                cudaTextureObject_t perlin){
    

    dim3 numBlocks, blockSize;

    if(is_persis){
        
        numBlocks = dim3(96);
        blockSize = dim3(256);
        
        int* d_counter = nullptr;
        cudaMalloc(&d_counter, sizeof(int)); // ck("d_pixel Malloc");

        cudaMemset(d_counter, 0, sizeof(int)); //ck("d_pixel Memset");
        
        raytraceKernelGL<<<numBlocks, blockSize>>>(
                WIDTH, 
                HEIGHT,
                pos, fwd, right, up,
                fov_y, 
                rs, 
                starmap, 
                perlin, 
                surface, 
                d_counter);

        // ck("Post-Kernel");

        cudaStreamSynchronize(0); //ck("StreamSync");
        cudaFree(d_counter); //ck("Free d_counter");
    
    } else {
    
        numBlocks = dim3((WIDTH + 15)/16, (HEIGHT + 15)/16);
        blockSize = dim3(16, 16);

        raytraceKernelGL<<<numBlocks, blockSize>>>(
            WIDTH, 
            HEIGHT,
            pos, fwd, right, up,
            fov_y, 
            rs, 
            starmap, 
            perlin, 
            surface, 
            nullptr);
        
        // ck("Post-Kernel");

        cudaStreamSynchronize(0); // ck("StreamSync");

    }
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
