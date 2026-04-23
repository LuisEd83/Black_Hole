#ifndef STATE_HEATMAP_H
#define STATE_HEATMAP_H

/*
 * state_heatmap.h — per-state dwell counter display for CUDA kernels
 *
 * Displays a live bar per RayResult state showing how many threads
 * have landed in each one. The host polls a device-side counter array
 * and redraws on a background thread.
 *
 * Compile: nvcc ... -lpthread
 */

#include <string>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <pthread.h>
#include <unistd.h>
#include <cuda_runtime.h>


#define SH_NUM_STATES 5

static const char* SH_STATE_NAMES[SH_NUM_STATES] = {
    "NONE",
    "HORIZON",
    "ESCAPE",
    "DISK",
    "FALLBACK"
};

/* ANSI colors per state */
static const char* SH_STATE_COLORS[SH_NUM_STATES] = {
    "\033[38;5;240m",   /* NONE     — dim gray   */
    "\033[38;5;196m",   /* HORIZON  — red        */
    "\033[38;5;45m",    /* ESCAPE   — cyan       */
    "\033[38;5;214m",   /* DISK     — orange     */
    "\033[38;5;135m",   /* FALLBACK — purple     */
};


#define SH_RESET  "\033[0m"
#define SH_DIM    "\033[2m"
#define SH_BOLD   "\033[1m"
#define SH_BAR_W  28



#define SH_RECORD(counter_array, ray_result) \
    atomicAdd(&(counter_array)[static_cast<int>(ray_result)], 1u)


typedef struct {

    unsigned int*  d_counts;      /* device pointer                   */
    unsigned int   total;         /* total rays launched              */
    volatile int            running;
    volatile int            ready;
    pthread_t      thread;
    pthread_mutex_t mutex;
    //pthread_cond_t cond;
    cudaStream_t poll_stream;
    bool verbose;

} SH_State;


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


static void sh_draw(unsigned int* h, unsigned int total) {

    unsigned int settled = 0;
    for (int i = 1; i < SH_NUM_STATES; i++) 
        settled += h[i];  // skip NONE=0

    /* move cursor up SH_NUM_STATES+2 lines to redraw in place */
    static int first_draw = 1;
    if (!first_draw)
        fprintf(stderr, "\033[%dA", SH_NUM_STATES + 2);
    
    first_draw = 0;

    float pct_done = total > 0 ? (float)settled / total * 100.f : 0.f;
   

    fprintf(stderr, SH_BOLD "  rays: %u / %u  (%.1f%% settled)\n" SH_RESET,
           settled, total, pct_done);

    fprintf(stderr, SH_DIM "  %-8s  %-*s  %s\n" SH_RESET,
           "state", SH_BAR_W, "distribution", "count");


    for(int i = 0; i < SH_NUM_STATES; i++){
        
        float frac  = (settled > 0) ? (float)h[i] / settled : 0.f;
        int   filled = (int)(frac * SH_BAR_W + 0.5f);
        int   empty  = SH_BAR_W - filled;

        char bar_filled[SH_BAR_W * 3 + 1]; bar_filled[0] = '\0';
        char bar_empty [SH_BAR_W * 3 + 1]; bar_empty [0] = '\0';

        for (int j = 0; j < filled; j++) 
            strcat(bar_filled, "█");

        for (int j = 0; j < empty;  j++) 
            strcat(bar_empty,  "░");

        fprintf(stderr, "  %s%-8s%s  %s%s%s%s%s%s  %u\n",
               SH_STATE_COLORS[i], SH_STATE_NAMES[i], SH_RESET,
               SH_STATE_COLORS[i], bar_filled, SH_RESET,
               SH_DIM,             bar_empty,  SH_RESET,
               h[i]);
    }

    fflush(stdout);
}

static void* sh_poll_thread2(void* arg){
    SH_State* sh = (SH_State*)arg;
    unsigned int h[SH_NUM_STATES] = {0};

    // cria contexto próprio para este thread — independente do flagSet do main
    cudaSetDevice(0);
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    cudaStream_t local_stream;
    cudaStreamCreateWithFlags(&local_stream, cudaStreamNonBlocking);
    
    cudaEvent_t ev;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);

    for (int i = 0; i < SH_NUM_STATES + 2; i++) fprintf(stderr, "\n");
    sh->ready = 1;

    while(1){
        if (!sh->running) break;

        cudaMemcpyAsync(h, sh->d_counts,
                        SH_NUM_STATES * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost,
                        local_stream);
        cudaEventRecord(ev, local_stream);

        while(cudaEventQuery(ev) == cudaErrorNotReady)
            usleep(50000);

        sh_draw(h, sh->total);
        usleep(1000000);
    }

    // draw final com dados completos
    cudaMemcpy(h, sh->d_counts,
               SH_NUM_STATES * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
  
    sh_draw(h, sh->total);
    

    cudaEventDestroy(ev);
    cudaStreamDestroy(local_stream);

    return nullptr;
}

static void* sh_poll_thread(void* arg){

    SH_State* sh = (SH_State*)arg;
    unsigned int h[SH_NUM_STATES] = {0};

    sh->ready = 1;


    while(1){

        // ─────────────────────────────────────────────────────────────────────────────────────────────────
        /*
        cudaMemcpy(h, sh->d_counts,
                   SH_NUM_STATES * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);

        */
        // ─────────────────────────────────────────────────────────────────────────────────────────────────

        if(!sh->running) 
            break;
           
        cudaMemcpyAsync(h, sh->d_counts,
                SH_NUM_STATES * sizeof(unsigned int),
                cudaMemcpyDeviceToHost,
                sh->poll_stream);

        cudaStreamSynchronize(sh->poll_stream); 

        sh_draw(h, sh->total);

        usleep(1000000);
    }

    return NULL;
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class StateHeatmap {
   
    private:
        SH_State sh_;

    public:
        StateHeatmap(unsigned int total_rays, bool verbose){

            memset(&sh_, 0, sizeof(sh_));

            sh_.total = total_rays;
            sh_.verbose = verbose;

            pthread_mutex_init(&sh_.mutex, NULL);
        }

        ~StateHeatmap(){
            pthread_mutex_destroy(&sh_.mutex);
        }


        void start(unsigned int* d_counts){

            sh_.d_counts = d_counts;
            cudaMemset(d_counts, 0, SH_NUM_STATES * sizeof(unsigned int));

            if (!sh_.verbose) return; 

            sh_.ready = 0;
            sh_.running = 1;

            cudaStreamCreateWithFlags(&sh_.poll_stream, cudaStreamNonBlocking);
            
            pthread_create(&sh_.thread, NULL, sh_poll_thread2, &sh_);
            
            //pthread_mutex_lock(&sh_.mutex);
            while(!sh_.ready)
                //pthread_cond_wait(&sh_.cond, &sh_.mutex);
                usleep(1000);
            //pthread_mutex_lock(&sh_.mutex);
        //
        }


        void stop(){

             if (!sh_.verbose) return; 

            sh_.running = 0;
            pthread_join(sh_.thread, NULL);

            cudaStreamDestroy(sh_.poll_stream);
        }

};

#endif
