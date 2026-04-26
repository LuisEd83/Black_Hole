#ifndef STATE_HEATMAP_H
#define STATE_HEATMAP_H

/* 
    info

    StateHeatmap:   pega os dados do kernel para gerar um display contínuo dos resultados.
    
        • uso de cudaStream, cudaSymbol, cudaContext.
            
            ◦ cudaStream:   uma sequência de operações CUDA. O que fazemos aqui é sincronizar 
                            todas as streams sob um mesmo padrão, o que torna a leitura dos 
                            dados possíveis. 

                                → stream A: [kernel--------------------]
                                → stream B: (NonBlocking): [memcpy][memcpy][memcpy]                               
                                
                                · ambos andam na mesma velocidade.

            ◦ cudaSymbol:   usamos o d_state_counts como uma variável __device__, o que é inacessível
                            em tempo de compilação ao host. Logo, precisamos montar um ponteiro para esse
                            símbolo CUDA, para então poder pegar os dados via memset, memcpy.
            
            ◦ cudaContext:  um contexto CUDA é a associação de operações em GPU (memória, streams, módulos) 
                            com uma thread de CPU. Toda thread de CPU tem um contexto único, mas isso resulta
                            em complicações que serão descritas a seguir.

        • o por quê desse uso:
            
            ◦   Essa abordagem foi usada para combinar com o uso 
                de flagSet, que por si só é uma ferramenta para amenizar as operações
                constantes, diminuindo o uso de CPU.
            
            ◦ deviceSync fazia a seguinte coisa:
                
                → Thread CPU associada ao contexto GPU fazia um spinlock: checava a condição do kernel várias vezes
                  por segundo. Isso escala para renders longos, logo, a CPU esquentava muito. 

            ◦ a alternativa:
                
                → o nome flagSet, que é um apelido para a ferramenta deviceScheduleBlockingSync, faz a thread
                  dormir até o kernel ser completo, em vez dela mesmo checar sua condição. Isso só pode ser feito
                  antes da criação de contexto CUDA, por isso toda a preocupação com a sequência dentro da 
                  main & kernel & poll_thread.


                main thread                     poll thread       
                ───────────────────────         ──────────────────────────────
                flagSet                         cudaSetDevice(0)
                launch kernel on stream         loop every 1s:
                cudaStreamSynchronize(...)      cudaMemcpyAsync(snap_stream)
                state.stop()                    cudaStreamSynchronize(snap)
                                                sh_draw(h, total)


*/



#include <cstddef>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <unistd.h>
#include <cuda_runtime.h>


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


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
    atomicAdd(&(counter_array)[static_cast<int>(ray_result)], 1u); \


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


typedef struct {

    bool verbose;
    unsigned int* d_counts;
    unsigned int total;    
    volatile int running;
    pthread_t thread;

    pthread_mutex_t mutex;

} SH_State;


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// desenha o display


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
   

    fprintf(stderr, SH_BOLD "    rays: %u / %u  (%.1f%% settled)\n" SH_RESET,
           settled, total, pct_done);

    fprintf(stderr, SH_DIM "    %-8s  %-*s  %s\n" SH_RESET,
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

        fprintf(stderr, "    %s%-8s%s  %s%s%s%s%s%s  %u\n",
               SH_STATE_COLORS[i], SH_STATE_NAMES[i], SH_RESET,
               SH_STATE_COLORS[i], bar_filled, SH_RESET,
               SH_DIM,             bar_empty,  SH_RESET,
               h[i]);
    }

    fflush(stdout);
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/* 
    info
    
    pipeline de comandos:

        · 1: cria um contexto próprio para essa thread com setDevice.
        
        · 2: cria um stream CUDA com a flag cudaStreamNonBlocking: isso
             nos possibilita não esperar pelo FIM DA STREAM principal:
             reads assíncronos.
        
        · 3: faz a cópia de memória dos counts assíncronamente com memcpyAsync.

        · 4: streamSyncronize espera apenas o memcpyAsync acabar.
        
        · 5: desenha.
*/


static void* sh_poll_thread(void* arg){
    SH_State* sh = (SH_State*)arg;
    unsigned int h[SH_NUM_STATES] = {0};
    
    cudaSetDevice(0); // cria contexto próprio para este thread — independente do flagSet do main
    
    cudaStream_t snap_stream;
    cudaStreamCreateWithFlags(&snap_stream, cudaStreamNonBlocking);

    while(sh->running){
        
        usleep(1000000);
        
        cudaMemcpyAsync(h, sh->d_counts,
                   SH_NUM_STATES * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost, snap_stream
                   );

        cudaStreamSynchronize(snap_stream);   
       

        pthread_mutex_lock(&sh->mutex);

            sh_draw(h, sh->total);

        pthread_mutex_unlock(&sh->mutex);
    
    }
    
    cudaStreamDestroy(snap_stream);
    return nullptr;
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class StateHeatmap {
   
    public:

        SH_State sh_;
        StateHeatmap(unsigned int total_rays, bool verbose){

            memset(&sh_, 0, sizeof(sh_));

            sh_.total = total_rays;
            sh_.verbose = verbose;

        }

        ~StateHeatmap(){
        }

        void start(unsigned int* d_counts){

            sh_.d_counts = d_counts;
            cudaMemset(d_counts, 0, SH_NUM_STATES * sizeof(unsigned int));

            if (!sh_.verbose) return; 
            sh_.running = 1;

            pthread_create(&sh_.thread, NULL, sh_poll_thread, &sh_);
        }
        
        void stop(){
            if (!sh_.verbose) return; 

            sh_.running = 0;
            pthread_join(sh_.thread, NULL);

            pthread_mutex_destroy(&sh_.mutex);
        }

};


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


#endif
