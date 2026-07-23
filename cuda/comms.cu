#include "../headers/comms.cuh"

__device__ unsigned int d_state_counts[SH_NUM_STATES];

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


unsigned int* getStateCountsPtr(){

    unsigned int* ptr = nullptr;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&ptr), d_state_counts);
    cudaMemset(ptr, 0, SH_NUM_STATES * sizeof(unsigned int));

    return ptr;
}


void activateSetFlags(){

    static bool flagSet = false;
    if (!flagSet) {
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        flagSet = true;
    }
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
