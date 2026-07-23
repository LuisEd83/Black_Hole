#pragma once

/*
    CPU backend declarations (macOS / BH_CPU_BACKEND=1).
*/

#include "platform.hpp"
#include <atomic>
#include <vector>

// ── Progresso do render por tiles ────────────────────────────────────────────
// O engine lê `done` a cada frame para saber quantos tiles já pode fazer upload.

struct TileProgress {
    std::atomic<int>  done{0};
    std::atomic<int>  next_pos{0};   // contador de posição para tile_done_order
    int               total{0};
    int               tiles_x{0};
    int               tile_w{64};
    int               tile_h{64};
    std::vector<int>  tile_done_order; // [pos] = índice do tile que terminou em `pos`
    std::atomic<bool> stop{false};    // sinaliza cancelamento para os workers
};

TileProgress& getTileProgress();

// ── Entry point ──────────────────────────────────────────────────────────────

void launchRaytraceCPU(void* pixels, int WIDTH, int HEIGHT,
                        double3 pos, double3 fwd, double3 right, double3 up,
                        double fov_y, double rs,
                        cudaTextureObject_t starmap, cudaTextureObject_t perlin);

// ── Stubs para helpers CUDA-only chamados por main.cpp ───────────────────────

void           activateSetFlags();
unsigned int*  getStateCountsPtr();
