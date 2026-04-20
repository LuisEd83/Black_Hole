#pragma once

#include <cuda_runtime.h>
#include <texture_types.h>

// objeto de textura - criado por perlinLoad(), usado pelo kernel
extern cudaTextureObject_t perlin;

// carrega data/disk3d.txt e cria a textura CUDA
// retorna false se falha

bool perlinLoad(const char* path);

// libera memória
void perlinFree();
