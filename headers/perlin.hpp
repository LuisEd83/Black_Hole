#pragma once
#include "platform.hpp"

// objeto de textura - criado por perlinLoad(), usado pelo kernel
extern cudaTextureObject_t perlin;

// carrega data/disk3d.txt e cria a textura CUDA
// retorna false se falha
bool perlinLoad(const char* path);

// retorna objeto para a main.
cudaTextureObject_t perlinGet();

// libera memória
void perlinFree();
