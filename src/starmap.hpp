#pragma once
 
#include <cuda_runtime.h>
 
// objeto de textura — criado por starmapLoad(), usado pelo kernel
extern cudaTextureObject_t starmap;
 
// carrega data/starmap.png e cria a textura CUDA
// retorna false se falhar
bool starmapLoad(const char* path);
 
// libera memória — chamar no shutdown
void starmapFree();
