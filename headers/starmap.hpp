#pragma once
#include "platform.hpp"

// objeto de textura — criado por starmapLoad(), usado pelo kernel
extern cudaTextureObject_t starmap;
 
// carrega data/starmap.png e cria a textura CUDA
// retorna false se falhar
bool starmapLoad(const char* path);

// retorna o objeto para a main.
cudaTextureObject_t starmapGet();

// libera memória — chamar no shutdown
void starmapFree();
