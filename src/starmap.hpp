#pragma once
#include "platform.hpp"

// Texture handle — CUDA object on Linux/Windows, CpuTexture* cast on macOS
extern cudaTextureObject_t starmap;

bool starmapLoad(const char* path);
void starmapFree();
