#pragma once
#include "platform.hpp"

// Texture handle — CUDA object on Linux/Windows, CpuTexture* cast on macOS
extern cudaTextureObject_t perlin;

bool perlinLoad(const char* path);
void perlinFree();
