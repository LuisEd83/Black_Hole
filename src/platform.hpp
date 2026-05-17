#pragma once

/*
    Platform adapter.

    Include this header instead of <cuda_runtime.h> in any host .cpp or .hpp
    that needs CUDA types.

      • macOS   → BH_CPU_BACKEND 1 — provides C++ type stubs; no CUDA headers.
      • Linux / Windows → BH_CPU_BACKEND 0 — includes the real CUDA headers.
*/

#if defined(__APPLE__)
// ── macOS: CPU backend ───────────────────────────────────────────────────────
#define BH_CPU_BACKEND 1

struct double3 { double x, y, z; };
struct float4  { float  x, y, z, w; };
struct uchar4  { unsigned char x, y, z, w; };

inline double3 make_double3(double x, double y, double z)      { return {x,y,z}; }
inline float4  make_float4(float x, float y, float z, float w) { return {x,y,z,w}; }
inline uchar4  make_uchar4(unsigned char r, unsigned char g,
                            unsigned char b, unsigned char a)   { return {r,g,b,a}; }

// Opaque texture handle: on macOS this is a reinterpret-cast CpuTexture*
using cudaTextureObject_t = unsigned long long;

// CUDA function qualifiers stripped on CPU
#define __device__
#define __host__
#define __global__
#define __forceinline__ inline

#else
// ── Linux / Windows: real CUDA ───────────────────────────────────────────────
#define BH_CPU_BACKEND 0
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#endif
