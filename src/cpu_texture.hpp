#pragma once

/*
    CpuTexture: CPU-side 2D texture with bilinear sampling.

    On macOS the opaque cudaTextureObject_t handle is actually a
    reinterpret-cast pointer to one of these structs.  Loaders (starmap.cpp,
    perlin.cpp) allocate a CpuTexture on the heap and cast its address to
    cudaTextureObject_t; the CPU ray tracer casts it back.
*/

#include "platform.hpp"
#include <cmath>
#include <algorithm>
#include <cstdlib>

struct CpuTexture {
    float* data   = nullptr;  // RGBA, normalized [0,1], row-major
    int    width  = 0;
    int    height = 0;
    bool   wrap_u = true;     // longitude axis: wrap
    bool   wrap_v = false;    // latitude  axis: clamp at poles

    CpuTexture() = default;
    ~CpuTexture() { delete[] data; }

    // Bilinear sampling — UV in [0,1]
    float4 sample(float u, float v) const {
        auto wrapf = [](float t) {
            t -= std::floor(t);
            if (t < 0.f) t += 1.f;
            return t;
        };

        if (wrap_u) u = wrapf(u);
        else        u = std::fmax(0.f, std::fmin(1.f, u));
        if (wrap_v) v = wrapf(v);
        else        v = std::fmax(0.f, std::fmin(1.f, v));

        float px = u * width  - 0.5f;
        float py = v * height - 0.5f;

        int x0 = (int)std::floor(px);
        int y0 = (int)std::floor(py);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float fx = px - x0;
        float fy = py - y0;

        auto cx = [&](int x) {
            if (wrap_u) return ((x % width)  + width)  % width;
            return std::max(0, std::min(width  - 1, x));
        };
        auto cy = [&](int y) {
            if (wrap_v) return ((y % height) + height) % height;
            return std::max(0, std::min(height - 1, y));
        };

        x0 = cx(x0); x1 = cx(x1);
        y0 = cy(y0); y1 = cy(y1);

        auto px4 = [&](int x, int y) { return data + (y * width + x) * 4; };

        float* c00 = px4(x0, y0);
        float* c10 = px4(x1, y0);
        float* c01 = px4(x0, y1);
        float* c11 = px4(x1, y1);

        float s = fx, t = fy;
        return {
            c00[0]*(1-s)*(1-t) + c10[0]*s*(1-t) + c01[0]*(1-s)*t + c11[0]*s*t,
            c00[1]*(1-s)*(1-t) + c10[1]*s*(1-t) + c01[1]*(1-s)*t + c11[1]*s*t,
            c00[2]*(1-s)*(1-t) + c10[2]*s*(1-t) + c01[2]*(1-s)*t + c11[2]*s*t,
            c00[3]*(1-s)*(1-t) + c10[3]*s*(1-t) + c01[3]*(1-s)*t + c11[3]*s*t,
        };
    }
};

inline CpuTexture* cpu_tex(cudaTextureObject_t h) {
    return reinterpret_cast<CpuTexture*>(static_cast<uintptr_t>(h));
}
