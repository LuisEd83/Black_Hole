#pragma once

#include "../../headers/geodesic.cuh"


// ─────────────────────────────────────────────────────────────────────────────────────────────────


__device__ void geodesicRHS(const Rays& s, double rhs[6], double rs);


__device__  void rk4Step(Rays& s, double dl, double rs);


// ─────────────────────────────────────────────────────────────────────────────────────────────────
