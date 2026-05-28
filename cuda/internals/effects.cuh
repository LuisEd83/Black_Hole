#pragma once

// ─────────────────────────────────────────────────────────────────────────────────────────────────

__device__ float dopplerShift(double phi, 
                              double r_current, 
                              double3 camera_pos, 
                              double rs);


__device__  float perlinNoise(  cudaTextureObject_t perlin, 
                                double r_current, 
                                double phi,
                                double disk_r1, 
                                double disk_r2);


__device__  float redShift(double r_current, 
                           double rs);


__device__  float diskEmissivity(   double r_current,
                                    double z_cartesiano,
                                    double disk_r1, 
                                    double disk_r2,
                                    double height_scale);


__device__  void temperatureToColor(float t_normalized, 
                                    float& disk_r, 
                                    float& disk_g, 
                                    float& disk_b);


// ─────────────────────────────────────────────────────────────────────────────────────────────────
