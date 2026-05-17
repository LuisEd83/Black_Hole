#pragma once

#include <string>
#include <chrono>
#include <fstream>

using Clock = std::chrono::high_resolution_clock;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

std::string printNow();
std::string estimatedEnd(Clock::time_point t0, double duration_ms);
double elapsedMs(Clock::time_point t0, Clock::time_point t1);
float getCpuTemp();
float getGpuTemp();

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
