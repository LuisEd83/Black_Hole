#pragma once

#include "../src/state_heatmap.hpp"

void setStatePtr(StateHeatmap* s);
unsigned int* getShDCounts();
void signalFrameReady();
void activateSetFlags();
unsigned int* getStateCountsPtr();
