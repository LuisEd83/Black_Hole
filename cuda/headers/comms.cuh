#pragma once

#include "../../src/distribution.hpp"

void setStatePtr(StateHeatmap* s);
unsigned int* getShDCounts();
void signalFrameReady();
void activateSetFlags();
unsigned int* getStateCountsPtr();
