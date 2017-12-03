#pragma once

#include <data_structures/Vector.h>
#include <variables/IntVariables.h>
#include <domains/IntDomainsRepresentations.h>
#include <statistics/Statistics.h>

struct IntBacktrackStack
{
    IntDomainsRepresentations* representations;

    Vector<IntDomainsRepresentations> backupsStacks;
    Vector<Vector<int>> levelsStacks;

    Statistics* stats;

    void initialize(IntDomainsRepresentations* representations, Statistics* stats);
    void deinitialize();

    cudaDevice void saveState(int backtrackLevel, MonotonicIntVector* changedDomains);
    cudaDevice void resetState(MonotonicIntVector* changedDomains);
    cudaDevice void restorePreviousState(int backtrackLevel);
    cudaDevice void clearState(int backtrackLevel);

    cudaDevice bool isDomainChanged(int variable);
};
