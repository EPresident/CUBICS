#pragma once

#include <data_structures/Vector.h>
#include <variables/IntVariables.h>
#include <domains/IntDomainsRepresentations.h>
#include <statistics/Statistics.h>

struct IntBacktrackStack
{
	IntVariables* variables;
    IntDomainsRepresentations* representations;

    Vector<IntDomainsRepresentations> backupsStacks;
    Vector<Vector<int>> levelsStacks;

    Statistics* stats;

    void initialize(IntVariables* vairables, Statistics* stats);
    void deinitialize();

    cudaDevice void saveState(int backtrackLevel, Vector<int>* changedDomains);
    cudaDevice void restoreState(int backtrackLevel, Vector<int>* changedDomains);
    cudaDevice void clearState(int backtrackLevel);

    cudaDevice bool isDomainChanged(int variable);
};
