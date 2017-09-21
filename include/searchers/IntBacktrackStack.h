#pragma once

#include <data_structures/Vector.h>
#include <variables/IntVariables.h>
#include <domains/IntDomainsRepresentations.h>

struct IntBacktrackStack
{
    IntDomainsRepresentations* representations;

    Vector<IntDomainsRepresentations> backupsStacks;
    Vector<Vector<int>> levelsStacks;

    void initialize(IntDomainsRepresentations* representations);
    void deinitialize();

    void saveState(int backtrackLevel);
    void restoreState(int backtrackLevel);
    void clearState(int backtrackLevel);

    bool isDomainChanged(int variable);
};
