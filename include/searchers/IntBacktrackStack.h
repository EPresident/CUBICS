#pragma once

#include <data_structures/Vector.h>
#include <variables/IntVariables.h>
#include <domains/IntDomainsRepresentations.h>

struct IntBacktrackStack
{
    IntDomainsRepresentations* representations;

    /** 
    * Backup of the values assigned to variables at different backtrack levels.
    * The i-th domain representation is for the i-th variable only;
    * This way more values for the same variable are stacked in the representation.
    */
    Vector<IntDomainsRepresentations> backupsStacks;
    /// For each variable the list of backtrack levels is stored.
    Vector<Vector<int>> levelsStacks;

    void initialize(IntDomainsRepresentations* representations);
    void deinitialize();

    /**
    * Saves the current domain representation for each variable in the stack,
    * with the given backtrack level (assuming there has been a change).
    */
    cudaDevice void saveState(int backtrackLevel);
    /**
    * Restores the domain representation with the given backtrack level,
    * assuming it is different.
    */
    cudaDevice void restoreState(int backtrackLevel);
    /**
    * Clears the state (domain representations) with the given backtrack level
    * from the stack.
    */
    cudaDevice void clearState(int backtrackLevel);

    /**
    * \return true if "variable" has a different domain (cardinality) in the state 
    * with the last backtrack level.
    */
    cudaDevice bool isDomainChanged(int variable);
};
