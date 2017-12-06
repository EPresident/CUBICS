#include <searchers/IntBacktrackStack.h>
#include <utils/KernelUtils.h>

void IntBacktrackStack::initialize(IntDomainsRepresentations* representations)
{
    this->representations = representations;

    backupsStacks.initialize(representations->bitvectors.size);
    backupsStacks.resize(representations->bitvectors.size);

    levelsStacks.initialize(representations->bitvectors.size);
    levelsStacks.resize(representations->bitvectors.size);

    for (int vi = 0; vi < backupsStacks.size; vi += 1)
    {
        backupsStacks[vi].initialize(VECTOR_INITIAL_CAPACITY);

        levelsStacks[vi].initialize();
    }
}

void IntBacktrackStack::deinitialize()
{
    for (int vi = 0; vi < backupsStacks.size; vi += 1)
    {
        backupsStacks[vi].deinitialize();
        levelsStacks[vi].deinitialize();
    }
    backupsStacks.deinitialize();
    levelsStacks.deinitialize();
}

/**
* Saves the current domain representation for each variable in the stack,
* with the given backtrack level (assuming there has been a change).
*/
cudaDevice void IntBacktrackStack::saveState(int backtrackLevel)
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < backupsStacks.size)
#else
    for (int vi = 0; vi < backupsStacks.size; vi += 1)
#endif
    {
        if (backtrackLevel == 0 || isDomainChanged(vi))
        {
            int min = representations->minimums[vi];
            int max = representations->maximums[vi];
            int offset = representations->offsets[vi];
            int version = representations->versions[vi];
            Vector<unsigned int>* bitvector = &representations->bitvectors[vi];
            backupsStacks[vi].push(min, max, offset, version, bitvector);

            levelsStacks[vi].push_back(backtrackLevel);
        }
    }
}

/**
* Restores the domain representation with the given backtrack level,
* assuming it is different.
*/
cudaDevice void IntBacktrackStack::restoreState(int backtrackLevel)
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < backupsStacks.size)
#else
    for (int vi = 0; vi < backupsStacks.size; vi += 1)
#endif
    {
        if (isDomainChanged(vi))
        {
            representations->minimums[vi] = backupsStacks[vi].minimums.back();
            representations->maximums[vi] = backupsStacks[vi].maximums.back();
            representations->offsets[vi] = backupsStacks[vi].offsets.back();
            representations->versions[vi] = backupsStacks[vi].versions.back();
            representations->bitvectors[vi].copy(&backupsStacks[vi].bitvectors.back());

        }
    }
}

/**
* Clears the state (domain representations) with the given backtrack level
* from the stack.
*/
cudaDevice void IntBacktrackStack::clearState(int backtrackLevel)
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < backupsStacks.size)
#else
    for (int vi = 0; vi < backupsStacks.size; vi += 1)
#endif
    {
        if (levelsStacks[vi].back() == backtrackLevel)
        {
            backupsStacks[vi].pop();
            levelsStacks[vi].pop_back();
        }
    }
}

/**
* \return true if "variable" has a different domain (cardinality) in the state 
* with the last backtrack level.
*/
cudaDevice bool IntBacktrackStack::isDomainChanged(int variable)
{
    return backupsStacks[variable].versions.back() != representations->versions[variable];
}
