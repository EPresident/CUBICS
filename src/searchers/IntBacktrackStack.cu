#include <algorithm>

#include <searchers/IntBacktrackStack.h>
#include <utils/KernelUtils.h>

void IntBacktrackStack::initialize(IntVariables* variables, Statistics* stats)
{
    this->variables = variables;
    representations = &variables->domains.representations;

    backupsStacks.initialize(representations->bitvectors.size);
    backupsStacks.resize(representations->bitvectors.size);

    levelsStacks.initialize(representations->bitvectors.size);
    levelsStacks.resize(representations->bitvectors.size);

    for (int vi = 0; vi < backupsStacks.size; vi += 1)
    {
        backupsStacks[vi].initialize(VECTOR_INITIAL_CAPACITY);
        int min = representations->minimums[vi];
        int max = representations->maximums[vi];
        int offset = representations->offsets[vi];
        int version = representations->versions[vi];
        Vector<unsigned int>* bitvector = &representations->bitvectors[vi];
        backupsStacks[vi].push(min, max, offset, version, bitvector);

        levelsStacks[vi].initialize();
        levelsStacks[vi].push_back(0);
    }

    this->stats = stats;
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

cudaDevice void IntBacktrackStack::saveState(int backtrackLevel, Vector<int>* changedDomains)
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < backupsStacks.size)
#else
    for (int vi = 0; vi < changedDomains->size; vi += 1)
#endif
    {
    	int variable = changedDomains->at(vi);
        if (isDomainChanged(variable))
        {
            int min = representations->minimums[variable];
            int max = representations->maximums[variable];
            int offset = representations->offsets[variable];
            int version = representations->versions[variable];
            Vector<unsigned int>* bitvector = &representations->bitvectors[variable];
            backupsStacks[vi].push(min, max, offset, version, bitvector);

            levelsStacks[vi].push_back(backtrackLevel);

            stats->maxStackSize = std::max(stats->maxStackSize, levelsStacks[vi].size - 1);
        }
    }
}

cudaDevice void IntBacktrackStack::restoreState(int backtrackLevel, Vector<int>* changedDomains)
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < backupsStacks.size)
#else
    for (int vi = 0; vi < changedDomains->size; vi += 1)
#endif
    {
    	int variable = changedDomains->at(vi);
        if (isDomainChanged(variable))
        {
            representations->minimums[variable] = backupsStacks[variable].minimums.back();
            representations->maximums[variable] = backupsStacks[variable].maximums.back();
            representations->offsets[variable] = backupsStacks[variable].offsets.back();
            representations->versions[variable] = backupsStacks[variable].versions.back();
            representations->bitvectors[variable].copy(&backupsStacks[variable].bitvectors.back());

        }
    }
}

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

cudaDevice bool IntBacktrackStack::isDomainChanged(int variable)
{
    bool changed = backupsStacks[variable].versions.back() != representations->versions[variable];
    return changed;
}
