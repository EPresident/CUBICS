#include <searchers/IntBacktrackStack.h>

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

void IntBacktrackStack::saveState(int backtrackLevel)
{
    for (int vi = 0; vi < backupsStacks.size; vi += 1)
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

void IntBacktrackStack::restoreState(int backtrackLevel)
{
    for (int vi = 0; vi < backupsStacks.size; vi += 1)
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

void IntBacktrackStack::clearState(int backtrackLevel)
{
    for (int vi = 0; vi < backupsStacks.size; vi += 1)
    {
        if (levelsStacks[vi].back() == backtrackLevel)
        {
            backupsStacks[vi].pop();
            levelsStacks[vi].pop_back();
        }
    }
}

bool IntBacktrackStack::isDomainChanged(int variable)
{
    return backupsStacks[variable].versions.back() != representations->versions[variable];
}
