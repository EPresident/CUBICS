#include <searchers/IntBacktrackStack.h>

void IntBacktrackStack::initialize(IntDomainsRepresentations* representations)
{
    this->representations = representations;

    backupsStacks.initialize(representations->bitvectors.size);
    backupsStacks.resize(representations->bitvectors.size);
    for (int vi = 0; vi < backupsStacks.size; vi += 1)
    {
        backupsStacks[vi].initialize(VECTOR_INITIAL_CAPACITY);
        int min = representations->minimums[vi];
        int max = representations->maximums[vi];
        int offset = representations->offsets[vi];
        int version = representations->versions[vi];
        Vector<unsigned int>* bitvector = &representations->bitvectors[vi];
        backupsStacks[vi].push(min, max, offset, version, bitvector);
    }

    levelsStacks.initialize(representations->bitvectors.size);
    levelsStacks.resize(representations->bitvectors.size);
    for (int vi = 0; vi < backupsStacks.size; vi += 1)
    {
        levelsStacks[vi].initialize();
    }
    for (int vi = 0; vi < backupsStacks.size; vi += 1)
    {
        levelsStacks[0].push_back(vi);
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

void IntBacktrackStack::saveState(int backtrackLevel, MonotonicIntVector* changedDomains)
{
    for (int i = 0; i < changedDomains->getSize(); i += 1)
    {
        int vi = changedDomains->at(i);

        int min = representations->minimums[vi];
        int max = representations->maximums[vi];
        int offset = representations->offsets[vi];
        int version = representations->versions[vi];
        Vector<unsigned int>* bitvector = &representations->bitvectors[vi];
        backupsStacks[vi].push(min, max, offset, version, bitvector);

        levelsStacks[backtrackLevel].push_back(vi);
    }
}

void IntBacktrackStack::resetState(MonotonicIntVector* changedDomains)
{
    for (int i = 0; i < changedDomains->getSize(); i += 1)
    {
        int vi = changedDomains->at(i);

        representations->minimums[vi] = backupsStacks[vi].minimums.back();
        representations->maximums[vi] = backupsStacks[vi].maximums.back();
        representations->offsets[vi] = backupsStacks[vi].offsets.back();
        representations->versions[vi] = backupsStacks[vi].versions.back();
        representations->bitvectors[vi].copy(&backupsStacks[vi].bitvectors.back());
    }
}

void IntBacktrackStack::restorePreviousState(int backtrackLevel)
{
    for (int i = 0; i < levelsStacks[backtrackLevel].size; i += 1)
    {
        int vi = levelsStacks[backtrackLevel][i];

        backupsStacks[vi].pop();

        representations->minimums[vi] = backupsStacks[vi].minimums.back();
        representations->maximums[vi] = backupsStacks[vi].maximums.back();
        representations->offsets[vi] = backupsStacks[vi].offsets.back();
        representations->versions[vi] = backupsStacks[vi].versions.back();
        representations->bitvectors[vi].copy(&backupsStacks[vi].bitvectors.back());

    }

    levelsStacks[backtrackLevel].clear();
}
