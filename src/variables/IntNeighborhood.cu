#include <variables/IntNeighborhood.h>
#include <utils/Utils.h>
#include <cassert>
#include <wrappers/Wrappers.h>

void IntNeighborhood::initialize(Vector<int>* neighbors, IntDomainsRepresentations* originalRepr, int constraints)
{
    this->count = neighbors->size;
    // Init bitvector
    neighMask.initialize(originalRepr->minimums.size, false);
    // Init domains representations
    neighRepr.initialize(count);
    // Init map
    map.initialize(count);
    // Init events
    events.initialize(count);
    // init actions
    neighActions.initialize(count);
    // get required blocks
    #ifdef GPU
    variablesBlocks = KernelUtils::getBlockCount(count, DEFAULT_BLOCK_SIZE, true);
    #endif
    // Init constraints-to-propagate flags
    constraintToPropagate.initialize(constraints);
    constraintToPropagate.resize(constraints);
    for (int ci = 0; ci < constraints; ci += 1)
    {
        constraintToPropagate[ci] = false;
    }
    // Push neighbors
    for (int i = 0; i < neighbors->size; i += 1)
    {
        int var = neighbors->at(i);
        // Update mask
        neighMask.set(var);
        // Update variable-representation map
        map.push_back(var);
        // Push to domain representation
        int min = originalRepr->minimums[var];
        int max = originalRepr->maximums[var];
        int offset = originalRepr->offsets[var];
        int version = originalRepr->versions[var];
        Vector<unsigned int>* bitvector = &originalRepr->bitvectors[var];
        neighRepr.push(min, max, offset, version, bitvector);
        // Push event
        events.push_back(EventTypes::Changed);
        // Push action
        neighActions.push();
    }
}

cudaHostDevice void IntNeighborhood::deinitialize()
{
    neighMask.deinitialize();
    neighRepr.deinitialize();
    map.deinitialize();
    events.deinitialize();
    neighActions.deinitialize();
    constraintToPropagate.deinitialize();
}

cudaDevice void IntNeighborhood::getBinding(int var, int* repr)
{
    #ifdef GPU
    int i = KernelUtils::getTaskIndex();
    if (i >= 0 and i < map.size)
    #else
    for (int i = 0; i < map.size; i += 1)
    #endif
    {
        if(map[i] == var)
        {
            *repr = i;
        }
    }
}

cudaDevice int IntNeighborhood::getRepresentationIndex(int var)
{
    int* reprIdx;
    MemUtils::malloc(&reprIdx);
    *reprIdx = -1;
    #ifdef GPU
        Wrappers::getBinding<<<variablesBlocks, DEFAULT_BLOCK_SIZE>>>(this, var, reprIdx);
        cudaDeviceSynchronize();
    #else
        getBinding(var, reprIdx);
    #endif
    #ifndef NDEBUG
        assert(*reprIdx >= 0);
    #endif
    return *reprIdx;
}
