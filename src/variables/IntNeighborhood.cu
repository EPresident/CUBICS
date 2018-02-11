#include <variables/IntNeighborhood.h>
#include <utils/Utils.h>
#include <cassert>
#include <wrappers/Wrappers.h>

cudaDevice void IntNeighborhood::initialize(int count)
{
    this->count = count;
    // Init bitvector
    neighMask.initialize(count, false);
    // Init domains representations
    neighRepr.initialize(count);
    // Init map
    map.initialize(count);
    map.resize(count);
    // Init events
    events.initialize(count);
    // init actions
    neighActions.initialize(count);
    // get required blocks
    #ifdef GPU
    variablesBlocks = KernelUtils::getBlockCount(count, DEFAULT_BLOCK_SIZE, true);
    #endif
}

cudaDevice void IntNeighborhood::deinitialize()
{
    neighMask.deinitialize();
    neighRepr.deinitialize();
    for(int i = 0; i < map.size; i += 1)
    {
        map[i].deinitialize();
    }
    map.deinitialize();
    events.deinitialize();
    neighActions.deinitialize();
}

cudaDevice void IntNeighborhood::pushNeighbors(Vector<int>* neighbors, IntDomainsRepresentations* originalRepr)
{
    #ifdef GPU
    int i = KernelUtils::getTaskIndex();
    if (i >= 0 and i < neighbors->size)
    #else
    for (int i = 0; i < neighbors->size; i += 1)
    #endif
    {
        int var = neighbors->at(i);
        // Update mask
        neighMask.set(var);
        // Update variable-representation map
        int idx = neighRepr.minimums.size;
        Vector<int> bind;
        bind.initialize(2);
        bind.push_back(var);
        bind.push_back(idx);
        map.push_back(bind);
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

cudaDevice void IntNeighborhood::getBinding(int var, int* repr)
{
    #ifdef GPU
    int i = KernelUtils::getTaskIndex();
    if (i >= 0 and i < map.size)
    #else
    for (int i = 0; i < map.size; i += 1)
    #endif
    {
        if(map[i][0] == var)
        {
            *repr = map[i][1];
        }
    }
}

cudaDevice int IntNeighborhood::getRepresentationIndex(int var)
{
    int* reprIdx;
    MemUtils::malloc(&reprIdx);
    *reprIdx = -1;
    Wrappers::getBinding<<<variablesBlocks, DEFAULT_BLOCK_SIZE>>>(this, var, reprIdx);
    #ifndef NDEBUG
    assert(*reprIdx >= 0);
    #endif
    return *reprIdx;
}
