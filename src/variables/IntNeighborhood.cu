#include <variables/IntNeighborhood.h>
#include <utils/KernelUtils.h>

cudaDevice void IntNeighborhood::initialize(int count/*, IntDomains* dom*/)
{
   this->count = count;
   //domains = dom;
   // Init bitvector
   neighMask.initialize(count, false);
   // Init domains representations
   neighRepr.initialize(count);
   // Init map
   map.initialize(count);
   map.resize(count);
    // get required blocks
    #ifdef GPU
    blocksRequired = KernelUtils::getBlockCount(count, DEFAULT_BLOCK_SIZE, true);
    #endif
}

cudaDevice void IntNeighborhood::deinitialize()
{
    //domains.deinitialize();
    neighMask.deinitialize();
    neighRepr.deinitialize();
    for(int i = 0; i < map.size; i += 1)
    {
        map[i].deinitialize();
    }
    map.deinitialize();
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
