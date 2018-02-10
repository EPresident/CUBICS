#include <algorithm>

#include <domains/IntDomainsActions.h>
#include <utils/Utils.h>

/// Initialize the struct for "count" variables.
cudaHostDevice void IntDomainsActions::initialize(int count)
{
    elementsToRemove.initialize(count);

    lowerbounds.initialize(count);
    upperbounds.initialize(count);

#ifdef GPU
    locks.initialize(count);
#endif
}

cudaHostDevice void IntDomainsActions::deinitialize()
{
    for (int i = 0; i < elementsToRemove.size; i += 1)
    {
        elementsToRemove[i].deinitialize();
    }
    elementsToRemove.deinitialize();

    lowerbounds.deinitialize();
    upperbounds.deinitialize();

#ifdef GPU
    locks.deinitialize();
#endif
}

/** 
* Add a new empty action, i.e. add room for a new element in 
* the vectors of the struct. 
*/
cudaHostDevice void IntDomainsActions::push()
{
    elementsToRemove.resize_by_one();
    elementsToRemove.back().initialize();

    lowerbounds.push_back(INT_MIN);
    upperbounds.push_back(INT_MAX);

#ifdef GPU
    locks.resize_by_one();
    locks.back().initialize();
#endif
}

/** 
* Clear the "index"-th action.
* This is typically called after the action is performed by IntDomains.
*/
cudaDevice void IntDomainsActions::clear(int index)
{
    elementsToRemove[index].clear();

    lowerbounds[index] = INT_MIN;
    upperbounds[index] = INT_MAX;
}

/** 
* Queue the action to remove the "val" value from the "index"-th domain.
* "val" must be within the bounds, and the operation is mutually exclusive
* on GPU.
* This is done by adding "val" to elementsToRemove["index"].
*/
cudaDevice void IntDomainsActions::removeElement(int index, int val)
{
    // Check bounds
    if (lowerbounds[index] <= val and val <= upperbounds[index])
    {
#ifdef GPU
        locks[index].lock(); // acquire lock, must be executed only once
#endif
        elementsToRemove[index].push_back(val);
#ifdef GPU
        locks[index].unlock();
#endif
    }
}

/** 
* Queue the action to remove the any value greater than "val" from 
* the "index"-th domain.
* "val" must be within the bounds.
* This is done by setting upperbounds["index"] to "val".
*/
cudaDevice void IntDomainsActions::removeAnyGreaterThan(int index, int val)
{
#ifdef GPU
    __threadfence();
    atomicMin(&upperbounds[index], val);
#else
    upperbounds[index] = std::min(val, upperbounds[index]);
#endif
}

/** 
* Queue the action to remove the any value smaller than "val" from 
* the "index"-th domain.
* "val" must be within the bounds.
* This is done by setting lowerbounds["index"] to "val".
*/
cudaDevice void IntDomainsActions::removeAnyLesserThan(int index, int val)
{
#ifdef GPU
    __threadfence();
    atomicMax(&lowerbounds[index], val);
#else
    lowerbounds[index] = std::max(val, lowerbounds[index]);
#endif
}
