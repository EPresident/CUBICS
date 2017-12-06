#include <algorithm>

#include <domains/IntDomainsActions.h>
#include <utils/Utils.h>

void IntDomainsActions::initialize(int count)
{
    elementsToRemove.initialize(count);

    lowerbounds.initialize(count);
    upperbounds.initialize(count);


    domainsWithActions.initialize(count);

#ifdef GPU
    locks.initialize(count);
#endif
}

void IntDomainsActions::deinitialize()
{
    for (int i = 0; i < elementsToRemove.size; i += 1)
    {
        elementsToRemove[i].deinitialize();
    }
    elementsToRemove.deinitialize();

    lowerbounds.deinitialize();
    upperbounds.deinitialize();


    domainsWithActions.deinitialize();

#ifdef GPU
    locks.deinitialize();
#endif
}

void IntDomainsActions::push()
{
    elementsToRemove.resize_by_one();
    elementsToRemove.back().initialize();

    lowerbounds.push_back(INT_MIN);
    upperbounds.push_back(INT_MAX);


    domainsWithActions.reinitialize(elementsToRemove.size);

#ifdef GPU
    locks.resize_by_one();
    locks.back().initialize();
#endif
}

cudaDevice void IntDomainsActions::clear(int index)
{
    elementsToRemove[index].clear();

    lowerbounds[index] = INT_MIN;
    upperbounds[index] = INT_MAX;
}

cudaDevice void IntDomainsActions::removeElement(int index, int val)
{
    if (lowerbounds[index] <= val and val <= upperbounds[index])
    {
#ifdef GPU
        locks[index].lock();
#endif
        elementsToRemove[index].push_back(val);
#ifdef GPU
        locks[index].unlock();
#endif
    }

    domainsWithActions.add(index);
}

cudaDevice void IntDomainsActions::removeAnyGreaterThan(int index, int val)
{
#ifdef GPU
    __threadfence();
    atomicMin(&upperbounds[index], val);
#else
    upperbounds[index] = std::min(val, upperbounds[index]);
#endif

    domainsWithActions.add(index);
}

cudaDevice void IntDomainsActions::removeAnyLesserThan(int index, int val)
{
#ifdef GPU
    __threadfence();
    atomicMax(&lowerbounds[index], val);
#else
    lowerbounds[index] = std::max(val, lowerbounds[index]);
#endif

    domainsWithActions.add(index);
}
