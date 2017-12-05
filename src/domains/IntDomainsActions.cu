#include <algorithm>

#include <domains/IntDomainsActions.h>
#include <utils/Utils.h>

void IntDomainsActions::initialize(int count)
{
	changedDomainsMask.initialize(count);
	changedDomainsMask.resize(count);
	for(int vi = 0; vi < changedDomainsMask.size; vi += 1)
	{
		changedDomainsMask[vi] = false;
	}

	changedDomains.initialize(count);;

    elementsToRemove.initialize(count);

    lowerbounds.initialize(count);
    upperbounds.initialize(count);

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

#ifdef GPU
    locks.deinitialize();
#endif

    changedDomainsMask.deinitialize();
    changedDomains.deinitialize();
}

void IntDomainsActions::push()
{
	changedDomainsMask.resize_by_one();
	changedDomainsMask.back() = false;

    elementsToRemove.resize_by_one();
    elementsToRemove.back().initialize();

    lowerbounds.push_back(INT_MIN);
    upperbounds.push_back(INT_MAX);

#ifdef GPU
    locks.resize_by_one();
    locks.back().initialize();
#endif
}

cudaDevice void IntDomainsActions::clearActions()
{
	for(int vi = 0; vi < changedDomains.size; vi += 1)
	{
		int index = changedDomains[vi];

		elementsToRemove[index].clear();

		lowerbounds[index] = INT_MIN;
		upperbounds[index] = INT_MAX;
	}
}

cudaDevice void IntDomainsActions::clearChangedDomains()
{
	for(int vi = 0; vi < changedDomains.size; vi += 1)
	{
		int index = changedDomains[vi];
		changedDomainsMask[index] = false;
	}

	changedDomains.clear();
}

cudaDevice void IntDomainsActions::clearAll()
{
	clearActions();
	clearChangedDomains();
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
        if(not changedDomainsMask[index])
        {
        	changedDomainsMask[index] = true;
        	changedDomains.push_back(index);
        }
    }
}

cudaDevice void IntDomainsActions::removeAnyGreaterThan(int index, int val)
{
#ifdef GPU
    __threadfence();
    atomicMin(&upperbounds[index], val);
#else
    upperbounds[index] = std::min(val, upperbounds[index]);
#endif
    if(not changedDomainsMask[index])
    {
    	changedDomainsMask[index] = true;
    	changedDomains.push_back(index);
    }
}

cudaDevice void IntDomainsActions::removeAnyLesserThan(int index, int val)
{
#ifdef GPU
    __threadfence();
    atomicMax(&lowerbounds[index], val);
#else
    lowerbounds[index] = std::max(val, lowerbounds[index]);
#endif
    if(not changedDomainsMask[index])
    {
    	changedDomainsMask[index] = true;
    	changedDomains.push_back(index);
    }
}

cudaDevice void IntDomainsActions::keepOnly(int index, int val)
{
	removeAnyGreaterThan(index, val);
	removeAnyLesserThan(index, val);
}
