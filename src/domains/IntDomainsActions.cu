#include <algorithm>

#include <domains/IntDomainsActions.h>
#include <utils/Utils.h>

void IntDomainsActions::initialize(int count)
{
    elementsToRemove.initialize(count);

    lowerbounds.initialize(count);
    upperbounds.initialize(count);
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
}

void IntDomainsActions::push()
{
    elementsToRemove.resize_by_one();
    elementsToRemove.back().initialize();

    lowerbounds.push_back(INT_MIN);
    upperbounds.push_back(INT_MAX);
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
        elementsToRemove[index].push_back(val);
    }
}

cudaDevice void IntDomainsActions::removeAnyGreaterThan(int index, int val)
{
    upperbounds[index] = std::min(val, upperbounds[index]);
}

cudaDevice void IntDomainsActions::removeAnyLesserThan(int index, int val)
{
    lowerbounds[index] = std::max(val, lowerbounds[index]);
}
