#include <domains/IntDomains.h>
#include <cassert>

void IntDomains::initialize(int count)
{
    events.initialize(count);

    representations.initialize(count);
    actions.initialize(count);
}

void IntDomains::deinitialize()
{
    events.deinitialize();

    representations.deinitialize();
    actions.deinitialize();
}

/// Add a new domain (for a new variable), ranging from "min" to "max".
void IntDomains::push(int min, int max)
{
    events.push_back(EventTypes::Changed);

    representations.push(min, max);
    actions.push();
}

cudaDevice bool IntDomains::isEmpty(int index, IntNeighborhood* nbh)
{
    if(nbh->isNeighbor(index))
    {
        return nbh->neighRepr.isEmpty(nbh->getRepresentationIndex(index));
    }
    else
    {
        return IntDomains::isEmpty(index);
    }
}

cudaDevice bool IntDomains::isSingleton(int index, IntNeighborhood* nbh)
{
    if(nbh->isNeighbor(index))
    {
        return nbh->neighRepr.isSingleton(nbh->getRepresentationIndex(index));
    }
    else
    {
        return IntDomains::isSingleton(index);
    }
}

cudaDevice unsigned int IntDomains::getApproximateCardinality(int index, IntNeighborhood* nbh)
{
    if(nbh->isNeighbor(index))
    {
        return nbh->neighRepr.getApproximateCardinality(nbh->getRepresentationIndex(index));
    }
    else
    {
        return IntDomains::getApproximateCardinality(index);
    }
}

cudaDevice int IntDomains::getMin(int index, IntNeighborhood* nbh)
{
    if(nbh->isNeighbor(index))
    {
        return nbh->neighRepr.minimums[nbh->getRepresentationIndex(index)];
    }
    else
    {
        return IntDomains::getMin(index);
    }
}

cudaDevice int IntDomains::getMax(int index, IntNeighborhood* nbh)
{
    if(nbh->isNeighbor(index))
    {
        return nbh->neighRepr.maximums[nbh->getRepresentationIndex(index)];
    }
    else
    {
        return IntDomains::getMax(index);
    }
}

/// Reduce the domain on the "index"-th variable to "value" (singleton).
cudaDevice void IntDomains::fixValue(int index, int value)
{
    assert(representations.contain(index, value));

    representations.keepOnly(index, value);
    events[index] = EventTypes::Changed;
}

cudaDevice void IntDomains::fixValue(int index, int value, IntNeighborhood* nbh)
{
    #ifndef NDEBUG
    assert(nbh->isNeighbor(index));
    #endif
    nbh->neighRepr.keepOnly(nbh->getRepresentationIndex(index), value);
    events[index] = EventTypes::Changed;  // FIXME? !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

}

/**
* Perform the domain reduction actions pertaining the "index"-th 
* variable/domain.
* Values outside the bounds and inside the "remove list" are dropped.
*/
cudaDevice void IntDomains::updateDomain(int index)
{
    unsigned int previousVersion = representations.versions[index];

    // Shave off any value outside the bounds
    representations.removeAnyGreaterThan(index, actions.upperbounds[index]);
    representations.removeAnyLesserThan(index, actions.lowerbounds[index]);

    // Remove single elements
    for (int ei = 0; ei < actions.elementsToRemove[index].size; ei += 1)
    {
        representations.remove(index, actions.elementsToRemove[index][ei]);
    }

    // Remove the action after it's been perforned
    actions.clear(index);

    // Push the appropriate events
    if (previousVersion != representations.versions[index])
    {
        events[index] = EventTypes::Changed;
    }
    else
    {
        events[index] = EventTypes::None;
    }
}

cudaDevice void IntDomains::updateDomain(int index, IntNeighborhood* nbh)
{
    #ifndef NDEBUG
    assert(nbh->isNeighbor(index));
    #endif
    
    int ridx {nbh->getRepresentationIndex(index)};
    
    unsigned int previousVersion = nbh->neighRepr.versions[ridx];

    // Shave off any value outside the bounds
    nbh->neighRepr.removeAnyGreaterThan(ridx, actions.upperbounds[index]);
    nbh->neighRepr.removeAnyLesserThan(ridx, actions.lowerbounds[index]);

    // Remove single elements
    for (int ei = 0; ei < actions.elementsToRemove[index].size; ei += 1)
    {
        nbh->neighRepr.remove(ridx, actions.elementsToRemove[index][ei]);
    }

    // Remove the action after it's been perforned
    actions.clear(index);

    // Push the appropriate events
    if (previousVersion != nbh->neighRepr.versions[ridx])
    {
        events[index] = EventTypes::Changed;
    }
    else
    {
        events[index] = EventTypes::None;
    }
}
