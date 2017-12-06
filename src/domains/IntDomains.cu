#include <domains/IntDomains.h>

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

/// Reduce the domain on the "index"-th variable to "value" (singleton).
cudaDevice void IntDomains::fixValue(int index, int value)
{
    assert(representations.contain(index, value));

    representations.keepOnly(index, value);
    events[index] = EventTypes::Changed;
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
