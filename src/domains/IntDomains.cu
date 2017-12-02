#include <domains/IntDomains.h>

void IntDomains::initialize(int count)
{
    events.initialize(count);

    representations.initialize(count);
    actions.initialize(count);

    changes.initialize(count);
}

void IntDomains::deinitialize()
{
    events.deinitialize();

    representations.deinitialize();
    actions.deinitialize();

    changes.deinitialize();
}

void IntDomains::push(int min, int max)
{
    events.push_back(EventTypes::Changed);

    changes.reinitialize(events.size);

    representations.push(min, max);
    actions.push();
}

cudaDevice void IntDomains::fixValue(int index, int value)
{
    assert(representations.contain(index, value));

    actions.removeAnyGreaterThan(index, value);
    actions.removeAnyLesserThan(index, value);

    update(index);
}

cudaDevice void IntDomains::update(int index)
{
    unsigned int previousVersion = representations.versions[index];

    representations.removeAnyGreaterThan(index, actions.upperbounds[index]);

    representations.removeAnyLesserThan(index, actions.lowerbounds[index]);

    for (int ei = 0; ei < actions.elementsToRemove[index].size; ei += 1)
    {
        representations.remove(index, actions.elementsToRemove[index][ei]);
    }

    actions.clear(index);

    if (previousVersion != representations.versions[index])
    {
        changes.add(index);
        events[index] = EventTypes::Changed;
    }
}
