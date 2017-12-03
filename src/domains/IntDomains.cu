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

void IntDomains::push(int min, int max)
{
    events.push_back(EventTypes::Initialized);

    representations.push(min, max);
    actions.push();
}

cudaDevice void IntDomains::fixValue(int index, int value)
{
    assert(representations.contain(index, value));

    representations.keepOnly(index, value);

    setEvent(index, ValueRemoved);
    setEvent(index, Istantiated);
}

cudaDevice void IntDomains::setEvent(int index, unsigned int event)
 {
     switch(event)
     {
         case ValueRemoved:
             events[index] |= ValueRemoved;
             break;
         case IncreasedMinimum:
             events[index] |= IncreasedMinimum;
             break;
         case DecreasedMaximums:
             events[index] |= DecreasedMaximums;
             break;
         case Istantiated:
             events[index] |= Istantiated;
             break;
         default:
             LogUtils::error(__PRETTY_FUNCTION__, "Invalid event type");
     }
 }

cudaDevice void IntDomains::updateDomain(int index)
{
    unsigned int previousVersion = representations.versions[index];
    int previousMin = getMin(index);
    int previousMax = getMax(index);

    representations.removeAnyGreaterThan(index, actions.upperbounds[index]);

    representations.removeAnyLesserThan(index, actions.lowerbounds[index]);

    for (int ei = 0; ei < actions.elementsToRemove[index].size; ei += 1)
    {
        representations.remove(index, actions.elementsToRemove[index][ei]);
    }

    actions.clear(index);

    if (previousVersion != representations.versions[index])
    {
        setEvent(index, ValueRemoved);

        if (previousMin < getMin(index))
        {
            setEvent(index, IncreasedMinimum);
        }

        if (getMax(index) < previousMax)
        {
            setEvent(index, DecreasedMaximums);
        }

        if (isSingleton(index))
        {
            setEvent(index, Istantiated);
        }
    }
}
