#pragma once

#include <data_structures/Vector.h>

struct IntDomainsActions
{
    Vector<Vector<int>> elementsToRemove;
    Vector<int> lowerbounds;
    Vector<int> upperbounds;

    void initialize(int count);
    void deinitialize();

    void push();

    void clear(int index);

    void removeElement(int index, int val);
    void removeAnyLesserThan(int index, int val);
    void removeAnyGreaterThan(int index, int val);
};
