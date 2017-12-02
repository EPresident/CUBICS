#pragma once

#include <data_structures/Vector.h>
#include <data_structures/MonotonicIntVector.h>

struct IntDomainsActions
{
    Vector<Vector<int>> elementsToRemove;
    Vector<int> lowerbounds;
    Vector<int> upperbounds;

    MonotonicIntVector domainsWithActions;

    void initialize(int count);
    void deinitialize();

    void push();

    cudaDevice void clear(int index);

    cudaDevice void removeElement(int index, int val);
    cudaDevice void removeAnyLesserThan(int index, int val);
    cudaDevice void removeAnyGreaterThan(int index, int val);
};
