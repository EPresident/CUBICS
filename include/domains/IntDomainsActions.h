#pragma once

#include <data_structures/Vector.h>
#include <data_structures/Lock.h>

struct IntDomainsActions
{
	Vector<bool> changedDomainsMask;
	Vector<int> changedDomains;

    Vector<Vector<int>> elementsToRemove;
    Vector<int> lowerbounds;
    Vector<int> upperbounds;
#ifdef GPU
    Vector<Lock> locks;
#endif

    void initialize(int count);
    void deinitialize();

    void push();

    cudaDevice void clearActions();
    cudaDevice void clearChangedDomains();
    cudaDevice void clearAll();

    cudaDevice void removeElement(int index, int val);
    cudaDevice void removeAnyLesserThan(int index, int val);
    cudaDevice void removeAnyGreaterThan(int index, int val);
    cudaDevice void keepOnly(int index, int val);
};
