#pragma once

#include <data_structures/Vector.h>
#include <domains/IntDomains.h>

struct IntVariables
{
    int count;

    IntDomains domains;
    Vector<Vector<int>> constraints;

    void initialize(int count);
    void deinitialize();

    void push(int min, int max);
};
