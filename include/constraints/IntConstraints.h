#pragma once

#include <data_structures/Vector.h>

struct IntConstraints
{
    int count;

    Vector<Vector<int>> variables;
    Vector<Vector<int>> parameters;
};
