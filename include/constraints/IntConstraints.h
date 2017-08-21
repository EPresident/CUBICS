#pragma once

#include <data_structures/Vector.h>
#include <variables/IntVariables.h>

#include <constraints/IntLinNe.h>

struct IntConstraints
{
    enum Type
    {
        IntLinNe
    };

    int count;

    Vector<int> types;

    Vector<Vector<int>> variables;
    Vector<Vector<int>> parameters;

    void initialize();
    void deinitialize();

    void push(int type);

    void propagate(int index, IntVariables* variables);
    bool satisfied(int index, IntVariables* variables);
};
