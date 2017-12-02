#pragma once

#include <data_structures/Vector.h>
#include <variables/IntVariables.h>

#include <constraints/IntLinNe.h>
#include <constraints/IntLinLe.h>
#include <constraints/IntOptLb.h>
#include <constraints/IntOptUb.h>
#include <constraints/IntLinEq.h>

struct IntConstraints
{
    enum Type
    {
        IntLinNe,
        IntLinLe,
        IntOptLb,
        IntOptUb,
        IntLinEq
    };

    int count;

    Vector<int> types;

    Vector<Vector<int>> variables;
    Vector<Vector<int>> parameters;

    void initialize();
    void deinitialize();

    void push(int type);

    cudaDevice void propagate(int index, IntVariables* variables);
    cudaDevice bool satisfied(int index, IntVariables* variables);
};
