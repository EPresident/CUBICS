#pragma once

#include <data_structures/Vector.h>
#include <variables/IntVariables.h>

struct IntVariablesChooser
{
    enum Type
    {
    };

    int type;

    IntVariables* variables;
    Vector<int>* backtrackState;

    void initialzie(int type, IntVariables* variables, Vector<int>* backtrackState);

    bool getVariable(int backtrackLevel, int* variable);
};
