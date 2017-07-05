#pragma once

#include <variables/IntVariables.h>

struct IntValuesChooser
{
    enum Type
    {
        InOrder
    };

    int type;
    IntVariables* variables;

    void initialzie(int type, IntVariables* variables);

    bool getFirstValue(int variable, int* firstValue);
    bool getNextValue(int variable, int currentValue, int* nextValue);
};
