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

    cudaDevice bool getFirstValue(int variable, int* firstValue);
    cudaDevice bool getNextValue(int variable, int currentValue, int* nextValue);
};
