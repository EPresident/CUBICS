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

    /**
    * Get the first value (following the chooser's criteria) for a variable.
    * \return true if successful.
    */
    cudaDevice bool getFirstValue(int variable, int* firstValue);
    /**
    * Get the next value (following the chooser's criteria) for a variable.
    * \return true if successful.
    */
    cudaDevice bool getNextValue(int variable, int currentValue, int* nextValue);
};
