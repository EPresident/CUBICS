#pragma once

#include <data_structures/Vector.h>
#include <variables/IntVariables.h>

struct IntVariablesChooser
{
    enum Type
    {
        InOrder,
        RestrictedInOrder
    };

    int type;

    IntVariables* variables;
    Vector<int>* backtrackState;

    void initialzie(int type, IntVariables* variables, Vector<int>* backtrackState);

    /** 
    * Get a reference to "variable" at the desired backtrack level.
    * \return true if successful.
    */  
    cudaDevice bool getVariable(int backtrackLevel, int* variable);
};
