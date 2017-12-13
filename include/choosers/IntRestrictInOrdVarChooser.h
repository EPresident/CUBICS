#pragma once

#include <choosers/IntVariablesChooser.h>

/**
* \brief In-order variable chooser, selecting from a restricted pool of
* integer variables.
*
* \author Elia
*/
namespace IntRestrictInOrdVarChooser
{   
    cudaDevice bool getVariable(IntVariablesChooser* variablesChooser,
                                int backtrackLevel, int* variable);
};
