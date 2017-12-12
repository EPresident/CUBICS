#pragma once

#include <choosers/IntVariablesChooser.h>

/**
* \brief In-order variable chooser, selecting from a restricted pool of
* integer variables.
*
* \author Elia
*/
struct IntRestrictInOrdVarChooser
{
    /** 
    * Pools of variables to select from.
    * If it's empty, then all variables are allowed.
    * Must be populated from wherever the IntVariablesChooser is 
    * created and used (e.g. the searcher).
    */
    Vector<int> variables;
    
    /**
    * Initialize the struct for a given expected variables pool size.
    * Defaults to -1 (i.e. unknown). 
    */
    void initialize(int expectedVarCount = -1);
    void deinitialize();
    cudaDevice bool getVariable(IntVariablesChooser* variablesChooser,
                                int backtrackLevel, int* variable);
    /**
    * Add a variable to the "choosable" variables pool.
    * \param variable the index of the variable to be added.
    */
    cudaDevice void addVariable(int variable);
}
