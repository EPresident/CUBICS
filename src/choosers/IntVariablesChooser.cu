#include <choosers/IntVariablesChooser.h>

#include <utils/Utils.h>
#include <choosers/IntInOrderVariablesChooser.h>
#include <choosers/IntRestrictInOrdVarChooser.h>

void IntVariablesChooser::initialzie(int type, IntVariables* variables, Vector<int>* backtrackState)
{
    this->type = type;
    this->variables = variables;
    this->backtrackState = backtrackState;
}

/** 
* Get a reference to "variable" at the desired backtrack level.
* \return true if successful.
*/ 
cudaDevice bool IntVariablesChooser::getVariable(int backtrackLevel, int* variable)
{
    switch (type)
    {
        case InOrder:
            return IntInOrderVariablesChooser::getVariable(this, backtrackLevel, variable);
        case RestrictedInOrder:
            return IntRestrictInOrdVarChooser::getVariable(this, backtrackLevel, variable);
        default:
            LogUtils::error(__PRETTY_FUNCTION__, "Invalid variable chooser type");
            return false;
    }
}
