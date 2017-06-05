#include <choosers/IntVariablesChooser.h>
#include <utils/Utils.h>

void IntVariablesChooser::initialzie(int type, IntVariables* variables, Vector<int>* backtrackState)
{
    this->type = type;
    this->variables = variables;
    this->backtrackState = backtrackState;
}

bool IntVariablesChooser::getVariable(int backtrackLevel, int* variable)
{
    switch (type)
    {
        default:
            LogUtils::error(__PRETTY_FUNCTION__, "Invalid variable chooser type");
            return false;
    }
}
