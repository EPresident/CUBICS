#pragma once

#include <variables/IntVariables.h>
#include <constraints/IntConstraints.h>
#include <choosers/IntVariablesChooser.h>
#include <choosers/IntValuesChooser.h>
#include <propagators/IntConstraintsPropagator.h>
#include <searchers/IntBacktrackStack.h>

struct IntBacktrackSearcher
{
    enum States
    {
        VariableNotChosen,
        VariableChosen,
        ValueChosen,
        SuccessfulPropagation,
        ValueChecked
    };

    int backtrackingState;
    int backtrackingLevel;

    int chosenVariable;
    int chosenValue;
    Vector<int> chosenVariables;
    Vector<int> chosenValues;

    IntVariables* variables;
    IntConstraints* constraints;

    IntBacktrackStack stack;

    IntVariablesChooser variablesChooser;
    IntValuesChooser valuesChooser;

    IntConstraintsPropagator propagator;

#ifdef GPU
    int varibalesBlockCount;
#endif

    void initialize(IntVariables* variables, IntConstraints* constraints);
    void deinitialize();

    cudaDevice bool getNextSolution();
};

