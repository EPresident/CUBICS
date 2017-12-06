#pragma once

#include <variables/IntVariables.h>
#include <constraints/IntConstraints.h>
#include <choosers/IntVariablesChooser.h>
#include <choosers/IntValuesChooser.h>
#include <propagators/IntConstraintsPropagator.h>
#include <searchers/IntBacktrackStack.h>
#include <flatzinc/flatzinc.h>
#include <statistics/Statistics.h>

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

    enum SearchType
    {
        Satisfiability,
        Maximization,
        Minimization
    };

    int searchType;
    int optVariable;
    int optConstraint;

    Statistics* stats;

    void initialize(FlatZinc::FlatZincModel* fzModel, Statistics* stats);
    void deinitialize();

    cudaDevice bool getNextSolution();

    cudaDevice void shrinkOptimizationBound();
};

