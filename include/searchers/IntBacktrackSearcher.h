#pragma once

#include <variables/IntVariables.h>
#include <constraints/IntConstraints.h>
#include <choosers/IntVariablesChooser.h>
#include <choosers/IntValuesChooser.h>
#include <propagators/IntConstraintsPropagator.h>
#include <searchers/IntBacktrackStack.h>
#include <flatzinc/flatzinc.h>
#include <utils/TimerUtils.h>

struct IntBacktrackSearcher
{
    enum States
    {
        VariableNotChosen, ///< Initial state
        VariableChosen, ///< A variable has been chosen
        ValueChosen, ///< A value for the variable has been chosen
        SuccessfulPropagation, ///< Propagation successful after fixing a variable
        ValueChecked ///< Need to find another value for the same variable
    };

    int backtrackingState;
    int backtrackingLevel;

    /// Current variable to be assigned.
    int chosenVariable;
    /// Value to assign to the chosen variable.
    int chosenValue;
    
    /// Indicates the variable assigned on each backtrack level.
    Vector<int> chosenVariables;
    /** 
    * Indicates the value of the assignment on each backtrack level;
    * chosenValues[i] is the value assigned to chosenVariables[i].
    */
    Vector<int> chosenValues;
    
    /// Device-Host timer
    TimerUtils timer;

    IntVariables* variables;
    IntConstraints* constraints;

    IntBacktrackStack stack;

    IntVariablesChooser variablesChooser;
    IntValuesChooser valuesChooser;

    IntConstraintsPropagator propagator;


#ifdef GPU
    /// CUDA blocks needed to handle all the variables
    int varibalesBlockCount;
#endif

    enum SearchType
    {
        Satisfiability,
        Maximization,
        Minimization
    };

    int searchType;
    /// The variable to be optimized
    int optVariable;
    /// Constraint to be optimized (cost function)
    int optConstraint;

    void initialize(FlatZinc::FlatZincModel* fzModel);
    void deinitialize();

    /**
    * Find the next solution, backtracking when needed.
    * \return true if a solution is found, false otherwise.
    */
    cudaDevice bool getNextSolution(long long timeout = LONG_LONG_MAX);

    /**
    * Require that the optimization variable ("optVariable") take a value
    * greater/smaller than its minumum/maximum value (for a 
    * maximization/minimization problem).
    */
    cudaDevice void shrinkOptimizationBound();
};

