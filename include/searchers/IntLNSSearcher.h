#pragma once

#include <searchers/IntBacktrackSearcher.h>
#include <random>
#include <utils/TimerUtils.h>
#include <data_structures/Lock.h>
#include <variables/IntNeighborhood.h>

/**
* Struct used to perform Large Neighborhood Search.
* Basically works like this: after a solution has been found, unassign n 
* variables at random and find the best/first improving solution from there 
* (using something like IntBacktrackSearcher). Repeat.
*
* \author Elia Calligaris
* \see IntBacktrackSearcher
*/
struct IntLNSSearcher
{
    enum States
    {        
        Initialized,
        VariableNotChosen,
        VariableChosen, ///< A variable has been chosen
        ValueChosen, ///< A value for the variable has been chosen
        SuccessfulPropagation, ///< Propagation successful after fixing a variable
        ValueChecked ///< Need to find another value for the same variable
    };

    Vector<int> LNSStates;
    /// PRNG seed
    unsigned int randSeed;
    /**
    * The percentage (between 0 and 1) of variables that will be unassigned.
    * This means that floor(unassignmentRate) variables will be randomly chosen
    * to have their assignment reverted.
    */
    double unassignmentRate;
    /// The number of variables that will be unassigned.
    int unassignAmount;
    
    /// Device-Host timers
    Vector<TimerUtils> timers;
    
    IntVariables* variables;
    IntConstraints* constraints;

    /// Domains after initial propagation (before solving start)
    IntDomainsRepresentations* originalDomains;
    /// Best solution found so far
    IntDomainsRepresentations* bestSolution;
    /// Mutex lock for \a bestSolution
    Lock bestSolLock;
    
    Vector<IntNeighborhood*> neighborhoods;
    
    #ifdef GPU
        /// CUDA blocks needed to handle all the variables
        int variablesBlockCount;
        /// CUDA blocks needed to handle all the variables (with divergence)
        int variablesBlockCountDivergence;
        /// CUDA blocks needed to handle all the neighbors
        int neighborsBlockCount;
    #endif
    
    //-----------------------------------------
    // Stuff for the backtrack search part
    //-----------------------------------------
    /// Indicates the variable assigned on each backtrack level.
    Vector<Vector<int>> chosenVariables;
    /** 
    * Indicates the value of the assignment on each backtrack level;
    * chosenValues[i] is the value assigned to chosenVariables[i].
    */
    Vector<Vector<int>> chosenValues;

    Vector<IntBacktrackStack*> stacks;

    IntValuesChooser valuesChooser;

    IntConstraintsPropagator propagator;

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
    
    /**
    * \brief Initialize the searcher.
    * \param unassignRate the percentage (between 0 and 1) of variables 
    * that will be unassigned.
    */
    void initialize(FlatZinc::FlatZincModel* fzModel, double unassignRate,
                   int numNeighborhoods, IntDomainsRepresentations* originalDomains);
    void deinitialize();

    /**
    * Find the next solution, unassigning variables when needed.
    * \return true if a solution is found, false otherwise.
    */
    cudaDevice bool getNextSolution(long long timeout);
    
    /**
    * Back up the best solution found so far by copying the 
    * current domain representation inside \a domainsBackup.
    * Beware that no optimality check are performed.
    */
    cudaDevice void saveBestSolution(IntNeighborhood* neighborhood);
    
    /**
    * Restore the best solution found so far, by overwriting the current
    * domains representation.
    */
    cudaDevice void restoreBestSolution();
};
