#pragma once

#include <searchers/IntBacktrackSearcher.h>


/**
* Struct used to perform Large Neighborhood Search.
* Basically works like this: after a solution has been found, unassign n 
* variables at random and find the best solution from there (using 
* something like IntBacktrackSearcher).
*
* \author Elia
* \see IntBacktrackSearcher
*/
struct IntLNSSearcher
{
    enum States
    {        
        Initialized,
        FirstSolutionFound,
        VariablesUnassigned,
        /// Subtree explored after the unassign
        NeighborhoodExplored
    };

    int LNSState;
    /**
    * The percentage (between 0 and 1) of variables that will be unassigned.
    * This means that floor(unassignmentRate) variables will be randomly chosen
    * to have their assignment reverted.
    */
    double unassignmentRate;
    /// LNS iterations done (i.e. variables unassignments)
    int iterationsDone;
    
    /// Indicates the variables to unassign.
    Vector<int> chosenVariables;
    
    IntVariables* variables;
    IntConstraints* constraints;

    IntBacktrackSearcher BTSearcher;

    IntVariablesChooser variablesChooser;



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
    
    /**
    * \brief Initialize the searcher.
    * \param unassignRate the percentage (between 0 and 1) of variables 
    * that will be unassigned.
    */
    void initialize(FlatZinc::FlatZincModel* fzModel, double unassignRate);
    void deinitialize();

    /**
    * Find the next solution, unassigning variables when needed.
    * \return true if a solution is found, false otherwise.
    */
    cudaDevice bool getNextSolution();
};
