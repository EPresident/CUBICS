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
        DoUnassignment,
        VariablesUnassigned
    };

    int LNSState;
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
    /// LNS iterations done (i.e. variables unassignments)
    int iterationsDone;
    /// LNS iterations to do
    int maxIterations;
    
    /// Indicates the variables to unassign.
    Vector<int> chosenVariables;
    
    IntVariables* variables;
    IntConstraints* constraints;

    IntBacktrackSearcher BTSearcher;

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
    void initialize(FlatZinc::FlatZincModel* fzModel, double unassignRate,
                   int iterations);
    void deinitialize();

    /**
    * Find the next solution, unassigning variables when needed.
    * \return true if a solution is found, false otherwise.
    */
    cudaDevice bool getNextSolution();
    
    /**
    * Choose the variables to unassign.
    * Populates the "chosenVariables" vector by shuffling the variables and taking
    * the first n, where n is N° of variables times the unassignment rate (floored).
    * The shuffle is done using the Fisher-Yates/Knuth algorithm.
    */
    //cudaDevice void chooseVariables();
};
