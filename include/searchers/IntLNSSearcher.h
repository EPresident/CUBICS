#pragma once

#include <searchers/IntBacktrackSearcher.h>

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
    /**
    * Here go the domains representation backups. The i-th 
    * representation is for the i-th variable, and has two "entries":
    * - the first is for the initial domain (before solving begins);
    * - the second is for the best solution found during the search.
    */
    Vector<IntDomainsRepresentations> domainsBackup;
    
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
    * Back up the initial domains by copying the current domain 
    * representation inside \a domainsBackup.
    */
    cudaDevice void backupInitialDomains();
    
    /**
    * Back up the best solution found so far by copying the 
    * current domain representation inside \a domainsBackup.
    * Beware that no optimality check are performed.
    */
    cudaDevice void saveBestSolution();
    
    /**
    * Restore the best solution found so far, by overwriting the current
    * domains representation.
    */
    cudaDevice void restoreBestSolution();
};
