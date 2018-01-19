#pragma once

#include <random>
#include <utils/TimerUtils.h>
#include <searchers/IntBacktrackSearcher.h>
#include <choosers/IntValuesChooser.h>

/**
* Small Neighborhood Brute Searcher
* 
* This searcher will perform a Generate&Test-flavoured version of LNS:
* small numbers of variables are unassigned (e.g. 3-5) and brute force
* search is done on the search-space, without constraint propagation.
* The intention of this approach is exploiting the SIMT architecture
* of CUDA to obtain a speedup, since Generate&Test is simple.
*
* \author Elia Calligaris
* \see IntLNSSearcher
*/
struct IntSNBSearcher
{
    enum States
    {        
        Initialized,
        DoUnassignment,
        Generate,
        Test,
        NextCandidate,
        FreeOptVar
    };

    IntBacktrackSearcher BTSearcher;

    int SNBSState;
    /// PRNG seed
    unsigned int randSeed;
    /**
    * The amount of variables that will be unassigned.
    * Increasing this has an exponential impact on performance.
    */
    int unassignAmount;
    /// LNS iterations done (i.e. variables unassignments)
    int iterationsDone;
    /// LNS iterations to do
    int maxIterations;
    
    /// Device-Host timer
    TimerUtils timer;
    
    /// Indicates the variables to unassign.
    Vector<int> chosenVariables;
    /// Value to assign to the chosen variable.
    int chosenValue;
    /// Values assigned to the chosen variables.
    Vector<int> chosenValues;
    /**
    * Number of neighborhood (chosen) variables that have been 
    * (re)assigned. It's like backtrackingLevel for the backtracking
    * searcher.
    * 
    * \see IntBacktrackSearcher::backtrackingLevel
    */
    int neighVarsAssigned;
    
    IntVariables* variables;
    IntConstraints* constraints;
    
    IntValuesChooser valuesChooser;
    
    IntConstraintsPropagator propagator;

    /**
    * Here go the domains representation backups. The i-th 
    * representation is for the i-th variable, and has two "entries":
    * - the first is for the initial domain (before solving begins);
    * - the second is for the best solution found during the search.
    */
    Vector<IntDomainsRepresentations> domainsBackup;
    
    #ifndef GPU
        /// Mersenne Twister PRNG
        std::mt19937 mt_rand;
    #endif
    #ifdef GPU
        /// State for the cuRAND PRNG library (GPU)
        curandState* cuRANDstate;
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
    void initialize(FlatZinc::FlatZincModel* fzModel, int unassignAmount,
                   int iterations);
    void deinitialize();

    /**
    * Find the next solution, unassigning variables when needed.
    * \return true if a solution is found, false otherwise.
    */
    cudaDevice bool getNextSolution(long timeout);
    
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
