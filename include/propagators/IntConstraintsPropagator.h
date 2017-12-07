#pragma once

#include <data_structures/Vector.h>
#include <variables/IntVariables.h>
#include <constraints/IntConstraints.h>

struct IntConstraintsPropagator
{
    IntVariables* variables;
    IntConstraints* constraints;

    /// Indicates if the i-th constraint has been propagated or not
    Vector<bool> constraintToPropagate;

    /// true if at least one domain has become empty
    bool someEmptyDomain;
    /// true if at least one constraint has to be (re)propagated
    bool someConstraintsToPropagate;
    bool allConstraintsSatisfied;

#ifdef GPU
    int constraintsBlockCountDivergence;
    int constraintsBlockCount;
    int variablesBlockCount;
#endif

    void initialize(IntVariables* variables, IntConstraints* constraints);
    void deinitialize();

    /**
    * Propagates all constraints.
    * \return true if no variable domains are made empty.
    */
    cudaDevice bool propagateConstraints();
    /**
    * Check if any constraint needs to be propagated, updating the appropriate
    * flags.
    * Propagation is required if any variable of a constraint has a "Changed" domain
    * event.
    * \see IntDomains
    */
    cudaDevice void setConstraintsToPropagate();
    /**
    * Propagates all constraints flagged in "constraintToPropagate", and flips
    * the respective flag.
    */
    cudaDevice void collectActions();
    /** 
    * Clears the domain events list.
    * \see IntDomains
    */
    cudaDevice void clearDomainsEvents();
    /**
    * Perform the domain reduction actions on all variables.
    * \see IntDomains
    */
    cudaDevice void updateDomains();
    /// Clears the "constraintToPropagate" vector.
    cudaHostDevice void clearConstraintsToPropagate();
    /// Checks if any domain has become empty, updating the "someEmptyDomain" flag.
    cudaDevice void checkEmptyDomains();

    /// \return true if all constraints are satisfied.
    cudaDevice bool verifyConstraints();
    /// Updates the "allConstraintsSatisfied" flag, scanning all constraints.
    cudaDevice void checkSatisfiedConstraints();
};
