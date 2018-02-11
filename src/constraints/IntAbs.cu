#include <cmath>
#include <algorithm>

#include <constraints/IntAbs.h>
#include <data_structures/Vector.h>
#include <constraints/IntConstraints.h>
/**
* \file Integer absolute value constraint.
* |a| = b      a,b are variables.
*/

/** 
* Enforce bounds-consistency for this constraint.
*/
cudaDevice void IntAbs::propagate(IntConstraints* constraints, int index, IntVariables* variables)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    //IntDomainsRepresentations* intDomRepr  = &variables->domains.representations;
    
    // 0 is the index of a, 1 is the index of b.
    int indexA = constraintVariables->at(0);
    int indexB = constraintVariables->at(1);
    
    // First off try to cut as much of the domains as possible;
    // This is achieved by looking at max/min values
    // ~ bounds consistency
    int maxA = variables->domains.getMax(indexA);
    int maxB = variables->domains.getMax(indexB);
    int minA = variables->domains.getMin(indexA);
    // b has to be positive, of course. |a| is always positive.
    if (variables->domains.getMin(indexB) < 0)
    {
        variables->domains.actions.removeAnyLesserThan(indexB,0);
    }
    // Check B to A
    // A's domain can be at most [-maxB,maxB]
    // (These below are domains, as an example)
    // A |----------0-----------|
    // B            0-------|
    // A'   |-------0-------|
    if( maxA > maxB )
    {
        variables->domains.actions.removeAnyGreaterThan(indexA,maxB);
        maxA = maxB;
    }
    if( minA < -maxB )
    {
        variables->domains.actions.removeAnyLesserThan(indexA,-maxB);
        minA = -maxB;
    }
    // Check A to B (is this really needed?)
    // B's domain can be at most [-n,n] where n = max{|minA|,|maxA|}
    // A        |---0--------|
    // B            0----------------|
    // B'           0--------|
    // or
    // A     |------0--|
    // B            0----------------|
    // B'           0------|
    #ifdef GPU
        int boundB; 
        if(abs(minA) < abs(maxA))
        {
           boundB = maxA;
        }
        else
        {
            boundB = minA;
        }
    #else
        int boundB {std::max(abs(minA),abs(maxA))};
    #endif
    
    if( maxB > boundB )
    {
        variables->domains.actions.removeAnyGreaterThan(indexB,boundB);
        maxB = boundB;
    }
    
}
cudaDevice void IntAbs::propagate(IntConstraints* constraints, int index, IntVariables* variables, IntNeighborhood* nbh)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    
    // 0 is the index of a, 1 is the index of b.
    int indexA = constraintVariables->at(0);
    int indexB = constraintVariables->at(1);
    // Are the variables in the neighborhood?
    bool neighA = nbh->isNeighbor(indexA);
    bool neighB = nbh->isNeighbor(indexB);
    // neighborhood representation indices
    int ridxA {-1};
    int ridxB {-1};
    if(neighA) ridxA = nbh->getRepresentationIndex(indexA);
    if(neighB) ridxB = nbh->getRepresentationIndex(indexB);
    
    // First off try to cut as much of the domains as possible;
    // This is achieved by looking at max/min values
    // ~ bounds consistency
    int maxA = variables->domains.getMax(indexA, nbh, ridxA);
    int maxB = variables->domains.getMax(indexB, nbh, ridxB);
    int minA = variables->domains.getMin(indexA, nbh, ridxA);
    // b has to be positive, of course. |a| is always positive.
    if (variables->domains.getMin(indexB, nbh, ridxB) < 0)
    {
        assert(neighB);
        nbh->neighActions.removeAnyLesserThan(ridxB, 0);
    }
    // Check B to A
    if( maxA > maxB )
    {
        assert(neighA);
        nbh->neighActions.removeAnyGreaterThan(ridxA,maxB);
        maxA = maxB;
    }
    if( minA < -maxB )
    {
        assert(neighA);
        nbh->neighActions.removeAnyLesserThan(indexA,-maxB);
        minA = -maxB;
    }
    // Check A to B
    #ifdef GPU
        int boundB; 
        if(abs(minA) < abs(maxA))
        {
           boundB = maxA;
        }
        else
        {
            boundB = minA;
        }
    #else
        int boundB {std::max(abs(minA),abs(maxA))};
    #endif
    
    if( maxB > boundB )
    {
        assert(neighB);
        nbh->neighActions.removeAnyGreaterThan(ridxB,boundB);
        maxB = boundB;
    }
    
}

cudaDevice bool IntAbs::satisfied(IntConstraints* constraints, int index, IntVariables* variables)
{
    //Satisfaction check is performed only when all variables are ground
    Vector<int>* constraintVariables = &constraints->variables[index];
    
    for (int i = 0; i < constraintVariables->size; i += 1)
    {
        if (not variables->domains.isSingleton(constraintVariables->at(i)))
        {
            return true;
        }
    }

    int a {variables->domains.getMin(constraintVariables->at(0))};
    int b {variables->domains.getMin(constraintVariables->at(1))};


    return (a < 0 && b == - a ) || (a >= 0 && b == a);
}
cudaDevice bool IntAbs::satisfied(IntConstraints* constraints, int index, IntVariables* variables, IntNeighborhood* nbh)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    // 0 is the index of a, 1 is the index of b.
    int& indexA = constraintVariables->at(0);
    int& indexB = constraintVariables->at(1);
    // Are the variables in the neighborhood?
    bool neighA = nbh->isNeighbor(indexA);
    bool neighB = nbh->isNeighbor(indexB);
    // neighborhood representation indices
    int ridxA {-1};
    int ridxB {-1};
    if(neighA) ridxA = nbh->getRepresentationIndex(indexA);
    if(neighB) ridxB = nbh->getRepresentationIndex(indexB);
    //Satisfaction check is performed only when all variables are ground
    if ( not variables->domains.isSingleton(indexA, nbh, ridxA) or
         not variables->domains.isSingleton(indexB, nbh, ridxB) )
    {
        return true;
    }

    int a {variables->domains.getMin(indexA, nbh, ridxA)};
    int b {variables->domains.getMin(indexB, nbh, ridxB)};


    return (a < 0 && b == - a ) || (a >= 0 && b == a);
}
