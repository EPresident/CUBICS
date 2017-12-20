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
    int boundB {std::max(abs(minA),abs(maxA))};
    if( maxB > boundB )
    {
        variables->domains.actions.removeAnyGreaterThan(indexB,boundB);
        maxB = boundB;
    }

    // Now remove single elements from the domains (more costly)
    // Check support for values of b
    // Easy: for each value of b we need to check two of a.
    /*int b = variables->domains.getMin(indexB);
    do
    {
        if( 
            ! intDomRepr->contain(indexA,b) &&
            ! intDomRepr->contain(indexA,-b) 
          )
        {
            variables->domains.actions.removeElement(indexB,b);
        }
        
    } while (intDomRepr->getNextValue(indexB,b,&b));
      
    // Check support for values of a
    // Even easier: for each val of a check one of b. 
    int a = variables->domains.getMin(indexA);
    do
    {
        if( 
            ( a < 0 && !intDomRepr->contain(indexB,-a) ) ||
            ( a >= 0 && !intDomRepr->contain(indexB, a) )
          )
        {
            variables->domains.actions.removeElement(indexA,a);
        }
    } while (intDomRepr->getNextValue(indexA,a,&a));*/
    
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
