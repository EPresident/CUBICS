#include <cmath>

#include <constraints/IntAbs.h>
#include <data_structures/Vector.h>
#include <constraints/IntConstraints.h>
/**
* \file Integer absolute value constraint.
* |a| = b      a,b are variables.
*/

/** 
* Enforce arc-consistency for this constraint.
* Arc consistency has been chosen because the cost of the propagation
* is only O(n) (i.e. linear) if n = max{domain(a), domain(b)};
* that means it should pay off to go for arc instead of bounds consistency!
*/
cudaDevice void IntAbs::propagate(IntConstraints* constraints, int index, IntVariables* variables)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    IntDomainsRepresentations* intDomRepr  = &variables->domains.representations;
    
    // 0 is the index of a, 1 is the index of b.
    int indexA = constraintVariables->at(0);
    int indexB = constraintVariables->at(1);
    
    //int maxA = variables->domains.getMax(indexA);
    //int maxB = variables->domains.getMax(indexB);

    // Check support for values of b
    // Easy: for each value of b we need to check two of a.
    int b = variables->domains.getMin(indexB);
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
    } while (intDomRepr->getNextValue(indexA,a,&a));
    
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
