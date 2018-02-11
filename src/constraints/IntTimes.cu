#include <cmath>
#include <cassert>

#include <constraints/IntTimes.h>
#include <data_structures/Vector.h>
#include <constraints/IntConstraints.h>
/**
 * \file Variables multiplication constraint.
 * 
 * x * y = z
 * x,y and z are variables.
 */
 
/** 
* Enforce bounds-consistency for this constraint.
* x * y = z
* Minimums and maximums of the variables must be supported by the others.
*
*/
cudaDevice void IntTimes::propagate(IntConstraints* constraints, int index, IntVariables* variables)
{
    //~ DEV NOTE (TODO): the code for the support verification of the variables
    //~ could be refactored in a function.
    Vector<int>* constraintVariables = &constraints->variables[index];
    IntDomainsRepresentations& intDomRepr  = variables->domains.representations;
    IntDomainsActions& intDomAct = variables->domains.actions;
    
    // Indices of the variables
    int varX {constraintVariables->at(0)};
    int varY {constraintVariables->at(1)};
    int varZ {constraintVariables->at(2)};

    int minX {variables->domains.getMin(varX)};
    int minY {variables->domains.getMin(varY)};
    int minZ {variables->domains.getMin(varZ)};
    int maxX {variables->domains.getMax(varX)};
    int maxY {variables->domains.getMax(varY)};
    int maxZ {variables->domains.getMax(varZ)};
    
    // ---> CLARIFICATION / REMINDER <---
    // Just to be 99.99999% clear!
    // In the following code this "convention" is adopted,
    // considering as an example the full range {-100..100}:
    // - lowest negative value: -100
    // - highest negative value: -1
    // - lowest positive value: 0
    // - highest positive value: 100
    //
    // --- Elia
    
    // Is the maximum value positive?
    bool xMaxIsPos {maxX >= 0};
    bool yMaxIsPos {maxY >= 0};
    bool zMaxIsPos {maxZ >= 0};
    // Is the minimum value negative?
    bool xMinIsNeg {minX < 0};
    bool yMinIsNeg {minY < 0};
    bool zMinIsNeg {minZ < 0};
    
    // Positive minimum values
    int posMinX = 0;
    if(minX < 0 and maxX >= 0) posMinX = 0 ;
    if(minX >= 0 and maxX > 0) posMinX = minX;
    int posMinY {variables->domains.getMin(varY)};
    if(minY < 0 and maxY >= 0) posMinY = 0;
    if(minY >= 0 and maxY > 0) posMinY = minY;
    int posMinZ {variables->domains.getMin(varZ)};
    if(minZ < 0 and maxZ >= 0) posMinZ = 0;
    if(minZ >= 0 and maxZ > 0) posMinZ = minZ;
    // Positive maximum values
    int posMaxX {maxX}; // if maxX < 0 then xMaxIsPos==false, this value is unused
    int posMaxY {maxY};
    int posMaxZ {maxZ};
    // Negative minimum values (i.e. farther from zero)
    int negMinX {minX};// if minX >= 0 then xMinIsNeg==false, this value is unused
    int negMinY {minY};
    int negMinZ {minZ};
    // Negative maximum values (i.e. closer to zero)
    int negMaxX {-1}; // if minX >= 0 then xMinIsNeg==false, this value is unused
    if(xMinIsNeg and !xMaxIsPos) negMaxX = maxX;
    int negMaxY {-1};
    if(yMinIsNeg and !yMaxIsPos) negMaxY = maxY;
    int negMaxZ {-1};
    if(zMinIsNeg and !zMaxIsPos) negMaxZ = maxZ;
    
    // ---------------------------------------------------------
    // Check z lower bound
    // ---------------------------------------------------------
    {
        int minVal { INT_MAX };
        // If x has negative values, get the lowest negative possible
        // multiplying by the greatest positive value in y
        // e.g. 100 * -100
        if(xMaxIsPos and yMinIsNeg and posMaxX > 0)
        {
            int p {posMaxX * negMinY};
            if(p < minVal)
            {
                minVal = p;
            }
        }
        // If x has negative values, get the lowest negative possible
        // multiplying by the greatest positive value in y
        // e.g. -100 * 100
        if(xMinIsNeg and yMaxIsPos and posMaxY > 0)
        {
            int p { negMinX * posMaxY };
            if(p < minVal)
            {
                minVal = p;
            }
        }
        // If x and y both have positive values, get the lowest positive 
        // value possible by multiplying the smallest positive values
        // e.g. 1 * 1
        if(xMaxIsPos and yMaxIsPos)
        {
            int p {posMinX * posMinY};
            if(p < minVal)
            {
                minVal = p;
            }
        }
        // If x and y both have negative values, get the lowest positive 
        // value possible by multiplying the smallest negative values
        // e.g. -1 * -1
        if(xMinIsNeg and yMinIsNeg)
        {
            int p { negMaxX * negMaxY };
            if(p < minVal)
            {
                minVal = p;
            }
        }
        
        #ifdef NDEBUG
            assert(minVal < INT_MAX);
        #endif
        if(minVal > minZ) intDomAct.removeAnyLesserThan(varZ, minVal);
    } //~
    // ---------------------------------------------------------
    // Check z upper bound
    // ---------------------------------------------------------
    {
        int maxVal { INT_MIN };
        // If x and y both have positive values, get the highest positive 
        // value possible by multiplying the greatest positive values
        // e.g. 100 * 100
        if(xMaxIsPos and yMaxIsPos)
        {
            int p {posMaxX * posMaxY};
            if( p > maxVal)
            {
                maxVal = p;
            }
        }
        // If x and y both have negative values, get the highest positive 
        // value possible by multiplying the lowest negative values
        // e.g. -100 * -100
        if(xMinIsNeg and yMinIsNeg)
        {
            int p { negMinX * negMinY };
            if(p > maxVal)
            {
                maxVal = p;
            }
        }
        // If y has negative values, get the highest negative possible
        // multiplying by the smallest positive value in x
        // e.g. 1 * -1
        if(xMaxIsPos and yMinIsNeg)
        {
            int p {posMinX * negMaxY};
            if( p > maxVal)
            {
                maxVal = p;
            }
        }
        // If x has negative values, get the highest negative possible
        // multiplying by the smallest positive value in y 
        // e.g. -1 * 1
        if(xMinIsNeg and yMaxIsPos)
        {
            int p { negMaxX * posMinY };
            if(p > maxVal)
            {
                maxVal = p;
            }
        }
        
        #ifdef NDEBUG
            assert(maxVal > INT_MIN);
        #endif
        if(maxVal < maxZ) intDomAct.removeAnyGreaterThan(varZ, maxVal);
    } //~
    
    // ---------------------------------------------------------
    // Check x lower bound
    // ---------------------------------------------------------
    {
        int minVal { INT_MAX };
        // If z has positive values,
        // and y has negative values,
        // get the smallest negative possible
        // Of course it makes sense only if z > y
        // e.g. 100 / -1
        if(zMaxIsPos and yMinIsNeg and posMaxZ > negMaxY)
        {
            int q {posMaxZ / negMaxY};
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // If z has negative values,
        // and y has positive values,
        // get the smallest negative possible
        // Of course it makes sense only if |z| > y
        // e.g. -100 / 1
        if(zMinIsNeg and yMaxIsPos and posMinY > 0 and -negMinZ > posMinY)
        {
            int q { negMinZ / posMinY };
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // If z and y have positive values,
        // get the smallest positive x possible
        // e.g. 1 / 100
        //--x = {0..n} y = {1..m} z = {1..k}
        //--posMinZ / posMaxY = 1 / m
        if(zMaxIsPos and yMaxIsPos and posMaxY != 0)
        {
            int d {posMaxY};
            if(posMaxY > posMinZ and posMinZ != 0) d = posMinZ;
            // This way q is one if posMinZ < posMaxY
            // but it stays zero if posMinZ==0
            int q {posMinZ / d};
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // If z and y have negative values,
        // get the smallest positive x possible
        // e.g. -1 / -100
        //   x = {0..n} y = {-100..m} z = {-1..k}
        if(zMinIsNeg and yMinIsNeg)
        {
            int q { negMaxZ / negMinY };
            if(q < 1) q = 1; // This way q is one if negMaxZ < negMinY
            if(q < minVal)
            {
                minVal = q;
            }
        }
        
        #ifdef NDEBUG
            assert(minVal < INT_MAX);
        #endif
        if(minVal > minX) intDomAct.removeAnyLesserThan(varX, minVal);
    } //~
    // ---------------------------------------------------------
    // Check x upper bound
    // ---------------------------------------------------------
    {
        int maxVal { INT_MIN };
        
        // If z and y have positive values,
        // get the biggest positive x possible
        // e.g. 100 / 1
        if(zMaxIsPos and yMaxIsPos and posMaxY > 0)
        {
            int d {posMinY} ;
            if(d == 0) d=1 ; 
            // This way I don't divide by zero
            // Since posMaxY > 0 I can divide by 1 instead
            int q {posMaxZ / d};
            if(posMaxZ % d != 0) q += 1; // ceiling
            if( q > maxVal)
            {
                maxVal = q;
            }
        }
        
        // If z and y have negative values,
        // get the biggest positive x possible
        // e.g. -100 / -1
        if(zMinIsNeg and yMinIsNeg)
        {
            int q { negMinZ / negMaxY };
            if(negMinZ % negMaxY != 0) q += 1; // ceiling
            if(q > maxVal)
            {
                maxVal = q;
            }
        }
        // If z has positive values,
        // and y has negative values,
        // get the biggest negative x possible (close to zero)
        // e.g. 1 / -100
        if(zMaxIsPos and yMinIsNeg)
        {
            int q {posMinZ / negMinY};
            //(posMinZ % negMaxY != 0) ? q -= 1; // ceiling
            if(q >= 0) q = -1; // make sure to stay negative (posMinZ can be 0)
            if( q > maxVal)
            {
                maxVal = q;
            }
        }
        // If z has negative values,
        // and y has positive values,
        // get the biggest negative x possible (close to zero)
        // e.g. -1 / 100
        if(zMinIsNeg and yMaxIsPos and posMaxY > 0)
        {
            int q = negMaxZ / posMaxY;
            //(negMaxZ % posMaxY != 0) ? q -= 1; // ceiling
            if(q >= 0) q = -1; // make sure to stay negative
            if(q > maxVal)
            {
                maxVal = q;
            }
        }
        
        #ifdef NDEBUG
            assert(maxVal > INT_MIN);
        #endif
        if(maxVal < maxX) intDomAct.removeAnyGreaterThan(varX, maxVal);
    } //~
    
    // ---------------------------------------------------------
    // Check y lower bound
    // ---------------------------------------------------------
    {
        int minVal { INT_MAX };
        // If z has positive values,
        // and x has negative values,
        // get the smallest negative y possible
        // Of course it makes sense only if z > x
        // e.g. 100 / -1
        if(zMaxIsPos and xMinIsNeg and posMaxZ > negMaxX)
        {
            int q {posMaxZ / negMaxX};
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // If z has negative values,
        // and x has positive values,
        // get the smallest negative y possible
        // Of course it makes sense only if |z| > x
        // e.g. -100 / 1
        if(zMinIsNeg and xMaxIsPos and posMinX > 0 and -negMinZ > posMinX)
        {
            int q { negMinZ / posMinX };
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // If z and x have positive values,
        // get the smallest positive y possible
        // e.g. 1 / 100
        if(zMaxIsPos and xMaxIsPos and posMaxX != 0)
        {
            int d {posMaxX};
            if(posMaxX > posMinZ and posMinZ != 0) d = posMinZ;
            // This way q is one if posMinZ < posMaxX
            // but it stays zero if posMinZ==0
            int q {posMinZ / d};
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // If z and x have negative values,
        // get the smallest positive y possible
        // e.g. -1 / -100
        if(zMinIsNeg and xMinIsNeg)
        {
            int q { negMaxZ / negMinX };
            if(q < 1) q = 1; // This way q is one if negMaxZ < negMinX
            if(q < minVal)
            {
                minVal = q;
            }
        }
        
        #ifdef NDEBUG
            assert(minVal < INT_MAX);
        #endif
        if(minVal > minY) intDomAct.removeAnyLesserThan(varY, minVal);
    } //~
    // ---------------------------------------------------------
    // Check y upper bound
    // ---------------------------------------------------------
    {
        int maxVal { INT_MIN };
        
        // If z and x have positive values,
        // get the biggest positive y possible
        // e.g. 100 / 1
        if(zMaxIsPos and xMaxIsPos and posMaxX > 0)
        {
            int d {posMinX} ;
            if(d == 0) d=1 ; 
            // This way I don't divide by zero
            // Since posMaxX > 0 I can divide by 1 instead
            int q {posMaxZ / d};
            if(posMaxZ % d != 0) q += 1; // ceiling
            if( q > maxVal)
            {
                maxVal = q;
            }
        }
        
        // If z and x have negative values,
        // get the biggest positive y possible
        // e.g. -100 / -1
        if(zMinIsNeg and xMinIsNeg)
        {
            int q { negMinZ / negMaxX };
            if(negMinZ % negMaxX != 0) q += 1; // ceiling
            if(q > maxVal)
            {
                maxVal = q;
            }
        }
        // If z has positive values,
        // and x has negative values,
        // get the biggest negative y possible (close to zero)
        // e.g. 1 / -100
        if(zMaxIsPos and xMinIsNeg)
        {
            int q {posMinZ / negMinX};
            if(q >= 0) q = -1; // make sure to stay negative (posMinZ can be 0)
            if( q > maxVal)
            {
                maxVal = q;
            }
        }
        // If z has negative values,
        // and x has positive values,
        // get the biggest negative y possible (close to zero)
        // e.g. -1 / 100
        if(zMinIsNeg and xMaxIsPos and posMaxX > 0)
        {
            int q = negMaxZ / posMaxX;
            if(q >= 0) q = -1; // make sure to stay negative
            if(q > maxVal)
            {
                maxVal = q;
            }
        }
        
        #ifdef NDEBUG
            assert(maxVal > INT_MIN);
        #endif
        if(maxVal < maxY) intDomAct.removeAnyGreaterThan(varY, maxVal);
    } //~

    //---------------------------------------------------------------------
    //---------------------------------------------------------------------
    //---------------------------------------------------------------------
    //----------       Bound values support check       -------------------
    //---------------------------------------------------------------------
    //---------------------------------------------------------------------
    //---------------------------------------------------------------------
    
    // ---------------------------------------------------------
    // Verify support for the minimum of x
    // ---------------------------------------------------------
    bool supported {false};
    int valY {minY};
    do // Iterating over values of x
    {
        do // Iterating over values of y
        {
            
            if( intDomRepr.contain(varZ, minX * valY) )
            {
                supported = true;
            }
        }while(!supported and intDomRepr.getNextValue(varY, valY, &valY));
    }while(!supported and intDomRepr.getNextValue(varX, minX, &minX));
    if(!supported)
    {
        // No support
        intDomRepr.removeAll(varX);
        return;
    }
    // Remove unsupported values
    intDomAct.removeAnyLesserThan(varX, minX);
    
    // ---------------------------------------------------------
    // Verify support for the minimum of y
    // ---------------------------------------------------------
    supported = false;
    int valX {minX};
    do  // Iterating over values of y
    {
        do // Iterating over values of x
        {
            if( intDomRepr.contain(varZ, valX * minY) )
            {
                supported = true;
            }
        }while(!supported and intDomRepr.getNextValue(varX, valX, &valX));
    }while(!supported and intDomRepr.getNextValue(varY, minY, &minY));
    if(!supported)
    {
        // No support
        intDomRepr.removeAll(varY);
        return;
    }
    // Remove unsupported values
    intDomAct.removeAnyLesserThan(varY, minY);
    
    // ---------------------------------------------------------
    // Verify support for the minimum of z
    // ---------------------------------------------------------
    supported = false;
    valX = variables->domains.getMin(varX);
    do // Iterating over values of z
    {
        do // Iterating over values of x
        {
            if( (minZ % valX == 0) and intDomRepr.contain(varY, minZ / valX) )
            {
                supported = true;
            }
        }while(!supported and intDomRepr.getNextValue(varX, valX, &valX));
    }while(!supported and intDomRepr.getNextValue(varZ, minZ, &minZ));
    if(!supported)
    {
        // No support
        intDomRepr.removeAll(varZ);
        return;
    }
    // Remove unsupported values
    intDomAct.removeAnyLesserThan(varZ, minZ);
    
    // ---------------------------------------------------------
    // Verify support for the maximum of x
    // ---------------------------------------------------------
    supported = false;
    valY = variables->domains.getMax(varY);
    do // Iterating over values of x
    {
        do // Iterating over values of y
        {
            if( intDomRepr.contain(varZ, maxX * valY) )
            {
                supported = true;
            }
        }while(!supported and intDomRepr.getPrevValue(varY, valY, &valY));
    }while(!supported and intDomRepr.getPrevValue(varX, maxX, &maxX));
    if(!supported)
    {
        // No support
        intDomRepr.removeAll(varX);
        return;
    }
    // Remove unsupported values
    intDomAct.removeAnyGreaterThan(varX, maxX);
    
    // ---------------------------------------------------------
    // Verify support for the maximum of y
    // ---------------------------------------------------------
    supported = false;
    valX = variables->domains.getMax(varX);
    do // Iterating over values of y
    {
        do // Iterating over values of x
        {
            if( intDomRepr.contain(varZ, valX * maxY) )
            {
                supported = true;
            }
        }while(!supported and intDomRepr.getPrevValue(varX, valX, &valX));
    }while(!supported and intDomRepr.getPrevValue(varY, maxY, &maxY));
    if(!supported)
    {
        // No support
        intDomRepr.removeAll(varY);
        return;
    }
    // Remove unsupported values
    intDomAct.removeAnyGreaterThan(varY, maxY);
    
    // ---------------------------------------------------------
    // Verify support for the maximum of z
    // ---------------------------------------------------------
    supported = false;
    valX = variables->domains.getMax(varX);
    do // Iterating over values of z
    {
        do // Iterating over values of x
        {
            if( (maxZ % valX == 0) and intDomRepr.contain(varY, maxZ / valX) )
            {
                supported = true;
            }
        }while(!supported and intDomRepr.getPrevValue(varX, valX, &valX));
    }while(!supported and intDomRepr.getPrevValue(varZ, maxZ, &maxZ));
    if(!supported)
    {
        // No support
        intDomRepr.removeAll(varZ);
        return;
    }
    // Remove unsupported values
    intDomAct.removeAnyGreaterThan(varZ, maxZ);
    
    #ifdef NDEBUG
        // assert check
    #endif
}
//-----------------------------------------------------------------------------------------------------------------
cudaDevice void IntTimes::propagate(IntConstraints* constraints, int index, IntVariables* variables, IntNeighborhood* nbh)
{
    //~ DEV NOTE (TODO): the code for the support verification of the variables
    //~ could be refactored in a function.
    Vector<int>* constraintVariables = &constraints->variables[index];
    IntDomainsRepresentations& intDomRepr  = nbh->neighRepr;
    IntDomainsActions& intDomAct = nbh->neighActions;
    
    // Indices of the variables
    int varX {constraintVariables->at(0)};
    int varY {constraintVariables->at(1)};
    int varZ {constraintVariables->at(2)};
    // Are the variables in the neighborhood?
    bool neighX = nbh->isNeighbor(varX);
    bool neighY = nbh->isNeighbor(varY);
    bool neighZ = nbh->isNeighbor(varZ);
    // neighborhood representation indices
    int ridxX {-1};
    int ridxY {-1};
    int ridxZ {-1};
    if(neighX) ridxX = nbh->getRepresentationIndex(varX);
    if(neighY) ridxY = nbh->getRepresentationIndex(varY);
    if(neighZ) ridxZ = nbh->getRepresentationIndex(varZ);

    int minX {variables->domains.getMin(varX, nbh, ridxX)};
    int minY {variables->domains.getMin(varY, nbh, ridxY)};
    int minZ {variables->domains.getMin(varZ, nbh, ridxZ)};
    int maxX {variables->domains.getMax(varX, nbh, ridxX)};
    int maxY {variables->domains.getMax(varY, nbh, ridxY)};
    int maxZ {variables->domains.getMax(varZ, nbh, ridxZ)};
    
    // ---> CLARIFICATION / REMINDER <---
    // Just to be 99.99999% clear!
    // In the following code this "convention" is adopted,
    // considering as an example the full range {-100..100}:
    // - lowest negative value: -100
    // - highest negative value: -1
    // - lowest positive value: 0
    // - highest positive value: 100
    //
    // --- Elia
    
    // Is the maximum value positive?
    bool xMaxIsPos {maxX >= 0};
    bool yMaxIsPos {maxY >= 0};
    bool zMaxIsPos {maxZ >= 0};
    // Is the minimum value negative?
    bool xMinIsNeg {minX < 0};
    bool yMinIsNeg {minY < 0};
    bool zMinIsNeg {minZ < 0};
    
    // Positive minimum values
    int posMinX = 0;
    if(minX < 0 and maxX >= 0) posMinX = 0 ;
    if(minX >= 0 and maxX > 0) posMinX = minX;
    int posMinY {variables->domains.getMin(varY, nbh, ridxY)};
    if(minY < 0 and maxY >= 0) posMinY = 0;
    if(minY >= 0 and maxY > 0) posMinY = minY;
    int posMinZ {variables->domains.getMin(varZ, nbh, ridxZ)};
    if(minZ < 0 and maxZ >= 0) posMinZ = 0;
    if(minZ >= 0 and maxZ > 0) posMinZ = minZ;
    // Positive maximum values
    int posMaxX {maxX}; // if maxX < 0 then xMaxIsPos==false, this value is unused
    int posMaxY {maxY};
    int posMaxZ {maxZ};
    // Negative minimum values (i.e. farther from zero)
    int negMinX {minX};// if minX >= 0 then xMinIsNeg==false, this value is unused
    int negMinY {minY};
    int negMinZ {minZ};
    // Negative maximum values (i.e. closer to zero)
    int negMaxX {-1}; // if minX >= 0 then xMinIsNeg==false, this value is unused
    if(xMinIsNeg and !xMaxIsPos) negMaxX = maxX;
    int negMaxY {-1};
    if(yMinIsNeg and !yMaxIsPos) negMaxY = maxY;
    int negMaxZ {-1};
    if(zMinIsNeg and !zMaxIsPos) negMaxZ = maxZ;
    
    // ---------------------------------------------------------
    // Check z lower bound
    // ---------------------------------------------------------
    if(neighZ)
    {
        int minVal { INT_MAX };
        // If x has negative values, get the lowest negative possible
        // multiplying by the greatest positive value in y
        // e.g. 100 * -100
        if(xMaxIsPos and yMinIsNeg and posMaxX > 0)
        {
            int p {posMaxX * negMinY};
            if(p < minVal)
            {
                minVal = p;
            }
        }
        // If x has negative values, get the lowest negative possible
        // multiplying by the greatest positive value in y
        // e.g. -100 * 100
        if(xMinIsNeg and yMaxIsPos and posMaxY > 0)
        {
            int p { negMinX * posMaxY };
            if(p < minVal)
            {
                minVal = p;
            }
        }
        // If x and y both have positive values, get the lowest positive 
        // value possible by multiplying the smallest positive values
        // e.g. 1 * 1
        if(xMaxIsPos and yMaxIsPos)
        {
            int p {posMinX * posMinY};
            if(p < minVal)
            {
                minVal = p;
            }
        }
        // If x and y both have negative values, get the lowest positive 
        // value possible by multiplying the smallest negative values
        // e.g. -1 * -1
        if(xMinIsNeg and yMinIsNeg)
        {
            int p { negMaxX * negMaxY };
            if(p < minVal)
            {
                minVal = p;
            }
        }
        
        #ifdef NDEBUG
            assert(minVal < INT_MAX);
        #endif
        if(minVal > minZ) 
        {
            assert(neighZ);
            intDomAct.removeAnyLesserThan(ridxZ, minVal);
        }
    } //~
    // ---------------------------------------------------------
    // Check z upper bound
    // ---------------------------------------------------------
    if(neighZ)
    {
        int maxVal { INT_MIN };
        // If x and y both have positive values, get the highest positive 
        // value possible by multiplying the greatest positive values
        // e.g. 100 * 100
        if(xMaxIsPos and yMaxIsPos)
        {
            int p {posMaxX * posMaxY};
            if( p > maxVal)
            {
                maxVal = p;
            }
        }
        // If x and y both have negative values, get the highest positive 
        // value possible by multiplying the lowest negative values
        // e.g. -100 * -100
        if(xMinIsNeg and yMinIsNeg)
        {
            int p { negMinX * negMinY };
            if(p > maxVal)
            {
                maxVal = p;
            }
        }
        // If y has negative values, get the highest negative possible
        // multiplying by the smallest positive value in x
        // e.g. 1 * -1
        if(xMaxIsPos and yMinIsNeg)
        {
            int p {posMinX * negMaxY};
            if( p > maxVal)
            {
                maxVal = p;
            }
        }
        // If x has negative values, get the highest negative possible
        // multiplying by the smallest positive value in y 
        // e.g. -1 * 1
        if(xMinIsNeg and yMaxIsPos)
        {
            int p { negMaxX * posMinY };
            if(p > maxVal)
            {
                maxVal = p;
            }
        }
        
        #ifdef NDEBUG
            assert(maxVal > INT_MIN);
        #endif
        if(maxVal < maxZ)
        {
            assert(neighZ);
            intDomAct.removeAnyGreaterThan(ridxZ, maxVal);
        }
    } //~
    
    // ---------------------------------------------------------
    // Check x lower bound
    // ---------------------------------------------------------
    if(neighX)
    {
        int minVal { INT_MAX };
        // If z has positive values,
        // and y has negative values,
        // get the smallest negative possible
        // Of course it makes sense only if z > y
        // e.g. 100 / -1
        if(zMaxIsPos and yMinIsNeg and posMaxZ > negMaxY)
        {
            int q {posMaxZ / negMaxY};
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // If z has negative values,
        // and y has positive values,
        // get the smallest negative possible
        // Of course it makes sense only if |z| > y
        // e.g. -100 / 1
        if(zMinIsNeg and yMaxIsPos and posMinY > 0 and -negMinZ > posMinY)
        {
            int q { negMinZ / posMinY };
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // If z and y have positive values,
        // get the smallest positive x possible
        // e.g. 1 / 100
        //--x = {0..n} y = {1..m} z = {1..k}
        //--posMinZ / posMaxY = 1 / m
        if(zMaxIsPos and yMaxIsPos and posMaxY != 0)
        {
            int d {posMaxY};
            if(posMaxY > posMinZ and posMinZ != 0) d = posMinZ;
            // This way q is one if posMinZ < posMaxY
            // but it stays zero if posMinZ==0
            int q {posMinZ / d};
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // If z and y have negative values,
        // get the smallest positive x possible
        // e.g. -1 / -100
        //   x = {0..n} y = {-100..m} z = {-1..k}
        if(zMinIsNeg and yMinIsNeg)
        {
            int q { negMaxZ / negMinY };
            if(q < 1) q = 1; // This way q is one if negMaxZ < negMinY
            if(q < minVal)
            {
                minVal = q;
            }
        }
        
        #ifdef NDEBUG
            assert(minVal < INT_MAX);
        #endif
        if(minVal > minX)
        {
            assert(neighX);
            intDomAct.removeAnyLesserThan(ridxX, minVal);
        }
    } //~
    // ---------------------------------------------------------
    // Check x upper bound
    // ---------------------------------------------------------
    if(neighX)
    {
        int maxVal { INT_MIN };
        
        // If z and y have positive values,
        // get the biggest positive x possible
        // e.g. 100 / 1
        if(zMaxIsPos and yMaxIsPos and posMaxY > 0)
        {
            int d {posMinY} ;
            if(d == 0) d=1 ; 
            // This way I don't divide by zero
            // Since posMaxY > 0 I can divide by 1 instead
            int q {posMaxZ / d};
            if(posMaxZ % d != 0) q += 1; // ceiling
            if( q > maxVal)
            {
                maxVal = q;
            }
        }
        
        // If z and y have negative values,
        // get the biggest positive x possible
        // e.g. -100 / -1
        if(zMinIsNeg and yMinIsNeg)
        {
            int q { negMinZ / negMaxY };
            if(negMinZ % negMaxY != 0) q += 1; // ceiling
            if(q > maxVal)
            {
                maxVal = q;
            }
        }
        // If z has positive values,
        // and y has negative values,
        // get the biggest negative x possible (close to zero)
        // e.g. 1 / -100
        if(zMaxIsPos and yMinIsNeg)
        {
            int q {posMinZ / negMinY};
            //(posMinZ % negMaxY != 0) ? q -= 1; // ceiling
            if(q >= 0) q = -1; // make sure to stay negative (posMinZ can be 0)
            if( q > maxVal)
            {
                maxVal = q;
            }
        }
        // If z has negative values,
        // and y has positive values,
        // get the biggest negative x possible (close to zero)
        // e.g. -1 / 100
        if(zMinIsNeg and yMaxIsPos and posMaxY > 0)
        {
            int q = negMaxZ / posMaxY;
            //(negMaxZ % posMaxY != 0) ? q -= 1; // ceiling
            if(q >= 0) q = -1; // make sure to stay negative
            if(q > maxVal)
            {
                maxVal = q;
            }
        }
        
        #ifdef NDEBUG
            assert(maxVal > INT_MIN);
        #endif
        if(maxVal < maxX)
        {
            assert(neighX);
            intDomAct.removeAnyGreaterThan(ridxX, maxVal);
        }
    } //~
    
    // ---------------------------------------------------------
    // Check y lower bound
    // ---------------------------------------------------------
    if(neighY)
    {
        int minVal { INT_MAX };
        // If z has positive values,
        // and x has negative values,
        // get the smallest negative y possible
        // Of course it makes sense only if z > x
        // e.g. 100 / -1
        if(zMaxIsPos and xMinIsNeg and posMaxZ > negMaxX)
        {
            int q {posMaxZ / negMaxX};
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // If z has negative values,
        // and x has positive values,
        // get the smallest negative y possible
        // Of course it makes sense only if |z| > x
        // e.g. -100 / 1
        if(zMinIsNeg and xMaxIsPos and posMinX > 0 and -negMinZ > posMinX)
        {
            int q { negMinZ / posMinX };
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // If z and x have positive values,
        // get the smallest positive y possible
        // e.g. 1 / 100
        if(zMaxIsPos and xMaxIsPos and posMaxX != 0)
        {
            int d {posMaxX};
            if(posMaxX > posMinZ and posMinZ != 0) d = posMinZ;
            // This way q is one if posMinZ < posMaxX
            // but it stays zero if posMinZ==0
            int q {posMinZ / d};
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // If z and x have negative values,
        // get the smallest positive y possible
        // e.g. -1 / -100
        if(zMinIsNeg and xMinIsNeg)
        {
            int q { negMaxZ / negMinX };
            if(q < 1) q = 1; // This way q is one if negMaxZ < negMinX
            if(q < minVal)
            {
                minVal = q;
            }
        }
        
        #ifdef NDEBUG
            assert(minVal < INT_MAX);
        #endif
        if(minVal > minY)
        {
            assert(neighY);
            intDomAct.removeAnyLesserThan(ridxY, minVal);
        }
    } //~
    // ---------------------------------------------------------
    // Check y upper bound
    // ---------------------------------------------------------
    if(neighY)
    {
        int maxVal { INT_MIN };
        
        // If z and x have positive values,
        // get the biggest positive y possible
        // e.g. 100 / 1
        if(zMaxIsPos and xMaxIsPos and posMaxX > 0)
        {
            int d {posMinX} ;
            if(d == 0) d=1 ; 
            // This way I don't divide by zero
            // Since posMaxX > 0 I can divide by 1 instead
            int q {posMaxZ / d};
            if(posMaxZ % d != 0) q += 1; // ceiling
            if( q > maxVal)
            {
                maxVal = q;
            }
        }
        
        // If z and x have negative values,
        // get the biggest positive y possible
        // e.g. -100 / -1
        if(zMinIsNeg and xMinIsNeg)
        {
            int q { negMinZ / negMaxX };
            if(negMinZ % negMaxX != 0) q += 1; // ceiling
            if(q > maxVal)
            {
                maxVal = q;
            }
        }
        // If z has positive values,
        // and x has negative values,
        // get the biggest negative y possible (close to zero)
        // e.g. 1 / -100
        if(zMaxIsPos and xMinIsNeg)
        {
            int q {posMinZ / negMinX};
            if(q >= 0) q = -1; // make sure to stay negative (posMinZ can be 0)
            if( q > maxVal)
            {
                maxVal = q;
            }
        }
        // If z has negative values,
        // and x has positive values,
        // get the biggest negative y possible (close to zero)
        // e.g. -1 / 100
        if(zMinIsNeg and xMaxIsPos and posMaxX > 0)
        {
            int q = negMaxZ / posMaxX;
            if(q >= 0) q = -1; // make sure to stay negative
            if(q > maxVal)
            {
                maxVal = q;
            }
        }
        
        #ifdef NDEBUG
            assert(maxVal > INT_MIN);
        #endif
        if(maxVal < maxY)
        {
            assert(neighY);
            intDomAct.removeAnyGreaterThan(ridxY, maxVal);
        }
    } //~

    //---------------------------------------------------------------------
    //---------------------------------------------------------------------
    //---------------------------------------------------------------------
    //----------       Bound values support check       -------------------
    //---------------------------------------------------------------------
    //---------------------------------------------------------------------
    //---------------------------------------------------------------------
    
    // ---------------------------------------------------------
    // Verify support for the minimum of x
    // ---------------------------------------------------------
    bool supported {false};
    int valY {minY};
    if(neighX)
    {
        do // Iterating over values of x
        {
            do // Iterating over values of y
            {
                
                if( intDomRepr.contain(ridxZ, minX * valY) )
                {
                    supported = true;
                }
            }while(!supported and intDomRepr.getNextValue(ridxY, valY, &valY));
        }while(!supported and intDomRepr.getNextValue(ridxX, minX, &minX));
        if(!supported)
        {
            // No support
            intDomAct.removeAnyGreaterThan(ridxX, INT_MIN);
            return;
        }
        // Remove unsupported values
        intDomAct.removeAnyLesserThan(ridxX, minX);
    }
    
    // ---------------------------------------------------------
    // Verify support for the minimum of y
    // ---------------------------------------------------------
    supported = false;
    int valX {minX};
    if(neighY)
    {
        do  // Iterating over values of y
        {
            do // Iterating over values of x
            {
                if( intDomRepr.contain(ridxZ, valX * minY) )
                {
                    supported = true;
                }
            }while(!supported and intDomRepr.getNextValue(ridxX, valX, &valX));
        }while(!supported and intDomRepr.getNextValue(ridxY, minY, &minY));
        if(!supported)
        {
            // No support
            intDomRepr.removeAll(ridxY);
            return;
        }
        // Remove unsupported values
        intDomAct.removeAnyLesserThan(ridxY, minY);
    }
    
    // ---------------------------------------------------------
    // Verify support for the minimum of z
    // ---------------------------------------------------------
    supported = false;
    if(neighZ)
    {
        valX = variables->domains.getMin(varX, nbh, ridxX);
        do // Iterating over values of z
        {
            do // Iterating over values of x
            {
                if( (minZ % valX == 0) and intDomRepr.contain(ridxY, minZ / valX) )
                {
                    supported = true;
                }
            }while(!supported and intDomRepr.getNextValue(ridxX, valX, &valX));
        }while(!supported and intDomRepr.getNextValue(ridxZ, minZ, &minZ));
        if(!supported)
        {
            // No support
            intDomRepr.removeAll(ridxZ);
            return;
        }
        // Remove unsupported values
        intDomAct.removeAnyLesserThan(ridxZ, minZ);
    }
    
    // ---------------------------------------------------------
    // Verify support for the maximum of x
    // ---------------------------------------------------------
    supported = false;
    if(neighX)
    {
        valY = variables->domains.getMax(varY, nbh, ridxY);
        do // Iterating over values of x
        {
            do // Iterating over values of y
            {
                if( intDomRepr.contain(ridxZ, maxX * valY) )
                {
                    supported = true;
                }
            }while(!supported and intDomRepr.getPrevValue(ridxY, valY, &valY));
        }while(!supported and intDomRepr.getPrevValue(ridxX, maxX, &maxX));
        if(!supported)
        {
            // No support
            intDomRepr.removeAll(ridxX);
            return;
        }
        // Remove unsupported values
        intDomAct.removeAnyGreaterThan(ridxX, maxX);
    }
    
    // ---------------------------------------------------------
    // Verify support for the maximum of y
    // ---------------------------------------------------------
    supported = false;
    if(neighY)
    {
        valX = variables->domains.getMax(varX, nbh, ridxX);
        do // Iterating over values of y
        {
            do // Iterating over values of x
            {
                if( intDomRepr.contain(ridxZ, valX * maxY) )
                {
                    supported = true;
                }
            }while(!supported and intDomRepr.getPrevValue(ridxX, valX, &valX));
        }while(!supported and intDomRepr.getPrevValue(ridxY, maxY, &maxY));
        if(!supported)
        {
            // No support
            intDomRepr.removeAll(ridxY);
            return;
        }
        // Remove unsupported values
        intDomAct.removeAnyGreaterThan(ridxY, maxY);
    }
    
    // ---------------------------------------------------------
    // Verify support for the maximum of z
    // ---------------------------------------------------------
    supported = false;
    if(neighZ)
    {
        valX = variables->domains.getMax(varX, nbh, ridxX);
        do // Iterating over values of z
        {
            do // Iterating over values of x
            {
                if( (maxZ % valX == 0) and intDomRepr.contain(ridxY, maxZ / valX) )
                {
                    supported = true;
                }
            }while(!supported and intDomRepr.getPrevValue(ridxX, valX, &valX));
        }while(!supported and intDomRepr.getPrevValue(ridxZ, maxZ, &maxZ));
        if(!supported)
        {
            // No support
            intDomRepr.removeAll(ridxZ);
            return;
        }
        // Remove unsupported values
        intDomAct.removeAnyGreaterThan(ridxZ, maxZ);
    }
    
    #ifdef NDEBUG
        // assert check
    #endif
}

cudaDevice bool IntTimes::satisfied(IntConstraints* constraints, int index, IntVariables* variables)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    // Indices of the variables
    int x {variables->domains.getMin(constraintVariables->at(0))};
    int y {variables->domains.getMin(constraintVariables->at(1))};
    int z {variables->domains.getMin(constraintVariables->at(2))};

    //Satisfaction check is performed only when all variables is ground
    for (int i = 0; i < constraintVariables->size; i += 1)
    {
        if (not variables->domains.isSingleton(constraintVariables->at(i)))
        {
            return true;
        }
    }

    return x * y == z;
}
cudaDevice bool IntTimes::satisfied(IntConstraints* constraints, int index, IntVariables* variables, IntNeighborhood* nbh)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    int& indexX = constraintVariables->at(0);
    int& indexY = constraintVariables->at(1);
    int& indexZ = constraintVariables->at(2);
    // Are the variables in the neighborhood?
    bool neighX = nbh->isNeighbor(indexX);
    bool neighY = nbh->isNeighbor(indexY);
    bool neighZ = nbh->isNeighbor(indexZ);
    // neighborhood representation indices
    int ridxX {-1};
    int ridxY {-1};
    int ridxZ {-1};
    if(neighX) ridxX = nbh->getRepresentationIndex(indexX);
    if(neighY) ridxY = nbh->getRepresentationIndex(indexY);
    if(neighZ) ridxZ = nbh->getRepresentationIndex(indexZ);
    //Satisfaction check is performed only when all variables are ground
    if ( not variables->domains.isSingleton(indexX, nbh, ridxX) or
         not variables->domains.isSingleton(indexY, nbh, ridxY) or 
         not variables->domains.isSingleton(indexZ, nbh, ridxZ) ) 
    {
        return true;
    }

    // Indices of the variables
    int x {variables->domains.getMin(constraintVariables->at(0), nbh, ridxX)};
    int y {variables->domains.getMin(constraintVariables->at(1), nbh, ridxY)};
    int z {variables->domains.getMin(constraintVariables->at(2), nbh, ridxZ)};

    return x * y == z;
}
