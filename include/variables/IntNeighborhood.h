#pragma once

#include <data_structures/Vector.h>
#include <data_structures/BitsArray.h>
#include <domains/IntDomains.h>

/**
* This struct represents a neighborhood of integer variables, i.e.
* a subset of the model's variables and their domains.
*/
struct IntNeighborhood
{
    /// Number of integer variables.
    int count;
    #ifdef GPU
    /// Blocks required to launch the kernels
    int blocksRequired;
    #endif
    /// Domain this neighborhood is based on
    //IntDomains* domains;
    /// Bitmask for each variable in the domain. 1 = is a neighbor.
    BitsArray neighMask;
    /// Representations of the domains of the neighbors
    IntDomainsRepresentations neighRepr;
    /// Map from variable name (number) to its representation in \a neighRepr
    Vector<Vector<int>> map;

    /// Allocate memory for \a count sized neighborhood
    cudaDevice void initialize(int count/*, IntDomains* dom*/);
    cudaDevice void deinitialize();

    /**
     * Add the neighbor variables to the neighborhood.
     * \param neighbors list of the variable composing the neighborhood
     * \param originalRepr the representation of the original domain of the variable
     */
    cudaDevice void pushNeighbors(Vector<int>* neighbors, IntDomainsRepresentations* originalRepr);
    
    /// Find which representation is for \a var and return it in \a repr
    cudaDevice void getBinding(int var, int* repr);
    
    /// Return true if \a var is in the neighborhood
    cudaDevice inline bool isNeighbor(int var)
    {
        return neighMask.get(var);
    }
};
