#pragma once

#include <data_structures/Vector.h>
#include <data_structures/BitsArray.h>
#include <domains/IntDomainsRepresentations.h>
#include <domains/IntDomainsActions.h>

/**
* This struct represents a neighborhood of integer variables, i.e.
* a subset of the model's variables and their domains.
*/
struct IntNeighborhood
{
    enum EventTypes
    {
        None,
        Changed ///< A domain has been changed
    };
    
    /// Number of integer variables.
    int count;
    #ifdef GPU
    /// Blocks required to launch the kernels with default block size
    int blocksRequired;
    #endif
    /// Bitmask for each variable in the domain. 1 = is a neighbor.
    BitsArray neighMask;
    /// Representations of the domains of the neighbors
    IntDomainsRepresentations neighRepr;
    /// Domain actions of the neighbors
    IntDomainsActions neighActions;
    /// Map from variable name (number) to its representation in \a neighRepr
    Vector<Vector<int>> map;
    /// A list of domain events (domain changed) in chronological order.
    Vector<int> events;

    /// Allocate memory for \a count sized neighborhood
    cudaDevice void initialize(int count);
    cudaDevice void deinitialize();

    /**
     * Add the neighbor variables to the neighborhood.
     * \param neighbors list of the variable composing the neighborhood
     * \param originalRepr the representation of the original domain of the variable
     */
    cudaDevice void pushNeighbors(Vector<int>* neighbors, IntDomainsRepresentations* originalRepr);
    
    /// Find which representation is for \a var and return it in \a repr (used for kernel)
    cudaDevice void getBinding(int var, int* repr);
    /// Find which representation is for \a var (used as a normal function)
    cudaDevice int getRepresentationIndex(int var);
    
    /// Return true if \a var is in the neighborhood
    cudaDevice inline bool isNeighbor(int var)
    {
        return neighMask.get(var);
    }
};
