#ifdef GPU
#pragma once

struct Lock
{
    int mutex;

    cudaHostDevice inline void initialize()
    {
        mutex = 0;
    }

    cudaDevice inline void lock()
    {
        while (atomicCAS(&mutex, 0, 1) != 0)
            ;
        __threadfence();
    }

    cudaDevice inline void unlock()
    {
        __threadfence();
        atomicExch(&mutex, 0);
    }
};
#endif
