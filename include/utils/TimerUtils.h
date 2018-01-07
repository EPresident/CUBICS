#pragma once
#include <cassert>
#ifndef GPU
    #include <chrono>
#endif

/**
 * Utility for measuring time intervals (in milliseconds) for both
 * CPU and GPU.
 * 
 * CPU measurement is done using the std::chrono library.
 * GPU measurement is done using CUDA events from the Driver API.
*/ 
struct TimerUtils
{
    #ifdef GPU
        cudaEvent_t startEvent, stopEvent;
    #else
        std::chrono::steady_clock::time_point startTime;
    #endif
    bool initialized {false};
    
    /**
     * Set the current time as thestarting time for the measurement,
     * which will be subtracted from the elapsed time.
    */
    cudaDevice inline void setStartTime()
    {
        initialized = true;
        #ifdef GPU
            cudaEventCreate(&startEvent);
            cudaEventCreate(&stopEvent);
            cudaEventRecord(start);
        #else
            startTime = std::chrono::steady_clock::now();
        #endif
    }
    
    /**
     * Calculate the elapsed time since \a startTime.
     * 
     * \return the elapsed time in milliseconds.
    */
    cudaDevice inline long getElapsedTime()
    {
        // Make sure \a setStartTime() has been called first!
        assert(initialized);
        
        long elapsedTime;
        #ifdef GPU
            cudaEventRecord(stopEvent);
            cudaEventSynchronize(stopEvent);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            elapsedTime = static_cast<long>(milliseconds);
        #else
            elapsedTime = 
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - startTime
                ).count();
        #endif
        return elapsedTime;
    }
};
