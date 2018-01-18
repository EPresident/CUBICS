#pragma once
#include <cassert>
#include <utils/GpuUtils.h>
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
        long long startTime;
        //int idx = threadIdx.x+blockDim.x*blockIdx.x;
        /// Device (GPU) clock speed
        int peak_clk{-1};
        bool initialized {false};
    #else
        std::chrono::steady_clock::time_point startTime;
    #endif
    bool started {false};
    
    #ifdef GPU
        cudaHostDevice void initialize();
    #endif
    
    /**
     * Set the current time as the starting time for the measurement,
     * which will be subtracted from the elapsed time.
    */
    cudaDevice void setStartTime();
    
    /**
     * Calculate the elapsed time since \a startTime.
     * 
     * \return the elapsed time in milliseconds.
    */
    cudaDevice long getElapsedTime();
};
