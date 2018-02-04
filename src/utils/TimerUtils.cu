#include <utils/TimerUtils.h>
#include <utils/LogUtils.h>
#ifndef NDEBUG
#include <iostream>
#endif
/**
 * Utility for measuring time intervals (in nanoseconds) for both
 * CPU and GPU.
 * 
 * CPU measurement is done using the std::chrono library.
 * GPU measurement is done using CUDA events from the Driver API.
*/ 

#ifdef GPU
cudaHostDevice void TimerUtils::initialize()
{
    initialized = true;
    const int DEVICE = 0; // this might need changing in a multi-device setting
    // Get clock speed
    cudaError_t err = cudaDeviceGetAttribute(&peak_clk,
        cudaDevAttrClockRate, DEVICE);
        
    if (err != cudaSuccess) 
    {
        LogUtils::error("TimeUtils"," can't get device clock!");
    }
}
#endif

/**
 * Set the current time as the starting time for the measurement,
 * which will be subtracted from the elapsed time.
*/
cudaDevice void TimerUtils::setStartTime()
{
    //~ //smallest difference test
    //~ #ifndef NDEBUG
    //~ long long t1 {clock64()};
    //~ long long t2 {clock64()};
    //~ long long diff {t2-t1};
    //~ diff *= 1000000;
    //~ printf("Diff: %ld - Peak Clock %d\n", diff, peak_clk);
    //~ long mdiff {static_cast<long>(diff/(double)peak_clk)};
    //~ printf("MDiff: %ld\n", mdiff);
    //~ #endif
    
    #ifdef GPU
        assert(initialized);
    #endif
    started = true;
    #ifdef GPU
        //startTime = clock64();
        unsigned long int ret;
        asm volatile("mov.u64 %0, %globaltimer;" : "=l"(ret) );
        startTime = static_cast<long long int>(ret);
    #else
        startTime = std::chrono::steady_clock::now();
    #endif
}

/**
 * Calculate the elapsed time since \a startTime.
 * 
 * \return the elapsed time in milliseconds.
*/
cudaDevice long long TimerUtils::getElapsedTime()
{
    //~ // Make sure \a setStartTime() has been called first!
    #ifndef NDEBUG
        assert(started);
    #endif
    
    long long elapsedTime;
    #ifdef GPU
        //long long clk = clock64();
        unsigned long int ret;
        asm volatile("mov.u64 %0, %globaltimer;" : "=l"(ret) );
        long long int clk = static_cast<long long int>(ret);
        long long diff(clk-startTime);
        elapsedTime = static_cast<long long>(diff*1000000/peak_clk);
    #else
        elapsedTime = 
            std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - startTime
            ).count();
    #endif
    #ifndef NDEBUG
        //asm volatile("mov.u64 %0, %globaltimer;" : "=l"(ret) );
        //printf("clk: %ld - sclk: %ld - diff: %ld\n",clk,startTime,diff,ret);
        if(elapsedTime < 1){
            printf("elapsedTime zero\n");
            elapsedTime = 1;
        }
        //printf("elapsedTime: %ld ns\n", elapsedTime);
        //assert(elapsedTime > 0);
    #endif
    return elapsedTime;
}
