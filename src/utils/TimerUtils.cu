#include <utils/TimerUtils.h>
#include <utils/LogUtils.h>

/**
 * Utility for measuring time intervals (in milliseconds) for both
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
    #ifdef GPU
        assert(initialized);
    #endif
    started = true;
    #ifdef GPU
        startTime = clock64();
    #else
        startTime = std::chrono::steady_clock::now();
    #endif
}

/**
 * Calculate the elapsed time since \a startTime.
 * 
 * \return the elapsed time in milliseconds.
*/
cudaDevice long TimerUtils::getElapsedTime()
{
    // Make sure \a setStartTime() has been called first!
    assert(started);
    
    long elapsedTime;
    #ifdef GPU
        long long diff(clock64()-startTime);
        elapsedTime = static_cast<long>(diff/(float)peak_clk);
    #else
        elapsedTime = 
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - startTime
            ).count();
    #endif
    return elapsedTime;
}
