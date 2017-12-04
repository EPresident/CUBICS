#include <chrono>
#include <iostream>
#include <cstdio>

#include <statistics/Statistics.h>

void Statistics::initialize()
{
    startElaborationTime = 0;
    endElaborationTime = 0;

    startSolveTime = 0;
    endSolveTime = 0;

    solutionsCount = 0;
    varibalesCount = 0;
    constraintsCount = 0;

    propagationsCount = 0;
    nodesCount = 0;
    failuresCount = 0;
    maxStackSize = 0;
}

void Statistics::setCurrentTime(size_t* time)
{
    *time = std::chrono::duration_cast < std::chrono::milliseconds > (std::chrono::steady_clock::now().time_since_epoch()).count();
}

void Statistics::setStartElaborationTime()
{
    setCurrentTime(&startElaborationTime);
}

void Statistics::setEndElaborationTime()
{
    setCurrentTime(&endElaborationTime);
}

void Statistics::setStartSolveTime()
{
    setCurrentTime(&startSolveTime);
}

void Statistics::setEndSolveTime()
{
    setCurrentTime(&endSolveTime);
}

void Statistics::print()
{
    std::cout.precision(3);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);

    std::cout << "%% Total time: ";
    size_t tmpElapsedTime = endElaborationTime - startElaborationTime;
    if (tmpElapsedTime > 1000)
    {
        std::cout << tmpElapsedTime / 1000 << " s" << std::endl;
    }
    else
    {
        float tmpElapsedTimeFloat = static_cast<float>(tmpElapsedTime) / static_cast<float>(1000);
        std::cout << tmpElapsedTimeFloat << " ms" << std::endl;
    }

    std::cout << "%% Solve time: ";
    tmpElapsedTime = endSolveTime - startSolveTime;
    if (tmpElapsedTime > 1000)
    {
        std::cout << tmpElapsedTime / 1000 << " s" << std::endl;
    }
    else
    {
        float tmpElapsedTimeFloat = static_cast<float>(tmpElapsedTime) / static_cast<float>(1000);
        std::cout << tmpElapsedTimeFloat << " ms" << std::endl;
    }

    std::cout.unsetf(std::ios::floatfield);

    std::cout << "%% Solutions: " << solutionsCount << std::endl;
    std::cout << "%% Variables: " << varibalesCount << std::endl;
    std::cout << "%% Constraints: " << constraintsCount << std::endl;

    std::cout << "%% Propagations: " << propagationsCount << std::endl;
    std::cout << "%% Nodes: " << nodesCount << std::endl;
    std::cout << "%% Failures: " << failuresCount << std::endl;
    std::cout << "%% Max stack size: " << maxStackSize << std::endl;

}
