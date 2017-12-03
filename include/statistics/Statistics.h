#pragma once

struct Statistics
{
    size_t startElaborationTime;
    size_t endElaborationTime;

    size_t startSolveTime;
    size_t endSolveTime;

    unsigned int solutionsCount;
    int varibalesCount;
    int constraintsCount;

    size_t propagationsCount;
    size_t nodesCount;
    size_t failuresCount;
    int maxStackSize;

    void initialize();

    void setCurrentTime(size_t* time);

    void setStartElaborationTime();
    void setEndElaborationTime();

    void setStartSolveTime();
    void setEndSolveTime();

    void print();
};
