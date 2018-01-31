#pragma once

#include <anyoption/anyoption.h>

/// Struct used to parse and store optional arguments.
struct Options
{
    enum SearchMode
    {
        Backtracking,
        LNS,
        SNBS
    };
    
    const char* name;
    const char* version;
    int mode;
    unsigned int solutionsCount;
    long timeout;
    unsigned int iterations;
    char* inputFile;
    double unassignRate;
    int unassignAmount;
    AnyOption *opt;

    void initialize();
    /// Parse optional arguments.
    void parseOptions(int argc, char * argv[]);
};
