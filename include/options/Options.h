#pragma once

#include <anyoption/anyoption.h>

/// Struct used to parse and store optional arguments.
struct Options
{

    const char* name;
    const char* version;
    unsigned int solutionsCount;
    long timeout;
    unsigned int iterations;
    char* inputFile;
    AnyOption *opt;

    void initialize();
    /// Parse optional arguments.
    void parseOptions(int argc, char * argv[]);
};
