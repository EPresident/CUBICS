#pragma once

#include <anyoption/anyoption.h>

struct Options
{

    const char* name;
    const char* version;
    unsigned int solutionsCount;
    char* inputFile;
    AnyOption *opt;
    bool printStats;

    void initialize();
    void parseOptions(int argc, char * argv[]);
};
