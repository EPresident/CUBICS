#!/bin/bash

function printHelp 
{
    echo "Usage: ./autoCMake.sh <platform type> <Build type>"
    echo "  platform types:"
    echo "    -c  CPU"
    echo "    -g  GPU"
    echo "  Build types:"
    echo "    -d  Debug"
    echo "    -r  Release"
    echo "Example: ./autoCMake.sh -c -d"
}

platformSelected="False"
buildTypeSelected="False"

if [ "$#" -ne 2 ];
then
    printHelp
    exit 1
fi

for arg in "$@" 
do
    case "$arg" in
        -c) 
            platform="CPU"
            platformSelected="True"
            ;;
        -g) 
            platform="GPU"
            platformSelected="True"
            ;;
        -r) 
            buildType="Release"
            buildTypeSelected="True"
            ;;
        -d) 
            buildType="Debug"
            buildTypeSelected="True"
            ;;
        -h|--help) 
            printHelp
            exit 0
            ;;
        *)
            echo "[ERROR] Unrecognized argument: $arg"
            printHelp
            exit 1
            ;;
    esac
done

if [ "$platformSelected" != "True" ];
then
    echo "[ERROR] Missing platform type"
    printHelp
    exit 1
fi

if [ "$buildTypeSelected" != "True" ];
then
    echo "[ERROR] Missing build type"
    printHelp
    exit 1
fi

if [ -d "./build" ]; 
then
    rm -rf ./build/*
else
    mkdir build
fi

cd build

cmake \
    -DPLATFORM=$platform \
    -DCMAKE_BUILD_TYPE=$buildType \
    ..
