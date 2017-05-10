# CUBICS
CUBICS (CUda BasIc Constraint Solver) is a constraint solver that can exploit GPU computational power in order to solve constraints problems. It is designed with simplicity and performances as primary goals. 

## Requirements
- CMake >= 3.5.1
- GCC >= 5.4

### Addional requirements for GPU version
- CUDA SDK >= 7.5
- nVIDIA GPU with [compute capability](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) >= 3.5

## Compilation
Execute the following commands
```sh
./autoCMake.sh -c -d
cd build
make
```
for a **C**PU **d**ebug version, or
```sh
./autoCMake.sh -g -r
cd build
make
```
for a **G**PU **r**elease version.

Fore more details about build configurations:
```sh
./autoCMake.sh --help
```
