# Description

This project contains General-purpose-sdk specific code and tools. it involves  the needed material (libs, docs, exampless), to allow a customer to:
* Write, debug and profile a standalone compute Kernel 
* Execute on EtSoC
* Recover results

## Building

In order to build you need to use a docker image:
````
./dock.py --image=convoke/ubuntu-18.04-et-sw-develop-stack prompt
````
This image contains the et sw stack allowing buliding GP-SDK from sources.

A `build.sh` script is provided to ease build process.

Conceptually the build phase can be divided in:

1. Building device kernel code
2. Building host examples
