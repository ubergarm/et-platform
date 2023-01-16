# Description

This project contains General Purpose Sdk specific code and tools. it involves  the needed material (libs, docs, exampless), to allow a customer to:
* Write, debug and profile a standalone compute Kernel 
* Execute on ET-SoC-1
* Recover results

for further information, please review companion documentation provided in docs folder.

## Building

In order to build you need to use a docker image:
```
et_docker --develop --network=host prompt
```
This image contains the et sw stack allowing buliding GP-SDK from sources.

Conceptually the build phase can be divided in:

1. Building device kernel code
```
mkdir device/build
cd device/build
cmake .. -DCMAKE_TOOLCHAIN_FILE=/usr/local/esperanto/.builds/device/conan_toolchain.cmake -DADDRESS:STRING=0x8006335000  -DCMAKE_BUILD_TYPE=Release  -DUSE_CONAN=ON

make all
```
2. Building host examples

```
cd host/build

cmake .. -DCMAKE_TOOLCHAIN_FILE=/usr/local/esperanto/.builds/host/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug -DUSE_CONAN=ON

make all

```

## Executing

Executing involves launching a host application that receives as a parameter the computational kernel to be executed on the device, as in the following examples:

```
sdk/hello_world_launcher -kernel-path=../../device/build/tests/print.elf -device-type=sysemu
sdk/hello_world_launcher -kernel-path=../../device/build/tests/print.elf -device-type=silicon
```

and we can see  traces from the device through dt2json application installed on the docker image:

```
dt2json traceKernels_dev0_0.bin -t
(...)
12362222;string;{plain_string};{"entryPoint_0,24 HELLO WORLD!!!!"}
```

