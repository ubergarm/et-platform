# Description

This project contains General-purpose-sdk specific code and tools. it involves  the needed material (libs, docs, exampless), to allow a customer to:
* Write, debug and profile a standalone compute Kernel 
* Execute on EtSoC
* Recover results

## Building (with Conan)

In order to build Conan you need to use a special docker image for conan development:
````
./dock.py --image=convoke/ubuntu-18.04-gcc7-conan
````
This image contains the bare minimum to build (gcc), conan and pre-configured conan-config with profiles for esperanto.

## Building Host (with Conan)

Inside the docker image:
```
cd host
conan install .  -pr:b default -pr:h linux-ubuntu18.04-x86_64-gcc7-release --build missing
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build .
```

## Deployment

At the moment we only create a tarball from Conan. It contains:

* Host includes & libraries (runtime, deviceLayer, sysemu, hostUtils, deviceApi, et-trace)
* Host tools (generic_launcher, sysemu executable)
* RISC-V gnu toolchain (cross-compiler, linker, gdb, etc..)
* (TODO) Device libs
