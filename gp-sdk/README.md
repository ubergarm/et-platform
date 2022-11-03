# Description

This project contains General-purpose-sdk specific code and tools. it involves  the needed material (libs, docs, exampless), to allow a customer to:
* Write, debug and profile a standalone compute Kernel 
* Execute on EtSoC
* Recover results

## Building Host

```
cd host
conan install .  -pr:b default -pr:h linux-ubuntu18.04-x86_64-gcc7-release --build missing
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build .
```