#!/bin/bash
rm -rf build
mkdir build
conan install .  -pr:b default -pr:h linux-ubuntu18.04-x86_64-gcc7-release --build missing
cd build
cmake .. -DUSE_CONAN=ON -DCMAKE_TOOLCHAIN_FILE=generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build .
