#!/bin/bash

set -x

start_time=$(date +%s)

rm -rf device/build host/build

# build device kernels
cd device
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=/usr/local/esperanto/.builds/device/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DADDRESS:STRING=0x8005b35000 -DCMAKE_C_COMPILER=riscv64-unknown-elf-gcc -DCMAKE_CXX_COMPILER=riscv64-unknown-elf-g++ -DCMAKE_ASM_COMPILER=riscv64-unknown-elf-gcc -DUSE_CONAN=ON
cmake --build .
cd ../..

# build host tools
cd host
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=/usr/local/esperanto/.builds/host/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CONAN=ON -G Ninja
cmake --build .

./sdk/basic_launcher -kernel_path=../../device/build/tests/print2/print2 -device-type=sysemu
dt2json traceKernels_dev0_0.bin -t

cd ../..

echo "done"
