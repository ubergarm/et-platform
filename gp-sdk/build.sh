#!/bin/bash

echo $OPTARGS
echo $OPTIND

version="v0.12.1"
device_api_version="0.6.0"
device_type="sysemu"

while getopts "v:d:r:" flag
do
  case $flag in
    v) version=${OPTARG};;
    d) device_api_version=${OPTARG};;
    r) device_type=${OPTARG};;
    *) echo 'Error in command line parsing' >&2
       exit 1
  esac
done

echo "TESTING with sdk version=${version}"
echo "Using device-api version=${device_api_version}"
echo "Running device_type=${device_type}"

set -x

start_time=$(date +%s)

rm -rf device/build host/build

# build device kernels
cd device
conan install conanfile_${version}.txt -pr:b default -pr:h baremetal-rv64-gcc8.2-release --remote conan-develop --build missing -if=build
source build/generators/conanbuild.sh
cmake --preset release -DADDRESS:STRING=0x8005b35000 -DCMAKE_C_COMPILER=riscv64-unknown-elf-gcc -DCMAKE_CXX_COMPILER=riscv64-unknown-elf-g++ -DCMAKE_ASM_COMPILER=riscv64-unknown-elf-gcc -DUSE_CONAN=ON
cmake --build --preset release
cd -

# build host tools
cd host
conan install .  -pr:b default -pr:h linux-ubuntu18.04-x86_64-gcc7-release --remote conan-develop --build missing -if=build -o gp-sdk-host:device_api_version=${device_api_version}
conan install .  -pr:b default -pr:h linux-ubuntu18.04-x86_64-gcc7-debug   --remote conan-develop --build missing -if=build -o gp-sdk-host:device_api_version=${device_api_version}
cmake --preset release -G Ninja
cmake --preset debug -G Ninja
cmake --build --preset release
cmake --build --preset debug

cd build

# install dt2json
conan install trace-utils/0.6.0@ -pr:b default -pr:h linux-ubuntu18.04-x86_64-gcc7-release --remote conan-develop --build missing -o trace-utils:with_cereal=False -g VirtualRunEnv
source conanrun.sh

# execute test

./Release/sdk/basic_launcher -kernel_path=../../device/build/Release/tests/print2/print2 -device-type=${device_type} $extra_flags

# parse trace
dt2json traceKernels_dev0.bin -t

end_time=$(date +%s)
# elapsed time with second resolution
elapsed="$(( end_time - start_time )) seconds"

echo "done"
