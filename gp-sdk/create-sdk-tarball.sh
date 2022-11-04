#!/bin/sh

set -x

# initial cleanup
rm -rf gp-sdk-0.1.0* gp-sdk-host-0.1.0/ riscv-gnu-toolchain/ zlib/

if [[ -f "gp-sdk-host.lock" ]]; then
    echo "Detected gp-sdk-host lockfile"
    CONAN_INSTALL_GP_SDK_HOST_EXTRA_PARAMS="--lockfile gp-sdk-host.lock"
else
    CONAN_INSTALL_GP_SDK_HOST_EXTRA_PARAMS="-pr:b default -pr:h linux-ubuntu18.04-x86_64-gcc7-release"
fi

conan install gp-sdk-host/0.1.0@ -if=gp-sdk-host-0.1.0 $CONAN_INSTALL_GP_SDK_HOST_EXTRA_PARAMS
conan install riscv-gnu-toolchain/20220720@ -pr:b default -pr:h linux-ubuntu18.04-x86_64-gcc7-release -g deploy

# create gp-sdk folder
mkdir -p gp-sdk-0.1.0

# prune gp-sdk-host
cd gp-sdk-host-0.1.0/
rm -rf include/boost include/cereal include/elfio include/g3log include/gflags include/glog include/gmock include/gtest include/lzma include/backtrace-supported.h include/backtrace.h include/libunwind* include/lzma.h include/unwind.h include/zconf.h include/zlib.h
rm -rf lib/cmake lib/libbacktrace* lib/libboost* lib/libbz2* lib/libcap* lib/libgflags* lib/libglog* lib/libgtest* lib/liblzma* lib/libunwind* lib/libz*
cd ..

# populate gp-sdk folder with gp-sdk-host contents
mv gp-sdk-host-0.1.0/* gp-sdk-0.1.0/

# populate gp-sdk folder with gcc cross-compiler
rsync -a riscv-gnu-toolchain/* gp-sdk-0.1.0/
rm -rf riscv-gnu-toolchain

tar -czf gp-sdk-0.1.0.tar.gz gp-sdk-0.1.0/

# cleanup
rm -rf gp-sdk-0.1.0/ gp-sdk-host-0.1.0/ riscv-gnu-toolchain/ zlib/
