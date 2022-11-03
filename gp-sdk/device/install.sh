#!/bin/bash
rm -rf build
mkdir -p build
cd build
conan install .. -pr:b default -pr:h baremetal-rv64-gcc8.2-release --build missing -g deploy