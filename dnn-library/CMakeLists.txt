#------------------------------------------------------------------------------
# Copyright (C) 2019, Esperanto Technologies Inc.
# The copyright to the computer program(s) herein is the
# property of Esperanto Technologies, Inc. All Rights Reserved.
# The program(s) may be used and/or copied only with
# the written permission of Esperanto Technologies and
# in accordance with the terms and conditions stipulated in the
# agreement/contract under which the program(s) have been supplied.
#------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.5)

project(dnn_lib VERSION 0.1.0 LANGUAGES CXX C)

find_package(LibDeviceCommon REQUIRED)
find_package(fake-fw REQUIRED)

set(INCLUDE_INSTALL_DIR include/ CACHE STRING "Include install dir")
set(LIB_INSTALL_DIR     lib/     CACHE STRING "Library install dir")

file(GLOB INSTALL_HEADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/AutoGenInstan.def"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/Float16.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/GenInstances.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/LibCommon.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/LibNodes.h"
)

option(CLANG_BUILD "Build using clang")

if(CLANG_BUILD)
  set(CMAKE_C_COMPILER "$ENV{LLVM_RVET}/bin/clang")
  set(CMAKE_CXX_COMPILER "$ENV{LLVM_RVET}/bin/clang")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g3 -O3 -mcmodel=medium -target riscv64 -march=rv64-et -mabi=lp64f")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --sysroot=$(RISCV)/riscv64-unknown-elf")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I$(RISCV)/riscv64-unknown-elf/include/c++/8.2.0/")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I$(RISCV)/riscv64-unknown-elf/include/c++/8.2.0/riscv64-unknown-elf")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}")
else()
  set(CMAKE_C_COMPILER "riscv64-unknown-elf-gcc")
  set(CMAKE_CXX_COMPILER "riscv64-unknown-elf-g++")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g3 -O2 -mcmodel=medany -march=rv64imf -mabi=lp64f")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 11)

message("${PROJECT_NAME}: Clang Build is set to ${CLANG_BUILD}")

add_library(dnn_lib STATIC
  src/AdaptiveAvgPoolInst.cpp
  src/AvgPoolInst.cpp
  src/BatchedAddInst.cpp
  src/BatchedReduceAddInst.cpp
  src/BatchOneHotInst.cpp
  src/Checksum.cpp
  src/ConvertToInst.cpp
  src/Convolution3DInst.cpp
  src/ConvolutionInst.cpp
  src/CopyInst.cpp
  src/CopyInstTensorized.cpp
  src/SyncopyInstTensorized.cpp
  src/CrossEntropyLossInst.cpp
  src/DequantizeInst.cpp
  src/ElementBoolInst.cpp
  src/ElementExpInst.cpp
  src/ElementInst.cpp
  src/ElementIsNaNInst.cpp
  src/ElementSelectInst.cpp
  src/ElementSingleInst.cpp
  src/EmbeddingBagInst.cpp
  src/ETSOCMaxSplatInst.cpp
  src/ExtractTensorInst.cpp
  src/FullyConnectedInst.cpp
  src/FusedRowwiseQuantizedSparseLengthsSum.cpp
  src/FusedRowwiseQuantizedSparseLengthsWeightedSum.cpp
  src/GatherInst.cpp
  src/GatherRangesInst.cpp
  src/InsertTensorInst.cpp
  src/IntLookupTable.cpp
  src/LengthsSumInst.cpp
  src/LengthsToRangesInst.cpp
  src/LocalResponseNormalizationInst.cpp
  src/MatMulInst.cpp
  src/MatMulInstTransposed.cpp
  src/MaxPoolInst.cpp
  src/ArgMaxInst.cpp
  src/ModuloInst.cpp
  src/QuantizeInst.cpp
  src/RescaleQuantizedInst.cpp
  src/RowwiseQuantizedFullyConnected.cpp
  src/RowwiseQuantizedSparseLengthsWeightedSum.cpp
  src/ScatterDataInst.cpp
  src/SigmoidInst.cpp
  src/SoftMaxInst1.cpp
  src/SoftMaxInst2.cpp
  src/SoftMaxInst.cpp
  src/SparseLengthsWeightedSumInst.cpp
  src/SparseToDenseInst.cpp
  src/SparseToDenseMaskInst.cpp
  src/SplatInst.cpp
  src/TanhInst.cpp
  src/TensorViewInst.cpp
  src/TopKInst.cpp
  src/TransposeInst.cpp
  src/libc.c
  src/printf.c
)

include_directories(include)

target_link_libraries(${PROJECT_NAME} device_common fake-fw)

target_include_directories(dnn_lib INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/dnn_lib>
)

install(FILES ${INSTALL_HEADERS}
  DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dnn_lib"
)

install(TARGETS dnn_lib
  DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
  EXPORT dnn_libConfig
)

# This makes the project importable from the install directory
install(EXPORT dnn_libConfig
  DESTINATION "${LIB_INSTALL_DIR}/cmake/dnn_lib"
)
