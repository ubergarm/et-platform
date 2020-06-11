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

set(GCC_SYSROOT_PATH /esperanto/minion/ CACHE STRING "Path to the gcc riscv toolchain sysroot folder")

# FIXME make this a required package
find_package(EsperantoDeviceMinionRuntime
  NO_CMAKE_PATH
  NO_CMAKE_ENVIRONMENT_PATH
  NO_DEFAULT_PATH
  )

if(NOT EsperantoDeviceMinionRuntime_FOUND)
  find_package(LibDeviceCommon REQUIRED)
  find_package(fake-fw REQUIRED)
endif()

set(INCLUDE_INSTALL_DIR include/ CACHE STRING "Include install dir")
set(LIB_INSTALL_DIR     lib/     CACHE STRING "Library install dir")

file(GLOB INSTALL_HEADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/AutoGenInstan.def"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/Float16.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/GenInstances.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/Operators.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/LibApi.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/LibCommon.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/LibNodes.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/LibTypes.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/LibUtils.h"  
  "${CMAKE_CURRENT_SOURCE_DIR}/include/LibTensor.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/LibTensorIterator.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/AdaptiveAvgPoolInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ArgMaxInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/AvgPoolInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/BatchOneHotInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/BatchedAddInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/BatchedReduceAddInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/Checksum.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ConvertToInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/Convolution3DInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ConvolutionInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/CopyInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/CopyInstTensorized.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/CrossEntropyLossInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/DequantizeInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/MaxSplatInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ElementBoolInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ElementExpInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ElementImmLogic.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ElementInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ElementIsNaNInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ElementSelectInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ElementSingleInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/EmbeddingBagInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ExtractTensorInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/FlipInst.h"    
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/FullyConnectedInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/FusedRowwiseQuantizedSparseLengthsSum.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/FusedRowwiseQuantizedSparseLengthsWeightedSum.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/GatherInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/GatherRangesInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/InsertTensorInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/IntLookupTableInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/LengthsSumInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/LengthsToRangesInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/LocalResponseNormalizationInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/MatMulInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/MaxPoolInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ModuloInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/QuantizeInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/RescaleQuantizedInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/RowwiseQuantizedFullyConnected.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/RowwiseQuantizedSparseLengthsWeightedSum.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ScatterDataInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/SigmoidInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/SoftMaxInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/SoftMaxInst1.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/SoftMaxInst2.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/SparseLengthsWeightedSumInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/SparseToDenseInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/SparseToDenseMaskInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/SplatInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/SyncopyInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/TanhInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/TensorViewInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/TopKInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/TransposeInst.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/inlining.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/internal/Addresser.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/internal/Converter.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/internal/Operator.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/internal/PackedBufferProxy.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/internal/Writer.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/internal/printf.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/internal/utils.h"
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
  set(CMAKE_C_COMPILER "${GCC_SYSROOT_PATH}/bin/riscv64-unknown-elf-gcc")
  set(CMAKE_CXX_COMPILER "${GCC_SYSROOT_PATH}/bin/riscv64-unknown-elf-g++")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g3 -O2 -mcmodel=medany -march=rv64imf -mabi=lp64f")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 14)

add_compile_options(-Werror -Wall)


message("${PROJECT_NAME}: Clang Build is set to ${CLANG_BUILD}")

add_library(dnn_lib STATIC
  src/AdaptiveAvgPoolInst.cpp
  src/ArgMaxInst.cpp
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
  src/CrossEntropyLossInst.cpp
  src/DequantizeInst.cpp
  src/ElementBoolInst.cpp
  src/ElementExpInst.cpp
  src/ElementImmLogic.cpp
  src/ElementInst.cpp
  src/ElementIsNaNInst.cpp
  src/ElementSelectInst.cpp
  src/ElementSingleInst.cpp
  src/EmbeddingBagInst.cpp
  src/ExtractTensorInst.cpp
#  src/FlipInst.cpp
  src/FullyConnectedInst.cpp
  src/FusedRowwiseQuantizedSparseLengthsSum.cpp
  src/FusedRowwiseQuantizedSparseLengthsWeightedSum.cpp
  src/GatherInst.cpp
  src/GatherRangesInst.cpp
  src/InsertTensorInst.cpp
  src/IntLookupTableInst.cpp
  src/LengthsSumInst.cpp
  src/LengthsToRangesInst.cpp
  src/LocalResponseNormalizationInst.cpp
  src/MatMulInst.cpp
  src/MaxSplatInst.cpp
  src/MaxPoolInst.cpp
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
  src/SyncopyInst.cpp
  src/TanhInst.cpp
  src/TensorViewInst.cpp
  src/TopKInst.cpp
  src/TransposeInst.cpp
  src/libc.c
  src/printf.c
  src/Float16.cpp
)

include_directories(include/)
include_directories(include/inlining)
include_directories(include/internal)

if(EsperantoDeviceMinionRuntime_FOUND)
  target_link_libraries(dnn_lib
    EsperantoDeviceMinionRuntime::device_common
    EsperantoDeviceMinionRuntime::fw_helpers
    )
else()
  target_link_libraries(${PROJECT_NAME} device_common fake-fw)
endif()

target_include_directories(dnn_lib INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/dnn_lib>
)

install(FILES ${INSTALL_HEADERS}
  DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dnn_lib"
)

### Add package information
include(CMakePackageConfigHelpers)

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE dnn_lib)

# Create the EsperantoNeuralizerConfig.cmake and EsperantoNeuralizerConfigVersion files
configure_package_config_file(dnn_libConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/dnn_libConfig.cmake
  INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
  INSTALL_DESTINATION ${LIB_INSTALL_DIR}/dnn_lib/cmake
  PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR)

set_target_properties(dnn_lib PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/include/dnn_lib"
)

# Add all targets to the build-tree export set
export(TARGETS dnn_lib NAMESPACE dnn_lib:: FILE "${PROJECT_BINARY_DIR}/dnnLibTargets.cmake")

## Build shared Neuralizer library
install(TARGETS dnn_lib
  EXPORT dnnLibTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  COMPONENT dnn-lib
)

# Install the dnn_libConfig.cmake and NeuralizerConfigVersion.cmake
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/dnn_libConfig.cmake"
  DESTINATION ${LIB_INSTALL_DIR}/cmake/dnn_lib
  COMPONENT dnn-lib
)

# Install the export set for use with the install-tree
install(EXPORT dnnLibTargets DESTINATION
  "${LIB_INSTALL_DIR}/cmake/dnn_lib" 
  COMPONENT dnn-lib
)

# Install the export set for use with the install-tree
install(EXPORT dnnLibTargets DESTINATION
  "${LIB_INSTALL_DIR}/cmake/dnn_lib" COMPONENT dnn-lib
)
