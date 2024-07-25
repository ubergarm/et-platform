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
project(dnnLibrary VERSION 0.11.1 LANGUAGES C CXX)

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

option(ENABLE_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

include(GNUInstallDirs)

find_package(et-common-libs REQUIRED COMPONENTS cm-umode)

file(GLOB LIB_DNN_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Float16.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Operators.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/LibCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/LibNodes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/LibTypes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/LibUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/LibTensor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/LibTensorIterator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/AdaptiveAvgPoolInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ArgMaxInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/AvgPoolInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/BatchOneHotInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/BatchedAddInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/BatchedReduceAddInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/BatchedReduceMinInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ChannelWiseQuantizedConvolutionInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ConvertToInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/Convolution3DInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ConvolutionInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ConvTransposeInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/CopyInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/CopyInstTensorized.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/CrossEntropyLossInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/CumSumInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/Dequantize4BitsColumnBlocksInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/Dequantize8BitsColumnBlocksInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/DequantizeInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/DenoiseMasks.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/MaxSplatInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ElementBoolInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ElementImmLogic.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ElementBinaryInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ElementSelectInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ElementSingleInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/EmbeddingBagInst.h
#  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/EmbeddingBagByteRowwiseOffsetsInst.h"
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ExtractTensorInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ETSOCGenericOpInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/FlipInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/FullyConnectedInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/FusedRowwiseQuantizedSparseLengthsSumInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/FusedRowwiseQuantizedSparseLengthsWeightedSumInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/GatherInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/GatherRangesInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ImageConditioning.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/InsertTensorInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/IntLookupTableInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/LengthsRangeFillInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/LengthsSumInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/LengthsToRangesInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/LocalResponseNormalizationInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/MatMulInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/MatMulInstTransposed.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/MaxPoolInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ModuloInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/NonMaxSuppressionInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/OneHotInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ProfileInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/QuantizeInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/RescaleQuantizedInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ResizeBilinearInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ResizeNearestInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/RowwiseQuantizedFullyConnectedInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/RowwiseQuantizedSparseLengthsWeightedSumInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ScatterDataInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/SoftMaxInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/SpaceToDepthInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/SparseLengthsSumInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/SparseLengthsWeightedSumInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/SparseToDenseInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/SparseToDenseMaskInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/SplatInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/SyncopyInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/TensorViewInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/TopKInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/TransposeInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/TriluInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/inlining.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/0_filterMask.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/1_filterMask.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/2_filterMask.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/3_filterMask.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/4_filterMask.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/5_filterMask.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/6_filterMask.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/7_filterMask.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/8_filterMask.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/9_filterMask.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/Addresser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/Addresser2.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/Compiler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/FFT.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/FFT2.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/FFTTables.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/LoadStore.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/LoadStore2.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/Operator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/Operator2.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/Writer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/Writer2.h
)

add_library(dnn_lib_host_generated_headers INTERFACE)
add_library(dnnLibrary::dnn_lib_host_generated_headers ALIAS dnn_lib_host_generated_headers)

target_include_directories(dnn_lib_host_generated_headers  
       INTERFACE $<INSTALL_INTERFACE:include/dnn_lib/host_headers>)


set_target_properties(dnn_lib_host_generated_headers PROPERTIES
    PUBLIC_HEADER  ${CMAKE_CURRENT_SOURCE_DIR}/include/host_headers/InstrTableGenerated.h
)

# Note: 2nd dnn_lib is to force consumers to namespace it: include <dnn_lib/InstrTableGenerated.h>
install (TARGETS dnn_lib_host_generated_headers
         EXPORT dnnLibraryTargets
         PUBLIC_HEADER DESTINATION  ${CMAKE_INSTALL_INCLUDEDIR}/dnn_lib/host_headers/dnn_lib) 

add_library(dnn_lib INTERFACE) #interface only libraries
         
target_link_libraries(dnn_lib
  INTERFACE
   et-common-libs::cm-umode
)

add_library(dnnLibrary::dnn_lib ALIAS dnn_lib)
target_include_directories(dnn_lib
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/inlining>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/internal>
        $<INSTALL_INTERFACE:include>
)
      
target_compile_features(dnn_lib INTERFACE cxx_std_17)
target_compile_options(dnn_lib
  INTERFACE
        -Wall
        $<$<BOOL:${ENABLE_WARNINGS_AS_ERRORS}>:-Werror>
        # Even w/o treating all warnings as arrors, there are some warnings we don't
        # allow at this compilation stage. This list wil grow as categories are cleaned up.
)
      
set_target_properties(dnn_lib PROPERTIES
    PUBLIC_HEADER "${LIB_DNN_PUBLIC_HEADERS}"
)

install(TARGETS dnn_lib 
    EXPORT dnnLibraryTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dnn_lib
)
install(EXPORT dnnLibraryTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dnnLibrary
    NAMESPACE dnnLibrary::
)

### Add package information
include(CMakePackageConfigHelpers)

configure_package_config_file(dnnLibraryConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/dnnLibraryConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dnnLibrary
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/dnnLibraryConfigVersion.cmake
    VERSION ${CMAKE_PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/dnnLibraryConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/dnnLibraryConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dnnLibrary
)

add_executable(dnnlibraryDummyExecutable
  src/LibNodes.cpp
  src/DummyLogCode.cpp
  src/main.cpp
)

target_link_libraries(dnnlibraryDummyExecutable
  PRIVATE
    dnn_lib
)

target_compile_options(dnnlibraryDummyExecutable
      PRIVATE
        -Wall
        -std=c++17
        $<$<BOOL:${ENABLE_WARNINGS_AS_ERRORS}>:-Werror>
        # Even w/o treating all warnings as arrors, there are some warnings we don't
        # allow at this compilation stage. This list wil grow as categories are cleaned up.
        -Werror=shadow 
        -Werror=ignored-qualifiers 
        -Werror=div-by-zero
        -Werror=unused-but-set-parameter
        -Werror=unused-but-set-variable
        -Werror=unused-parameter
        -Werror=unused-variable
        -Werror=float-conversion
        -Werror=double-promotion
        -Werror=cast-align
        -Werror=maybe-uninitialized
        -Werror=parentheses
        -Werror=unused-local-typedefs
        -Werror=sign-compare
        -Werror=narrowing
      )
