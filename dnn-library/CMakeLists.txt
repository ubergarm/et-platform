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
project(dnnLibrary VERSION 0.1.0 LANGUAGES C CXX)

message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

option(ENABLE_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

include(GNUInstallDirs)

find_package(et-common-libs REQUIRED COMPONENTS cm-umode)


file(GLOB LIB_DNN_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/Fully-macro-defs.S
    ${CMAKE_CURRENT_SOURCE_DIR}/include/AutoGenInstan.def
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Float16.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/GenInstances.h
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
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/DequantizeInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/MaxSplatInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ElementBoolInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ElementImmLogic.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ElementBinaryInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ElementSelectInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ElementSingleInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/EmbeddingBagInst.h
#  "${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/EmbeddingBagByteRowwiseOffsetsInst.h"
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/ExtractTensorInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/FlipInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/FullyConnectedInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/FusedRowwiseQuantizedSparseLengthsSumInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/FusedRowwiseQuantizedSparseLengthsWeightedSumInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/GatherInst.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/GatherRangesInst.h
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
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inlining/inlining.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/Addresser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/LoadStore.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/Operator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/PackedBufferProxy.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/Writer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/libc_stub/assert.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/libc_stub/stdlib.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/internal/libc_stub/string.h
)

add_library(dnn_lib STATIC
    src/AdaptiveAvgPoolInst.cpp
    src/ArgMaxInst.cpp
    src/AvgPoolInst.cpp
    src/BatchedAddInst.cpp
    src/BatchedReduceAddInst.cpp
    src/BatchedReduceMinInst.cpp
    src/BatchOneHotInst.cpp
    src/ChannelWiseQuantizedConvolutionInst.cpp
    src/ConvertToInst.cpp
    src/Convolution3DInst.cpp
    src/ConvolutionInst.cpp
    src/ConvTransposeInst.cpp
    src/CopyInst.cpp
    src/CrossEntropyLossInst.cpp
    src/CumSumInst.cpp
    src/DequantizeInst.cpp
    src/ElementAddInst.cpp
    src/ElementCmpEQInst.cpp
    src/ElementCmpLTEInst.cpp
    src/ElementCmpLTInst.cpp
    src/ElementDivInst.cpp
    src/ElementLogInst.cpp
    src/ElementMaxInst.cpp
    src/ElementMinInst.cpp
    src/ElementMulInst.cpp
    src/ElementPowInst.cpp
    src/ElementSubInst.cpp
    src/Int8ConverterInst.cpp
    src/ElementExpInst.cpp
    src/ElementIsNaNInst.cpp
    src/ElementSelectInst.cpp
    src/EmbeddingBagInst.cpp
#   src/EmbeddingBagByteRowwiseOffsetsInst.cpp
    src/ExtractTensorInst.cpp
    src/FlipInst.cpp
    src/FullyConnectedInst.cpp
    src/FusedRowwiseQuantizedSparseLengthsSumInst.cpp
    src/FusedRowwiseQuantizedSparseLengthsWeightedSumInst.cpp
    src/GatherInst.cpp
    src/GatherRangesInst.cpp
    src/InsertTensorInst.cpp
    src/IntLookupTableInst.cpp
    src/LengthsRangeFillInst.cpp
    src/LengthsSumInst.cpp
    src/LengthsToRangesInst.cpp
    src/LocalResponseNormalizationInst.cpp
    src/MatMulInst.cpp
    src/MaxSplatInst.cpp
    src/MaxPoolInst.cpp
    src/ModuloInst.cpp
    src/NonMaxSuppressionInst.cpp
    src/QuantizeInst.cpp
    src/RescaleQuantizedInst.cpp
    src/ResizeBilinearInst.cpp
    src/ResizeNearestInst.cpp
    src/RowwiseQuantizedFullyConnectedInst.cpp
    src/RowwiseQuantizedSparseLengthsWeightedSumInst.cpp
    src/ScatterDataInst.cpp
    src/SigmoidInst.cpp
    src/SoftMaxInst.cpp
    src/SpaceToDepthInst.cpp
    src/SparseLengthsWeightedSumInst.cpp
    src/SparseToDenseInst.cpp
    src/SparseToDenseMaskInst.cpp
    src/SplatInst.cpp
    src/SyncopyInst.cpp
    src/TanhInst.cpp
    src/TensorViewInst.cpp
    src/TopKInst.cpp
    src/TransposeInst.cpp
    src/Float16.cpp
    src/libc_stub/stdlib.c
    src/libc_stub/string.c
)
add_library(dnnLibrary::dnn_lib ALIAS dnn_lib)
target_include_directories(dnn_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/inlining>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/internal>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/internal/libc_stub>
        $<INSTALL_INTERFACE:include/dnn_lib>
)
target_compile_features(dnn_lib PRIVATE cxx_std_17)
target_compile_options(dnn_lib
    PRIVATE
        -Wall
        $<$<BOOL:${ENABLE_WARNINGS_AS_ERRORS}>:-Werror>
)


# Even w/o treating all warnings as arrors, there are some warnings we don't
# allow at this compilation stage. This list wil grow as categories are cleaned up.
target_compile_options(dnn_lib
    PRIVATE
        -Werror=shadow 
        -Werror=ignored-qualifiers 
        -Werror=div-by-zero
)

target_link_libraries(dnn_lib
    PRIVATE
        et-common-libs::cm-umode
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
