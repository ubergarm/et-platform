#------------------------------------------------------------------------------
# Copyright (C) 2022, Esperanto Technologies Inc.
# The copyright to the computer program(s) herein is the
# property of Esperanto Technologies, Inc. All Rights Reserved.
# The program(s) may be used and/or copied only with
# the written permission of Esperanto Technologies and
# in accordance with the terms and conditions stipulated in the
# agreement/contract under which the program(s) have been supplied.
#------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.5)

project(gp-sdk-device VERSION 1.0.0 LANGUAGES C CXX ASM)

option(USE_CONAN "Use connan packages" OFF)

if (NOT USE_CONAN)
  if (NOT ET_SWPLATFORM_DIR)
    set(ET_SWPLATFORM_DIR ${CMAKE_SOURCE_DIR}/../../../)
  endif()

  if (NOT ET_INSTALL_DIR)
    set(ET_INSTALL_DIR ${ET_SWPLATFORM_DIR}/build/install-staging)
  endif()

  if (NOT CMAKE_MODULE_PATH)
    set (CMAKE_MODULE_PATH ${ET_SWPLATFORM_DIR}/cmake)
  endif()

  set(et-common-libs_DIR ${ET_INSTALL_DIR}/sw-platform-riscv-sysroot/lib/cmake/et-common-libs)
  set(etsoc_hal_DIR ${ET_INSTALL_DIR}/sw-platform-riscv-sysroot/lib/cmake/etsoc_hal)
  set(esperantoTrace_DIR ${ET_INSTALL_DIR}/sw-platform-riscv-sysroot/lib/cmake/esperantoTrace)
  set(dnnLibrary_DIR ${ET_INSTALL_DIR}/sw-platform-riscv-sysroot/lib/cmake/dnnLibrary)
endif()

if (NOT ADDRESS)
  message(FATAL_ERROR "ADRESS parameter is missing. Please provide a value (e.g: -DADDRESS:STRING=0x8005b35000")
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(et-common-libs REQUIRED COMPONENTS cm-umode)
find_package(dnnLibrary)

include("${CMAKE_SOURCE_DIR}/riscv_helpers.cmake")

if (NOT ET_SDK_TOOLS_PATH)
  set(ET_SDK_TOOLS_PATH ${CMAKE_SOURCE_DIR}/../tools)
endif()

add_subdirectory(sdk)
add_subdirectory(tests)

# Users of this CMakeLists.txt can provide an additional subdirectory by passing build arguments,
# as opposed to modifying this CMakeLists.txt. This CMake configuration mechanism is intended for
# consumers of the GP-SDK to build custom kernels leveraging the same build recipe from the
# internal GP-SDK tests. The accepted parameters are:
#
# - CUSTOM_KERNELS_SRC_DIR: directory where the sources and the corresponding CMakeLists.txt
#   are located. It can be located out of this tree.
#
# - CUSTOM_KERNELS_BIN_DIR: directory where the binary files will be stored. If the directory
#   does not exist it gets created.
#
# If one such pair of arguments is provided it is handled at this stage.

if (CUSTOM_KERNELS_SRC_DIR AND CUSTOM_KERNELS_BIN_DIR)
    add_subdirectory(${CUSTOM_KERNELS_SRC_DIR} ${CUSTOM_KERNELS_BIN_DIR})
endif()

# Users can also hack their way by just editing this file and have their own project added as a
# subdirectory here without any extra arguments being passed. That build method is suggested in
# the tutorial from the GP-SDK user guide but has also been labeled as not ideal for production by
# some users of the GP-SDK.
#
# add_subdirectory(tutorial_example)
