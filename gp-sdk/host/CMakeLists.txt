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
project(gp-sdk-host VERSION 1.0.0 LANGUAGES C CXX)

option(USE_CONAN "Use connan packages" OFF)
option(BUILD_TESTS "Build tests (adds gtest dependency)" OFF)
option(BUILD_DOCS "Enable/disable building documentation" ON)



include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)


message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

if (NOT USE_CONAN)
  message(WARNING "Before open sourcing we should remove all these hardcoded paths")

  # gp-sdkk build on the ML build tree
  # 

  message(STATUS "Building gp-sdk on top of ML build tree. " )
  if (NOT ET_SWPLATFORM_DIR)
    set(ET_SWPLATFORM_DIR ${CMAKE_SOURCE_DIR}/../../../)
   endif()

  if (NOT ET_INSTALL_DIR)
    set(ET_INSTALL_DIR ${ET_SWPLATFORM_DIR}/build/install-staging)
  endif()

  if (NOT CMAKE_MODULE_PATH)
    set (CMAKE_MODULE_PATH ${ET_SWPLATFORM_DIR}/cmake)
  endif()

  add_compile_options(-DET_INSTALL_DIR="""${ET_INSTALL_DIR}""")

  set(deviceApi_DIR ${ET_INSTALL_DIR}/sw-platform-x86_64-sysroot/lib/cmake/deviceApi)
  set(deviceLayer_DIR ${ET_INSTALL_DIR}/sw-platform-x86_64-sysroot/lib/cmake/deviceLayer)
  set(linuxDriver_DIR ${ET_INSTALL_DIR}/sw-platform-x86_64-sysroot/lib/cmake/linuxDriver)
  set(hostUtils_DIR ${ET_INSTALL_DIR}/sw-platform-x86_64-sysroot/lib/cmake/hostUtils)
  set(sw-sysemu_DIR ${ET_INSTALL_DIR}/sw-platform-x86_64-sysroot/lib/cmake/sw-sysemu)
  set(base_DIR ${ET_INSTALL_DIR}/sw-platform-x86_64-sysroot/lib/cmake/base)
  set(runtime_DIR ${ET_INSTALL_DIR}/sw-platform-x86_64-sysroot/lib/cmake/runtime)
  set(esperantoTrace_DIR ${ET_INSTALL_DIR}/sw-platform-x86_64-sysroot/lib/cmake/esperantoTrace)

  message(STATUS "Build type: " ${CMAKE_BUILD_TYPE} )
  message(STATUS "SDK root: " ${ET_INSTALL_DIR} )
  message(STATUS "deviceApi_DIR is " ${deviceApi_DIR})
  message(STATUS "deviceLayer_DIR is " ${deviceLayer_DIR})
  message(STATUS "linuxDriver_DIR is " ${linuxDriver_DIR})
  message(STATUS "hostUtils_DIR is " ${hostUtils_DIR})
  message(STATUS "sw-sysemu_DIR is " ${sw-sysemu_DIR})
  message(STATUS "base_DIR is " ${base_DIR})
  message(STATUS "runtime_DIR is " ${runtime_DIR})

  add_definitions(-DWITH_SYSEMU_PATHS)
endif()

find_package(deviceApi REQUIRED)
find_package(deviceLayer REQUIRED)
find_package(runtime REQUIRED)
find_package(esperantoTrace REQUIRED)
find_package(FFTW3f REQUIRED)
find_package(gflags REQUIRED)

find_package(gflags REQUIRED)
add_subdirectory(sdk)

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()

## project documentation
if (BUILD_DOCS)
    add_subdirectory(${CMAKE_SOURCE_DIR}/../docs/host ${CMAKE_SOURCE_DIR}/../docs/host/build)
endif()

# project-specific directories to be added here
