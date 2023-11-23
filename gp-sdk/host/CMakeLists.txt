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

project(gp-sdk-host VERSION 1.4.0 LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

option(BUILD_DOCS "Enable/disable building documentation" OFF)
option(BUILD_TESTS "Build example launchers" ON)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

find_package(deviceApi REQUIRED)
find_package(deviceLayer REQUIRED)
find_package(runtime REQUIRED)
find_package(esperantoTrace REQUIRED)
find_package(FFTW3f REQUIRED)

include(SanitizerSupport)

add_subdirectory(sdk)

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()

## project documentation
if (BUILD_DOCS)
    add_subdirectory(${CMAKE_SOURCE_DIR}/../docs/host ${CMAKE_SOURCE_DIR}/../docs/host/build)
endif()

# project-specific directories to be added here
