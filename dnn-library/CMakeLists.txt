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

project(dnn_lib VERSION 0.1.0 LANGUAGES CXX)

find_package(LibDeviceCommon REQUIRED)
find_package(fake-fw REQUIRED)

set(INCLUDE_INSTALL_DIR include/ CACHE STRING "Include install dir")
set(LIB_INSTALL_DIR     lib/     CACHE STRING "Library install dir")

file(GLOB INSTALL_HEADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/AutoGenInstan.def"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/Float16.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/LibCommon.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/LibNodes.h"
)

option(CLANG_BUILD "Build using clang")

if(CLANG_BUILD)
  set(CMAKE_CXX_COMPILER "$ENV{LLVM_RVET}/bin/clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mcmodel=medium -target riscv64 -march=rv64-et -mabi=lp64f")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --sysroot=$(RISCV)/riscv64-unknown-elf")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I$(RISCV)/riscv64-unknown-elf/include/c++/8.2.0/")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I$(RISCV)/riscv64-unknown-elf/include/c++/8.2.0/riscv64-unknown-elf")
else()
  set(CMAKE_CXX_COMPILER "riscv64-unknown-elf-g++")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mcmodel=medany -march=rv64imf -mabi=lp64f")
endif()

message("${PROJECT_NAME}: Clang Build is set to ${CLANG_BUILD}")

add_library(dnn_lib STATIC
  src/LibNodes.cpp
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
