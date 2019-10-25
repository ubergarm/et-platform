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

project(dnn_lib VERSION 0.1.0 LANGUAGES CXX C ASM)

if (CLANG_BUILD)
  set(CLANG_BUILD_DEF 1)
else()
  set(CLANG_BUILD_DEF "")
endif(CLANG_BUILD)

message("Clang Build is set to ${CLANG_BUILD_DEF}")

add_custom_target(dnn_lib ALL
  COMMAND $(MAKE) -C ${CMAKE_CURRENT_SOURCE_DIR}
                     CLANG_BUILD=${CLANG_BUILD_DEF}
                     BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}/build
                     SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/src
                     all
  VERBATIM)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
  DESTINATION ${CMAKE_INSTALL_PREFIX}/dnn_lib
  COMPONENT dnn_lib)
