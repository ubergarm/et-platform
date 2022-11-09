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

project(gp-sdk-device VERSION 0.1.0 LANGUAGES C CXX ASM)

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

add_subdirectory(sdk)
add_subdirectory(tests)
