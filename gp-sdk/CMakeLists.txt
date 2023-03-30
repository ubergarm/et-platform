cmake_minimum_required(VERSION 3.5)

project(gp-sdk VERSION 1.1.0 LANGUAGES C CXX ASM)

if(NOT DEFINED ENV{ET_SDK_HOME})
  message(
    FATAL_ERROR
      "Missing $ET_SDK_HOME. Please run from inside the et_docker environment")
endif()

include(ExternalProject)

if (NOT ADDRESS)
  message(FATAL_ERROR "ADDRESS parameter is missing. Please provide a value for the predicted device load address (e.g: -DADDRESS=0x8006335000")
endif()

ExternalProject_Add(
  host
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/host"
  CMAKE_ARGS
    -DCMAKE_TOOLCHAIN_FILE=$ENV{ET_SDK_HOME}/.builds/host/conan_toolchain.cmake
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/local)

ExternalProject_Add(
  device
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/device"
  CMAKE_ARGS
    -DCMAKE_TOOLCHAIN_FILE=$ENV{ET_SDK_HOME}/.builds/device/conan_toolchain.cmake
    -DCMAKE_BUILD_TYPE=Release
    -DADDRESS:STRING=${ADDRESS}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/local)
