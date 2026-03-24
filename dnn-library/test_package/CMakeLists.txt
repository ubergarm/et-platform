cmake_minimum_required(VERSION 3.8)
project(test_package LANGUAGES CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(dnnLibrary REQUIRED)

if (WITH_HOST_HEADERS)
    if (NOT TARGET dnnLibrary::dnn_lib_host_generated_headers)
        message(FATAL_ERROR "Missing target: dnnLibrary::dnn_lib_host_generated_headers")
    endif()

    get_target_property(DNNLIB_HOST_HEADERS_INCLUDE_DIR dnnLibrary::dnn_lib_host_generated_headers INTERFACE_INCLUDE_DIRECTORIES)

    if (NOT DNNLIB_HOST_HEADERS_INCLUDE_DIR)
        message(FATAL_ERROR "Missing INTERFACE_INCLUDE_DIRECTORIES for target: dnnLibrary::dnn_lib_host_generated_headers")
    endif()

    message(STATUS "OK: dnnlibrary with_host_headers --> 1) target dnnLibrary::dnn_lib_host_generated_headers is available 2) variable DNNLIB_HOST_HEADERS_INCLUDE_DIR is defined (${DNNLIB_HOST_HEADERS_INCLUDE_DIR})")
endif()


if (WITH_DEVICE_HEADERS)
    add_executable(test_package_device test_package_device.cpp)
    target_link_libraries(test_package_device PRIVATE dnnLibrary::dnnLibrary)
endif()
