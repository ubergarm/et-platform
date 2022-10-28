cmake_minimum_required(VERSION 3.5)
project(examples VERSION 0.1.0 LANGUAGES CXX C)

if (NOT ET_SWPLATFORM_DIR)
  set(ET_SWPLATFORM_DIR ${CMAKE_SOURCE_DIR}/../../../)
endif()

if (NOT ET_INSTALL_DIR)
  set(ET_INSTALL_DIR ${ET_SWPLATFORM_DIR}/build/install-staging)
endif()

if (NOT CMAKE_MODULE_PATH)
  set (CMAKE_MODULE_PATH ${ET_SWPLATFORM_DIR}/cmake)
endif()

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-O0 -g)
  message(STATUS "Disabling all the optimizations in debug build")
else()
  message(STATUS "Not overriding any options")
endif()

add_compile_options(-DET_INSTALL_DIR="""${ET_INSTALL_DIR}""")

set(deviceApi_DIR ${ET_INSTALL_DIR}/sw-platform-x86_64-sysroot/lib/cmake/deviceApi)
set(deviceLayer_DIR ${ET_INSTALL_DIR}/sw-platform-x86_64-sysroot/lib/cmake/deviceLayer)
set(linuxDriver_DIR ${ET_INSTALL_DIR}/sw-platform-x86_64-sysroot/lib/cmake/linuxDriver)
set(hostUtils_DIR ${ET_INSTALL_DIR}/sw-platform-x86_64-sysroot/lib/cmake/hostUtils)
set(sw-sysemu_DIR ${ET_INSTALL_DIR}/sw-platform-x86_64-sysroot/lib/cmake/sw-sysemu)
set(base_DIR ${ET_INSTALL_DIR}/sw-platform-x86_64-sysroot/lib/cmake/base)
set(runtime_DIR ${ET_INSTALL_DIR}/sw-platform-x86_64-sysroot/lib/cmake/runtime)

message(STATUS "Build type: " ${CMAKE_BUILD_TYPE} )
message(STATUS "SDK root: " ${ET_INSTALL_DIR} )
message(STATUS "deviceApi_DIR is " ${deviceApi_DIR})
message(STATUS "deviceLayer_DIR is " ${deviceLayer_DIR})
message(STATUS "linuxDriver_DIR is " ${linuxDriver_DIR})
message(STATUS "hostUtils_DIR is " ${hostUtils_DIR})
message(STATUS "sw-sysemu_DIR is " ${sw-sysemu_DIR})
message(STATUS "base_DIR is " ${base_DIR})
message(STATUS "runtime_DIR is " ${runtime_DIR})

add_subdirectory(sdk)
add_subdirectory(tests)

