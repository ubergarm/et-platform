#------------------------------------------------------------------------------
# Copyright (C) 2019, Esperanto Technologies Inc.
# The copyright to the computer program(s) herein is the
# property of Esperanto Technologies, Inc. All Rights Reserved.
# The program(s) may be used and/or copied only with
# the written permission of Esperanto Technologies and
# in accordance with the terms and conditions stipulated in the
# agreement/contract under which the program(s) have been supplied.
#------------------------------------------------------------------------------


find_package(et-common-libs REQUIRED COMPONENTS cm-umode)
find_package(dnnLibrary REQUIRED)

add_library(etsoc_crt0 STATIC
  lib/deviceGpSdk.cc
  lib/crt0.S
)
target_include_directories(etsoc_crt0
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include
)

target_compile_options(etsoc_crt0 PRIVATE -fno-exceptions -falign-functions=64 -O3 -g3)
target_link_libraries(etsoc_crt0 PUBLIC etgcc etm)

add_library(etgcc STATIC
  libetgcc/divsf.c
)
target_include_directories(etgcc
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/libetgcc/include
)

target_compile_options(etgcc PUBLIC -mno-fdiv)

add_library(etm STATIC
  libetm/ef_sqrtf.c
)
target_include_directories(etm
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/libetm/include
)
target_compile_options(etm PUBLIC -mno-fdiv)



add_library(libcpp_throw_stub STATIC libcpp_throw_stub/functexcept.cc)
target_link_libraries(libcpp_throw_stub
    PRIVATE
        et-common-libs::cm-umode
)


set_target_properties(etsoc_crt0 PROPERTIES 
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
target_link_libraries(etsoc_crt0
    PUBLIC
        et-common-libs::cm-umode
        libcpp_throw_stub
)



