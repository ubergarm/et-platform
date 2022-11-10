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

add_library(crt_shared STATIC
  lib/deviceGpSdk.cc
  lib/SyncComputeNode.cc
  lib/crt0.S
)
target_include_directories(crt_shared
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include
)
set_target_properties(crt_shared PROPERTIES 
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
target_link_libraries(crt_shared
    PUBLIC
        et-common-libs::cm-umode
)


#set(LINKER_SCRIPT_ABS_PATH ${CMAKE_CURRENT_LIST_DIR}/lib/linker.ld)

# Add explicit dependency on linker script when linking target
#set_target_properties(test1 PROPERTIES LINK_DEPENDS ${LINKER_SCRIPT_ABS_PATH})
#set(MAP_FILE test1.map)

# Use custom linker script #-lm -lgcc --> libgcc.a/so
#set(ELF_EXE_LINKER_FLAGS "-nostdlib -nostartfiles -Wl,--gc-sections -Xlinker -Map=${MAP_FILE} -e _start -Wl,--start-group -lm -lgcc -T ${LINKER_SCRIPT_ABS_PATH}")


#set_target_properties(test1 PROPERTIES LINK_FLAGS ${ELF_EXE_LINKER_FLAGS})
