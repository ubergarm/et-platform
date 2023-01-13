#------------------------------------------------------------------------------
# Copyright (C) 2022, Esperanto Technologies Inc.
# The copyright to the computer program(s) herein is the
# property of Esperanto Technologies, Inc. All Rights Reserved.
# The program(s) may be used and/or copied only with
# the written permission of Esperanto Technologies and
# in accordance with the terms and conditions stipulated in the
# agreement/contract under which the program(s) have been supplied.
#------------------------------------------------------------------------------

math(EXPR DEBUG_ADDRESS "${ADDRESS} + 0x1000" OUTPUT_FORMAT HEXADECIMAL)
message(STATUS "Base address used to relink (debug) ELFs -> ${DEBUG_ADDRESS}")

#
# Adds a riscv executable prepared for being executed as a kernel on ET-SoC-1.
# @param TARGET_NAME name of the cmake target to be created.
# @param TARGET_SOURCES_LIST space-separated list of source files
# Notes:
# - This will also create a secondary target `${TARGET_NAME}_dbg`.
#   The debug target is relinked using a runtime-predicted address and should be used with debuggers.
# - The created target is a standard cmake target, so compilation options and dependencies  can be
#   set with standard cmake functions (target_link_libraries, target_include_directories, etc.).
#
macro(add_etsoc_riscv_executable TARGET_NAME TARGET_SOURCES_LIST) 

  # merge parameter list
  set(TARGET_SOURCES ${TARGET_SOURCES_LIST} ${ARGN} ) 

  set(LINKER_SCRIPT_ABS_PATH ${GP_SDK_LINKER_SCRIPT})

  add_executable(${TARGET_NAME} ${TARGET_SOURCES})
  
  # enabling exports allows crating a dependeny with the host app to 
  # reach xxx_kernel_arguments header exported by the kernel
  set_property(TARGET ${TARGET_NAME} PROPERTY ENABLE_EXPORTS 1)
 
  target_link_libraries(${TARGET_NAME}
    et-common-libs::cm-umode
  )

  set(ELF_EXE_LINKER_FLAGS_BASE "-nostdlib -nostartfiles -Wl,--gc-sections  -e _start -Wl,--start-group -lm -lgcc")
  
  set(ELF_EXE_LINKER_FLAGS "${ELF_EXE_LINKER_FLAGS_BASE} -T ${LINKER_SCRIPT_ABS_PATH} -Wl,--defsym=BASE_ADDRESS=0")
  set(ELF_EXE_LINKER_FLAGS_DBG "${ELF_EXE_LINKER_FLAGS_BASE} -T ${LINKER_SCRIPT_ABS_PATH} -Wl,--defsym=BASE_ADDRESS=${DEBUG_ADDRESS}")

  target_compile_options(${TARGET_NAME} PRIVATE -fno-exceptions -falign-functions=64 -O3 -g3 -Wstack-usage=4096 -DNDEBUG)
  target_compile_definitions(${TARGET_NAME} PRIVATE "-DNDEBUG")

  set_target_properties(${TARGET_NAME}
    PROPERTIES
    LINK_DEPENDS ${LINKER_SCRIPT_ABS_PATH}
    LINK_FLAGS ${ELF_EXE_LINKER_FLAGS}
  )
  
  # adding a second "debug" target (identical, but linked using a different base address)
  set(DEBUG_TARGET "${TARGET_NAME}_dbg")
  
  add_executable(${DEBUG_TARGET} $<TARGET_PROPERTY:${TARGET_NAME},SOURCES>)
  target_link_libraries(${DEBUG_TARGET} PRIVATE $<TARGET_PROPERTY:${TARGET_NAME},LINK_LIBRARIES>) 
  target_include_directories(${DEBUG_TARGET} PRIVATE $<TARGET_PROPERTY:${TARGET_NAME},INCLUDE_DIRECTORIES>) 

  target_compile_options(${DEBUG_TARGET}  PRIVATE $<TARGET_PROPERTY:${TARGET_NAME},COMPILE_OPTIONS>)
  target_compile_definitions(${DEBUG_TARGET} PRIVATE   $<TARGET_PROPERTY:${TARGET_NAME},COMPILE_DEFINITIONS>)

  set_target_properties(${DEBUG_TARGET}  
     PROPERTIES  
     LINK_DEPENDS ${LINKER_SCRIPT_ABS_PATH}
     LINK_FLAGS  ${ELF_EXE_LINKER_FLAGS_DBG})

  add_custom_command(TARGET ${DEBUG_TARGET}
    POST_BUILD
    COMMAND ${GP_SDK_TOOLS_PATH}/scripts/check_unimplemented_instructions.sh ${DEBUG_TARGET}
  )


endmacro(add_etsoc_riscv_executable)

