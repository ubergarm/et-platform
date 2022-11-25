#------------------------------------------------------------------------------
# Copyright (C) 2019, Esperanto Technologies Inc.
# The copyright to the computer program(s) herein is the
# property of Esperanto Technologies, Inc. All Rights Reserved.
# The program(s) may be used and/or copied only with
# the written permission of Esperanto Technologies and
# in accordance with the terms and conditions stipulated in the
# agreement/contract under which the program(s) have been supplied.
#------------------------------------------------------------------------------

macro(add_etsoc_riscv_executable TARGET_NAME TARGET_SOURCES) 

  set(LINKER_SCRIPT_ABS_PATH ${PROJECT_SOURCE_DIR}/sdk/lib/linker.ld)
  
  set(MAP_FILE ${TARGET_NAME}.map)

  add_executable(${TARGET_NAME} ${TARGET_SOURCES})

  if(EXISTS "${PROJECT_SOURCE_DIR}/tests/${TARGET_NAME}/include")
    target_include_directories(${TARGET_NAME} PRIVATE "${TARGET_NAME}/include")
  endif()
  
  target_link_libraries(${TARGET_NAME}
    PRIVATE
    et-common-libs::cm-umode
  )
  
  #@TODO we should commonoalize the options into some flags set to avoid repeating them.
  set(ELF_EXE_LINKER_FLAGS "-nostdlib -nostartfiles -Wl,--gc-sections -Xlinker -Map=${MAP_FILE} -e _start -Wl,--start-group -lm -lgcc -T ${LINKER_SCRIPT_ABS_PATH}")
  
  set_target_properties(${TARGET_NAME}
    PROPERTIES
    SUFFIX  ".elf"
    COMPILE_FLAGS "-fno-exceptions -DNDEBUG -falign-functions=64 -O3 -g3"
    LINK_DEPENDS ${LINKER_SCRIPT_ABS_PATH}
    LINK_FLAGS ${ELF_EXE_LINKER_FLAGS}
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}" #bin
  )
  
  #Mov map file to the proper folder.
  add_custom_command(TARGET ${TARGET_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rename "${CMAKE_CURRENT_BINARY_DIR}/${MAP_FILE}"
    "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}/${MAP_FILE}"
  )

  set(LINKER_SCRIPT_ABS_PATH  ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}/debug_linker.ld)
  set(DEBUG_MAP_FILE "${TARGET_NAME}_dbg.map")
  set(ELF_EXE_LINKER_FLAGS "-nostdlib -nostartfiles -Wl,--gc-sections -Xlinker -Map=${DEBUG_MAP_FILE} -e _start -Wl,--start-group -lm -lgcc -T ${LINKER_SCRIPT_ABS_PATH}")
  set(DEBUG_TARGET "${TARGET_NAME}_dbg")

  add_executable(${DEBUG_TARGET} ${TARGET_SOURCES})

  if(EXISTS "${PROJECT_SOURCE_DIR}/tests/${TARGET_NAME}/include")
    target_include_directories(${DEBUG_TARGET} PRIVATE "${TARGET_NAME}/include")
  endif() 

  target_link_libraries(${DEBUG_TARGET}
    PRIVATE
    et-common-libs::cm-umode  
  )

  set_target_properties(${DEBUG_TARGET}
    PROPERTIES
    SUFFIX  ".elf"
    COMPILE_FLAGS "-fno-exceptions -DNDEBUG -falign-functions=64 -O3 -g3"
    LINK_DEPENDS ${LINKER_SCRIPT_ABS_PATH}
    LINK_FLAGS ${ELF_EXE_LINKER_FLAGS}
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}" #bin
  )

  add_custom_target("GenLinker_${DEBUG_TARGET}"
    COMMAND sed '0,/0x00000001/s//${DEBUG_ADDRESS}/' ${PROJECT_SOURCE_DIR}/sdk/lib/linker.ld > ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}/debug_linker.ld    
  )
  
  add_dependencies(${DEBUG_TARGET} "GenLinker_${DEBUG_TARGET}")

  add_custom_command(TARGET ${DEBUG_TARGET}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rename "${CMAKE_CURRENT_BINARY_DIR}/${DEBUG_MAP_FILE}"
    "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}/${DEBUG_MAP_FILE}"    
    COMMAND ${CMAKE_COMMAND} -E rm "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}/debug_linker.ld"
  )

endmacro(add_etsoc_riscv_executable)

macro(add_etsoc_riscv_target_link TARGET_NAME TARGET_LIBS)

  target_link_libraries(${TARGET_NAME}
    PRIVATE
    "${TARGET_LIBS}"
  )

  target_link_libraries("${TARGET_NAME}_dbg"
    PRIVATE
    "${TARGET_LIBS}"
  )

endmacro(add_etsoc_riscv_target_link)

macro(add_etsoc_riscv_include TARGET_NAME INCLUDES)
  target_include_directories(${TARGET_NAME} PRIVATE ${INCLUDES})
  target_include_directories("${TARGET_NAME}_dbg" PRIVATE ${INCLUDES})
endmacro(add_etsoc_riscv_include)
