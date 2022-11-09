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
  
  target_link_libraries(${TARGET_NAME}
    PRIVATE
    crt_shared
    et-common-libs::cm-umode
  )

  set(ELF_EXE_LINKER_FLAGS "-nostdlib -nostartfiles -Wl,--gc-sections -Xlinker -Map=${MAP_FILE} -e _start -Wl,--start-group -lm -lgcc -T ${LINKER_SCRIPT_ABS_PATH}")
  
  set_target_properties(${TARGET_NAME}
    PROPERTIES
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

  set(LINKER_SCRIPT_ABS_PATH  ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}/relinker.ld)
  set(MAP_FILE "${TARGET_NAME}_dbg.map")
  set(ELF_EXE_LINKER_FLAGS "-nostdlib -nostartfiles -Wl,--gc-sections -Xlinker -Map=${MAP_FILE} -e _start -Wl,--start-group -lm -lgcc -T ${LINKER_SCRIPT_ABS_PATH}")
  set(RELINK_TARGET "${TARGET_NAME}_dbg")
  
  add_executable(${RELINK_TARGET} ${TARGET_SOURCES})

  target_link_libraries(${RELINK_TARGET}
    PRIVATE
    crt_shared
    et-common-libs::cm-umode
  )

  set_target_properties(${RELINK_TARGET}
    PROPERTIES
    LINK_DEPENDS ${LINKER_SCRIPT_ABS_PATH}
    LINK_FLAGS ${ELF_EXE_LINKER_FLAGS}
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}" #bin
  )

  add_custom_target("GenLinker_${RELINK_TARGET}"
    COMMAND sed '0,/0x00000001/s//${ADDRESS}/' ${PROJECT_SOURCE_DIR}/sdk/lib/linker.ld > ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}/relinker.ld    
  )

  add_dependencies(${RELINK_TARGET} "GenLinker_${RELINK_TARGET}")

  add_custom_command(TARGET ${RELINK_TARGET}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rename "${CMAKE_CURRENT_BINARY_DIR}/${MAP_FILE}"
    "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}/${MAP_FILE}"    
    COMMAND ${CMAKE_COMMAND} -E rm "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}/relinker.ld"
  )
  
endmacro(add_etsoc_riscv_executable)
