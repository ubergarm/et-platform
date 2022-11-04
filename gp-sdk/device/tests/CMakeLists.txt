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

set(LINKER_SCRIPT_ABS_PATH ${CMAKE_CURRENT_LIST_DIR}/../sdk/lib/linker.ld)

file( GLOB TEST_NAME  ${PROJECT_SOURCE_DIR}/tests/*/*.cc )

foreach( testsourcefile ${TEST_NAME} )
  get_filename_component(EXECUTABLE_NAME ${testsourcefile} NAME_WE)
  get_filename_component(EXECUTABLE_NAME_DIRECTORY ${testsourcefile} DIRECTORY) 

  file(RELATIVE_PATH
    ITEM_PATH_REL # Output variable
    "${PROJECT_SOURCE_DIR}/tests" # Base directory
    ${EXECUTABLE_NAME_DIRECTORY} # Absolute path to the file
  )
    
  set(CURRENT_TARGET_NAME "${ITEM_PATH_REL}_${EXECUTABLE_NAME}")   
  set(MAP_FILE ${EXECUTABLE_NAME}.map)
  #WIP we should find to pass this linker flags in friendly CMAKE way.
  set(ELF_EXE_LINKER_FLAGS "-nostdlib -nostartfiles -Wl,--gc-sections -Xlinker -Map=${MAP_FILE} -e _start -Wl,--start-group -lm -lgcc -T ${LINKER_SCRIPT_ABS_PATH}")

  add_executable(${CURRENT_TARGET_NAME}  ${testsourcefile})
  
  target_link_libraries(${CURRENT_TARGET_NAME}
    PRIVATE
    crt_shared
    et-common-libs::cm-umode
  )
    
  set_target_properties(${CURRENT_TARGET_NAME}
    PROPERTIES
    LINK_DEPENDS ${LINKER_SCRIPT_ABS_PATH}
    LINK_FLAGS ${ELF_EXE_LINKER_FLAGS}
    #ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/${ITEM_PATH_REL}}/lib"
    #LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/${ITEM_PATH_REL}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${ITEM_PATH_REL}" #bin
  )   
    
  #Rename output to the ${EXECUTABLE_NAME}
  add_custom_command(TARGET "${ITEM_PATH_REL}_${EXECUTABLE_NAME}" POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E rename "${CMAKE_CURRENT_BINARY_DIR}/${ITEM_PATH_REL}/${CURRENT_TARGET_NAME}"
                    "${CMAKE_CURRENT_BINARY_DIR}/${ITEM_PATH_REL}/${EXECUTABLE_NAME}")

                  
  #Mov map file to the proper folder.
  add_custom_command(TARGET "${ITEM_PATH_REL}_${EXECUTABLE_NAME}" POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E rename "${CMAKE_CURRENT_BINARY_DIR}/${MAP_FILE}"
                     "${CMAKE_CURRENT_BINARY_DIR}/${ITEM_PATH_REL}/${MAP_FILE}")

endforeach( testsourcefile ${TEST_NAME} )
