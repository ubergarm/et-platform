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

  #@TODO we should commonoalize the options into some flags set to avoid repeating them.
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

  set(LINKER_SCRIPT_ABS_PATH  ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}/debug_linker.ld)
  set(DEBUG_MAP_FILE "${TARGET_NAME}_dbg.map")
  set(ELF_EXE_LINKER_FLAGS "-nostdlib -nostartfiles -Wl,--gc-sections -Xlinker -Map=${DEBUG_MAP_FILE} -e _start -Wl,--start-group -lm -lgcc -T ${LINKER_SCRIPT_ABS_PATH}")
  set(DEBUG_TARGET "${TARGET_NAME}_dbg")
  
  add_executable(${DEBUG_TARGET} ${TARGET_SOURCES})

  target_link_libraries(${DEBUG_TARGET}
    PRIVATE
    crt_shared
    et-common-libs::cm-umode
  )

  set_target_properties(${DEBUG_TARGET}
    PROPERTIES
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


# There a few immediate issues to consider:

# It only considers CMAKE_CXX_FLAGS (common to all build types),
# not CMAKE_CXX_FLAGS_<DEBUG|RELEASE> etc.
# The macros can only handle one flag at a time
# -remove_flag_from_file() can only handle one target and input file as input at a time.
# -remove_flag_from_file() expects a filename, not a path. Passing, say,
#     source/foobar/foobar.cpp will not work, since source/foobar/foobar.cpp will be
#     compared against foobar.cpp. This can be fixed by using get_source_file_property()
#     and the LOCATION property and placing a precondition that _file is a full path.
#   -What happens if a target have two or more files with the same name?
# - remove_flag_from_file() can probably be optimized and improved greatly.
# The call to separate_arguments() assumes Unix.
# Most of these should be fairly easy to fix.

#
# Applies CMAKE_CXX_FLAGS to all targets in the current CMake directory.
# After this operation, CMAKE_CXX_FLAGS is cleared.
#
#  This macro first creates a list from CMAKE_CXX_FLAGS, then it gets a list of all
# targets and applies CMAKE_CXX_FLAGS to each of the targets. Finally, CMAKE_CXX_FLAGS
# is cleared. _flag_sync_required is used to indicate if we need to force a rewrite
# of cached variables.
macro(apply_global_cxx_flags_to_all_targets)
    separate_arguments(_global_cxx_flags_list UNIX_COMMAND ${CMAKE_CXX_FLAGS})
    
    get_property(_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_target ${_targets})
      target_compile_options(${_target} PUBLIC ${_global_cxx_flags_list})
      message(STATUS "for target ${_target} apply_compile_options as Public  -> ${_global_cxx_flags_list}")      
    endforeach()
    unset(CMAKE_CXX_FLAGS)
    set(_flag_sync_required TRUE)
endmacro()

#
# Removes the specified compile flag from the specified target.
#   _target     - The target to remove the compile flag from
#   _flag       - The compile flag to remove
#
# Pre: apply_global_cxx_flags_to_all_targets() must be invoked.
#
# The idea is to first obtain the compile options from the target to which the file
# belongs, and then applying said options to all source files in that target, which
# allows us to manipulate the compile flags for individual files. We do this by
# maintaining a cached list of compile flags for each file we want to remove flags
# from, and when a remove is requested, we remove it from the cached list and then
# re-apply the remaining flags. The compile options for the target itself is cleared.
macro(remove_flag_from_target _target _flag)
    get_target_property(_target_cxx_flags ${_target} COMPILE_OPTIONS)
#    message(STATUS "Removed from target ${_target} this Flags->${_flag} inside ${_target_cxx_flags}")
    if(_target_cxx_flags)
      list(REMOVE_ITEM _target_cxx_flags ${_flag})
      message(STATUS "Once flag is removed ${_target_cxx_flags}")
        set_target_properties(${_target} PROPERTIES COMPILE_OPTIONS "")      
        set_target_properties(${_target} PROPERTIES COMPILE_OPTIONS "${_target_cxx_flags}")
    endif()
    get_target_property(_newtarget_cxx_flags ${_target} COMPILE_OPTIONS)
    message(STATUS "COMPILE_OPTIONS after removed flag ${_newtarget_cxx_flags}")    
endmacro()

#
# Removes the specified compiler flag from the specified file.
#   _target     - The target that _file belongs to
#   _file       - The file to remove the compiler flag from
#   _flag       - The compiler flag to remove.
#
# Pre: apply_global_cxx_flags_to_all_targets() must be invoked.
#
macro(remove_flag_from_file _target _file _flag)
    get_target_property(_target_sources ${_target} SOURCES)
    # Check if a sync is required, in which case we'll force a rewrite of the cache variables.
    if(_flag_sync_required)
        unset(_cached_${_target}_cxx_flags CACHE)
        unset(_cached_${_target}_${_file}_cxx_flags CACHE)
    endif()
    get_target_property(_${_target}_cxx_flags ${_target} COMPILE_OPTIONS)
    # On first entry, cache the target compile flags and apply them to each source file
    # in the target.
    if(NOT _cached_${_target}_cxx_flags)
        # Obtain and cache the target compiler options, then clear them.
        get_target_property(_target_cxx_flags ${_target} COMPILE_OPTIONS)
        set(_cached_${_target}_cxx_flags "${_target_cxx_flags}" CACHE INTERNAL "")
        set_target_properties(${_target} PROPERTIES COMPILE_OPTIONS "")
        # Apply the target compile flags to each source file.
        foreach(_source_file ${_target_sources})
            # Check for pre-existing flags set by set_source_files_properties().
            get_source_file_property(_source_file_cxx_flags ${_source_file} COMPILE_FLAGS)
            if(_source_file_cxx_flags)
                separate_arguments(_source_file_cxx_flags UNIX_COMMAND ${_source_file_cxx_flags})
                list(APPEND _source_file_cxx_flags "${_target_cxx_flags}")
            else()
                set(_source_file_cxx_flags "${_target_cxx_flags}")
            endif()
            # Apply the compile flags to the current source file.
            string(REPLACE ";" " " _source_file_cxx_flags_string "${_source_file_cxx_flags}")
            set_source_files_properties(${_source_file} PROPERTIES COMPILE_FLAGS "${_source_file_cxx_flags_string}")
        endforeach()
    endif()
    list(FIND _target_sources ${_file} _file_found_at)
    if(_file_found_at GREATER -1)
        if(NOT _cached_${_target}_${_file}_cxx_flags)
            # Cache the compile flags for the specified file.
            # This is the list that we'll be removing flags from.
            get_source_file_property(_source_file_cxx_flags ${_file} COMPILE_FLAGS)
            separate_arguments(_source_file_cxx_flags UNIX_COMMAND ${_source_file_cxx_flags})
            set(_cached_${_target}_${_file}_cxx_flags ${_source_file_cxx_flags} CACHE INTERNAL "")
        endif()
        # Remove the specified flag, then re-apply the rest.
        list(REMOVE_ITEM _cached_${_target}_${_file}_cxx_flags ${_flag})
        string(REPLACE ";" " " _cached_${_target}_${_file}_cxx_flags_string "${_cached_${_target}_${_file}_cxx_flags}")
        set_source_files_properties(${_file} PROPERTIES COMPILE_FLAGS "${_cached_${_target}_${_file}_cxx_flags_string}")
    endif()
endmacro()

#if _flag is given as a list we have to redo the macro in order to get the whole list of options
#and added the new ones using COMPILE_OPTIONS instead of COMPILE_FLAGS
macro(add_flag_to_target _target _flag)
  message(STATUS "Adding flag ${_flag}")
  set_target_properties(${_target} PROPERTIES COMPILE_FlAGS "${_flag}")
endmacro()
