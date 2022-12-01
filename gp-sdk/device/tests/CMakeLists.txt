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
find_package(dnnLibrary)

include("${CMAKE_SOURCE_DIR}/riscv_helpers.cmake")

#Adding tests

add_etsoc_riscv_executable(print.elf print/print.cc)
target_link_libraries(print.elf  etsoc_crt0)

add_etsoc_riscv_executable(print2.elf print2/print2.cc)
target_link_libraries(print2.elf  etsoc_crt0)

add_etsoc_riscv_executable(bss.elf kernelLaunch/resetBSS.cc)
target_link_libraries(bss.elf etsoc_crt0)

add_etsoc_riscv_executable(data.elf kernelLaunch/resetData.cc)
target_link_libraries(data.elf etsoc_crt0)

add_etsoc_riscv_executable(fftKernel.elf fftKernel/src/uberKernel_crt.S 
                                         fftKernel/src/uberKernel.cc
                                         fftKernel/src/testOperator_compute.cc
                                         fftKernel/src/testOperator_w_pref.cc 
                                         fftKernel/src/testOperator_inst_pref.cc 
                                         fftKernel/src/testOperator_act_pref.cc 
                                         fftKernel/src/SyncComputeNode.cc )

target_link_libraries(fftKernel.elf dnnLibrary::dnn_lib)
target_include_directories(fftKernel.elf PRIVATE fftKernel/include)
