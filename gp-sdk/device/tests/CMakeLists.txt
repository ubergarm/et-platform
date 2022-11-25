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

math(EXPR DEBUG_ADDRESS "${ADDRESS} + 0x1000" OUTPUT_FORMAT HEXADECIMAL)
message(STATUS "address used for Debugging-> ${DEBUG_ADDRESS}")

#Adding tests

add_etsoc_riscv_executable(print.elf "print/print.cc")
add_etsoc_riscv_target_link(print.elf crt_shared)

add_etsoc_riscv_executable(print2.elf "print2/print2.cc")
add_etsoc_riscv_target_link(print2.elf crt_shared)

add_etsoc_riscv_executable(bss.elf "kernelLaunch/resetBSS.cc")
add_etsoc_riscv_target_link(bss.elf crt_shared)

add_etsoc_riscv_executable(data.elf kernelLaunch/resetData.cc)
add_etsoc_riscv_target_link(data.elf crt_shared)

set(FFT_SOURCES fftKernel/src/uberKernel_crt.S fftKernel/src/uberKernel.cc fftKernel/src/testOperator_compute.cc fftKernel/src/testOperator_w_pref.cc fftKernel/src/testOperator_inst_pref.cc fftKernel/src/testOperator_act_pref.cc fftKernel/src/SyncComputeNode.cc)
add_etsoc_riscv_executable(fftKernel.elf "${FFT_SOURCES}")
add_etsoc_riscv_target_link(fftKernel.elf dnnLibrary::dnn_lib)
set(FFT_INCLUDES fftKernel/include)
add_etsoc_riscv_include(fftKernel.elf "${FFT_INCLUDES}")
