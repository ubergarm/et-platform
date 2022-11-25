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

include(tests.cmake)

math(EXPR DEBUG_ADDRESS "${ADDRESS} + 0x1000" OUTPUT_FORMAT HEXADECIMAL)
message(STATUS "address used for Debugging-> ${DEBUG_ADDRESS}")

#Adding tests

add_etsoc_riscv_executable(print "print/print.cc")
add_etsoc_riscv_target_link(print crt_shared)

add_etsoc_riscv_executable(print2 "print2/print2.cc")
add_etsoc_riscv_target_link(print2 crt_shared)

add_etsoc_riscv_executable(bss "kernelLaunch/resetBSS.cc")
add_etsoc_riscv_target_link(bss crt_shared)

add_etsoc_riscv_executable(data kernelLaunch/resetData.cc)
add_etsoc_riscv_target_link(data crt_shared)

set(FFT_SOURCES fftKernel/src/uberKernel_crt.S fftKernel/src/uberKernel.cc fftKernel/src/testOperator_compute.cc fftKernel/src/testOperator_w_pref.cc fftKernel/src/testOperator_inst_pref.cc fftKernel/src/testOperator_act_pref.cc fftKernel/src/SyncComputeNode.cc)
add_etsoc_riscv_executable(fftKernel "${FFT_SOURCES}")
add_etsoc_riscv_target_link(fftKernel dnnLibrary::dnn_lib)
