#------------------------------------------------------------------------------
# Copyright (C) 2022, Esperanto Technologies Inc.
# The copyright to the computer program(s) herein is the
# property of Esperanto Technologies, Inc. All Rights Reserved.
# The program(s) may be used and/or copied only with
# the written permission of Esperanto Technologies and
# in accordance with the terms and conditions stipulated in the
# agreement/contract under which the program(s) have been supplied.
#------------------------------------------------------------------------------

add_etsoc_riscv_executable(print.elf print/print.cc)
target_link_libraries(print.elf  etsoc_crt0)
list(APPEND KERNEL_FILES print)

add_etsoc_riscv_executable(print2.elf print2/print2.cc)
target_link_libraries(print2.elf  etsoc_crt0)
list(APPEND KERNEL_FILES print2)

add_etsoc_riscv_executable(bss.elf kernelLaunch/resetBSS.cc)
target_link_libraries(bss.elf etsoc_crt0)
list(APPEND KERNEL_FILES bss)

add_etsoc_riscv_executable(data.elf kernelLaunch/resetData.cc)
target_link_libraries(data.elf etsoc_crt0)
list(APPEND KERNEL_FILES data)

add_etsoc_riscv_executable(hang.elf hang/hang.cc)
target_link_libraries(hang.elf etsoc_crt0)
list(APPEND KERNEL_FILES hang)

add_etsoc_riscv_executable(exception.elf exception/exception.cc)
target_link_libraries(exception.elf etsoc_crt0)
list(APPEND KERNEL_FILES exception)

add_etsoc_riscv_executable(saxpy_scalar.elf saxpy/saxpy.cpp)
target_link_libraries(saxpy_scalar.elf etsoc_crt0)
target_include_directories(saxpy_scalar.elf INTERFACE .)
list(APPEND KERNEL_FILES saxpy_scalar)

add_etsoc_riscv_executable(saxpy_vector.elf saxpy/saxpy.cpp)
target_link_libraries(saxpy_vector.elf etsoc_crt0)
target_include_directories(saxpy_scalar.elf INTERFACE .)
target_compile_definitions(saxpy_vector.elf PRIVATE "-DSAXPY_VECTOR")
list(APPEND KERNEL_FILES saxpy_vector)
list(APPEND KERNEL_FILES saxpy_scalar)

add_etsoc_riscv_executable(saxpy_intrinsics.elf saxpy/saxpy_intrinsics.cpp)
target_link_libraries(saxpy_intrinsics.elf etsoc_crt0)
target_include_directories(saxpy_intrinsics.elf INTERFACE .)
list(APPEND KERNEL_FILES saxpy_intrinsics)

add_etsoc_riscv_executable(saxpy_profiling.elf saxpy/saxpy_profiling.cpp)
target_link_libraries(saxpy_profiling.elf etsoc_crt0)
target_include_directories(saxpy_profiling.elf INTERFACE .)
list(APPEND KERNEL_FILES saxpy_profiling)

add_etsoc_riscv_executable(fftKernel.elf fftKernel/src/fft_entry.cc)
target_link_libraries(fftKernel.elf  etsoc_crt0
                                     dnnLibrary::dnn_lib)
target_include_directories(fftKernel.elf PUBLIC fftKernel/include)
target_compile_definitions(fftKernel.elf PRIVATE "-DGPSDK")
list(APPEND KERNEL_FILES fftKernel)

add_etsoc_riscv_executable(syncDeviceBasic.elf barrier/syncDeviceBasic.cpp)
target_link_libraries(syncDeviceBasic.elf etsoc_crt0)
list(APPEND KERNEL_FILES syncDeviceBasic)

add_etsoc_riscv_executable(syncMinion.elf barrier/syncMinion.cpp)
target_link_libraries(syncMinion.elf etsoc_crt0)
list(APPEND KERNEL_FILES syncMinion)

add_etsoc_riscv_executable(syncAll.elf barrier/syncAll.cpp)
target_link_libraries(syncAll.elf etsoc_crt0)
list(APPEND KERNEL_FILES syncAll)

add_etsoc_riscv_executable(syncShire2EP.elf barrier/syncShire2EP.cpp)
target_link_libraries(syncShire2EP.elf etsoc_crt0)
list(APPEND KERNEL_FILES syncShire2EP)

add_etsoc_riscv_executable(txfma.elf txfma/txfma.cpp)
target_link_libraries(txfma.elf etsoc_crt0)
target_include_directories(txfma.elf INTERFACE .)
list(APPEND KERNEL_FILES txfma)

add_etsoc_riscv_executable(oneTrapOnSync.elf trapping/oneTrapOnSync.cpp)
target_link_libraries(oneTrapOnSync.elf etsoc_crt0)
list(APPEND KERNEL_FILES oneTrapOnSync)

add_etsoc_riscv_executable(busy10sec.elf busy10sec/busy10sec.cc)
target_link_libraries(busy10sec.elf etsoc_crt0)
list(APPEND KERNEL_FILES busy10sec)

add_etsoc_riscv_executable(check_pmc.elf check_pmc/check_pmc.cc)
target_link_libraries(check_pmc.elf etsoc_crt0)
list(APPEND KERNEL_FILES check_pmc)

add_etsoc_riscv_executable(fnodiv.elf softfloat/fnodiv.cpp)
target_link_libraries(fnodiv.elf etsoc_crt0)
list(APPEND KERNEL_FILES fnodiv)

add_etsoc_riscv_executable(c_constructors.elf linker/c_constructors.cc)
target_link_libraries(c_constructors.elf etsoc_crt0)
list(APPEND KERNEL_FILES c_constructors)

add_etsoc_riscv_executable(cpp_constructors.elf linker/cpp_constructors.cc)
target_link_libraries(cpp_constructors.elf etsoc_crt0)
list(APPEND KERNEL_FILES cpp_constructors)

add_etsoc_riscv_executable(gp.elf linker/gp.cc)
target_link_libraries(gp.elf etsoc_crt0)
list(APPEND KERNEL_FILES gp)

add_etsoc_riscv_executable(c_tls.elf linker/c_tls.cc)
target_link_libraries(c_tls.elf etsoc_crt0)
list(APPEND KERNEL_FILES c_tls)

add_etsoc_riscv_executable(cpp_tls.elf linker/cpp_tls.cc)
target_link_libraries(cpp_tls.elf etsoc_crt0)
list(APPEND KERNEL_FILES cpp_tls)

add_etsoc_riscv_executable(external_tls.elf linker/external_tls_def.cc linker/external_tls_decl.cc)
target_link_libraries(external_tls.elf etsoc_crt0)
list(APPEND KERNEL_FILES external_tls)

add_etsoc_riscv_executable(profiling_simple.elf profiling/profiling_simple.cc)
target_link_libraries(profiling_simple.elf etsoc_crt0)
list(APPEND KERNEL_FILES profiling_simple)

add_etsoc_riscv_executable(profiling_stress.elf profiling/profiling_stress.cc)
target_link_libraries(profiling_stress.elf etsoc_crt0)
list(APPEND KERNEL_FILES profiling_stress)

add_etsoc_riscv_executable(cacheops_flush.elf cacheops/flush.cc)
target_link_libraries(cacheops_flush.elf etsoc_crt0)
list(APPEND KERNEL_FILES cacheops_flush)

add_etsoc_riscv_executable(autogen_matmul.elf autogenMatmul/autogen_matmul.cpp)
target_link_libraries(autogen_matmul.elf etsoc_crt0 matmul_f32_1024_512_512)
list(APPEND KERNEL_FILES autogen_matmul)

add_etsoc_riscv_executable(fail_abort.elf fail/fail_abort.cc)
target_link_libraries(fail_abort.elf  etsoc_crt0)
list(APPEND KERNEL_FILES fail_abort)

add_etsoc_riscv_executable(fail_assert.elf fail/fail_assert.cc)
target_link_libraries(fail_assert.elf  etsoc_crt0)
list(APPEND KERNEL_FILES fail_assert)

add_etsoc_riscv_executable(variableStrings.elf variableStrings/variableStrings.cc)
target_link_libraries(variableStrings.elf  etsoc_crt0)
list(APPEND KERNEL_FILES variableStrings)

add_etsoc_riscv_executable(sysemu_fatal.elf sysemu_fatal/sysemu_fatal.cc)
target_link_libraries(sysemu_fatal.elf  etsoc_crt0)
list(APPEND KERNEL_FILES sysemu_fatal)

add_etsoc_riscv_executable(tracing_factorial.elf tracing/tracing_factorial.cc)
target_link_libraries(tracing_factorial.elf  etsoc_crt0)
list(APPEND KERNEL_FILES tracing_factorial)

add_etsoc_riscv_executable(tracing_busywait.elf tracing/tracing_busywait.cc)
target_link_libraries(tracing_busywait.elf  etsoc_crt0)
list(APPEND KERNEL_FILES tracing_busywait)

foreach(KERNEL ${KERNEL_FILES})
  install(TARGETS ${KERNEL}.elf ${KERNEL}.elf_dbg DESTINATION kernels)
endforeach()