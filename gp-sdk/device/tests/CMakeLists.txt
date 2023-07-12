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

add_etsoc_riscv_executable(print2.elf print2/print2.cc)
target_link_libraries(print2.elf  etsoc_crt0)

add_etsoc_riscv_executable(bss.elf kernelLaunch/resetBSS.cc)
target_link_libraries(bss.elf etsoc_crt0)

add_etsoc_riscv_executable(data.elf kernelLaunch/resetData.cc)
target_link_libraries(data.elf etsoc_crt0)

add_etsoc_riscv_executable(hang.elf hang/hang.cc)
target_link_libraries(hang.elf etsoc_crt0)

add_etsoc_riscv_executable(exception.elf exception/exception.cc)
target_link_libraries(exception.elf etsoc_crt0)

add_etsoc_riscv_executable(saxpy_scalar.elf saxpy/saxpy.cpp)
target_link_libraries(saxpy_scalar.elf etsoc_crt0)
target_include_directories(saxpy_scalar.elf INTERFACE .)

add_etsoc_riscv_executable(saxpy_vector.elf saxpy/saxpy.cpp)
target_link_libraries(saxpy_vector.elf etsoc_crt0)
target_include_directories(saxpy_scalar.elf INTERFACE .)
target_compile_definitions(saxpy_vector.elf PRIVATE "-DSAXPY_VECTOR")

add_etsoc_riscv_executable(saxpy_profiling.elf saxpy/saxpy_profiling.cpp)
target_link_libraries(saxpy_profiling.elf etsoc_crt0)
target_include_directories(saxpy_profiling.elf INTERFACE .)

add_etsoc_riscv_executable(fftKernel.elf fftKernel/src/fft_entry.cc)
target_link_libraries(fftKernel.elf  etsoc_crt0
                                     dnnLibrary::dnn_lib)
target_include_directories(fftKernel.elf PUBLIC fftKernel/include)
target_compile_definitions(fftKernel.elf PRIVATE "-DGPSDK")

add_etsoc_riscv_executable(syncDeviceBasic.elf barrier/syncDeviceBasic.cpp)
target_link_libraries(syncDeviceBasic.elf etsoc_crt0)

add_etsoc_riscv_executable(syncMinion.elf barrier/syncMinion.cpp)
target_link_libraries(syncMinion.elf etsoc_crt0)

add_etsoc_riscv_executable(syncAll.elf barrier/syncAll.cpp)
target_link_libraries(syncAll.elf etsoc_crt0)

add_etsoc_riscv_executable(syncShire2EP.elf barrier/syncShire2EP.cpp)
target_link_libraries(syncShire2EP.elf etsoc_crt0)

add_etsoc_riscv_executable(txfma.elf txfma/txfma.cpp)
target_link_libraries(txfma.elf etsoc_crt0)
target_include_directories(txfma.elf INTERFACE .)

add_etsoc_riscv_executable(oneTrapOnSync.elf trapping/oneTrapOnSync.cpp)
target_link_libraries(oneTrapOnSync.elf etsoc_crt0)

add_etsoc_riscv_executable(busy10sec.elf busy10sec/busy10sec.cc)
target_link_libraries(busy10sec.elf etsoc_crt0)

add_etsoc_riscv_executable(check_pmc.elf check_pmc/check_pmc.cc)
target_link_libraries(check_pmc.elf etsoc_crt0)

add_etsoc_riscv_executable(fnodiv.elf softfloat/fnodiv.cpp)
target_link_libraries(fnodiv.elf etsoc_crt0)

add_etsoc_riscv_executable(c_constructors.elf linker/c_constructors.cc)
target_link_libraries(c_constructors.elf etsoc_crt0)

add_etsoc_riscv_executable(cpp_constructors.elf linker/cpp_constructors.cc)
target_link_libraries(cpp_constructors.elf etsoc_crt0)

add_etsoc_riscv_executable(gp.elf linker/gp.cc)
target_link_libraries(gp.elf etsoc_crt0)

add_etsoc_riscv_executable(c_tls.elf linker/c_tls.cc)
target_link_libraries(c_tls.elf etsoc_crt0)

add_etsoc_riscv_executable(cpp_tls.elf linker/cpp_tls.cc)
target_link_libraries(cpp_tls.elf etsoc_crt0)


add_etsoc_riscv_executable(external_tls.elf linker/external_tls_def.cc linker/external_tls_decl.cc)
target_link_libraries(external_tls.elf etsoc_crt0)

add_etsoc_riscv_executable(profiling_simple.elf profiling/profiling_simple.cc)
target_link_libraries(profiling_simple.elf etsoc_crt0)

add_etsoc_riscv_executable(profiling_stress.elf profiling/profiling_stress.cc)
target_link_libraries(profiling_stress.elf etsoc_crt0)

add_etsoc_riscv_executable(cacheops_flush.elf cacheops/flush.cc)
target_link_libraries(cacheops_flush.elf etsoc_crt0)

add_etsoc_riscv_executable(autogen_matmul.elf autogenMatmul/autogen_matmul.cpp)
target_link_libraries(autogen_matmul.elf etsoc_crt0 matmul_f32_1024_512_512)

add_etsoc_riscv_executable(fail_abort.elf fail/fail_abort.cc)
target_link_libraries(fail_abort.elf  etsoc_crt0)

add_etsoc_riscv_executable(fail_assert.elf fail/fail_assert.cc)
target_link_libraries(fail_assert.elf  etsoc_crt0)

add_etsoc_riscv_executable(variableStrings.elf variableStrings/variableStrings.cc)
target_link_libraries(variableStrings.elf  etsoc_crt0)

foreach(kernel print print2 bss data saxpy_scalar saxpy_vector saxpy_profiling fftKernel syncDeviceBasic syncAll syncMinion syncShire2EP txfma oneTrapOnSync busy10sec check_pmc fnodiv c_constructors cpp_constructors gp c_tls cpp_tls external_tls profiling_simple profiling_stress cacheops_flush fail_abort fail_assert variableStrings)
  install(TARGETS ${kernel}.elf ${kernel}.elf_dbg DESTINATION kernels)
endforeach()
