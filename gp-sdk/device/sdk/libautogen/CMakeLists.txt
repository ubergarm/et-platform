#------------------------------------------------------------------------------
# Copyright (c) 2025 Ainekko, Co.
# SPDX-License-Identifier: Apache-2.0
#------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.5)

add_library(matmul_f32_1024_512_512 STATIC
 matmul_f32_1024_512_512/src/SyncComputeNode.cc
 matmul_f32_1024_512_512/src/uberKernel.cc
 matmul_f32_1024_512_512/src/MatmulCmd_compute.cc
 matmul_f32_1024_512_512/src/MatmulCmd_w_pref.cc
 matmul_f32_1024_512_512/src/MatmulCmd_act_pref.cc
 matmul_f32_1024_512_512/src/MatmulCmd_inst_pref.cc)

 target_link_libraries(matmul_f32_1024_512_512 etsoc_crt0 dnnLibrary::dnn_lib)

 target_include_directories(matmul_f32_1024_512_512 PRIVATE matmul_f32_1024_512_512/include helpers)
 target_include_directories(matmul_f32_1024_512_512 PUBLIC ../sdk/)

 target_compile_options(matmul_f32_1024_512_512 PRIVATE -fno-exceptions -falign-functions=64 -O3 -g3)