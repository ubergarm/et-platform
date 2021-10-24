/*-------------------------------------------------------------------------
 * Copyright (C) 2021, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

// Local
#include "LibApiImplSel.h"
#include "LibTensor.h"
#include "dnnLibraryApi/LibApi.h"

// STD
#include <cctype>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace dnn_lib {

// Instruction properties. This is the internal implementation
struct instrConfigInt {
  using implStateVector = std::vector<std::vector<operandState>>;
  using selFnc = size_t (*)(std::vector<LibTensor*>&, std::vector<LibTensor*>&);

  std::string name;
  size_t nrOutputTensors; // number of output and in/out tensor operands
  size_t nrInputTensors;  // number of input tensor operands
  std::vector<instrMembers> members;
  uint64_t templateMask;
  std::vector<std::string> versions;
  selFnc implSel;

  implStateVector stateL1;
  implStateVector stateL2;
  implStateVector stateCB;
  std::vector<uint64_t> evictAvailableMask;
};

// The instruction table contains all the instructions available in the DNN Library
static std::vector<instrConfigInt> instrConfigTable;
static bool tableInitialized = false;

// This function initializes the table with all the contents
void initInstrConfigTable() {
  // ET_adaptiveavgpool
  instrConfigInt instConfigInt{
    "AdaptiveAvgPool",      // name
    1,                      // # outs
    1,                      // # ins
    {},                     // members
    2,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_allocactivation
  instConfigInt = instrConfigInt{
    "notImplemented", // name
    0,                // # outs
    0,                // # ins
    {},               // members
    0,                // template param mask
    {},               // impl versions
    nullptr,          // custom impl selector
    // L1 states per impl
    {{operandState::untouched}},
    // L2 states per impl
    {{operandState::untouched}},
    // CB states per impl
    {{operandState::untouched}},
    {0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_argmax
  instConfigInt = instrConfigInt{
    "ArgMax",                                         // name
    1,                                                // # outs
    1,                                                // # ins
    {instrMembers::mbAxis, instrMembers::mbKeepDims}, // members
    3,                                                // template param mask
    {},                                               // impl versions
    implSel::defaultSel<1>,                           // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_avgpool
  instConfigInt = instrConfigInt{
    "AvgPool", // name
    1,         // # outs
    1,         // # ins
    {instrMembers::mbKernels, instrMembers::mbStrides, instrMembers::mbPads, instrMembers::mbLayout,
     instrMembers::mbCountIncludePads}, // members
    1,                                  // template param mask
    {"Threaded"},                       // impl versions
    implSel::AvgPool,                   // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}, {operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}, {operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}, {operandState::untouched, operandState::untouched}}},
    {0x0, 0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_avgpool
  instConfigInt = instrConfigInt{
    "AvgPool", // name
    1,         // # outs
    1,         // # ins
    {instrMembers::mbKernels, instrMembers::mbStrides, instrMembers::mbPads, instrMembers::mbLayout,
     instrMembers::mbCountIncludePads}, // members
    1,                                  // template param mask
    {"Threaded"},                       // impl versions
    implSel::AvgPool,                   // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}, {operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}, {operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}, {operandState::untouched, operandState::untouched}}},
    {0x0, 0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_batchedadd
  instConfigInt = instrConfigInt{
    "BatchedAdd",           // name
    1,                      // # outs
    2,                      // # ins
    {},                     // members
    7,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_batchedreduceadd
  instConfigInt = instrConfigInt{
    "BatchedReduceAdd",     // name
    1,                      // # outs
    1,                      // # ins
    {instrMembers::mbAxis}, // members
    2,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_batchedreducemin
  instConfigInt = instrConfigInt{
    "BatchedReduceMin",     // name
    1,                      // # outs
    1,                      // # ins
    {instrMembers::mbAxes}, // members
    2,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_batchonehot
  instConfigInt = instrConfigInt{
    "BatchOneHot",          // name
    1,                      // # outs
    3,                      // # ins
    {},                     // members
    1,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_convertto
  instConfigInt = instrConfigInt{
    "ConvertTo",            // name
    1,                      // # outs
    1,                      // # ins
    {},                     // members
    3,                      // template param mask
    {"Vectorized"},         // impl versions
    implSel::defaultSel<2>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}, {operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}, {operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}, {operandState::untouched, operandState::untouched}}},
    {0x0, 0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_int8converter
  instConfigInt = instrConfigInt{
    "Int8Converter",        // name
    1,                      // # outs
    1,                      // # ins
    {},                     // members
    2,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_channelwisequantizedconvolution
  instConfigInt = instrConfigInt{
    "ChannelWiseQuantizedConvolution", // name
    1,                                 // # outs
    7,                                 // # ins
    {instrMembers::mbKernels, instrMembers::mbStrides, instrMembers::mbPads, instrMembers::mbGroup,
     instrMembers::mbDilation, instrMembers::mbFusedActivation, instrMembers::mbFusedActivationArgs}, // members
    9,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean,
       operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean,
       operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched,
       operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_channelwisequantizedconvolution3d
  instConfigInt = instrConfigInt{
    "notImplemented", // name
    0,                // # outs
    0,                // # ins
    {},               // members
    0,                // template param mask
    {},               // impl versions
    nullptr,          // custom impl selector
    // L1 states per impl
    {{operandState::untouched}},
    // L2 states per impl
    {{operandState::untouched}},
    // CB states per impl
    {{operandState::untouched}},
    {0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_convolution
  instConfigInt = instrConfigInt{
    "Convolution", // name
    1,             // # outs
    3,             // # ins
    {instrMembers::mbKernels, instrMembers::mbStrides, instrMembers::mbPads, instrMembers::mbGroup,
     instrMembers::mbDilation, instrMembers::mbFusedActivation, instrMembers::mbFusedActivationArgs}, // members
    9,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_convolution3d
  instConfigInt = instrConfigInt{
    "Convolution3D",                                                                                 // name
    1,                                                                                               // # outs
    3,                                                                                               // # ins
    {instrMembers::mbKernels, instrMembers::mbStrides, instrMembers::mbPads, instrMembers::mbGroup}, // members
    1,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_convtranspose
  instConfigInt = instrConfigInt{
    "ConvTranspose", // name
    1,               // # outs
    3,               // # ins
    {instrMembers::mbKernels, instrMembers::mbStrides, instrMembers::mbPads, instrMembers::mbGroup,
     instrMembers::mbDilation}, // members
    1,                          // template param mask
    {},                         // impl versions
    implSel::defaultSel<1>,     // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_copy
  instConfigInt = instrConfigInt{
    "Copy",         // name
    1,              // # outs
    1,              // # ins
    {},             // members
    1,              // template param mask
    {"Tensorized"}, // impl versions
    implSel::Copy,  // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}, {operandState::untouched, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}, {operandState::untouched, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}, {operandState::dirty, operandState::untouched}}},
    {0x1, 0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_crossentropyloss
  instConfigInt = instrConfigInt{
    "CrossEntropyLoss",     // name
    1,                      // # outs
    2,                      // # ins
    {},                     // members
    2,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_cumsum
  instConfigInt = instrConfigInt{
    "CumSum",                                             // name
    1,                                                    // # outs
    1,                                                    // # ins
    {instrMembers::mbExclusive, instrMembers::mbReverse}, // members
    2,                                                    // template param mask
    {},                                                   // impl versions
    implSel::defaultSel<1>,                               // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_deallocactivation
  instConfigInt = instrConfigInt{
    "notImplemented", // name
    0,                // # outs
    0,                // # ins
    {},               // members
    0,                // template param mask
    {},               // impl versions
    nullptr,          // custom impl selector
    // L1 states per impl
    {{operandState::untouched}},
    // L2 states per impl
    {{operandState::untouched}},
    // CB states per impl
    {{operandState::untouched}},
    {0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_debugprint
  instConfigInt = instrConfigInt{
    "notImplemented", // name
    0,                // # outs
    0,                // # ins
    {},               // members
    0,                // template param mask
    {},               // impl versions
    nullptr,          // custom impl selector
    // L1 states per impl
    {{operandState::untouched}},
    // L2 states per impl
    {{operandState::untouched}},
    // CB states per impl
    {{operandState::untouched}},
    {0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_dequantize
  instConfigInt = instrConfigInt{
    "Dequantize",           // name
    1,                      // # outs
    1,                      // # ins
    {},                     // members
    3,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_elementadd
  instConfigInt = instrConfigInt{
    "ElementAdd",           // name
    1,                      // # outs
    2,                      // # ins
    {},                     // members
    1,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_elementand
  instConfigInt = instrConfigInt{
    "notImplemented", // name
    0,                // # outs
    0,                // # ins
    {},               // members
    0,                // template param mask
    {},               // impl versions
    nullptr,          // custom impl selector
    // L1 states per impl
    {{operandState::untouched}},
    // L2 states per impl
    {{operandState::untouched}},
    // CB states per impl
    {{operandState::untouched}},
    {0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_elementcmpeq
  instConfigInt = instrConfigInt{
    "ElementCmpEQ",        // name
    1,                     // # outs
    2,                     // # ins
    {},                    // members
    6,                     // template param mask
    {"Vectorized"},        // impl versions
    implSel::ElementCmpEQ, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean},
      {operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean},
      {operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched},
      {operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0, 0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_elementcmplte
  instConfigInt = instrConfigInt{
    "ElementCmpLTE",        // name
    1,                      // # outs
    2,                      // # ins
    {},                     // members
    6,                      // template param mask
    {"Vectorized"},         // impl versions
    implSel::ElementCmpLTE, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean},
      {operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean},
      {operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched},
      {operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0, 0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_elementcmplt
  instConfigInt = instrConfigInt{
    "ElementCmpLT",        // name
    1,                     // # outs
    2,                     // # ins
    {},                    // members
    6,                     // template param mask
    {"Vectorized"},        // impl versions
    implSel::ElementCmpLT, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean},
      {operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean},
      {operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched},
      {operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0, 0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_elementdiv
  instConfigInt = instrConfigInt{
    "ElementDiv",           // name
    1,                      // # outs
    2,                      // # ins
    {},                     // members
    1,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_elementexp
  instConfigInt = instrConfigInt{
    "ElementExp",           // name
    1,                      // # outs
    1,                      // # ins
    {},                     // members
    2,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_elementisnan
  instConfigInt = instrConfigInt{
    "ElementIsNaN",         // name
    1,                      // # outs
    1,                      // # ins
    {},                     // members
    2,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_elementlog
  instConfigInt = instrConfigInt{
    "ElementLog",           // name
    1,                      // # outs
    1,                      // # ins
    {},                     // members
    3,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_elementmax
  instConfigInt = instrConfigInt{
    "ElementMax",           // name
    1,                      // # outs
    2,                      // # ins
    {},                     // members
    1,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_elementmin
  instConfigInt = instrConfigInt{
    "ElementMin",           // name
    1,                      // # outs
    2,                      // # ins
    {},                     // members
    1,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_elementmul
  instConfigInt = instrConfigInt{
    "ElementMul",           // name
    1,                      // # outs
    2,                      // # ins
    {},                     // members
    1,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_elementor
  instConfigInt = instrConfigInt{
    "notImplemented", // name
    0,                // # outs
    0,                // # ins
    {},               // members
    0,                // template param mask
    {},               // impl versions
    nullptr,          // custom impl selector
    // L1 states per impl
    {{operandState::untouched}},
    // L2 states per impl
    {{operandState::untouched}},
    // CB states per impl
    {{operandState::untouched}},
    {0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_elementpow
  instConfigInt = instrConfigInt{
    "ElementPow",           // name
    1,                      // # outs
    2,                      // # ins
    {},                     // members
    1,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_elementselect
  instConfigInt = instrConfigInt{
    "ElementSelect",        // name
    1,                      // # outs
    3,                      // # ins
    {},                     // members
    1,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_elementsub
  instConfigInt = instrConfigInt{
    "ElementSub",           // name
    1,                      // # outs
    2,                      // # ins
    {},                     // members
    1,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_elementxor
  instConfigInt = instrConfigInt{
    "notImplemented", // name
    0,                // # outs
    0,                // # ins
    {},               // members
    0,                // template param mask
    {},               // impl versions
    nullptr,          // custom impl selector
    // L1 states per impl
    {{operandState::untouched}},
    // L2 states per impl
    {{operandState::untouched}},
    // CB states per impl
    {{operandState::untouched}},
    {0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_embeddingbag
  instConfigInt = instrConfigInt{
    "EmbeddingBag",                 // name
    1,                              // # outs
    4,                              // # ins
    {instrMembers::mbHasEndOffset}, // members
    1,                              // template param mask
    {},                             // impl versions
    implSel::defaultSel<1>,         // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched,
       operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_embeddingbagbyterowwiseoffsets
  instConfigInt = instrConfigInt{
    "notImplemented", // name
    0,                // # outs
    0,                // # ins
    {},               // members
    0,                // template param mask
    {},               // impl versions
    nullptr,          // custom impl selector
    // L1 states per impl
    {{operandState::untouched}},
    // L2 states per impl
    {{operandState::untouched}},
    // CB states per impl
    {{operandState::untouched}},
    {0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_emptyoperator
  instConfigInt = instrConfigInt{
    "notImplemented", // name
    0,                // # outs
    0,                // # ins
    {},               // members
    0,                // template param mask
    {},               // impl versions
    nullptr,          // custom impl selector
    // L1 states per impl
    {{operandState::untouched}},
    // L2 states per impl
    {{operandState::untouched}},
    // CB states per impl
    {{operandState::untouched}},
    {0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_maxsplat
  instConfigInt = instrConfigInt{
    "MaxSplat",              // name
    1,                       // # outs
    1,                       // # ins
    {instrMembers::mbValue}, // members
    2,                       // template param mask
    {"Aligned32Bytes"},      // impl versions
    implSel::MaxSplat,       // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}, {operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}, {operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}, {operandState::untouched, operandState::untouched}}},
    {0x0, 0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_extracttensor
  instConfigInt = instrConfigInt{
    "ExtractTensor",           // name
    1,                         // # outs
    1,                         // # ins
    {instrMembers::mbOffsets}, // members
    2,                         // template param mask
    {},                        // impl versions
    implSel::defaultSel<1>,    // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_flip
  instConfigInt = instrConfigInt{
    "Flip",                 // name
    1,                      // # outs
    1,                      // # ins
    {instrMembers::mbAxis}, // members
    2,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_flushL3
  instConfigInt = instrConfigInt{
    "notImplemented", // name
    0,                // # outs
    0,                // # ins
    {},               // members
    0,                // template param mask
    {},               // impl versions
    nullptr,          // custom impl selector
    // L1 states per impl
    {{operandState::untouched}},
    // L2 states per impl
    {{operandState::untouched}},
    // CB states per impl
    {{operandState::untouched}},
    {0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_fullyconnected
  instConfigInt = instrConfigInt{
    "FullyConnected",       // name
    1,                      // # outs
    3,                      // # ins
    {},                     // members
    9,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_fusedrowwisequantizedsparselengthsweightedsum
  instConfigInt = instrConfigInt{
    "FusedRowwiseQuantizedSparseLengthsWeightedSum", // name
    1,                                               // # outs
    4,                                               // # ins
    {},                                              // members
    1,                                               // template param mask
    {},                                              // impl versions
    implSel::defaultSel<1>,                          // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched,
       operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_fusedrowwisequantizedsparselengthssum
  instConfigInt = instrConfigInt{
    "FusedRowwiseQuantizedSparseLengthsSum", // name
    1,                                       // # outs
    3,                                       // # ins
    {},                                      // members
    1,                                       // template param mask
    {},                                      // impl versions
    implSel::defaultSel<1>,                  // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_gather
  instConfigInt = instrConfigInt{
    "Gather",                    // name
    1,                           // # outs
    2,                           // # ins
    {instrMembers::mbBatchDims}, // members
    6,                           // template param mask
    {},                          // impl versions
    implSel::defaultSel<1>,      // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x1} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_gatherranges
  instConfigInt = instrConfigInt{
    "GatherRanges",         // name
    2,                      // # outs
    2,                      // # ins
    {},                     // members
    12,                     // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_inserttensor
  instConfigInt = instrConfigInt{
    "InsertTensor",                                                         // name
    1,                                                                      // # outs
    1,                                                                      // # ins
    {instrMembers::mbOffsets, instrMembers::mbCount, instrMembers::mbAxis}, // members
    2,                                                                      // template param mask
    {"Threaded"},                                                           // impl versions
    implSel::InsertTensor,                                                  // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}, {operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}, {operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}, {operandState::untouched, operandState::untouched}}},
    {0x1, 0x1} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_intlookuptable
  instConfigInt = instrConfigInt{
    "IntLookupTable",       // name
    1,                      // # outs
    2,                      // # ins
    {},                     // members
    0,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_lengthsrangefill
  instConfigInt = instrConfigInt{
    "LengthsRangeFill",     // name
    1,                      // # outs
    1,                      // # ins
    {},                     // members
    1,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_lengthssum
  instConfigInt = instrConfigInt{
    "LengthsSum",           // name
    1,                      // # outs
    2,                      // # ins
    {},                     // members
    2,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_lengthstoranges
  instConfigInt = instrConfigInt{
    "LengthsToRanges",      // name
    1,                      // # outs
    1,                      // # ins
    {},                     // members
    2,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_localresponsenormalization
  instConfigInt = instrConfigInt{
    "LocalResponseNormalization",                                                                     // name
    2,                                                                                                // # outs
    1,                                                                                                // # ins
    {instrMembers::mbHalfWindowSize, instrMembers::mbAlpha, instrMembers::mbBeta, instrMembers::mbK}, // members
    2,                                   // template param mask
    {"Vectorized"},                      // impl versions
    implSel::LocalResponseNormalization, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean},
      {operandState::dirty, operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean},
      {operandState::dirty, operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched},
      {operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0, 0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_matmul
  instConfigInt = instrConfigInt{
    "MatMul",               // name
    1,                      // # outs
    2,                      // # ins
    {},                     // members
    2,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_maxpool
  instConfigInt = instrConfigInt{
    "MaxPool",                                                                                        // name
    1,                                                                                                // # outs
    1,                                                                                                // # ins
    {instrMembers::mbKernels, instrMembers::mbStrides, instrMembers::mbPads, instrMembers::mbLayout}, // members
    3,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_maxpoolwithargmax
  instConfigInt = instrConfigInt{
    "MaxPoolWithArgMax",                                                                              // name
    2,                                                                                                // # outs
    1,                                                                                                // # ins
    {instrMembers::mbKernels, instrMembers::mbStrides, instrMembers::mbPads, instrMembers::mbLayout}, // members
    5,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_modulo
  instConfigInt = instrConfigInt{
    "Modulo",                                                     // name
    1,                                                            // # outs
    1,                                                            // # ins
    {instrMembers::mbDivisor, instrMembers::mbSignFollowDivisor}, // members
    2,                                                            // template param mask
    {},                                                           // impl versions
    implSel::defaultSel<1>,                                       // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_nonmaxsuppression
  instConfigInt = instrConfigInt{
    "NonMaxSuppression", // name
    2,                   // # outs
    2,                   // # ins
    {instrMembers::mbCenterPointBox, instrMembers::mbMaxOutputBoxesPerClass, instrMembers::mbIouThreshold,
     instrMembers::mbScoreThreshold, instrMembers::mbIsTFVersion4}, // members
    1,                                                              // template param mask
    {},                                                             // impl versions
    implSel::defaultSel<1>,                                         // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_quantizationprofile
  instConfigInt = instrConfigInt{
    "notImplemented", // name
    0,                // # outs
    0,                // # ins
    {},               // members
    0,                // template param mask
    {},               // impl versions
    nullptr,          // custom impl selector
    // L1 states per impl
    {{operandState::untouched}},
    // L2 states per impl
    {{operandState::untouched}},
    // CB states per impl
    {{operandState::untouched}},
    {0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_quantize
  instConfigInt = instrConfigInt{
    "Quantize",             // name
    1,                      // # outs
    1,                      // # ins
    {},                     // members
    3,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_rescalequantized
  instConfigInt = instrConfigInt{
    "RescaleQuantized",     // name
    1,                      // # outs
    1,                      // # ins
    {},                     // members
    2,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_resizebilinear
  instConfigInt = instrConfigInt{
    "ResizeBilinear",           // name
    1,                          // # outs
    1,                          // # ins
    {instrMembers::mbRszScale}, // members
    1,                          // template param mask
    {},                         // impl versions
    implSel::defaultSel<1>,     // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_resizenearest
  instConfigInt = instrConfigInt{
    "ResizeNearest",            // name
    1,                          // # outs
    1,                          // # ins
    {instrMembers::mbRszScale}, // members
    1,                          // template param mask
    {},                         // impl versions
    implSel::defaultSel<1>,     // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_rowwisequantizedfullyconnected
  instConfigInt = instrConfigInt{
    "RowwiseQuantizedFullyConnected",        // name
    1,                                       // # outs
    5,                                       // # ins
    {},                                      // members
    0,                                       // template param mask
    {"Aligned32Bytes"},                      // impl versions
    implSel::RowwiseQuantizedFullyConnected, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean,
       operandState::clean},
      {operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean,
       operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean,
       operandState::clean},
      {operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean,
       operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched,
       operandState::untouched, operandState::untouched},
      {operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched,
       operandState::untouched, operandState::untouched}}},
    {0x0, 0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_rowwisequantizedsparselengthsweightedsum
  instConfigInt = instrConfigInt{
    "RowwiseQuantizedSparseLengthsWeightedSum",        // name
    1,                                                 // # outs
    6,                                                 // # ins
    {},                                                // members
    33,                                                // template param mask
    {"Vectorized"},                                    // impl versions
    implSel::RowwiseQuantizedSparseLengthsWeightedSum, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean,
       operandState::clean, operandState::clean},
      {operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean,
       operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean,
       operandState::clean, operandState::clean},
      {operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean,
       operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched,
       operandState::untouched, operandState::untouched, operandState::untouched},
      {operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched,
       operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0, 0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_scatterdata
  instConfigInt = instrConfigInt{
    "ScatterData",          // name
    1,                      // # outs
    2,                      // # ins
    {},                     // members
    1,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_sigmoid
  instConfigInt = instrConfigInt{
    "Sigmoid",              // name
    1,                      // # outs
    1,                      // # ins
    {},                     // members
    2,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_softmax
  instConfigInt = instrConfigInt{
    "SoftMax",        // name
    1,                // # outs
    1,                // # ins
    {},               // members
    2,                // template param mask
    {"Vectorized"},   // impl versions
    implSel::SoftMax, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}, {operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}, {operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}, {operandState::untouched, operandState::untouched}}},
    {0x0, 0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_spacetodepth
  instConfigInt = instrConfigInt{
    "SpaceToDepth",              // name
    1,                           // # outs
    1,                           // # ins
    {instrMembers::mbBlockSize}, // members
    1,                           // template param mask
    {},                          // impl versions
    implSel::defaultSel<1>,      // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_sparselengthssum
  instConfigInt = instrConfigInt{
    "SparseLengthsSum",     // name
    1,                      // # outs
    3,                      // # ins
    {},                     // members
    5,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_sparselengthsweightedsum
  instConfigInt = instrConfigInt{
    "SparseLengthsWeightedSum",        // name
    1,                                 // # outs
    4,                                 // # ins
    {},                                // members
    10,                                // template param mask
    {"Threaded"},                      // impl versions
    implSel::SparseLengthsWeightedSum, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean},
      {operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean},
      {operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched,
       operandState::untouched},
      {operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched,
       operandState::untouched}}},
    {0x0, 0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_sparsetodense
  instConfigInt = instrConfigInt{
    "SparseToDense",        // name
    1,                      // # outs
    2,                      // # ins
    {},                     // members
    1,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_sparsetodensemask
  instConfigInt = instrConfigInt{
    "SparseToDenseMask",    // name
    1,                      // # outs
    4,                      // # ins
    {instrMembers::mbMask}, // members
    1,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched,
       operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_splat
  instConfigInt = instrConfigInt{
    "Splat",                 // name
    1,                       // # outs
    0,                       // # ins
    {instrMembers::mbValue}, // members
    1,                       // template param mask
    {},                      // impl versions
    implSel::defaultSel<1>,  // custom impl selector
    // L1 states per impl
    {{{operandState::dirty}}},
    // L2 states per impl
    {{{operandState::dirty}}},
    // CB states per impl
    {{{operandState::untouched}}},
    {0x1} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_sync
  instConfigInt = instrConfigInt{
    "notImplemented", // name
    0,                // # outs
    0,                // # ins
    {},               // members
    0,                // template param mask
    {},               // impl versions
    nullptr,          // custom impl selector
    // L1 states per impl
    {{operandState::untouched}},
    // L2 states per impl
    {{operandState::untouched}},
    // CB states per impl
    {{operandState::untouched}},
    {0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_syncopy
  instConfigInt = instrConfigInt{
    "Syncopy",                    // name
    1,                            // # outs
    1,                            // # ins
    {instrMembers::mbSyncOffset}, // members
    2,                            // template param mask
    {},                           // impl versions
    implSel::defaultSel<1>,       // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_tanh
  instConfigInt = instrConfigInt{
    "Tanh",                 // name
    1,                      // # outs
    1,                      // # ins
    {},                     // members
    2,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_tensorview
  instConfigInt = instrConfigInt{
    "TensorView",              // name
    1,                         // # outs
    1,                         // # ins
    {instrMembers::mbOffsets}, // members
    1,                         // template param mask
    {},                        // impl versions
    implSel::defaultSel<1>,    // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_topk
  instConfigInt = instrConfigInt{
    "TopK",                 // name
    2,                      // # outs
    1,                      // # ins
    {instrMembers::mbTopK}, // members
    4,                      // template param mask
    {},                     // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_touch
  instConfigInt = instrConfigInt{
    "notImplemented", // name
    0,                // # outs
    0,                // # ins
    {},               // members
    0,                // template param mask
    {},               // impl versions
    nullptr,          // custom impl selector
    // L1 states per impl
    {{operandState::untouched}},
    // L2 states per impl
    {{operandState::untouched}},
    // CB states per impl
    {{operandState::untouched}},
    {0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_traceevent
  instConfigInt = instrConfigInt{
    "notImplemented", // name
    0,                // # outs
    0,                // # ins
    {},               // members
    0,                // template param mask
    {},               // impl versions
    nullptr,          // custom impl selector
    // L1 states per impl
    {{operandState::untouched}},
    // L2 states per impl
    {{operandState::untouched}},
    // CB states per impl
    {{operandState::untouched}},
    {0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_transpose
  instConfigInt = instrConfigInt{
    "Transpose",               // name
    1,                         // # outs
    1,                         // # ins
    {instrMembers::mbShuffle}, // members
    2,                         // template param mask
    {"Aligned32Bytes"},        // impl versions
    implSel::Transpose,        // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}, {operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}, {operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}, {operandState::untouched, operandState::untouched}}},
    {0x0, 0x0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_inhosttransform
  instConfigInt = instrConfigInt{
    "notImplemented", // name
    0,                // # outs
    0,                // # ins
    {},               // members
    0,                // template param mask
    {},               // impl versions
    nullptr,          // custom impl selector
    // L1 states per impl
    {{operandState::untouched}},
    // L2 states per impl
    {{operandState::untouched}},
    // CB states per impl
    {{operandState::untouched}},
    {0} // evict available mask
  };
  instrConfigTable.push_back(instConfigInt);

  // ET_outhosttransform
  instConfigInt = instrConfigInt{
    "notImplemented", // name
    0,                // # outs
    0,                // # ins
    {},               // members
    0,                // template param mask
    {},               // impl versions
    nullptr,          // custom impl selector
    // L1 states per impl
    {{operandState::untouched}},
    // L2 states per impl
    {{operandState::untouched}},
    // CB states per impl
    {{operandState::untouched}},
    {0} // evict available mask
  };
}

operandState instrConfig::getOperandStateL1(size_t implIdx, size_t operand) {
  assert(operand < nrOutputTensors + nrInputTensors);
  return stateL1[implIdx][operand];
}
operandState instrConfig::getOperandStateL2(size_t implIdx, size_t operand) {
  assert(operand < nrOutputTensors + nrInputTensors);
  return stateL2[implIdx][operand];
}
operandState instrConfig::getOperandStateCB(size_t implIdx, size_t operand) {
  assert(operand < nrOutputTensors + nrInputTensors);
  return stateCB[implIdx][operand];
}
bool instrConfig::getOperandAutoEvict(size_t implIdx, size_t operand) {
  assert(operand < nrOutputTensors + nrInputTensors);
  return ((evictAvailableMask[implIdx] >> operand) & 1);
}
operandState instrConfig::getSrcStateL1(size_t implIdx, size_t idx) {
  return getOperandStateL1(implIdx, idx + nrOutputTensors);
}
operandState instrConfig::getSrcStateL2(size_t implIdx, size_t idx) {
  return getOperandStateL2(implIdx, idx + nrOutputTensors);
}
operandState instrConfig::getSrcStateCB(size_t implIdx, size_t idx) {
  return getOperandStateCB(implIdx, idx + nrOutputTensors);
}
bool instrConfig::getSrcAutoEvict(size_t implIdx, size_t idx) {
  return getOperandAutoEvict(implIdx, idx + nrOutputTensors);
}
operandState instrConfig::getDstStateL1(size_t implIdx, size_t idx) {
  assert(idx < nrOutputTensors);
  return getOperandStateL1(implIdx, idx);
}
operandState instrConfig::getDstStateL2(size_t implIdx, size_t idx) {
  assert(idx < nrOutputTensors);
  return getOperandStateL2(implIdx, idx);
}
operandState instrConfig::getDstStateCB(size_t implIdx, size_t idx) {
  assert(idx < nrOutputTensors);
  return getOperandStateCB(implIdx, idx);
}
bool instrConfig::getDstAutoEvict(size_t implIdx, size_t idx) {
  assert(idx < nrOutputTensors);
  return getOperandAutoEvict(implIdx, idx);
}

//////////////

bool caseInsCharCompare(char a, char b) {
  return (std::toupper(a) == std::toupper(b));
}

bool caseInsCompare(const std::string& s1, const std::string& s2) {
  return ((s1.size() == s2.size()) && equal(s1.begin(), s1.end(), s2.begin(), caseInsCharCompare));
}

bool getInstrConfig(const std::string& operatorName, instrConfigInt& instConfig) {
  // Initializes the table for the first access
  if (!tableInitialized) {
    initInstrConfigTable();
    tableInitialized = true;
  }

  // Linear access through all elements
  for (auto it = instrConfigTable.begin(); it != instrConfigTable.end(); ++it) {
    if (caseInsCompare(it->name, operatorName)) {
      instConfig = *it;
      return true;
    }
  }
  return false;
}

bool getInstrConfig(const std::string& operatorName, instrConfig& instConfig) {
  instrConfigInt instConfigInt;

  // Looks for the internal config
  if (!getInstrConfig(operatorName, instConfigInt)) {
    return false;
  }

  // Converts to the API config
  instConfig.name = instConfigInt.name;
  instConfig.nrOutputTensors = instConfigInt.nrOutputTensors;
  instConfig.nrInputTensors = instConfigInt.nrInputTensors;
  instConfig.members = instConfigInt.members;
  instConfig.templateMask = instConfigInt.templateMask;
  instConfig.versions = instConfigInt.versions;
  instConfig.stateL1 = instConfigInt.stateL1;
  instConfig.stateL2 = instConfigInt.stateL2;
  instConfig.stateCB = instConfigInt.stateCB;
  instConfig.evictAvailableMask = instConfigInt.evictAvailableMask;

  return true;
}

size_t getInstrNumCycles(const std::string& operatorName, size_t assignedMinions, const std::vector<Tensor>& operands) {
  size_t result = 2000;
  if (caseInsCompare("FusedRowwiseQuantizedSparseLengthsWeightedSum", operatorName)) {
    // Computes lines to read from DDR
    size_t lineSize = operands[0].strides[0];
    size_t lineCL = (size_t)ceilf((float)lineSize / 64.0f); // Converts to cachelines
    size_t lookups = operands[1].sizes[0];
    // Data read from the embedding
    size_t totalBytes = lineCL * lookups * 64;
    // Data read to get lookup indices and the weight
    // TODO: need to get the lookup index size and weight size
    totalBytes += lookups * 8 + lookups * 4;

    // Cycles is amount of CL to read by cycles per CL
    // TODO: gather this info from device
    float ddrBytesPerCycle = 133.0f * 0.55f;
    float ddrBytesPerMinion = ddrBytesPerCycle / 1024.0f;
    float minions = (float)assignedMinions;
    float ddrCycles = (float)totalBytes / ddrBytesPerMinion / minions;

    // Total amount of time is fixed time plus ddr time
    result = 4000 + (size_t)ddrCycles;
  } else if (caseInsCompare("FusedRowwiseQuantizedSparseLengthsSum", operatorName)) {
    // Computes lines to read from DDR
    size_t lineSize = operands[0].strides[0];
    size_t lineCL = (size_t)ceilf((float)lineSize / 64.0f); // Converts to cachelines
    size_t lookups = operands[1].sizes[0];
    // Data read from the embedding
    size_t totalBytes = lineCL * lookups * 64;
    // Data read to get lookup indices and the weight
    // TODO: need to get the lookup index size
    totalBytes += lookups * 8;

    // Cycles is amount of CL to read by cycles per CL
    // TODO: gather this info from device
    float ddrBytesPerCycle = 133.0f * 0.55f;
    float ddrBytesPerMinion = ddrBytesPerCycle / 1024.0f;
    float minions = (float)assignedMinions;
    float ddrCycles = (float)totalBytes / ddrBytesPerMinion / minions;

    // Total amount of time is fixed time plus ddr time
    result = 4000 + (size_t)ddrCycles;
  }

  return result;
}

size_t getImplementation(const std::string& operatorName, const std::vector<Tensor>& outOperands,
                         const std::vector<Tensor>& inOperands) {
  instrConfigInt instConfig;
  // If the operator doesn't exist, return a negative
  if (!getInstrConfig(operatorName, instConfig)) {
    return (size_t)-1;
  }

  // Converts the Tensor to LibTensor
  std::vector<std::unique_ptr<LibTensor>> outOperandsConverted, inOperandsConverted;
  std::vector<LibTensor*> outOperandsConvertedPtr, inOperandsConvertedPtr;

  for (auto& operand : outOperands) {
    auto libTensor = std::make_unique<dnn_lib::LibTensor>(operand);
    outOperandsConvertedPtr.push_back(libTensor.get());
    outOperandsConverted.push_back(std::move(libTensor));
  }
  for (auto& operand : inOperands) {
    auto libTensor = std::make_unique<dnn_lib::LibTensor>(operand);
    inOperandsConvertedPtr.push_back(libTensor.get());
    inOperandsConverted.push_back(std::move(libTensor));
  }
  return instConfig.implSel(outOperandsConvertedPtr, inOperandsConvertedPtr);
}

} // end namespace dnn_lib
