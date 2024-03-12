#ifndef INSTR_TABLE_GENERATED_H
#define INSTR_TABLE_GENERATED_H

// clang-format off

// File automatically generated with:
//  ./libManager.py --swplatform-root ../../../ --excel libManager.xlsx --cacheState cacheState.xlsx
//  cwd=/local/home/fgispert/sw-platform/host-software/dnnLibrary/scripts

// Manual changes will be detected by CI


static const std::vector<InstrConfigInt> instrConfigTable = { 
  // ET_adaptiveavgpool
  {
    "AdaptiveAvgPool", // name
    1, // # outs
    1,  // # ins
    {}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_argmax
  {
    "ArgMax", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbAxis, instrMembers::mbKeepDims}, // members
    3, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_avgpool
  {
    "AvgPool", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbKernels, instrMembers::mbStrides, instrMembers::mbPads, instrMembers::mbLayout, instrMembers::mbCountIncludePads}, // members
    1, // template param mask
    {"Threaded"}, // impl versions
    implSel::AvgPool, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean},
        {operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean},
        {operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched},
        {operandState::untouched, operandState::untouched}}},
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_batchedadd
  {
    "BatchedAdd", // name
    1, // # outs
    2,  // # ins
    {}, // members
    7, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_batchedreduceadd
  {
    "BatchedReduceAdd", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbAxis}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::untouched, operandState::clean}}},
    // L2 states per impl
    {{{operandState::untouched, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x1}  // global store mask
  },

  // ET_batchedreducemin
  {
    "BatchedReduceMin", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbAxes}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_batchonehot
  {
    "BatchOneHot", // name
    1, // # outs
    3,  // # ins
    {}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_convertto
  {
    "ConvertTo", // name
    1, // # outs
    1,  // # ins
    {}, // members
    3, // template param mask
    {"Vectorized"}, // impl versions
    implSel::defaultSel<2>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean},
        {operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean},
        {operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched},
        {operandState::untouched, operandState::untouched}}},
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_int8converter
  {
    "Int8Converter", // name
    1, // # outs
    1,  // # ins
    {}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_channelwisequantizedconvolution
  {
    "ChannelWiseQuantizedConvolution", // name
    1, // # outs
    7,  // # ins
    {instrMembers::mbKernels, instrMembers::mbStrides, instrMembers::mbPads, instrMembers::mbGroup, instrMembers::mbDilation, instrMembers::mbFusedActivation, instrMembers::mbFusedActivationArgs}, // members
    9, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_convolution
  {
    "Convolution", // name
    1, // # outs
    3,  // # ins
    {instrMembers::mbKernels, instrMembers::mbStrides, instrMembers::mbPads, instrMembers::mbGroup, instrMembers::mbDilation, instrMembers::mbFusedActivation, instrMembers::mbFusedActivationArgs}, // members
    9, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_convolution3d
  {
    "Convolution3D", // name
    1, // # outs
    3,  // # ins
    {instrMembers::mbKernels, instrMembers::mbStrides, instrMembers::mbPads, instrMembers::mbGroup}, // members
    9, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_convtranspose
  {
    "ConvTranspose", // name
    1, // # outs
    3,  // # ins
    {instrMembers::mbKernels, instrMembers::mbStrides, instrMembers::mbPads, instrMembers::mbGroup, instrMembers::mbDilation}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_copy
  {
    "Copy", // name
    1, // # outs
    1,  // # ins
    {}, // members
    1, // template param mask
    {"Tensorized"}, // impl versions
    implSel::Copy, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean},
        {operandState::untouched, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean},
        {operandState::untouched, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched},
        {operandState::dirty, operandState::untouched}}},
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_crossentropyloss
  {
    "CrossEntropyLoss", // name
    1, // # outs
    2,  // # ins
    {}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_cumsum
  {
    "CumSum", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbExclusive, instrMembers::mbReverse, instrMembers::mbAxis}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_dequantize
  {
    "Dequantize", // name
    1, // # outs
    1,  // # ins
    {}, // members
    3, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_dequantize4bitscolumnblocks
  {
    "Dequantize4BitsColumnBlocks", // name
    1, // # outs
    3,  // # ins
    {}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_dequantize8bitscolumnblocks
  {
    "Dequantize8BitsColumnBlocks", // name
    1, // # outs
    3,  // # ins
    {}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_elementadd
  {
    "ElementAdd", // name
    1, // # outs
    2,  // # ins
    {}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_elementand
  {
    "ElementAnd", // name
    1, // # outs
    2,  // # ins
    {}, // members
    0, // template param mask
    {"Threaded"}, // impl versions
    implSel::defaultSel<2>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched},
        {operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_elementcmpeq
  {
    "ElementCmpEQ", // name
    1, // # outs
    2,  // # ins
    {}, // members
    6, // template param mask
    {"Vectorized"}, // impl versions
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
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_elementcmpneq
  {
    "ElementCmpNEQ", // name
    1, // # outs
    2,  // # ins
    {}, // members
    6, // template param mask
    {"Vectorized"}, // impl versions
    implSel::ElementCmpNEQ, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched},
        {operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_elementcmplte
  {
    "ElementCmpLTE", // name
    1, // # outs
    2,  // # ins
    {}, // members
    6, // template param mask
    {"Vectorized"}, // impl versions
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
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_elementcmplt
  {
    "ElementCmpLT", // name
    1, // # outs
    2,  // # ins
    {}, // members
    6, // template param mask
    {"Vectorized"}, // impl versions
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
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_elementcos
  {
    "ElementCos", // name
    1, // # outs
    1,  // # ins
    {}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_elementdiv
  {
    "ElementDiv", // name
    1, // # outs
    2,  // # ins
    {}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_elementerf
  {
    "ElementErf", // name
    1, // # outs
    1,  // # ins
    {}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_elementexp
  {
    "ElementExp", // name
    1, // # outs
    1,  // # ins
    {}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_elementisnan
  {
    "ElementIsNaN", // name
    1, // # outs
    1,  // # ins
    {}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_elementlog
  {
    "ElementLog", // name
    1, // # outs
    1,  // # ins
    {}, // members
    3, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_elementmax
  {
    "ElementMax", // name
    1, // # outs
    2,  // # ins
    {}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_elementmin
  {
    "ElementMin", // name
    1, // # outs
    2,  // # ins
    {}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_elementmul
  {
    "ElementMul", // name
    1, // # outs
    2,  // # ins
    {}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_elementneg
  {
    "ElementNeg", // name
    1, // # outs
    1,  // # ins
    {}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_elementnot
  {
    "ElementNot", // name
    1, // # outs
    1,  // # ins
    {}, // members
    0, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_elementor
  {
    "ElementOr", // name
    1, // # outs
    2,  // # ins
    {}, // members
    0, // template param mask
    {"Threaded"}, // impl versions
    implSel::defaultSel<2>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched},
        {operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_elementpow
  {
    "ElementPow", // name
    1, // # outs
    2,  // # ins
    {}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_elementselect
  {
    "ElementSelect", // name
    1, // # outs
    3,  // # ins
    {}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_elementsin
  {
    "ElementSin", // name
    1, // # outs
    1,  // # ins
    {}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_elementsub
  {
    "ElementSub", // name
    1, // # outs
    2,  // # ins
    {}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_elementxor
  {
    "ElementXor", // name
    1, // # outs
    2,  // # ins
    {}, // members
    0, // template param mask
    {"Threaded"}, // impl versions
    implSel::defaultSel<2>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched},
        {operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_embeddingbag
  {
    "EmbeddingBag", // name
    1, // # outs
    4,  // # ins
    {instrMembers::mbHasEndOffset}, // members
    11, // template param mask
    {"Vectorized", "Fastpath"}, // impl versions
    implSel::EmbeddingBag, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched},
        {operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched},
        {operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0, 0x0, 0x0}, // evict available mask
    {0x0, 0x0, 0x0}  // global store mask
  },

  // ET_maxsplat
  {
    "MaxSplat", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbValueBits}, // members
    2, // template param mask
    {"Aligned32Bytes"}, // impl versions
    implSel::MaxSplat, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean},
        {operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean},
        {operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched},
        {operandState::untouched, operandState::untouched}}},
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_extracttensor
  {
    "ExtractTensor", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbOffsets}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_flip
  {
    "Flip", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbAxis}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_fullyconnected
  {
    "FullyConnected", // name
    1, // # outs
    3,  // # ins
    {}, // members
    9, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::untouched, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::untouched, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x1}  // global store mask
  },

  // ET_fusedrowwisequantizedsparselengthsweightedsum
  {
    "FusedRowwiseQuantizedSparseLengthsWeightedSum", // name
    1, // # outs
    4,  // # ins
    {}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_fusedrowwisequantizedsparselengthssum
  {
    "FusedRowwiseQuantizedSparseLengthsSum", // name
    1, // # outs
    3,  // # ins
    {}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_gather
  {
    "Gather", // name
    1, // # outs
    2,  // # ins
    {instrMembers::mbBatchDims}, // members
    6, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x1}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_gatherranges
  {
    "GatherRanges", // name
    2, // # outs
    2,  // # ins
    {}, // members
    12, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_inserttensor
  {
    "InsertTensor", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbOffsets, instrMembers::mbCount, instrMembers::mbAxis}, // members
    2, // template param mask
    {"Threaded"}, // impl versions
    implSel::InsertTensor, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean},
        {operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean},
        {operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched},
        {operandState::untouched, operandState::untouched}}},
    {0x1, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_intlookuptable
  {
    "IntLookupTable", // name
    1, // # outs
    2,  // # ins
    {}, // members
    0, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_lengthsrangefill
  {
    "LengthsRangeFill", // name
    1, // # outs
    1,  // # ins
    {}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_lengthssum
  {
    "LengthsSum", // name
    1, // # outs
    2,  // # ins
    {}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_lengthstoranges
  {
    "LengthsToRanges", // name
    1, // # outs
    1,  // # ins
    {}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_localresponsenormalization
  {
    "LocalResponseNormalization", // name
    2, // # outs
    1,  // # ins
    {instrMembers::mbHalfWindowSize, instrMembers::mbAlpha, instrMembers::mbBeta, instrMembers::mbK}, // members
    2, // template param mask
    {"Vectorized"}, // impl versions
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
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_matmul
  {
    "MatMul", // name
    1, // # outs
    2,  // # ins
    {}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::untouched, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::untouched, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x1}  // global store mask
  },

  // ET_maxpool
  {
    "MaxPool", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbKernels, instrMembers::mbStrides, instrMembers::mbPads, instrMembers::mbLayout}, // members
    3, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_maxpoolwithargmax
  {
    "MaxPoolWithArgMax", // name
    2, // # outs
    1,  // # ins
    {instrMembers::mbKernels, instrMembers::mbStrides, instrMembers::mbPads, instrMembers::mbLayout}, // members
    5, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_modulo
  {
    "Modulo", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbDivisor, instrMembers::mbSignFollowDivisor}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_nonmaxsuppression
  {
    "NonMaxSuppression", // name
    2, // # outs
    2,  // # ins
    {instrMembers::mbCenterPointBox, instrMembers::mbMaxOutputBoxesPerClass, instrMembers::mbIouThreshold, instrMembers::mbScoreThreshold, instrMembers::mbIsTFVersion4}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_profile
  {
    "Profile", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbSyncOffset}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_quantize
  {
    "Quantize", // name
    1, // # outs
    1,  // # ins
    {}, // members
    3, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_rescalequantized
  {
    "RescaleQuantized", // name
    1, // # outs
    1,  // # ins
    {}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_resizebilinear
  {
    "ResizeBilinear", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbRszScale}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_resizenearest
  {
    "ResizeNearest", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbRszScale}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_rowwisequantizedfullyconnected
  {
    "RowwiseQuantizedFullyConnected", // name
    1, // # outs
    5,  // # ins
    {}, // members
    0, // template param mask
    {"Aligned32Bytes"}, // impl versions
    implSel::RowwiseQuantizedFullyConnected, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched},
        {operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_rowwisequantizedsparselengthsweightedsum
  {
    "RowwiseQuantizedSparseLengthsWeightedSum", // name
    1, // # outs
    6,  // # ins
    {}, // members
    33, // template param mask
    {"Vectorized"}, // impl versions
    implSel::RowwiseQuantizedSparseLengthsWeightedSum, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched},
        {operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_scatterdata
  {
    "ScatterData", // name
    1, // # outs
    2,  // # ins
    {}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_sigmoid
  {
    "Sigmoid", // name
    1, // # outs
    1,  // # ins
    {}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_softmax
  {
    "SoftMax", // name
    1, // # outs
    1,  // # ins
    {}, // members
    2, // template param mask
    {"Vectorized"}, // impl versions
    implSel::SoftMax, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean},
        {operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean},
        {operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched},
        {operandState::untouched, operandState::untouched}}},
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_spacetodepth
  {
    "SpaceToDepth", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbBlockSize}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_sparselengthssum
  {
    "SparseLengthsSum", // name
    1, // # outs
    3,  // # ins
    {}, // members
    5, // template param mask
    {"Threaded"}, // impl versions
    implSel::SparseLengthsSum, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched},
        {operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_sparselengthsweightedsum
  {
    "SparseLengthsWeightedSum", // name
    1, // # outs
    4,  // # ins
    {}, // members
    10, // template param mask
    {"Threaded"}, // impl versions
    implSel::SparseLengthsWeightedSum, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean},
        {operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched},
        {operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_sparsetodense
  {
    "SparseToDense", // name
    1, // # outs
    2,  // # ins
    {}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_sparsetodensemask
  {
    "SparseToDenseMask", // name
    1, // # outs
    4,  // # ins
    {instrMembers::mbMask}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_splat
  {
    "Splat", // name
    1, // # outs
    0,  // # ins
    {instrMembers::mbValueBits}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty}}},
    // L2 states per impl
    {{{operandState::dirty}}},
    // CB states per impl
    {{{operandState::untouched}}},
    {0x1}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_syncopy
  {
    "Syncopy", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbSyncOffset}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_tanh
  {
    "Tanh", // name
    1, // # outs
    1,  // # ins
    {}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_tensorview
  {
    "TensorView", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbOffsets}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_topk
  {
    "TopK", // name
    2, // # outs
    1,  // # ins
    {instrMembers::mbTopK}, // members
    4, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_transpose
  {
    "Transpose", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbShuffle}, // members
    2, // template param mask
    {"Aligned32Bytes"}, // impl versions
    implSel::Transpose, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean},
        {operandState::dirty, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean},
        {operandState::dirty, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched},
        {operandState::untouched, operandState::untouched}}},
    {0x0, 0x0}, // evict available mask
    {0x0, 0x0}  // global store mask
  },

  // ET_trilu
  {
    "Trilu", // name
    1, // # outs
    2,  // # ins
    {instrMembers::mbUpper}, // members
    2, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::clean, operandState::clean}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  },

  // ET_etsocgenericop
  {
    "ETSOCGenericOp", // name
    1, // # outs
    1,  // # ins
    {instrMembers::mbGenericOperation}, // members
    1, // template param mask
    {}, // impl versions
    implSel::defaultSel<1>, // custom impl selector
    // L1 states per impl
    {{{operandState::dirty, operandState::untouched}}},
    // L2 states per impl
    {{{operandState::dirty, operandState::untouched}}},
    // CB states per impl
    {{{operandState::untouched, operandState::untouched}}},
    {0x0}, // evict available mask
    {0x0}  // global store mask
  } 
}; 
#endif // INSTR_TABLE_GENERATED_H
