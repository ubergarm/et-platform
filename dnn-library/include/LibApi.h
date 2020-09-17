#ifndef _LIB_API_H_
#define _LIB_API_H_

#include "LibApiImplSel.h"

namespace dnn_lib {
  
  enum instrMembers
  {
   mbInvalid,
   mbHalfWindowSize,
   mbAlpha,
   mbBeta,
   mbK,
   mbTopK,
   mbDivisor,
   mbSignFollowDivisor,
   mbAxis,
   mbKeepDims,
   mbKernels,
   mbStrides,
   mbPads,
   mbGroup,
   mbOffsets,
   mbShuffle,
   mbMask,
   mbBatchDims,
   mbCount,
   mbValue,
   mbSyncOffset,
   mbExclusive,
   mbReverse,
   mbBlockSize,
   mbAxes,
   mbRszScale,
   mbHasEndOffset,
   mbTransposed,
   mbDilation,
   mbCenterPointBox,
   mbMaxOutputBoxesPerClass,
   mbIouThreshold,
   mbScoreThreshold,
   mbIsTFVersion4,
   mbMaxMembers
  };

  static constexpr size_t maxImplVersions = 4;
  static constexpr size_t maxInstrConfigStrLen = 256;
  static constexpr size_t maxNrOperands = 12;

  enum class operandState { stale, dirty, clean, invalid};
  
  struct instrConfig {
    using operandStateArray = std::array<operandState, maxNrOperands>;
    using implStateArray = std::array<operandStateArray, maxImplVersions + 1>;
    using sel_fnc_t = size_t (*)(std::vector<LibTensor*> &, std::vector<LibTensor*> &);
    
    char name[maxInstrConfigStrLen];
    size_t nrOutputTensors; // number of output and in/out tensor operands
    size_t nrInputTensors;  // number of input tensor operands
    std::array<instrMembers, mbMaxMembers> members;
    uint64_t templateMask;
    std::array<char[maxInstrConfigStrLen], maxImplVersions> versions;
    sel_fnc_t implSel;

    implStateArray stateL1;
    implStateArray stateL2;
    implStateArray stateCB;
    std::array<uint64_t,maxImplVersions + 1>  evictAvailableMask;
    
    // functions to retrieve operand information
    operandState getOperandStateL1(size_t implIdx, size_t operand) {
      assert( operand < nrOutputTensors + nrInputTensors);
      return stateL1[implIdx][operand];
    } 
    operandState getOperandStateL2(size_t implIdx, size_t operand) {
      assert( operand < nrOutputTensors + nrInputTensors);
      return stateL2[implIdx][operand];
    }
    operandState getOperandStateCB(size_t implIdx, size_t operand) {
      assert( operand < nrOutputTensors + nrInputTensors);
      return stateCB[implIdx][operand];
    }
    bool getOperandAutoEvict(size_t implIdx, size_t operand) {
      assert( operand < nrOutputTensors + nrInputTensors);
      return ((evictAvailableMask[implIdx] >> operand) & 1);
    }
    
    // and same as before, but index is either input or output
    operandState getSrcStateL1(size_t implIdx, size_t idx) {
      return getOperandStateL1(implIdx, idx + nrOutputTensors);
    }
    operandState getSrcStateL2(size_t implIdx, size_t idx) {
      return getOperandStateL1(implIdx, idx + nrOutputTensors);
    }
    operandState getSrcStateCB(size_t implIdx, size_t idx) {
      return getOperandStateL1(implIdx, idx + nrOutputTensors);
    }
    bool getSrcAutoEvict(size_t implIdx, size_t idx) {
      return getOperandAutoEvict(implIdx, idx + nrOutputTensors);
    }

    operandState getDstStateL1(size_t implIdx, size_t idx) {
      assert( idx < nrOutputTensors );
      return getOperandStateL1(implIdx, idx);
    }
    operandState getDstStateL2(size_t implIdx, size_t idx) {
      assert( idx < nrOutputTensors );
      return getOperandStateL1(implIdx, idx);
    }
    operandState getDstStateCB(size_t implIdx, size_t idx) {
      assert( idx < nrOutputTensors );
      return getOperandStateL1(implIdx, idx);
    }
    bool getDstAutoEvict(size_t implIdx, size_t idx) {
      assert( idx < nrOutputTensors );
      return getOperandAutoEvict(implIdx, idx);
    }


    
  };

  static constexpr instrConfig instrConfigTable []  =
    {
     // INSTR_CONFIG_TABLE_BEGIN
     /**** ET_adaptiveavgpool ****/
     { "AdaptiveAvgPool", // name
       1, // # outs
       1,  // # ins
       {}, // members
       2, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_allocactivation ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {}, // impl versions
       nullptr, // custom impl selector
       // L1 states per impl
       {{operandState::invalid}},
       // L2 states per impl
       {{operandState::invalid}},
       // CB states per impl
       {{operandState::invalid}},
       {0} // evict available mask
     },
     /**** ET_argmax ****/
     { "ArgMax", // name
       1, // # outs
       1,  // # ins
       {mbAxis, mbKeepDims}, // members
       2, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_avgpool ****/
     { "AvgPool", // name
       1, // # outs
       1,  // # ins
       {mbKernels, mbStrides, mbPads}, // members
       1, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_batchedadd ****/
     { "BatchedAdd", // name
       1, // # outs
       2,  // # ins
       {}, // members
       7, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_batchedreduceadd ****/
     { "BatchedReduceAdd", // name
       1, // # outs
       1,  // # ins
       {mbAxis}, // members
       2, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_batchedreducemin ****/
     { "BatchedReduceMin", // name
       1, // # outs
       1,  // # ins
       {mbAxes}, // members
       2, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_batchonehot ****/
     { "BatchOneHot", // name
       1, // # outs
       3,  // # ins
       {}, // members
       1, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_checksum ****/
     { "Checksum", // name
       0, // # outs
       1,  // # ins
       {}, // members
       1, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_convertto ****/
     { "ConvertTo", // name
       1, // # outs
       1,  // # ins
       {}, // members
       3, // template param mask
       {"Vectorized"}, // impl versions
       implSel::ConvertTo, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_int8converter ****/
     { "Int8Converter", // name
       1, // # outs
       1,  // # ins
       {}, // members
       2, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_channelwisequantizedconvolution ****/
     { "ChannelWiseQuantizedConvolution", // name
       1, // # outs
       7,  // # ins
       {mbKernels, mbStrides, mbPads, mbGroup, mbDilation}, // members
       9, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_channelwisequantizedconvolution3d ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {}, // impl versions
       nullptr, // custom impl selector
       // L1 states per impl
       {{operandState::invalid}},
       // L2 states per impl
       {{operandState::invalid}},
       // CB states per impl
       {{operandState::invalid}},
       {0} // evict available mask
     },
     /**** ET_convolution ****/
     { "Convolution", // name
       1, // # outs
       3,  // # ins
       {mbKernels, mbStrides, mbPads, mbGroup, mbDilation}, // members
       9, // template param mask
       {"Vectorized"}, // impl versions
       implSel::Convolution, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_convolution3d ****/
     { "Convolution3D", // name
       1, // # outs
       3,  // # ins
       {mbKernels, mbStrides, mbPads, mbGroup}, // members
       1, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_convtranspose ****/
     { "ConvTranspose", // name
       1, // # outs
       3,  // # ins
       {mbKernels, mbStrides, mbPads, mbGroup, mbDilation}, // members
       1, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_copy ****/
     { "Copy", // name
       1, // # outs
       1,  // # ins
       {}, // members
       1, // template param mask
       {"Tensorized"}, // impl versions
       implSel::Copy, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::clean, operandState::clean}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::clean, operandState::clean}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::dirty, operandState::clean}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_crc ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {}, // impl versions
       nullptr, // custom impl selector
       // L1 states per impl
       {{operandState::invalid}},
       // L2 states per impl
       {{operandState::invalid}},
       // CB states per impl
       {{operandState::invalid}},
       {0} // evict available mask
     },
     /**** ET_crossentropyloss ****/
     { "CrossEntropyLoss", // name
       1, // # outs
       2,  // # ins
       {}, // members
       2, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_cumsum ****/
     { "CumSum", // name
       1, // # outs
       1,  // # ins
       {mbExclusive, mbReverse}, // members
       2, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_deallocactivation ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {}, // impl versions
       nullptr, // custom impl selector
       // L1 states per impl
       {{operandState::invalid}},
       // L2 states per impl
       {{operandState::invalid}},
       // CB states per impl
       {{operandState::invalid}},
       {0} // evict available mask
     },
     /**** ET_debugprint ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {}, // impl versions
       nullptr, // custom impl selector
       // L1 states per impl
       {{operandState::invalid}},
       // L2 states per impl
       {{operandState::invalid}},
       // CB states per impl
       {{operandState::invalid}},
       {0} // evict available mask
     },
     /**** ET_dequantize ****/
     { "Dequantize", // name
       1, // # outs
       1,  // # ins
       {}, // members
       3, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_elementadd ****/
     { "ElementAdd", // name
       1, // # outs
       2,  // # ins
       {}, // members
       7, // template param mask
       {"Vectorized"}, // impl versions
       implSel::ElementAdd, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_elementandi ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {}, // impl versions
       nullptr, // custom impl selector
       // L1 states per impl
       {{operandState::invalid}},
       // L2 states per impl
       {{operandState::invalid}},
       // CB states per impl
       {{operandState::invalid}},
       {0} // evict available mask
     },
     /**** ET_elementcmpeq ****/
     { "ElementCmpEQ", // name
       1, // # outs
       2,  // # ins
       {}, // members
       6, // template param mask
       {"Vectorized"}, // impl versions
       implSel::ElementCmpEQ, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_elementcmplte ****/
     { "ElementCmpLTE", // name
       1, // # outs
       2,  // # ins
       {}, // members
       6, // template param mask
       {"Vectorized"}, // impl versions
       implSel::ElementCmpLTE, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_elementcmplt ****/
     { "ElementCmpLT", // name
       1, // # outs
       2,  // # ins
       {}, // members
       6, // template param mask
       {"Vectorized"}, // impl versions
       implSel::ElementCmpLT, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_elementdiv ****/
     { "ElementDiv", // name
       1, // # outs
       2,  // # ins
       {}, // members
       7, // template param mask
       {"Vectorized"}, // impl versions
       implSel::ElementDiv, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_elementexp ****/
     { "ElementExp", // name
       1, // # outs
       1,  // # ins
       {}, // members
       2, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_elementisnan ****/
     { "ElementIsNaN", // name
       1, // # outs
       1,  // # ins
       {}, // members
       2, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_elementlog ****/
     { "ElementLog", // name
       1, // # outs
       1,  // # ins
       {}, // members
       3, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_elementmax ****/
     { "ElementMax", // name
       1, // # outs
       2,  // # ins
       {}, // members
       7, // template param mask
       {"Vectorized"}, // impl versions
       implSel::ElementMax, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_elementmin ****/
     { "ElementMin", // name
       1, // # outs
       2,  // # ins
       {}, // members
       7, // template param mask
       {"Vectorized"}, // impl versions
       implSel::ElementMin, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_elementmul ****/
     { "ElementMul", // name
       1, // # outs
       2,  // # ins
       {}, // members
       7, // template param mask
       {"Vectorized"}, // impl versions
       implSel::ElementMul, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_elementori ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {}, // impl versions
       nullptr, // custom impl selector
       // L1 states per impl
       {{operandState::invalid}},
       // L2 states per impl
       {{operandState::invalid}},
       // CB states per impl
       {{operandState::invalid}},
       {0} // evict available mask
     },
     /**** ET_elementpow ****/
     { "ElementPow", // name
       1, // # outs
       2,  // # ins
       {}, // members
       7, // template param mask
       {"Vectorized"}, // impl versions
       implSel::ElementPow, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_elementselect ****/
     { "ElementSelect", // name
       1, // # outs
       3,  // # ins
       {}, // members
       1, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_elementsub ****/
     { "ElementSub", // name
       1, // # outs
       2,  // # ins
       {}, // members
       7, // template param mask
       {"Vectorized"}, // impl versions
       implSel::ElementSub, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_elementxori ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {}, // impl versions
       nullptr, // custom impl selector
       // L1 states per impl
       {{operandState::invalid}},
       // L2 states per impl
       {{operandState::invalid}},
       // CB states per impl
       {{operandState::invalid}},
       {0} // evict available mask
     },
     /**** ET_embeddingbag ****/
     { "EmbeddingBag", // name
       1, // # outs
       4,  // # ins
       {mbHasEndOffset}, // members
       1, // template param mask
       {"Vectorized"}, // impl versions
       implSel::EmbeddingBag, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_embeddingbagbyterowwiseoffsets ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {}, // impl versions
       nullptr, // custom impl selector
       // L1 states per impl
       {{operandState::invalid}},
       // L2 states per impl
       {{operandState::invalid}},
       // CB states per impl
       {{operandState::invalid}},
       {0} // evict available mask
     },
     /**** ET_emptyoperator ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {}, // impl versions
       nullptr, // custom impl selector
       // L1 states per impl
       {{operandState::invalid}},
       // L2 states per impl
       {{operandState::invalid}},
       // CB states per impl
       {{operandState::invalid}},
       {0} // evict available mask
     },
     /**** ET_maxsplat ****/
     { "MaxSplat", // name
       1, // # outs
       1,  // # ins
       {mbValue}, // members
       2, // template param mask
       {"Aligned32Bytes"}, // impl versions
       implSel::MaxSplat, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_extracttensor ****/
     { "ExtractTensor", // name
       1, // # outs
       1,  // # ins
       {mbOffsets}, // members
       2, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_flip ****/
     { "Flip", // name
       1, // # outs
       1,  // # ins
       {mbAxis}, // members
       2, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_flushL3 ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {}, // impl versions
       nullptr, // custom impl selector
       // L1 states per impl
       {{operandState::invalid}},
       // L2 states per impl
       {{operandState::invalid}},
       // CB states per impl
       {{operandState::invalid}},
       {0} // evict available mask
     },
     /**** ET_fullyconnected ****/
     { "FullyConnected", // name
       1, // # outs
       3,  // # ins
       {}, // members
       15, // template param mask
       {"Vectorized"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_fusedrowwisequantizedsparselengthsweightedsum ****/
     { "FusedRowwiseQuantizedSparseLengthsWeightedSum", // name
       1, // # outs
       4,  // # ins
       {}, // members
       1, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_fusedrowwisequantizedsparselengthssum ****/
     { "FusedRowwiseQuantizedSparseLengthsSum", // name
       1, // # outs
       3,  // # ins
       {}, // members
       1, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_gather ****/
     { "Gather", // name
       1, // # outs
       2,  // # ins
       {mbBatchDims}, // members
       6, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_gatherranges ****/
     { "GatherRanges", // name
       2, // # outs
       2,  // # ins
       {}, // members
       12, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_inserttensor ****/
     { "InsertTensor", // name
       1, // # outs
       1,  // # ins
       {mbOffsets, mbCount, mbAxis}, // members
       2, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_intlookuptable ****/
     { "IntLookupTable", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_lengthsrangefill ****/
     { "LengthsRangeFill", // name
       1, // # outs
       1,  // # ins
       {}, // members
       1, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_lengthssum ****/
     { "LengthsSum", // name
       1, // # outs
       2,  // # ins
       {}, // members
       2, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_lengthstoranges ****/
     { "LengthsToRanges", // name
       1, // # outs
       1,  // # ins
       {}, // members
       2, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_localresponsenormalization ****/
     { "LocalResponseNormalization", // name
       2, // # outs
       1,  // # ins
       {mbHalfWindowSize, mbAlpha, mbBeta, mbK}, // members
       2, // template param mask
       {"Vectorized"}, // impl versions
       implSel::LocalResponseNormalization, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_matmul ****/
     { "MatMul", // name
       1, // # outs
       2,  // # ins
       {mbTransposed}, // members
       2, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_maxpool ****/
     { "MaxPool", // name
       1, // # outs
       1,  // # ins
       {mbKernels, mbStrides, mbPads}, // members
       3, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_maxpoolwithargmax ****/
     { "MaxPoolWithArgMax", // name
       2, // # outs
       1,  // # ins
       {mbKernels, mbStrides, mbPads}, // members
       5, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_modulo ****/
     { "Modulo", // name
       1, // # outs
       1,  // # ins
       {mbDivisor, mbSignFollowDivisor}, // members
       2, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_nonmaxsuppression ****/
     { "NonMaxSuppression", // name
       2, // # outs
       2,  // # ins
       {mbCenterPointBox, mbMaxOutputBoxesPerClass, mbIouThreshold, mbScoreThreshold, mbIsTFVersion4}, // members
       1, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_quantizationprofile ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {}, // impl versions
       nullptr, // custom impl selector
       // L1 states per impl
       {{operandState::invalid}},
       // L2 states per impl
       {{operandState::invalid}},
       // CB states per impl
       {{operandState::invalid}},
       {0} // evict available mask
     },
     /**** ET_quantize ****/
     { "Quantize", // name
       1, // # outs
       1,  // # ins
       {}, // members
       3, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_rescalequantized ****/
     { "RescaleQuantized", // name
       1, // # outs
       1,  // # ins
       {}, // members
       2, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_resizebilinear ****/
     { "ResizeBilinear", // name
       1, // # outs
       1,  // # ins
       {mbRszScale}, // members
       1, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_resizenearest ****/
     { "ResizeNearest", // name
       1, // # outs
       1,  // # ins
       {mbRszScale}, // members
       1, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_rowwisequantizedfullyconnected ****/
     { "RowwiseQuantizedFullyConnected", // name
       1, // # outs
       5,  // # ins
       {}, // members
       0, // template param mask
       {"Aligned32Bytes"}, // impl versions
       implSel::RowwiseQuantizedFullyConnected, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_rowwisequantizedsparselengthsweightedsum ****/
     { "RowwiseQuantizedSparseLengthsWeightedSum", // name
       1, // # outs
       6,  // # ins
       {}, // members
       33, // template param mask
       {"Vectorized"}, // impl versions
       implSel::RowwiseQuantizedSparseLengthsWeightedSum, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_scatterdata ****/
     { "ScatterData", // name
       1, // # outs
       2,  // # ins
       {}, // members
       1, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_sigmoid ****/
     { "Sigmoid", // name
       1, // # outs
       1,  // # ins
       {}, // members
       2, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_softmax ****/
     { "SoftMax", // name
       1, // # outs
       1,  // # ins
       {}, // members
       2, // template param mask
       {"Vectorized"}, // impl versions
       implSel::SoftMax, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_spacetodepth ****/
     { "SpaceToDepth", // name
       1, // # outs
       1,  // # ins
       {mbBlockSize}, // members
       1, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_sparselengthssum ****/
     { "SparseLengthsSum", // name
       1, // # outs
       3,  // # ins
       {}, // members
       5, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_sparselengthsweightedsum ****/
     { "SparseLengthsWeightedSum", // name
       1, // # outs
       4,  // # ins
       {}, // members
       10, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_sparsetodense ****/
     { "SparseToDense", // name
       1, // # outs
       2,  // # ins
       {}, // members
       1, // template param mask
       {"Vectorized"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_sparsetodensemask ****/
     { "SparseToDenseMask", // name
       1, // # outs
       4,  // # ins
       {mbMask}, // members
       1, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_splat ****/
     { "Splat", // name
       1, // # outs
       0,  // # ins
       {mbValue}, // members
       1, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_sync ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {}, // impl versions
       nullptr, // custom impl selector
       // L1 states per impl
       {{operandState::invalid}},
       // L2 states per impl
       {{operandState::invalid}},
       // CB states per impl
       {{operandState::invalid}},
       {0} // evict available mask
     },
     /**** ET_syncopy ****/
     { "Syncopy", // name
       1, // # outs
       1,  // # ins
       {mbSyncOffset}, // members
       2, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_tanh ****/
     { "Tanh", // name
       1, // # outs
       1,  // # ins
       {}, // members
       2, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_tensorview ****/
     { "TensorView", // name
       1, // # outs
       1,  // # ins
       {mbOffsets}, // members
       1, // template param mask
       {}, // impl versions
       implSel::defaultSel<1>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid}}},
       {0x0} // evict available mask
     },
     /**** ET_topk ****/
     { "TopK", // name
       2, // # outs
       1,  // # ins
       {mbTopK}, // members
       4, // template param mask
       {"Threaded"}, // impl versions
       implSel::defaultSel<2>, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     },
     /**** ET_touch ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {}, // impl versions
       nullptr, // custom impl selector
       // L1 states per impl
       {{operandState::invalid}},
       // L2 states per impl
       {{operandState::invalid}},
       // CB states per impl
       {{operandState::invalid}},
       {0} // evict available mask
     },
     /**** ET_traceevent ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {}, // impl versions
       nullptr, // custom impl selector
       // L1 states per impl
       {{operandState::invalid}},
       // L2 states per impl
       {{operandState::invalid}},
       // CB states per impl
       {{operandState::invalid}},
       {0} // evict available mask
     },
     /**** ET_transpose ****/
     { "Transpose", // name
       1, // # outs
       1,  // # ins
       {mbShuffle}, // members
       2, // template param mask
       {"Aligned32Bytes"}, // impl versions
       implSel::Transpose, // custom impl selector
       // L1 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // L2 states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       // CB states per impl
       {{{operandState::invalid, operandState::invalid},
        {operandState::invalid, operandState::invalid}}},
       {0x0, 0x0} // evict available mask
     }
     // INSTR_CONFIG_TABLE_END
    };


}
#endif
