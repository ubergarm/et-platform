#ifndef _LIB_API_H_
#define _LIB_API_H_

namespace dnn_lib {
  enum instrMembers
  {
   mbInvalid,
   mbHalfWindowSize,
   mbAlpha,
   mbBeta,
   mbK,
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
   mbMaxMembers
  };

  static constexpr size_t maxImplVersions = 8;
  static constexpr size_t maxInstrConfigStrLen = 256;
  
  struct instrConfig {
    char name[maxInstrConfigStrLen];
    size_t nrOutputTensors; // number of output and in/out tensor operands
    size_t nrInputTensors;  // number of input tensor operands
    std::array<instrMembers, mbMaxMembers> members;
    uint64_t templateMask;
    std::array<char[maxInstrConfigStrLen], maxImplVersions> versions;
    //TODO: add best version selector
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
       {} // impl versions
     },
     /**** ET_allocactivation ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_argmax ****/
     { "ArgMax", // name
       1, // # outs
       1,  // # ins
       {mbAxis, mbKeepDims}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_avgpool ****/
     { "AvgPool", // name
       1, // # outs
       1,  // # ins
       {mbKernels, mbStrides, mbPads}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_batchedadd ****/
     { "BatchedAdd", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded", "i8i32", "i8i32Threaded"} // impl versions
     },
     /**** ET_batchedreduceadd ****/
     { "BatchedReduceAdd", // name
       1, // # outs
       1,  // # ins
       {mbAxis}, // members
       0, // template param mask
       {"Threaded", "Int8", "Int8Threaded"} // impl versions
     },
     /**** ET_batchonehot ****/
     { "BatchOneHot", // name
       1, // # outs
       3,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_checksum ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_convertto ****/
     { "ConvertTo", // name
       1, // # outs
       1,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded", "Vectorized"} // impl versions
     },
     /**** ET_int8converter ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_convolution ****/
     { "Convolution", // name
       1, // # outs
       3,  // # ins
       {mbKernels, mbStrides, mbPads, mbGroup}, // members
       0, // template param mask
       {"Threaded", "Vectorized"} // impl versions
     },
     /**** ET_convolution3d ****/
     { "Convolution3D", // name
       1, // # outs
       3,  // # ins
       {mbKernels, mbStrides, mbPads, mbGroup}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_copy ****/
     { "Copy", // name
       1, // # outs
       1,  // # ins
       {}, // members
       0, // template param mask
       {"Tensorized", "Threaded", "Vectorized"} // impl versions
     },
     /**** ET_crc ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_crossentropyloss ****/
     { "CrossEntropyLoss", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_deallocactivation ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_debugprint ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_dequantize ****/
     { "Dequantize", // name
       1, // # outs
       1,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_elementadd ****/
     { "ElementAdd", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded", "Vectorized"} // impl versions
     },
     /**** ET_elementandi ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_elementcmpeq ****/
     { "ElementCmpEQ", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded", "Vectorized"} // impl versions
     },
     /**** ET_elementcmplte ****/
     { "ElementCmpLTE", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded", "Vectorized"} // impl versions
     },
     /**** ET_elementcmplt ****/
     { "ElementCmpLT", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded", "Vectorized"} // impl versions
     },
     /**** ET_elementdiv ****/
     { "ElementDiv", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded", "Vectorized"} // impl versions
     },
     /**** ET_elementexp ****/
     { "ElementExp", // name
       1, // # outs
       1,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_elementisnan ****/
     { "ElementIsNaN", // name
       1, // # outs
       1,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_elementlog ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_elementmax ****/
     { "ElementMax", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded", "Vectorized"} // impl versions
     },
     /**** ET_elementmin ****/
     { "ElementMin", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded", "Vectorized"} // impl versions
     },
     /**** ET_elementmul ****/
     { "ElementMul", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded", "Vectorized"} // impl versions
     },
     /**** ET_elementori ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_elementpow ****/
     { "ElementPow", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded", "Vectorized"} // impl versions
     },
     /**** ET_elementselect ****/
     { "ElementSelect", // name
       1, // # outs
       3,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_elementsub ****/
     { "ElementSub", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded", "Vectorized"} // impl versions
     },
     /**** ET_elementxori ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_embeddingbag ****/
     { "EmbeddingBag", // name
       1, // # outs
       4,  // # ins
       {}, // members
       0, // template param mask
       {"FloatTy"} // impl versions
     },
     /**** ET_emptyoperator ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_etsocfullyconnected ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_etsocmaxsplat ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_extracttensor ****/
     { "ExtractTensor", // name
       1, // # outs
       1,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_flip ****/
     { "Flip", // name
       1, // # outs
       1,  // # ins
       {mbAxis}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_flushL3 ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_fullyconnected ****/
     { "FullyConnected", // name
       1, // # outs
       3,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded", "Vectorized"} // impl versions
     },
     /**** ET_fusedrowwisequantizedsparselengthsweightedsum ****/
     { "FusedRowwiseQuantizedSparseLengthsWeightedSum", // name
       1, // # outs
       4,  // # ins
       {}, // members
       0, // template param mask
       {"FloatTy", "FloatTyThreaded", "FloatTyVectorized", "FloatTyVectorized"} // impl versions
     },
     /**** ET_fusedrowwisequantizedsparselengthssum ****/
     { "FusedRowwiseQuantizedSparseLengthsSum", // name
       1, // # outs
       3,  // # ins
       {}, // members
       0, // template param mask
       {"FloatTyThreaded", "FloatTyVectorized"} // impl versions
     },
     /**** ET_gather ****/
     { "Gather", // name
       1, // # outs
       2,  // # ins
       {mbBatchDims}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_gatherranges ****/
     { "GatherRanges", // name
       2, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_inserttensor ****/
     { "InsertTensor", // name
       1, // # outs
       1,  // # ins
       {mbOffsets, mbCount}, // members
       0, // template param mask
       {"Threaded", "Threaded"} // impl versions
     },
     /**** ET_intlookuptable ****/
     { "IntLookupTable", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Int8QTy", "Int8QTyThreaded"} // impl versions
     },
     /**** ET_lengthssum ****/
     { "LengthsSum", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_lengthstoranges ****/
     { "LengthsToRanges", // name
       1, // # outs
       1,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_localresponsenormalization ****/
     { "LocalResponseNormalization", // name
       2, // # outs
       1,  // # ins
       {mbHalfWindowSize, mbAlpha, mbBeta, mbK}, // members
       0, // template param mask
       {"Threaded", "Vectorized"} // impl versions
     },
     /**** ET_matmul ****/
     { "MatMul", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded", "Vectorized", "Transposed", "ThreadedTransposed", "VectorizedTransposed"} // impl versions
     },
     /**** ET_maxpool ****/
     { "MaxPool", // name
       1, // # outs
       1,  // # ins
       {mbKernels, mbStrides, mbPads}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_maxpoolwithargmax ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_modulo ****/
     { "Modulo", // name
       1, // # outs
       1,  // # ins
       {mbDivisor, mbSignFollowDivisor}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_quantizationprofile ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_quantize ****/
     { "Quantize", // name
       1, // # outs
       1,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_rescalequantized ****/
     { "RescaleQuantized", // name
       1, // # outs
       1,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_rowwisequantizedfullyconnected ****/
     { "RowwiseQuantizedFullyConnected", // name
       1, // # outs
       5,  // # ins
       {}, // members
       0, // template param mask
       {"Int8QTy", "Int8QTyThreaded", "Int8QTyVectorized", "Int8QTyAligned32Bytes"} // impl versions
     },
     /**** ET_rowwisequantizedsparselengthsweightedsum ****/
     { "RowwiseQuantizedSparseLengthsWeightedSum", // name
       1, // # outs
       6,  // # ins
       {}, // members
       0, // template param mask
       {"FloatTy", "FloatTyThreaded", "FloatTyVectorized"} // impl versions
     },
     /**** ET_scatterdata ****/
     { "ScatterData", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_sigmoid ****/
     { "Sigmoid", // name
       1, // # outs
       1,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_softmax ****/
     { "SoftMax", // name
       1, // # outs
       1,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded", "Vectorized", "Threaded1", "Vectorized1", "2", "Threaded2"} // impl versions
     },
     /**** ET_sparselengthsweightedsum ****/
     { "SparseLengthsWeightedSum", // name
       1, // # outs
       4,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_sparsetodense ****/
     { "SparseToDense", // name
       1, // # outs
       2,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded", "Vectorized"} // impl versions
     },
     /**** ET_sparsetodensemask ****/
     { "SparseToDenseMask", // name
       1, // # outs
       4,  // # ins
       {mbMask}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_splat ****/
     { "Splat", // name
       1, // # outs
       0,  // # ins
       {mbValue}, // members
       0, // template param mask
       {"Threaded", "Vectorized"} // impl versions
     },
     /**** ET_sync ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_syncopy ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_tanh ****/
     { "Tanh", // name
       1, // # outs
       1,  // # ins
       {}, // members
       0, // template param mask
       {"Threaded"} // impl versions
     },
     /**** ET_tensorview ****/
     { "TensorView", // name
       0, // # outs
       1,  // # ins
       {mbOffsets}, // members
       0, // template param mask
       {"Threaded", "Vectorized"} // impl versions
     },
     /**** ET_topk ****/
     { "TopK", // name
       2, // # outs
       1,  // # ins
       {mbK}, // members
       0, // template param mask
       {"Threaded_all", "Threaded_k4", "Threaded_k4", "Threaded_k8"} // impl versions
     },
     /**** ET_touch ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_traceevent ****/
     { "notImplemented", // name
       0, // # outs
       0,  // # ins
       {}, // members
       0, // template param mask
       {} // impl versions
     },
     /**** ET_transpose ****/
     { "Transpose", // name
       1, // # outs
       1,  // # ins
       {mbShuffle}, // members
       0, // template param mask
       {"Threaded", "Vectorized", "Aligned32Bytes"} // impl versions
     }
     // INSTR_CONFIG_TABLE_END
    };
}
#endif
