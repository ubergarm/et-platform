
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibExtractTensorInst(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibExtractTensorInst<in0Type>(out0, in0, Offsets, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibExtractTensorInstThreaded(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibExtractTensorInstThreaded<in0Type>(out0, in0, Offsets, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibExtractTensorInst<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInstThreaded<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInstThreaded<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInstThreaded<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInstThreaded<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInstThreaded<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInstThreaded<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
