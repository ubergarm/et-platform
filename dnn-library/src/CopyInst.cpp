
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibCopyInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibCopyInst<out0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibCopyInstThreaded(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibCopyInstThreaded<out0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibCopyInstVectorized(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibCopyInstVectorized<out0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibCopyInstTensorized(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibCopyInstTensorized<out0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibCopyInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstThreaded<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstThreaded<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstThreaded<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstThreaded<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstThreaded<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstThreaded<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstVectorized<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstVectorized<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstVectorized<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstVectorized<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstVectorized<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstVectorized<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstTensorized<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstTensorized<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstTensorized<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstTensorized<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstTensorized<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstTensorized<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
