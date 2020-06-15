
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibTopKInst(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTopKInst<FloatTy>(out0, out1, in0, K, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibTopKInst"Threaded_all"(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTopKInst"Threaded_all"<FloatTy>(out0, out1, in0, K, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibTopKInst"Threaded_k4"(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTopKInst"Threaded_k4"<FloatTy>(out0, out1, in0, K, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibTopKInst"Threaded_k8"(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTopKInst"Threaded_k8"<FloatTy>(out0, out1, in0, K, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibTopKInst<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst<Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst<Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst<Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst<Int16QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_all"<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_all"<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_all"<Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_all"<Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_all"<Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_all"<Int16QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k4"<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k4"<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k4"<Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k4"<Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k4"<Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k4"<Int16QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k4"<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k4"<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k4"<Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k4"<Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k4"<Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k4"<Int16QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k8"<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k8"<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k8"<Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k8"<Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k8"<Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst"Threaded_k8"<Int16QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
