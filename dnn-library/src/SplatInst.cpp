
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibSplatInst(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSplatInst<FloatTy>(out0, Value, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibSplatInst"Threaded"(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSplatInst"Threaded"<FloatTy>(out0, Value, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibSplatInst"Vectorized"(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSplatInst"Vectorized"<FloatTy>(out0, Value, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSplatInst<FloatTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst<Float16Ty>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst<Int8QTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst<Int64ITy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst<Int32ITy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst<Int16QTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst"Threaded"<FloatTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst"Threaded"<Float16Ty>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst"Threaded"<Int8QTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst"Threaded"<Int64ITy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst"Threaded"<Int32ITy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst"Threaded"<Int16QTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst"Vectorized"<FloatTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst"Vectorized"<Float16Ty>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst"Vectorized"<Int8QTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst"Vectorized"<Int64ITy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst"Vectorized"<Int32ITy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst"Vectorized"<Int16QTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
