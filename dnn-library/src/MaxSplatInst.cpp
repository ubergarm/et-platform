
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibMaxSplatInst(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMaxSplatInst<FloatTy>(out0, in0, Value, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibMaxSplatInst"Threaded"(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMaxSplatInst"Threaded"<FloatTy>(out0, in0, Value, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibMaxSplatInst"Vectorized"(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMaxSplatInst"Vectorized"<FloatTy>(out0, in0, Value, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibMaxSplatInst"Aligned32Bytes"(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMaxSplatInst"Aligned32Bytes"<FloatTy>(out0, in0, Value, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibMaxSplatInst<FloatTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst<Float16Ty>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst<Int8QTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst<Int64ITy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst<Int32ITy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst<Int16QTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Vectorized"<FloatTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Vectorized"<Float16Ty>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Vectorized"<Int8QTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Vectorized"<Int64ITy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Vectorized"<Int32ITy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Vectorized"<Int16QTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Aligned32Bytes"<FloatTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Aligned32Bytes"<Float16Ty>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Aligned32Bytes"<Int8QTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Aligned32Bytes"<Int64ITy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Aligned32Bytes"<Int32ITy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst"Aligned32Bytes"<Int16QTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
