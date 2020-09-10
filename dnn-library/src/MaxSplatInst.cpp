
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibMaxSplatInst(LibTensor* out0, LibTensor* in0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMaxSplatInst<in0Type>(out0, in0, Value, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibMaxSplatInstAligned32Bytes(LibTensor* out0, LibTensor* in0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMaxSplatInstAligned32Bytes<in0Type>(out0, in0, Value, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibMaxSplatInst<FloatTy>(LibTensor* out0, LibTensor* in0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInstAligned32Bytes<FloatTy>(LibTensor* out0, LibTensor* in0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInstAligned32Bytes<Float16Ty>(LibTensor* out0, LibTensor* in0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInstAligned32Bytes<Int8QTy>(LibTensor* out0, LibTensor* in0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInstAligned32Bytes<Int64ITy>(LibTensor* out0, LibTensor* in0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInstAligned32Bytes<Int32ITy>(LibTensor* out0, LibTensor* in0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInstAligned32Bytes<Int16QTy>(LibTensor* out0, LibTensor* in0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
