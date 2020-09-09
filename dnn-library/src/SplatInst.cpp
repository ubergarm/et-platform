
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibSplatInst(LibTensor* out0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSplatInst<out0Type>(out0, Value, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSplatInst<FloatTy>(LibTensor* out0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst<Float16Ty>(LibTensor* out0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst<Int8QTy>(LibTensor* out0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst<Int64ITy>(LibTensor* out0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst<Int32ITy>(LibTensor* out0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst<Int16QTy>(LibTensor* out0, const float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
