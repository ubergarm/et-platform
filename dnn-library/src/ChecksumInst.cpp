
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibChecksumInst(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibChecksumInst<FloatTy>(in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibChecksumInst<FloatTy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibChecksumInst<Float16Ty>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibChecksumInst<Int8QTy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibChecksumInst<Int64ITy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibChecksumInst<Int32ITy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibChecksumInst<Int16QTy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
