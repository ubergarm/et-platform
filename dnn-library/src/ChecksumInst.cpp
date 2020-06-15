
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibChecksumInst(LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibChecksumInst<FloatTy>(in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibChecksumInst<FloatTy>(LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibChecksumInst<Float16Ty>(LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibChecksumInst<Int8QTy>(LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibChecksumInst<Int64ITy>(LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibChecksumInst<Int32ITy>(LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibChecksumInst<Int16QTy>(LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
