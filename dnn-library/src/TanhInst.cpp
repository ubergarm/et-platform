
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibTanhInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTanhInst<FloatTy>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibTanhInst"Threaded"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTanhInst"Threaded"<FloatTy>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibTanhInst<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTanhInst<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTanhInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTanhInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
