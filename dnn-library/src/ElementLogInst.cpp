
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibElementLogInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementLogInst<out0Type, in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementLogInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementLogInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
