
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibModuloInst(LibTensor* out0, LibTensor* in0, const uint64_t Divisor, const bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibModuloInst<in0Type>(out0, in0, Divisor, SignFollowDivisor, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibModuloInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t Divisor, const bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibModuloInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t Divisor, const bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
