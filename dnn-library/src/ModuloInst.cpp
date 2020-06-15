
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibModuloInst(LibTensor* out0, LibTensor* in0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibModuloInst<Int64ITy>(out0, in0, Divisor, SignFollowDivisor, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibModuloInstThreaded(LibTensor* out0, LibTensor* in0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibModuloInstThreaded<Int64ITy>(out0, in0, Divisor, SignFollowDivisor, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibModuloInst<Int64ITy>(LibTensor* out0, LibTensor* in0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibModuloInst<Int32ITy>(LibTensor* out0, LibTensor* in0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibModuloInstThreaded<Int64ITy>(LibTensor* out0, LibTensor* in0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibModuloInstThreaded<Int32ITy>(LibTensor* out0, LibTensor* in0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
