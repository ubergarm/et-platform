
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibModuloInst(LibTensor* out0, LibTensor* out0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibModuloInst<Int64ITy>(out0, in0, Divisor, SignFollowDivisor, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibModuloInst"Threaded"(LibTensor* out0, LibTensor* out0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibModuloInst"Threaded"<Int64ITy>(out0, in0, Divisor, SignFollowDivisor, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibModuloInst<Int64ITy>(LibTensor* out0, LibTensor* out0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibModuloInst<Int32ITy>(LibTensor* out0, LibTensor* out0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibModuloInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibModuloInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
