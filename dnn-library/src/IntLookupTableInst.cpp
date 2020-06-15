
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <>
  void fwdLibIntLookupTableInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibIntLookupTableInst<>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <>
  void fwdLibIntLookupTableInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibIntLookupTableInst"Threaded"<>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibIntLookupTableInst<>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibIntLookupTableInst"Threaded"<>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
