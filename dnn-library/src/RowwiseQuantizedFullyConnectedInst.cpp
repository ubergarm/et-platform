
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <>
  void fwdLibRowwiseQuantizedFullyConnectedInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInst<>(out0, in0, in1, in2, in3, in4, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <>
  void fwdLibRowwiseQuantizedFullyConnectedInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInst"Threaded"<>(out0, in0, in1, in2, in3, in4, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <>
  void fwdLibRowwiseQuantizedFullyConnectedInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInst"Vectorized"<>(out0, in0, in1, in2, in3, in4, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <>
  void fwdLibRowwiseQuantizedFullyConnectedInst"Aligned32Bytes"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInst"Aligned32Bytes"<>(out0, in0, in1, in2, in3, in4, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibRowwiseQuantizedFullyConnectedInst<>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedFullyConnectedInst"Threaded"<>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedFullyConnectedInst"Vectorized"<>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedFullyConnectedInst"Aligned32Bytes"<>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
