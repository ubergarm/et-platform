
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <>
  void fwdLibRowwiseQuantizedFullyConnectedInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInst<>(out0, in0, in1, in2, in3, in4, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <>
  void fwdLibRowwiseQuantizedFullyConnectedInstThreaded(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInstThreaded<>(out0, in0, in1, in2, in3, in4, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <>
  void fwdLibRowwiseQuantizedFullyConnectedInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInstVectorized<>(out0, in0, in1, in2, in3, in4, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <>
  void fwdLibRowwiseQuantizedFullyConnectedInstAligned32Bytes(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInstAligned32Bytes<>(out0, in0, in1, in2, in3, in4, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibRowwiseQuantizedFullyConnectedInst<>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedFullyConnectedInstThreaded<>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedFullyConnectedInstVectorized<>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedFullyConnectedInstAligned32Bytes<>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
