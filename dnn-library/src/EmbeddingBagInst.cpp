
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibEmbeddingBagInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibEmbeddingBagInst<out0Type>(out0, in0, in1, in2, in3, HasEndOffset, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibEmbeddingBagInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibEmbeddingBagInstVectorized<out0Type>(out0, in0, in1, in2, in3, HasEndOffset, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibEmbeddingBagInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstVectorized<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstVectorized<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
