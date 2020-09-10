
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out1Type>
  void fwdLibLocalResponseNormalizationInst(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibLocalResponseNormalizationInst<out1Type>(out0, out1, in0, HalfWindowSize, Alpha, Beta, K, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out1Type>
  void fwdLibLocalResponseNormalizationInstVectorized(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibLocalResponseNormalizationInstVectorized<out1Type>(out0, out1, in0, HalfWindowSize, Alpha, Beta, K, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibLocalResponseNormalizationInst<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLocalResponseNormalizationInst<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLocalResponseNormalizationInstVectorized<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLocalResponseNormalizationInstVectorized<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
