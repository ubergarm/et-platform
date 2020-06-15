
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out1Type>
  void fwdLibLocalResponseNormalizationInst(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibLocalResponseNormalizationInst<FloatTy>(out0, out1, in0, HalfWindowSize, Alpha, Beta, K, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out1Type>
  void fwdLibLocalResponseNormalizationInst"Threaded"(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibLocalResponseNormalizationInst"Threaded"<FloatTy>(out0, out1, in0, HalfWindowSize, Alpha, Beta, K, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out1Type>
  void fwdLibLocalResponseNormalizationInst"Vectorized"(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibLocalResponseNormalizationInst"Vectorized"<FloatTy>(out0, out1, in0, HalfWindowSize, Alpha, Beta, K, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibLocalResponseNormalizationInst<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLocalResponseNormalizationInst<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLocalResponseNormalizationInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLocalResponseNormalizationInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLocalResponseNormalizationInst"Vectorized"<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLocalResponseNormalizationInst"Vectorized"<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
