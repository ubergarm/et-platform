
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibNonMaxSuppressionInst(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const int64_t CenterPointBox, const int64_t MaxOutputBoxesPerClass, const float IouThreshold, const float ScoreThreshold, const bool IsTFVersion4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibNonMaxSuppressionInst<out0Type>(out0, out1, in0, in1, CenterPointBox, MaxOutputBoxesPerClass, IouThreshold, ScoreThreshold, IsTFVersion4, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibNonMaxSuppressionInst<Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const int64_t CenterPointBox, const int64_t MaxOutputBoxesPerClass, const float IouThreshold, const float ScoreThreshold, const bool IsTFVersion4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibNonMaxSuppressionInst<Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const int64_t CenterPointBox, const int64_t MaxOutputBoxesPerClass, const float IouThreshold, const float ScoreThreshold, const bool IsTFVersion4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
