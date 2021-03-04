
#include "LibNodes.h"

namespace dnn_lib {
////////////////////////////////////////////////////////////////////////////////
// Forward call to corresponding dnn_lib::inlining implementations
////////////////////////////////////////////////////////////////////////////////

template <ElemKind in0Type>
void fwdLibSoftMaxInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset,
                       const uint32_t assignedMinions) {
  dnn_lib::inlining::fwdLibSoftMaxInst<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
}

////////////////////////////////////////////////////////////////////////////////
// Template specializations (declared with 'extern template' in LibNodes.h)
////////////////////////////////////////////////////////////////////////////////

template <ElemKind in0Type>
void fwdLibSoftMaxInstVectorized(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset,
                                 const uint32_t assignedMinions) {
  dnn_lib::inlining::fwdLibSoftMaxInstVectorized<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
}

////////////////////////////////////////////////////////////////////////////////
// Template specializations (declared with 'extern template' in LibNodes.h)
////////////////////////////////////////////////////////////////////////////////
template void fwdLibSoftMaxInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags,
                                         const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags,
                                           const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst<BFloat16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags,
                                            const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInstVectorized<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags,
                                                   const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInstVectorized<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags,
                                                     const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInstVectorized<BFloat16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags,
                                                      const uint32_t minionOffset, const uint32_t assignedMinions);
} // namespace dnn_lib
