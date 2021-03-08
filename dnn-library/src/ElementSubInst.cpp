
#include "LibNodes.h"

namespace dnn_lib {
////////////////////////////////////////////////////////////////////////////////
// Forward call to corresponding dnn_lib::inlining implementations
////////////////////////////////////////////////////////////////////////////////

template <ElemKind out0Type>
void fwdLibElementSubInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags,
                          const uint32_t minionOffset, const uint32_t assignedMinions) {
  dnn_lib::inlining::fwdLibElementSubInst<out0Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
}

////////////////////////////////////////////////////////////////////////////////
// Template specializations (declared with 'extern template' in LibNodes.h)
////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementSubInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags,
                                            const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementSubInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags,
                                              const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementSubInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags,
                                            const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementSubInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags,
                                             const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementSubInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags,
                                             const uint32_t minionOffset, const uint32_t assignedMinions);
} // namespace dnn_lib
