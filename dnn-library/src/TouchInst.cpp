
#include "LibNodes.h"

namespace dnn_lib {
////////////////////////////////////////////////////////////////////////////////
// Forward call to corresponding dnn_lib::inlining implementations
////////////////////////////////////////////////////////////////////////////////

template <ElemKind out0Type>
void fwdLibTouchInst(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset,
                     const uint32_t assignedMinions) {
  dnn_lib::inlining::fwdLibTouchInst<out0Type>(out0, flags, minionOffset, assignedMinions);
}

////////////////////////////////////////////////////////////////////////////////
// Template specializations (declared with 'extern template' in LibNodes.h)
////////////////////////////////////////////////////////////////////////////////
template void fwdLibTouchInst<FloatTy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset,
                                       const uint32_t assignedMinions);
template void fwdLibTouchInst<Float16Ty>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset,
                                         const uint32_t assignedMinions);
template void fwdLibTouchInst<BFloat16Ty>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset,
                                          const uint32_t assignedMinions);
template void fwdLibTouchInst<Int8QTy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset,
                                       const uint32_t assignedMinions);
template void fwdLibTouchInst<Int32ITy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset,
                                        const uint32_t assignedMinions);
template void fwdLibTouchInst<Int64ITy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset,
                                        const uint32_t assignedMinions);
template void fwdLibTouchInst<BoolTy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset,
                                      const uint32_t assignedMinions);
template void fwdLibTouchInst<Int16QTy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset,
                                        const uint32_t assignedMinions);
template void fwdLibTouchInst<Int32QTy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset,
                                        const uint32_t assignedMinions);
template void fwdLibTouchInst<UInt8QTy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset,
                                        const uint32_t assignedMinions);
} // namespace dnn_lib
