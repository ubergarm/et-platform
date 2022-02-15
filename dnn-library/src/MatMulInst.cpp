
#include "LibNodes.h"

namespace dnn_lib {
////////////////////////////////////////////////////////////////////////////////
// Forward call to corresponding dnn_lib::inlining implementations
////////////////////////////////////////////////////////////////////////////////

template <ElemKind in0Type>
void fwdLibMatMulInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags,
                      const uint32_t minionOffset, const uint32_t assignedMinions) {
  dnn_lib::inlining::fwdLibMatMulInst<in0Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
}

////////////////////////////////////////////////////////////////////////////////
// Template specializations (declared with 'extern template' in LibNodes.h)
////////////////////////////////////////////////////////////////////////////////
template void fwdLibMatMulInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags,
                                        const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags,
                                          const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags,
                                        const uint32_t minionOffset, const uint32_t assignedMinions);

// FIXME. [SW-11008] at the moment there is no support for those data types  a compile-time error willl happen if used
#if 0
template void fwdLibMatMulInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags,
                                         const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags,
                                         const uint32_t minionOffset, const uint32_t assignedMinions);
#endif
template void fwdLibMatMulInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags,
                                         const uint32_t minionOffset, const uint32_t assignedMinions);
} // namespace dnn_lib