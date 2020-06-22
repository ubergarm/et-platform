
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibMatMulInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMatMulInst<in0Type>(out0, in0, in1, Transposed, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibMatMulInstThreaded(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMatMulInstThreaded<in0Type>(out0, in0, in1, Transposed, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibMatMulInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMatMulInstVectorized<in0Type>(out0, in0, in1, Transposed, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibMatMulInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInstThreaded<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInstThreaded<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInstThreaded<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInstThreaded<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInstThreaded<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInstThreaded<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInstVectorized<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInstVectorized<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInstVectorized<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInstVectorized<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInstVectorized<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInstVectorized<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Transposed, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
