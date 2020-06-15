
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibSparseToDenseMaskInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSparseToDenseMaskInst<FloatTy>(out0, in0, in1, in2, in3, Mask, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibSparseToDenseMaskInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSparseToDenseMaskInst"Threaded"<FloatTy>(out0, in0, in1, in2, in3, Mask, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSparseToDenseMaskInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
