
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementMinInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementMinInst<FloatTy,FloatTy,FloatTy>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementMinInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementMinInst"Threaded"<FloatTy,FloatTy,FloatTy>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementMinInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementMinInst"Vectorized"<FloatTy,FloatTy,FloatTy>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementMinInst<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst"Threaded"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst"Threaded"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst"Threaded"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst"Threaded"<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst"Threaded"<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst"Vectorized"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst"Vectorized"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst"Vectorized"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst"Vectorized"<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst"Vectorized"<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
