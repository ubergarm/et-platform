
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibSoftMaxInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSoftMaxInst<FloatTy>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibSoftMaxInst"Threaded"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSoftMaxInst"Threaded"<FloatTy>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibSoftMaxInst"Vectorized"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSoftMaxInst"Vectorized"<FloatTy>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibSoftMaxInst"Threaded1"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSoftMaxInst"Threaded1"<FloatTy>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibSoftMaxInst"Vectorized1"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSoftMaxInst"Vectorized1"<FloatTy>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibSoftMaxInst"2"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSoftMaxInst"2"<FloatTy>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibSoftMaxInst"Threaded2"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSoftMaxInst"Threaded2"<FloatTy>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSoftMaxInst<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Vectorized"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Vectorized"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Vectorized"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Vectorized"<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Vectorized"<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Vectorized"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded1"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded1"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded1"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded1"<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded1"<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded1"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Vectorized1"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Vectorized1"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Vectorized1"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Vectorized1"<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Vectorized1"<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Vectorized1"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"2"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"2"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"2"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"2"<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"2"<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"2"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded2"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded2"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded2"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded2"<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded2"<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst"Threaded2"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
