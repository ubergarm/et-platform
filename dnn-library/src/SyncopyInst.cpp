
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibSyncopyInst(LibTensor* out0, LibTensor* in0, uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSyncopyInst<FloatTy>(out0, in0, SyncOffset, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSyncopyInst<FloatTy>(LibTensor* out0, LibTensor* in0, uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSyncopyInst<Float16Ty>(LibTensor* out0, LibTensor* in0, uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSyncopyInst<Int8QTy>(LibTensor* out0, LibTensor* in0, uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSyncopyInst<Int64ITy>(LibTensor* out0, LibTensor* in0, uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSyncopyInst<Int32ITy>(LibTensor* out0, LibTensor* in0, uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSyncopyInst<Int16QTy>(LibTensor* out0, LibTensor* in0, uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
