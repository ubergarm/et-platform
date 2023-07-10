// clang-format off

// File automatically generated with:
//  ./libManager.py --swplatform-root ../../.. --excel libManager.xlsx --cacheState cacheState.xlsx
//  cwd=/home/clesmes/sw-platform/host-software/dnnLibrary/scripts

// Manual changes will be detected by CI


#include "LibNodes.h"

 namespace dnn_lib { 
  // 
  // Section: AdaptiveAvgPool
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibAdaptiveAvgPoolInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibAdaptiveAvgPoolInst<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibAdaptiveAvgPoolInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAdaptiveAvgPoolInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAdaptiveAvgPoolInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAdaptiveAvgPoolInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAdaptiveAvgPoolInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAdaptiveAvgPoolInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ArgMax
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibArgMaxInst(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibArgMaxInst<out0Type, in0Type>(out0, in0, Axis, KeepDims, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibArgMaxInst<Int64ITy,FloatTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibArgMaxInst<Int32ITy,FloatTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibArgMaxInst<Int64ITy,Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibArgMaxInst<Int32ITy,Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibArgMaxInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibArgMaxInst<Int64ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibArgMaxInst<Int64ITy,Int32QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibArgMaxInst<Int32ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibArgMaxInst<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibArgMaxInst<Int32ITy,Int32QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: AvgPool
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibAvgPoolInst(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const bool CountIncludePads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibAvgPoolInst<out0Type>(out0, in0, Kernels, Strides, Pads, Layout, CountIncludePads, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibAvgPoolInstThreaded(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const bool CountIncludePads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibAvgPoolInstThreaded<out0Type>(out0, in0, Kernels, Strides, Pads, Layout, CountIncludePads, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibAvgPoolInst<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const bool CountIncludePads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const bool CountIncludePads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const bool CountIncludePads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInstThreaded<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const bool CountIncludePads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInstThreaded<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const bool CountIncludePads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInstThreaded<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const bool CountIncludePads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: BatchedAdd
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibBatchedAddInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchedAddInst<out0Type, in0Type, in1Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibBatchedAddInst<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedAddInst<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedAddInst<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedAddInst<Int8QTy,Int8QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedAddInst<Int16QTy,Int16QTy,Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedAddInst<Int16QTy,Int16QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: BatchedReduceAdd
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibBatchedReduceAddInst(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchedReduceAddInst<in0Type>(out0, in0, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibBatchedReduceAddInst<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: BatchedReduceMin
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibBatchedReduceMinInst(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchedReduceMinInst<in0Type>(out0, in0, Axes, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibBatchedReduceMinInst<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: BatchOneHot
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibBatchOneHotInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchOneHotInst<out0Type>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibBatchOneHotInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchOneHotInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchOneHotInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchOneHotInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchOneHotInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchOneHotInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ConvertTo
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibConvertToInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibConvertToInst<out0Type, in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibConvertToInstVectorized(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibConvertToInstVectorized<out0Type, in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibConvertToInst<Float16Ty,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<FloatTy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<Int32ITy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<Int32ITy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<Int32ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<Int64ITy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<Int64ITy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<Int64ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<FloatTy,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<Float16Ty,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<Int64ITy,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<Int32ITy,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<BoolTy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<BoolTy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<BoolTy,Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<BoolTy,Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInst<BoolTy,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<Float16Ty,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<FloatTy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<Int32ITy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<Int32ITy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<Int32ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<Int64ITy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<Int64ITy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<Int64ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<FloatTy,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<Float16Ty,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<Int64ITy,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<Int32ITy,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<BoolTy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<BoolTy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<BoolTy,Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<BoolTy,Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvertToInstVectorized<BoolTy,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: Int8Converter
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibInt8ConverterInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibInt8ConverterInst<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibInt8ConverterInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInt8ConverterInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ChannelWiseQuantizedConvolution
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in2Type>
  void fwdLibChannelWiseQuantizedConvolutionInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, LibTensor* in6, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibChannelWiseQuantizedConvolutionInst<out0Type, in2Type>(out0, in0, in1, in2, in3, in4, in5, in6, Kernels, Strides, Pads, Group, Dilation, FusedActivation, FusedActivationArgs, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibChannelWiseQuantizedConvolutionInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, LibTensor* in6, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibChannelWiseQuantizedConvolutionInst<Int8QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, LibTensor* in6, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: Convolution
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in2Type>
  void fwdLibConvolutionInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibConvolutionInst<out0Type, in2Type>(out0, in0, in1, in2, Kernels, Strides, Pads, Group, Dilation, FusedActivation, FusedActivationArgs, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibConvolutionInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst<Int8QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst<Int16QTy,Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst<Int16QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: Convolution3D
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in2Type>
  void fwdLibConvolution3DInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibConvolution3DInst<out0Type, in2Type>(out0, in0, in1, in2, Kernels, Strides, Pads, Group, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibConvolution3DInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Int8QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Int16QTy,Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Int16QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ConvTranspose
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibConvTransposeInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibConvTransposeInst<out0Type>(out0, in0, in1, in2, Kernels, Strides, Pads, Group, Dilation, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibConvTransposeInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: Copy
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibCopyInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibCopyInst<out0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibCopyInstTensorized(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibCopyInstTensorized<out0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibCopyInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstTensorized<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstTensorized<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstTensorized<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstTensorized<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstTensorized<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCopyInstTensorized<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: CrossEntropyLoss
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibCrossEntropyLossInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibCrossEntropyLossInst<in0Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibCrossEntropyLossInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCrossEntropyLossInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: CumSum
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibCumSumInst(LibTensor* out0, LibTensor* in0, const bool Exclusive, const bool Reverse, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibCumSumInst<in0Type>(out0, in0, Exclusive, Reverse, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibCumSumInst<FloatTy>(LibTensor* out0, LibTensor* in0, const bool Exclusive, const bool Reverse, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCumSumInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const bool Exclusive, const bool Reverse, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCumSumInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const bool Exclusive, const bool Reverse, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCumSumInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const bool Exclusive, const bool Reverse, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: Dequantize
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibDequantizeInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibDequantizeInst<out0Type, in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibDequantizeInst<FloatTy,Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibDequantizeInst<FloatTy,UInt8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibDequantizeInst<FloatTy,Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibDequantizeInst<FloatTy,Int32QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibDequantizeInst<Float16Ty,Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibDequantizeInst<Float16Ty,UInt8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibDequantizeInst<Float16Ty,Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibDequantizeInst<Float16Ty,Int32QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementAdd
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibElementAddInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementAddInst<out0Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementAddInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementCmpEQ
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementCmpEQInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementCmpEQInst<in0Type, in1Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementCmpEQInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementCmpEQInstVectorized<in0Type, in1Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementCmpEQInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpEQInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpEQInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpEQInst<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpEQInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpEQInstVectorized<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpEQInstVectorized<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpEQInstVectorized<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpEQInstVectorized<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpEQInstVectorized<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementCmpNEQ
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementCmpNEQInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementCmpNEQInst<in0Type, in1Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementCmpNEQInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementCmpNEQInstVectorized<in0Type, in1Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementCmpNEQInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpNEQInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpNEQInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpNEQInst<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpNEQInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpNEQInstVectorized<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpNEQInstVectorized<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpNEQInstVectorized<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpNEQInstVectorized<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpNEQInstVectorized<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementCmpLTE
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementCmpLTEInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementCmpLTEInst<in0Type, in1Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementCmpLTEInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementCmpLTEInstVectorized<in0Type, in1Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementCmpLTEInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTEInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTEInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTEInst<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTEInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTEInstVectorized<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTEInstVectorized<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTEInstVectorized<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTEInstVectorized<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTEInstVectorized<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementCmpLT
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementCmpLTInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementCmpLTInst<in0Type, in1Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementCmpLTInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementCmpLTInstVectorized<in0Type, in1Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementCmpLTInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTInst<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTInstVectorized<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTInstVectorized<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTInstVectorized<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTInstVectorized<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCmpLTInstVectorized<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementCos
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibElementCosInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementCosInst<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementCosInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementCosInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementDiv
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibElementDivInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementDivInst<out0Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementDivInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementDivInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementDivInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementDivInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementErf
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibElementErfInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementErfInst<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementErfInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementErfInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementExp
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibElementExpInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementExpInst<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementExpInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementExpInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementIsNaN
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibElementIsNaNInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementIsNaNInst<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementIsNaNInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementIsNaNInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementLog
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibElementLogInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementLogInst<out0Type, in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementLogInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementLogInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementMax
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibElementMaxInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementMaxInst<out0Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementMaxInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMaxInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMaxInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMaxInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMaxInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementMin
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibElementMinInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementMinInst<out0Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementMinInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMinInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementMul
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibElementMulInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementMulInst<out0Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementMulInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementNeg
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibElementNegInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementNegInst<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementNegInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementNegInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementNot
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  
  void fwdLibElementNotInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementNotInst(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  // 
  // Section: ElementPow
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibElementPowInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementPowInst<out0Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementPowInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementPowInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementSelect
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibElementSelectInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementSelectInst<out0Type>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementSelectInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementSelectInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementSelectInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementSelectInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementSelectInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementSelectInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementSin
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibElementSinInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementSinInst<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementSinInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementSinInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ElementSub
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibElementSubInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementSubInst<out0Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementSubInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementSubInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementSubInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementSubInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementSubInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: EmbeddingBag
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type, ElemKind in2Type>
  void fwdLibEmbeddingBagInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibEmbeddingBagInst<out0Type, in0Type, in2Type>(out0, in0, in1, in2, in3, HasEndOffset, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type, ElemKind in2Type>
  void fwdLibEmbeddingBagInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibEmbeddingBagInstVectorized<out0Type, in0Type, in2Type>(out0, in0, in1, in2, in3, HasEndOffset, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type, ElemKind in2Type>
  void fwdLibEmbeddingBagInstFastpath(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibEmbeddingBagInstFastpath<out0Type, in0Type, in2Type>(out0, in0, in1, in2, in3, HasEndOffset, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibEmbeddingBagInst<FloatTy,FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInst<Float16Ty,Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInst<FloatTy,Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInst<Float16Ty,Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInst<FloatTy,FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInst<Float16Ty,Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInst<FloatTy,Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInst<Float16Ty,Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstVectorized<FloatTy,FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstVectorized<Float16Ty,Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstVectorized<FloatTy,Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstVectorized<Float16Ty,Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstVectorized<FloatTy,FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstVectorized<Float16Ty,Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstVectorized<FloatTy,Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstVectorized<Float16Ty,Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstFastpath<FloatTy,FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstFastpath<Float16Ty,Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstFastpath<FloatTy,Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstFastpath<Float16Ty,Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstFastpath<FloatTy,FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstFastpath<Float16Ty,Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstFastpath<FloatTy,Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibEmbeddingBagInstFastpath<Float16Ty,Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: MaxSplat
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibMaxSplatInst(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMaxSplatInst<in0Type>(out0, in0, ValueBits, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibMaxSplatInstAligned32Bytes(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMaxSplatInstAligned32Bytes<in0Type>(out0, in0, ValueBits, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibMaxSplatInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInstAligned32Bytes<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInstAligned32Bytes<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInstAligned32Bytes<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInstAligned32Bytes<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInstAligned32Bytes<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxSplatInstAligned32Bytes<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ExtractTensor
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibExtractTensorInst(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibExtractTensorInst<in0Type>(out0, in0, Offsets, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibExtractTensorInst<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: Flip
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibFlipInst(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFlipInst<in0Type>(out0, in0, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibFlipInst<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFlipInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFlipInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFlipInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFlipInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFlipInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFlipInst<Int32QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: FullyConnected
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in2Type>
  void fwdLibFullyConnectedInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFullyConnectedInst<out0Type, in2Type>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibFullyConnectedInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFullyConnectedInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFullyConnectedInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFullyConnectedInst<Int8QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFullyConnectedInst<Int8QTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFullyConnectedInst<Int16QTy,Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFullyConnectedInst<Int16QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: FusedRowwiseQuantizedSparseLengthsWeightedSum
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst<out0Type>(out0, in0, in1, in2, in3, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: FusedRowwiseQuantizedSparseLengthsSum
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsSumInst<out0Type>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: Gather
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type, ElemKind in1Type>
  void fwdLibGatherInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibGatherInst<in0Type, in1Type>(out0, in0, in1, BatchDims, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibGatherInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<UInt8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Int16QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<UInt8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Int16QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: GatherRanges
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type, ElemKind in1Type>
  void fwdLibGatherRangesInst(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibGatherRangesInst<in0Type, in1Type>(out0, out1, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibGatherRangesInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<UInt8QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Int16QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<UInt8QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Int16QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: InsertTensor
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibInsertTensorInst(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibInsertTensorInst<in0Type>(out0, in0, Offsets, Count, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibInsertTensorInstThreaded(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibInsertTensorInstThreaded<in0Type>(out0, in0, Offsets, Count, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibInsertTensorInst<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInstThreaded<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInstThreaded<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInstThreaded<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInstThreaded<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInstThreaded<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInstThreaded<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: IntLookupTable
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  
  void fwdLibIntLookupTableInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibIntLookupTableInst(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  // 
  // Section: LengthsRangeFill
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibLengthsRangeFillInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibLengthsRangeFillInst<out0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibLengthsRangeFillInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: LengthsSum
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibLengthsSumInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibLengthsSumInst<in0Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibLengthsSumInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsSumInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsSumInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsSumInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsSumInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsSumInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: LengthsToRanges
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibLengthsToRangesInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibLengthsToRangesInst<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibLengthsToRangesInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: LocalResponseNormalization
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out1Type>
  void fwdLibLocalResponseNormalizationInst(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibLocalResponseNormalizationInst<out1Type>(out0, out1, in0, HalfWindowSize, Alpha, Beta, K, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out1Type>
  void fwdLibLocalResponseNormalizationInstVectorized(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibLocalResponseNormalizationInstVectorized<out1Type>(out0, out1, in0, HalfWindowSize, Alpha, Beta, K, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibLocalResponseNormalizationInst<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLocalResponseNormalizationInst<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLocalResponseNormalizationInstVectorized<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLocalResponseNormalizationInstVectorized<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: MatMul
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibMatMulInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMatMulInst<in0Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibMatMulInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMatMulInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: MaxPool
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibMaxPoolInst(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMaxPoolInst<out0Type, in0Type>(out0, in0, Kernels, Strides, Pads, Layout, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibMaxPoolInst<FloatTy, FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<Float16Ty, Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<UInt8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<Int8QTy,UInt8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<UInt8QTy,UInt8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: MaxPoolWithArgMax
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibMaxPoolWithArgMaxInst(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMaxPoolWithArgMaxInst<out0Type, in0Type>(out0, out1, in0, Kernels, Strides, Pads, Layout, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibMaxPoolWithArgMaxInst<FloatTy, FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolWithArgMaxInst<Float16Ty, Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolWithArgMaxInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolWithArgMaxInst<UInt8QTy,Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolWithArgMaxInst<Int8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolWithArgMaxInst<UInt8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolWithArgMaxInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: Modulo
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibModuloInst(LibTensor* out0, LibTensor* in0, const uint64_t Divisor, const bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibModuloInst<in0Type>(out0, in0, Divisor, SignFollowDivisor, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibModuloInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t Divisor, const bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibModuloInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t Divisor, const bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: NonMaxSuppression
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibNonMaxSuppressionInst(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t CenterPointBox, const uint64_t MaxOutputBoxesPerClass, const float IouThreshold, const float ScoreThreshold, const bool IsTFVersion4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibNonMaxSuppressionInst<out0Type>(out0, out1, in0, in1, CenterPointBox, MaxOutputBoxesPerClass, IouThreshold, ScoreThreshold, IsTFVersion4, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibNonMaxSuppressionInst<Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t CenterPointBox, const uint64_t MaxOutputBoxesPerClass, const float IouThreshold, const float ScoreThreshold, const bool IsTFVersion4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibNonMaxSuppressionInst<Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t CenterPointBox, const uint64_t MaxOutputBoxesPerClass, const float IouThreshold, const float ScoreThreshold, const bool IsTFVersion4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: Quantize
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibQuantizeInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibQuantizeInst<out0Type, in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibQuantizeInst<Int8QTy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibQuantizeInst<UInt8QTy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibQuantizeInst<Int16QTy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibQuantizeInst<Int32QTy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibQuantizeInst<Int8QTy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibQuantizeInst<UInt8QTy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibQuantizeInst<Int16QTy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibQuantizeInst<Int32QTy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: RescaleQuantized
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibRescaleQuantizedInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRescaleQuantizedInst<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibRescaleQuantizedInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRescaleQuantizedInst<UInt8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRescaleQuantizedInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRescaleQuantizedInst<Int32QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ResizeBilinear
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibResizeBilinearInst(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibResizeBilinearInst<out0Type>(out0, in0, RszScale, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibResizeBilinearInst<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeBilinearInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeBilinearInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeBilinearInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeBilinearInst<Int32QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeBilinearInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeBilinearInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ResizeNearest
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibResizeNearestInst(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibResizeNearestInst<out0Type>(out0, in0, RszScale, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibResizeNearestInst<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeNearestInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeNearestInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeNearestInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeNearestInst<Int32QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeNearestInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeNearestInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: RowwiseQuantizedFullyConnected
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  
  void fwdLibRowwiseQuantizedFullyConnectedInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInst(out0, in0, in1, in2, in3, in4, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  
  void fwdLibRowwiseQuantizedFullyConnectedInstAligned32Bytes(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInstAligned32Bytes(out0, in0, in1, in2, in3, in4, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  // 
  // Section: RowwiseQuantizedSparseLengthsWeightedSum
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in4Type>
  void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<out0Type, in4Type>(out0, in0, in1, in2, in3, in4, in5, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in4Type>
  void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized<out0Type, in4Type>(out0, in0, in1, in2, in3, in4, in5, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ScatterData
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibScatterDataInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibScatterDataInst<out0Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibScatterDataInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibScatterDataInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibScatterDataInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibScatterDataInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibScatterDataInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibScatterDataInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: Sigmoid
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibSigmoidInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSigmoidInst<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSigmoidInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSigmoidInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: SoftMax
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibSoftMaxInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSoftMaxInst<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibSoftMaxInstVectorized(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSoftMaxInstVectorized<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSoftMaxInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInst<BFloat16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInstVectorized<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInstVectorized<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSoftMaxInstVectorized<BFloat16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: SpaceToDepth
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibSpaceToDepthInst(LibTensor* out0, LibTensor* in0, const uint32_t BlockSize, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSpaceToDepthInst<out0Type>(out0, in0, BlockSize, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSpaceToDepthInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint32_t BlockSize, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSpaceToDepthInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint32_t BlockSize, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSpaceToDepthInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint32_t BlockSize, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSpaceToDepthInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint32_t BlockSize, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: SparseLengthsSum
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in1Type>
  void fwdLibSparseLengthsSumInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSparseLengthsSumInst<out0Type, in1Type>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in1Type>
  void fwdLibSparseLengthsSumInstThreaded(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSparseLengthsSumInstThreaded<out0Type, in1Type>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSparseLengthsSumInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsSumInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsSumInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsSumInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsSumInst<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsSumInst<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsSumInstThreaded<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsSumInstThreaded<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsSumInstThreaded<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsSumInstThreaded<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsSumInstThreaded<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsSumInstThreaded<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: SparseLengthsWeightedSum
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type, ElemKind in2Type>
  void fwdLibSparseLengthsWeightedSumInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSparseLengthsWeightedSumInst<in0Type, in2Type>(out0, in0, in1, in2, in3, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type, ElemKind in2Type>
  void fwdLibSparseLengthsWeightedSumInstThreaded(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSparseLengthsWeightedSumInstThreaded<in0Type, in2Type>(out0, in0, in1, in2, in3, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSparseLengthsWeightedSumInst<FloatTy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<Float16Ty, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<Int8QTy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<Int64ITy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<Int32ITy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<Int16QTy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInstThreaded<FloatTy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInstThreaded<Float16Ty, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInstThreaded<Int8QTy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInstThreaded<Int64ITy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInstThreaded<Int32ITy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInstThreaded<Int16QTy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInstThreaded<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInstThreaded<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInstThreaded<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: SparseToDense
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibSparseToDenseInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSparseToDenseInst<out0Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSparseToDenseInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: SparseToDenseMask
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibSparseToDenseMaskInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSparseToDenseMaskInst<out0Type>(out0, in0, in1, in2, in3, Mask, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSparseToDenseMaskInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: Splat
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibSplatInst(LibTensor* out0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSplatInst<out0Type>(out0, ValueBits, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSplatInst<FloatTy>(LibTensor* out0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst<Float16Ty>(LibTensor* out0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst<Int8QTy>(LibTensor* out0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst<Int64ITy>(LibTensor* out0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst<Int32ITy>(LibTensor* out0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSplatInst<Int16QTy>(LibTensor* out0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: Syncopy
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibSyncopyInst(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSyncopyInst<in0Type>(out0, in0, SyncOffset, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSyncopyInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSyncopyInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSyncopyInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSyncopyInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSyncopyInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSyncopyInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: Tanh
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibTanhInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTanhInst<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibTanhInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTanhInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: TensorView
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibTensorViewInst(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTensorViewInst<out0Type>(out0, in0, Offsets, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibTensorViewInst<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: TopK
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibTopKInst(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTopKInst<in0Type>(out0, out1, in0, TopK, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibTopKInst<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst<Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst<Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst<Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst<Int16QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: Transpose
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibTransposeInst(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTransposeInst<in0Type>(out0, in0, Shuffle, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibTransposeInstAligned32Bytes(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTransposeInstAligned32Bytes<in0Type>(out0, in0, Shuffle, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibTransposeInst<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<Int32ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<Int16QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: Trilu
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibTriluInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Upper, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTriluInst<in0Type>(out0, in0, in1, Upper, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibTriluInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Upper, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTriluInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Upper, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTriluInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Upper, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTriluInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Upper, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTriluInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Upper, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTriluInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Upper, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

  // 
  // Section: ETSOCGenericOp
  //

  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibETSOCGenericOpInst(LibTensor* out0, LibTensor* in0, const uint32_t GenericOperation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibETSOCGenericOpInst<out0Type>(out0, in0, GenericOperation, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibETSOCGenericOpInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint32_t GenericOperation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib 
