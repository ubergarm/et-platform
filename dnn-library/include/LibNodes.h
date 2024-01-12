#ifndef LIB_NODES_H
#define LIB_NODES_H

// clang-format off

// File automatically generated with:
//  ./libManager.py --swplatform-root /home/nivard/sw-platform/ --excel libManager.xlsx --cacheState cacheState.xlsx
//  cwd=/local/home/nivard/sw-platform/host-software/dnnLibrary/scripts

// Manual changes will be detected by CI



#include "ETSOCGenericOpInst.h"
#include "LibTensor.h"
#include "inlining.h"

namespace dnn_lib {
static constexpr size_t default_kernels_size = 2;
static constexpr size_t default_pads_size = 4;
static constexpr size_t default_mask_size = max_tensor_dimensions;
static constexpr size_t default_axes_size = max_tensor_dimensions;
static constexpr size_t default_rszscale_size = max_tensor_dimensions;
static constexpr size_t default_dilation_size = 2;
static constexpr size_t default_fusedActivationArgs = 10;

/****************************************************************************
*  AdaptiveAvgPool implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibAdaptiveAvgPoolInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibAdaptiveAvgPoolInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAdaptiveAvgPoolInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAdaptiveAvgPoolInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAdaptiveAvgPoolInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAdaptiveAvgPoolInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAdaptiveAvgPoolInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ArgMax implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibArgMaxInst(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibArgMaxInst<Int64ITy,FloatTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibArgMaxInst<Int32ITy,FloatTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibArgMaxInst<Int64ITy,Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibArgMaxInst<Int32ITy,Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibArgMaxInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibArgMaxInst<Int64ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibArgMaxInst<Int64ITy,Int32QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibArgMaxInst<Int32ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibArgMaxInst<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibArgMaxInst<Int32ITy,Int32QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  AvgPool implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibAvgPoolInst(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const bool CountIncludePads, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type>
void fwdLibAvgPoolInstThreaded(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const bool CountIncludePads, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibAvgPoolInst<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const bool CountIncludePads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const bool CountIncludePads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const bool CountIncludePads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInstThreaded<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const bool CountIncludePads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInstThreaded<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const bool CountIncludePads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInstThreaded<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const bool CountIncludePads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  BatchedAdd implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibBatchedAddInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibBatchedAddInst<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedAddInst<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedAddInst<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedAddInst<Int8QTy,Int8QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedAddInst<Int16QTy,Int16QTy,Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedAddInst<Int16QTy,Int16QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  BatchedReduceAdd implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibBatchedReduceAddInst(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibBatchedReduceAddInst<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  BatchedReduceMin implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibBatchedReduceMinInst(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibBatchedReduceMinInst<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceMinInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceMinInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceMinInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceMinInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceMinInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  BatchOneHot implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibBatchOneHotInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibBatchOneHotInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchOneHotInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchOneHotInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchOneHotInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchOneHotInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchOneHotInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ConvertTo implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibConvertToInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibConvertToInstVectorized(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibConvertToInst<Float16Ty,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<FloatTy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Int32ITy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Int32ITy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Int32ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Int64ITy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Int64ITy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Int64ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<FloatTy,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Float16Ty,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Int64ITy,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Int32ITy,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<BoolTy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<BoolTy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<BoolTy,Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<BoolTy,Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<BoolTy,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<Float16Ty,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<FloatTy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<Int32ITy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<Int32ITy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<Int32ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<Int64ITy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<Int64ITy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<Int64ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<FloatTy,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<Float16Ty,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<Int64ITy,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<Int32ITy,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<BoolTy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<BoolTy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<BoolTy,Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<BoolTy,Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInstVectorized<BoolTy,BoolTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  Int8Converter implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibInt8ConverterInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibInt8ConverterInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInt8ConverterInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ChannelWiseQuantizedConvolution implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in2Type>
void fwdLibChannelWiseQuantizedConvolutionInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, LibTensor* in6, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibChannelWiseQuantizedConvolutionInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, LibTensor* in6, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibChannelWiseQuantizedConvolutionInst<Int8QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, LibTensor* in6, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  Convolution implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in2Type>
void fwdLibConvolutionInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibConvolutionInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolutionInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolutionInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolutionInst<Int8QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolutionInst<Int16QTy,Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolutionInst<Int16QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const size_t FusedActivation, const std::array<float, default_fusedActivationArgs> & FusedActivationArgs, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  Convolution3D implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in2Type>
void fwdLibConvolution3DInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibConvolution3DInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolution3DInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolution3DInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolution3DInst<Int8QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolution3DInst<Int16QTy,Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolution3DInst<Int16QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ConvTranspose implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibConvTransposeInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibConvTransposeInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Group, const std::array<uint32_t, default_dilation_size> & Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  Copy implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibCopyInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type>
void fwdLibCopyInstTensorized(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibCopyInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInstTensorized<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInstTensorized<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInstTensorized<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInstTensorized<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInstTensorized<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInstTensorized<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  CrossEntropyLoss implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibCrossEntropyLossInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibCrossEntropyLossInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCrossEntropyLossInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  CumSum implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibCumSumInst(LibTensor* out0, LibTensor* in0, const bool Exclusive, const bool Reverse, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibCumSumInst<FloatTy>(LibTensor* out0, LibTensor* in0, const bool Exclusive, const bool Reverse, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCumSumInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const bool Exclusive, const bool Reverse, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCumSumInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const bool Exclusive, const bool Reverse, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCumSumInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const bool Exclusive, const bool Reverse, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  Dequantize implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibDequantizeInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibDequantizeInst<FloatTy,Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst<FloatTy,UInt8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst<FloatTy,Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst<FloatTy,Int32QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst<Float16Ty,Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst<Float16Ty,UInt8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst<Float16Ty,Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst<Float16Ty,Int32QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementAdd implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibElementAddInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementAddInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementAnd implementations
****************************************************************************/
// declarations

void fwdLibElementAndInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

void fwdLibElementAndInstThreaded(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations

/****************************************************************************
*  ElementCmpEQ implementations
****************************************************************************/
// declarations
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibElementCmpEQInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibElementCmpEQInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementCmpEQInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInstVectorized<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInstVectorized<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInstVectorized<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInstVectorized<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInstVectorized<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementCmpNEQ implementations
****************************************************************************/
// declarations
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibElementCmpNEQInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibElementCmpNEQInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementCmpNEQInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpNEQInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpNEQInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpNEQInst<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpNEQInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpNEQInstVectorized<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpNEQInstVectorized<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpNEQInstVectorized<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpNEQInstVectorized<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpNEQInstVectorized<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementCmpLTE implementations
****************************************************************************/
// declarations
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibElementCmpLTEInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibElementCmpLTEInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementCmpLTEInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInstVectorized<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInstVectorized<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInstVectorized<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInstVectorized<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInstVectorized<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementCmpLT implementations
****************************************************************************/
// declarations
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibElementCmpLTInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibElementCmpLTInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementCmpLTInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInstVectorized<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInstVectorized<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInstVectorized<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInstVectorized<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInstVectorized<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementCos implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibElementCosInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementCosInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCosInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementDiv implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibElementDivInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementDivInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementDivInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementDivInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementDivInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementErf implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibElementErfInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementErfInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementErfInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementExp implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibElementExpInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementExpInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementExpInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementIsNaN implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibElementIsNaNInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementIsNaNInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementIsNaNInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementLog implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibElementLogInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementLogInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementLogInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementMax implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibElementMaxInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementMaxInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementMin implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibElementMinInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementMinInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementMul implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibElementMulInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementMulInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementNeg implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibElementNegInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementNegInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementNegInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementNot implementations
****************************************************************************/
// declarations

void fwdLibElementNotInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations

/****************************************************************************
*  ElementOr implementations
****************************************************************************/
// declarations

void fwdLibElementOrInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

void fwdLibElementOrInstThreaded(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations

/****************************************************************************
*  ElementPow implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibElementPowInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementPowInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementPowInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementSelect implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibElementSelectInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementSelectInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSelectInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSelectInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSelectInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSelectInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSelectInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementSin implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibElementSinInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementSinInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSinInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementSub implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibElementSubInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementSubInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ElementXor implementations
****************************************************************************/
// declarations

void fwdLibElementXorInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

void fwdLibElementXorInstThreaded(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations

/****************************************************************************
*  EmbeddingBag implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type, ElemKind in2Type>
void fwdLibEmbeddingBagInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in2Type>
void fwdLibEmbeddingBagInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in2Type>
void fwdLibEmbeddingBagInstFastpath(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibEmbeddingBagInst<FloatTy,FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInst<Float16Ty,Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInst<FloatTy,Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInst<Float16Ty,Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInst<FloatTy,FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInst<Float16Ty,Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInst<FloatTy,Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInst<Float16Ty,Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInstVectorized<FloatTy,FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInstVectorized<Float16Ty,Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInstVectorized<FloatTy,Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInstVectorized<Float16Ty,Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInstVectorized<FloatTy,FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInstVectorized<Float16Ty,Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInstVectorized<FloatTy,Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInstVectorized<Float16Ty,Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInstFastpath<FloatTy,FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInstFastpath<Float16Ty,Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInstFastpath<FloatTy,Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInstFastpath<Float16Ty,Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInstFastpath<FloatTy,FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInstFastpath<Float16Ty,Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInstFastpath<FloatTy,Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibEmbeddingBagInstFastpath<Float16Ty,Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const bool HasEndOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  MaxSplat implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibMaxSplatInst(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibMaxSplatInstAligned32Bytes(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibMaxSplatInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInstAligned32Bytes<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInstAligned32Bytes<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInstAligned32Bytes<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInstAligned32Bytes<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInstAligned32Bytes<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInstAligned32Bytes<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ExtractTensor implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibExtractTensorInst(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibExtractTensorInst<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibExtractTensorInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibExtractTensorInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibExtractTensorInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibExtractTensorInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibExtractTensorInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  Flip implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibFlipInst(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibFlipInst<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFlipInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFlipInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFlipInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFlipInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFlipInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFlipInst<Int32QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  FullyConnected implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in2Type>
void fwdLibFullyConnectedInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibFullyConnectedInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFullyConnectedInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFullyConnectedInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFullyConnectedInst<Int8QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFullyConnectedInst<Int8QTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFullyConnectedInst<Int16QTy,Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFullyConnectedInst<Int16QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  FusedRowwiseQuantizedSparseLengthsWeightedSum implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  FusedRowwiseQuantizedSparseLengthsSum implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  Gather implementations
****************************************************************************/
// declarations
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibGatherInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibGatherInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<UInt8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<Int16QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<UInt8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<Int16QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  GatherRanges implementations
****************************************************************************/
// declarations
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibGatherRangesInst(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibGatherRangesInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<UInt8QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<Int16QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<UInt8QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<Int16QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  InsertTensor implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibInsertTensorInst(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibInsertTensorInstThreaded(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibInsertTensorInst<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInstThreaded<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInstThreaded<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInstThreaded<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInstThreaded<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInstThreaded<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInstThreaded<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  IntLookupTable implementations
****************************************************************************/
// declarations

void fwdLibIntLookupTableInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations

/****************************************************************************
*  LengthsRangeFill implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibLengthsRangeFillInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibLengthsRangeFillInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  LengthsSum implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibLengthsSumInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibLengthsSumInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsSumInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsSumInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsSumInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsSumInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsSumInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  LengthsToRanges implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibLengthsToRangesInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibLengthsToRangesInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  LocalResponseNormalization implementations
****************************************************************************/
// declarations
template <ElemKind out1Type>
void fwdLibLocalResponseNormalizationInst(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out1Type>
void fwdLibLocalResponseNormalizationInstVectorized(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibLocalResponseNormalizationInst<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLocalResponseNormalizationInst<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLocalResponseNormalizationInstVectorized<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLocalResponseNormalizationInstVectorized<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t HalfWindowSize, const float Alpha, const float Beta, const float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  MatMul implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibMatMulInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibMatMulInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  MaxPool implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibMaxPoolInst(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibMaxPoolInst<FloatTy, FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst<Float16Ty, Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst<UInt8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst<Int8QTy,UInt8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst<UInt8QTy,UInt8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  MaxPoolWithArgMax implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibMaxPoolWithArgMaxInst(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibMaxPoolWithArgMaxInst<FloatTy, FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolWithArgMaxInst<Float16Ty, Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolWithArgMaxInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolWithArgMaxInst<UInt8QTy,Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolWithArgMaxInst<Int8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolWithArgMaxInst<UInt8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolWithArgMaxInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  Modulo implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibModuloInst(LibTensor* out0, LibTensor* in0, const uint64_t Divisor, const bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibModuloInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t Divisor, const bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibModuloInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t Divisor, const bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  NonMaxSuppression implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibNonMaxSuppressionInst(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t CenterPointBox, const uint64_t MaxOutputBoxesPerClass, const float IouThreshold, const float ScoreThreshold, const bool IsTFVersion4, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibNonMaxSuppressionInst<Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t CenterPointBox, const uint64_t MaxOutputBoxesPerClass, const float IouThreshold, const float ScoreThreshold, const bool IsTFVersion4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibNonMaxSuppressionInst<Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t CenterPointBox, const uint64_t MaxOutputBoxesPerClass, const float IouThreshold, const float ScoreThreshold, const bool IsTFVersion4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  Profile implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibProfileInst(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibProfileInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibProfileInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibProfileInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  Quantize implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibQuantizeInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibQuantizeInst<Int8QTy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibQuantizeInst<UInt8QTy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibQuantizeInst<Int16QTy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibQuantizeInst<Int32QTy,FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibQuantizeInst<Int8QTy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibQuantizeInst<UInt8QTy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibQuantizeInst<Int16QTy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibQuantizeInst<Int32QTy,Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  RescaleQuantized implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibRescaleQuantizedInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibRescaleQuantizedInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRescaleQuantizedInst<UInt8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRescaleQuantizedInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRescaleQuantizedInst<Int32QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ResizeBilinear implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibResizeBilinearInst(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibResizeBilinearInst<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibResizeBilinearInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibResizeBilinearInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibResizeBilinearInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibResizeBilinearInst<Int32QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibResizeBilinearInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibResizeBilinearInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ResizeNearest implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibResizeNearestInst(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibResizeNearestInst<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibResizeNearestInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibResizeNearestInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibResizeNearestInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibResizeNearestInst<Int32QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibResizeNearestInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibResizeNearestInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  RowwiseQuantizedFullyConnected implementations
****************************************************************************/
// declarations

void fwdLibRowwiseQuantizedFullyConnectedInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

void fwdLibRowwiseQuantizedFullyConnectedInstAligned32Bytes(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations

/****************************************************************************
*  RowwiseQuantizedSparseLengthsWeightedSum implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in4Type>
void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in4Type>
void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ScatterData implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibScatterDataInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibScatterDataInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibScatterDataInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibScatterDataInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibScatterDataInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibScatterDataInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibScatterDataInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  Sigmoid implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibSigmoidInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibSigmoidInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSigmoidInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  SoftMax implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibSoftMaxInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibSoftMaxInstVectorized(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibSoftMaxInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst<BFloat16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInstVectorized<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInstVectorized<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInstVectorized<BFloat16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  SpaceToDepth implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibSpaceToDepthInst(LibTensor* out0, LibTensor* in0, const uint32_t BlockSize, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibSpaceToDepthInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint32_t BlockSize, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSpaceToDepthInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint32_t BlockSize, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSpaceToDepthInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint32_t BlockSize, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSpaceToDepthInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint32_t BlockSize, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  SparseLengthsSum implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in1Type>
void fwdLibSparseLengthsSumInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in1Type>
void fwdLibSparseLengthsSumInstThreaded(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibSparseLengthsSumInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsSumInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsSumInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsSumInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsSumInst<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsSumInst<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsSumInstThreaded<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsSumInstThreaded<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsSumInstThreaded<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsSumInstThreaded<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsSumInstThreaded<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsSumInstThreaded<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  SparseLengthsWeightedSum implementations
****************************************************************************/
// declarations
template <ElemKind in0Type, ElemKind in2Type>
void fwdLibSparseLengthsWeightedSumInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type, ElemKind in2Type>
void fwdLibSparseLengthsWeightedSumInstThreaded(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibSparseLengthsWeightedSumInst<FloatTy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst<Float16Ty, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst<Int8QTy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst<Int64ITy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst<Int32ITy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst<Int16QTy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInstThreaded<FloatTy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInstThreaded<Float16Ty, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInstThreaded<Int8QTy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInstThreaded<Int64ITy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInstThreaded<Int32ITy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInstThreaded<Int16QTy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInstThreaded<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInstThreaded<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInstThreaded<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  SparseToDense implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibSparseToDenseInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibSparseToDenseInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  SparseToDenseMask implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibSparseToDenseMaskInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibSparseToDenseMaskInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseMaskInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseMaskInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseMaskInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseMaskInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseMaskInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  Splat implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibSplatInst(LibTensor* out0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibSplatInst<FloatTy>(LibTensor* out0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst<Float16Ty>(LibTensor* out0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst<Int8QTy>(LibTensor* out0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst<Int64ITy>(LibTensor* out0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst<Int32ITy>(LibTensor* out0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst<Int16QTy>(LibTensor* out0, const uint64_t ValueBits, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  Syncopy implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibSyncopyInst(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibSyncopyInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSyncopyInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSyncopyInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSyncopyInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSyncopyInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSyncopyInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  Tanh implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibTanhInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibTanhInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTanhInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  TensorView implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibTensorViewInst(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibTensorViewInst<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  TopK implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibTopKInst(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibTopKInst<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst<Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst<Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst<Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst<Int16QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  Transpose implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibTransposeInst(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibTransposeInstAligned32Bytes(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibTransposeInst<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInstAligned32Bytes<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInstAligned32Bytes<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInstAligned32Bytes<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInstAligned32Bytes<Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInstAligned32Bytes<Int32ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInstAligned32Bytes<Int16QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  Trilu implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibTriluInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Upper, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibTriluInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Upper, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTriluInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Upper, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTriluInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Upper, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTriluInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Upper, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTriluInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Upper, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTriluInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const bool Upper, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
*  ETSOCGenericOp implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibETSOCGenericOpInst(LibTensor* out0, LibTensor* in0, const uint32_t GenericOperation, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibETSOCGenericOpInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint32_t GenericOperation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // namespace dnn_lib

#endif // LIB_NODES_H

