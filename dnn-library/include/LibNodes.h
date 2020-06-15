// File automatically generated with:
//  ./libManager.py --glow-dir ../../esperanto-glow/ --excel libManager.xlsx --hostsw-dir ../../
//  cwd=/home/sebastia/Esperanto/clean2/dnn_lib/scripts

#ifndef LIBNODES_H_
#define LIBNODES_H_

#include "LibTensor.h"

namespace dnn_lib {
static constexpr size_t default_kernels_size = 2;
static constexpr size_t default_mask_size = max_tensor_dimensions;

/****************************************************************************
/* AdaptiveAvgPool implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibAdaptiveAvgPoolInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibAdaptiveAvgPoolInst<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAdaptiveAvgPoolInst<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAdaptiveAvgPoolInst<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAdaptiveAvgPoolInst<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAdaptiveAvgPoolInst<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAdaptiveAvgPoolInst<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ArgMax implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibArgMaxInst(LibTensor* out0, LibTensor* out0, dim_t Axis, bool KeepDims, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibArgMaxInst<FloatTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibArgMaxInst<Float16Ty>(LibTensor* out0, LibTensor* out0, dim_t Axis, bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibArgMaxInst<Int8QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibArgMaxInst<Int64ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibArgMaxInst<Int32ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibArgMaxInst<Int16QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* AvgPool implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibAvgPoolInst(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibAvgPoolInst"Threaded"(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibAvgPoolInst<FloatTy, FloatTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInst<Float16Ty, Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInst<UInt8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInst<Int8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInst<UInt8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInst<Int64Ty,Int64Ty>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInst"Threaded"<FloatTy, FloatTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInst"Threaded"<Float16Ty, Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInst"Threaded"<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInst"Threaded"<UInt8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInst"Threaded"<Int8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInst"Threaded"<UInt8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibAvgPoolInst"Threaded"<Int64Ty,Int64Ty>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* BatchedAdd implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibBatchedAddInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibBatchedAddInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibBatchedAddInst<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedAddInst<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedAddInst<Int8Qty,Int8Qty,Int8Qty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedAddInst<Int8Qty,Int8Qty,Int32QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedAddInst<Int16QTy,Int16QTy,Int16Qty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedAddInst<Int16QTy,Int16QTy,Int32QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedAddInst"Threaded"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedAddInst"Threaded"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedAddInst"Threaded"<Int8Qty,Int8Qty,Int8Qty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedAddInst"Threaded"<Int8Qty,Int8Qty,Int32QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedAddInst"Threaded"<Int16QTy,Int16QTy,Int16Qty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedAddInst"Threaded"<Int16QTy,Int16QTy,Int32QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* BatchedReduceAdd implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibBatchedReduceAddInst(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibBatchedReduceAddInst"Threaded"(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibBatchedReduceAddInst"Int8"(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibBatchedReduceAddInst"Int8Threaded"(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibBatchedReduceAddInst<FloatTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst<Float16Ty>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst<Int8QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst<Int64ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst<Int32ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst<Int16QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Int8"<FloatTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Int8"<Float16Ty>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Int8"<Int8QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Int8"<Int64ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Int8"<Int32ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Int8"<Int16QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Int8Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Int8Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Int8Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Int8Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Int8Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchedReduceAddInst"Int8Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* BatchOneHot implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibBatchOneHotInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type>
void fwdLibBatchOneHotInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibBatchOneHotInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchOneHotInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchOneHotInst<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchOneHotInst<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchOneHotInst<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchOneHotInst<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchOneHotInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchOneHotInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchOneHotInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchOneHotInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchOneHotInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibBatchOneHotInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* Checksum implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibChecksumInst(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibChecksumInst<FloatTy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibChecksumInst<Float16Ty>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibChecksumInst<Int8QTy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibChecksumInst<Int64ITy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibChecksumInst<Int32ITy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibChecksumInst<Int16QTy>(LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ConvertTo implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibConvertToInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibConvertToInst"Threaded"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibConvertToInst"Vectorized"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibConvertToInst<Float16Ty,FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<FloatTy,Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Int32ITy,Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Int32ITy,FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Int32ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Int64ITy,Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Int64ITy,FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst<Int64ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Threaded"<Float16Ty,FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Threaded"<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Threaded"<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Threaded"<FloatTy,Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Threaded"<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Threaded"<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Threaded"<Int32ITy,Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Threaded"<Int32ITy,FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Threaded"<Int32ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Threaded"<Int64ITy,Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Threaded"<Int64ITy,FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Threaded"<Int64ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Vectorized"<Float16Ty,FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Vectorized"<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Vectorized"<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Vectorized"<FloatTy,Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Vectorized"<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Vectorized"<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Vectorized"<Int32ITy,Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Vectorized"<Int32ITy,FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Vectorized"<Int32ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Vectorized"<Int64ITy,Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Vectorized"<Int64ITy,FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvertToInst"Vectorized"<Int64ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* Int8Converter implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibInt8ConverterInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibInt8ConverterInst<>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* Convolution implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibConvolutionInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibConvolutionInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibConvolutionInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibConvolutionInst<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolutionInst<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolutionInst<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolutionInst<Int16QTy,Int16QTy,Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolutionInst"Threaded"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolutionInst"Threaded"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolutionInst"Threaded"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolutionInst"Threaded"<Int16QTy,Int16QTy,Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolutionInst"Vectorized"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolutionInst"Vectorized"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolutionInst"Vectorized"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolutionInst"Vectorized"<Int16QTy,Int16QTy,Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* Convolution3D implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibConvolution3DInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type>
void fwdLibConvolution3DInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibConvolution3DInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolution3DInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolution3DInst<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolution3DInst<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolution3DInst<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolution3DInst<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolution3DInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolution3DInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolution3DInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolution3DInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolution3DInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibConvolution3DInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* Copy implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibCopyInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type>
void fwdLibCopyInst"Threaded"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type>
void fwdLibCopyInst"Vectorized"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type>
void fwdLibCopyInst"Tensorized"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibCopyInst<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Vectorized"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Vectorized"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Vectorized"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Vectorized"<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Vectorized"<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Vectorized"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Tensorized"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Tensorized"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Tensorized"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Tensorized"<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Tensorized"<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCopyInst"Tensorized"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* CrossEntropyLoss implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibCrossEntropyLossInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibCrossEntropyLossInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibCrossEntropyLossInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCrossEntropyLossInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCrossEntropyLossInst<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCrossEntropyLossInst<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCrossEntropyLossInst<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCrossEntropyLossInst<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCrossEntropyLossInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCrossEntropyLossInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCrossEntropyLossInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCrossEntropyLossInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCrossEntropyLossInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibCrossEntropyLossInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* Dequantize implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibDequantizeInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibDequantizeInst"Threaded"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibDequantizeInst<FloatTy,Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst<FloatTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst<FloatTy,Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst<FloatTy,Int32QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst<Float16Ty,Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst<Float16Ty,UInt8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst<Float16Ty,Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst<Float16Ty,Int32QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst"Threaded"<FloatTy,Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst"Threaded"<FloatTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst"Threaded"<FloatTy,Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst"Threaded"<FloatTy,Int32QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst"Threaded"<Float16Ty,Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst"Threaded"<Float16Ty,UInt8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst"Threaded"<Float16Ty,Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibDequantizeInst"Threaded"<Float16Ty,Int32QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ElementAdd implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementAddInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementAddInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementAddInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementAddInst<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst"Threaded"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst"Threaded"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst"Threaded"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst"Threaded"<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst"Threaded"<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst"Vectorized"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst"Vectorized"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst"Vectorized"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst"Vectorized"<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementAddInst"Vectorized"<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ElementCmpEQ implementations
****************************************************************************/
// declarations
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibElementCmpEQInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibElementCmpEQInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibElementCmpEQInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementCmpEQInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst"Threaded"<FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst"Threaded"<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst"Threaded"<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst"Threaded"<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst"Threaded"<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst"Vectorized"<FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst"Vectorized"<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst"Vectorized"<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst"Vectorized"<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpEQInst"Vectorized"<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ElementCmpLTE implementations
****************************************************************************/
// declarations
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibElementCmpLTEInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibElementCmpLTEInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibElementCmpLTEInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementCmpLTEInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst"Threaded"<FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst"Threaded"<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst"Threaded"<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst"Threaded"<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst"Threaded"<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst"Vectorized"<FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst"Vectorized"<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst"Vectorized"<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst"Vectorized"<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTEInst"Vectorized"<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ElementCmpLT implementations
****************************************************************************/
// declarations
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibElementCmpLTInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibElementCmpLTInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibElementCmpLTInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementCmpLTInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst"Threaded"<FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst"Threaded"<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst"Threaded"<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst"Threaded"<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst"Threaded"<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst"Vectorized"<FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst"Vectorized"<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst"Vectorized"<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst"Vectorized"<Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementCmpLTInst"Vectorized"<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ElementDiv implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementDivInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementDivInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementDivInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementDivInst<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementDivInst<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementDivInst<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementDivInst<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementDivInst"Threaded"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementDivInst"Threaded"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementDivInst"Threaded"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementDivInst"Threaded"<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementDivInst"Vectorized"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementDivInst"Vectorized"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementDivInst"Vectorized"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementDivInst"Vectorized"<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ElementExp implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibElementExpInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementExpInst<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementExpInst<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ElementIsNaN implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibElementIsNaNInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibElementIsNaNInst"Threaded"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementIsNaNInst<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementIsNaNInst<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementIsNaNInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementIsNaNInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ElementLog implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibElementLogInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibElementLogInst"Threaded"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibElementLogInst"Vectorized"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementLogInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementLogInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementLogInst"Threaded"<FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementLogInst"Threaded"<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementLogInst"Vectorized"<FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementLogInst"Vectorized"<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ElementMax implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementMaxInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementMaxInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementMaxInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementMaxInst<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst"Threaded"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst"Threaded"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst"Threaded"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst"Threaded"<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst"Threaded"<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst"Vectorized"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst"Vectorized"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst"Vectorized"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst"Vectorized"<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMaxInst"Vectorized"<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ElementMin implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementMinInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementMinInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementMinInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementMinInst<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst"Threaded"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst"Threaded"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst"Threaded"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst"Threaded"<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst"Threaded"<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst"Vectorized"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst"Vectorized"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst"Vectorized"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst"Vectorized"<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMinInst"Vectorized"<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ElementMul implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementMulInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementMulInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementMulInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementMulInst<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst"Threaded"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst"Threaded"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst"Threaded"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst"Threaded"<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst"Threaded"<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst"Vectorized"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst"Vectorized"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst"Vectorized"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst"Vectorized"<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementMulInst"Vectorized"<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ElementPow implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementPowInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementPowInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementPowInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementPowInst<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementPowInst<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementPowInst"Threaded"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementPowInst"Threaded"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementPowInst"Vectorized"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementPowInst"Vectorized"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ElementSelect implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibElementSelectInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibElementSelectInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementSelectInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSelectInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSelectInst<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSelectInst<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSelectInst<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSelectInst<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSelectInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSelectInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSelectInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSelectInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSelectInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSelectInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ElementSub implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementSubInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementSubInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibElementSubInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibElementSubInst<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst"Threaded"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst"Threaded"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst"Threaded"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst"Threaded"<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst"Threaded"<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst"Vectorized"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst"Vectorized"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst"Vectorized"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst"Vectorized"<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibElementSubInst"Vectorized"<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* EmbeddingBag implementations
****************************************************************************/
// declarations
template <>
void fwdLibEmbeddingBagInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibEmbeddingBagInst<>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* MaxSplat implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibMaxSplatInst(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibMaxSplatInst"Threaded"(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibMaxSplatInst"Vectorized"(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibMaxSplatInst"Aligned32Bytes"(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibMaxSplatInst<FloatTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst<Float16Ty>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst<Int8QTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst<Int64ITy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst<Int32ITy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst<Int16QTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Vectorized"<FloatTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Vectorized"<Float16Ty>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Vectorized"<Int8QTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Vectorized"<Int64ITy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Vectorized"<Int32ITy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Vectorized"<Int16QTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Aligned32Bytes"<FloatTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Aligned32Bytes"<Float16Ty>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Aligned32Bytes"<Int8QTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Aligned32Bytes"<Int64ITy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Aligned32Bytes"<Int32ITy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxSplatInst"Aligned32Bytes"<Int16QTy>(LibTensor* out0, LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ExtractTensor implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibExtractTensorInst(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibExtractTensorInst"Threaded"(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibExtractTensorInst<FloatTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibExtractTensorInst<Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibExtractTensorInst<Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibExtractTensorInst<Int64ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibExtractTensorInst<Int32ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibExtractTensorInst<Int16QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibExtractTensorInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibExtractTensorInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibExtractTensorInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibExtractTensorInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibExtractTensorInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibExtractTensorInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* Flip implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibFlipInst(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibFlipInst<FloatTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFlipInst<Float16Ty>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFlipInst<Int8QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFlipInst<Int64ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFlipInst<Int32ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFlipInst<Int16QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* FullyConnected implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibFullyConnectedInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibFullyConnectedInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
void fwdLibFullyConnectedInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibFullyConnectedInst<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFullyConnectedInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFullyConnectedInst"Vectorized"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* FusedRowwiseQuantizedSparseLengthsWeightedSum implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type>
void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTy"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type>
void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyThreaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type>
void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyVectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTy"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTy"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyThreaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyThreaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyVectorized"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyVectorized"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyVectorized"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyVectorized"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* FusedRowwiseQuantizedSparseLengthsSum implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type>
void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type>
void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst"Vectorized"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst"Vectorized"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* Gather implementations
****************************************************************************/
// declarations
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibGatherInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibGatherInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibGatherInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<UInt8QTy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<Int16QTy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<UInt8QTy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst<Int16QTy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst"Threaded"<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst"Threaded"<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst"Threaded"<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst"Threaded"<UInt8QTy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst"Threaded"<Int16QTy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst"Threaded"<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst"Threaded"<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst"Threaded"<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst"Threaded"<UInt8QTy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherInst"Threaded"<Int16QTy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* GatherRanges implementations
****************************************************************************/
// declarations
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibGatherRangesInst(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type, ElemKind in1Type>
void fwdLibGatherRangesInst"Threaded"(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibGatherRangesInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<UInt8QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<Int16QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<UInt8QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst<Int16QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst"Threaded"<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst"Threaded"<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst"Threaded"<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst"Threaded"<UInt8QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst"Threaded"<Int16QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst"Threaded"<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst"Threaded"<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst"Threaded"<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst"Threaded"<UInt8QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibGatherRangesInst"Threaded"<Int16QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* InsertTensor implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibInsertTensorInst(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibInsertTensorInst"Threaded"(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibInsertTensorInst<FloatTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInst<Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInst<Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInst<Int64ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInst<Int32ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInst<Int16QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibInsertTensorInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* IntLookupTable implementations
****************************************************************************/
// declarations
template <>
void fwdLibIntLookupTableInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <>
void fwdLibIntLookupTableInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibIntLookupTableInst<>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibIntLookupTableInst"Threaded"<>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* LengthsSum implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibLengthsSumInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibLengthsSumInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibLengthsSumInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsSumInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsSumInst<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsSumInst<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsSumInst<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsSumInst<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsSumInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsSumInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsSumInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsSumInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsSumInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsSumInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* LengthsToRanges implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibLengthsToRangesInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibLengthsToRangesInst"Threaded"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibLengthsToRangesInst<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsToRangesInst<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsToRangesInst<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsToRangesInst<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsToRangesInst<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsToRangesInst<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsToRangesInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsToRangesInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsToRangesInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsToRangesInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsToRangesInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLengthsToRangesInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* LocalResponseNormalization implementations
****************************************************************************/
// declarations
template <ElemKind out1Type>
void fwdLibLocalResponseNormalizationInst(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out1Type>
void fwdLibLocalResponseNormalizationInst"Threaded"(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out1Type>
void fwdLibLocalResponseNormalizationInst"Vectorized"(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibLocalResponseNormalizationInst<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLocalResponseNormalizationInst<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLocalResponseNormalizationInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLocalResponseNormalizationInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLocalResponseNormalizationInst"Vectorized"<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibLocalResponseNormalizationInst"Vectorized"<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* out0, uint32_t HalfWindowSize, float Alpha, float Beta, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* MatMul implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibMatMulInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibMatMulInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibMatMulInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibMatMulInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst"Vectorized"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst"Vectorized"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst"Vectorized"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst"Vectorized"<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst"Vectorized"<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMatMulInst"Vectorized"<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* MaxPool implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibMaxPoolInst(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibMaxPoolInst"Threaded"(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibMaxPoolInst<FloatTy, FloatTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst<Float16Ty, Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst<UInt8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst<Int8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst<UInt8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst<Int64Ty,Int64Ty>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst"Threaded"<FloatTy, FloatTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst"Threaded"<Float16Ty, Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst"Threaded"<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst"Threaded"<UInt8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst"Threaded"<Int8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst"Threaded"<UInt8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibMaxPoolInst"Threaded"<Int64Ty,Int64Ty>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* Modulo implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibModuloInst(LibTensor* out0, LibTensor* out0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibModuloInst"Threaded"(LibTensor* out0, LibTensor* out0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibModuloInst<Int64ITy>(LibTensor* out0, LibTensor* out0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibModuloInst<Int32ITy>(LibTensor* out0, LibTensor* out0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibModuloInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibModuloInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, uint64_t Divisor, bool SignFollowDivisor, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* Quantize implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibQuantizeInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type>
void fwdLibQuantizeInst"Threaded"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibQuantizeInst<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibQuantizeInst<UInt8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibQuantizeInst<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibQuantizeInst<Int32QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibQuantizeInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibQuantizeInst"Threaded"<UInt8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibQuantizeInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibQuantizeInst"Threaded"<Int32QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* RescaleQuantized implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibRescaleQuantizedInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibRescaleQuantizedInst"Threaded"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibRescaleQuantizedInst<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRescaleQuantizedInst<UInt8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRescaleQuantizedInst<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRescaleQuantizedInst<Int32QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRescaleQuantizedInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRescaleQuantizedInst"Threaded"<UInt8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRescaleQuantizedInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRescaleQuantizedInst"Threaded"<Int32QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* RowwiseQuantizedFullyConnected implementations
****************************************************************************/
// declarations
template <>
void fwdLibRowwiseQuantizedFullyConnectedInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <>
void fwdLibRowwiseQuantizedFullyConnectedInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <>
void fwdLibRowwiseQuantizedFullyConnectedInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <>
void fwdLibRowwiseQuantizedFullyConnectedInst"Aligned32Bytes"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibRowwiseQuantizedFullyConnectedInst<>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRowwiseQuantizedFullyConnectedInst"Threaded"<>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRowwiseQuantizedFullyConnectedInst"Vectorized"<>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRowwiseQuantizedFullyConnectedInst"Aligned32Bytes"<>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* RowwiseQuantizedSparseLengthsWeightedSum implementations
****************************************************************************/
// declarations
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type, ElemKind in0Type>
void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<FloatTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<Float16Ty,UInt8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst"Threaded"<FloatTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst"Threaded"<Float16Ty,UInt8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst"Vectorized"<FloatTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst"Vectorized"<Float16Ty,UInt8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* ScatterData implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibScatterDataInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibScatterDataInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibScatterDataInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibScatterDataInst<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibScatterDataInst<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibScatterDataInst<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibScatterDataInst<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* Sigmoid implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibSigmoidInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibSigmoidInst"Threaded"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibSigmoidInst<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSigmoidInst<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSigmoidInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSigmoidInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* SoftMax implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibSoftMaxInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibSoftMaxInst"Threaded"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibSoftMaxInst"Vectorized"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibSoftMaxInst"Threaded1"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibSoftMaxInst"Vectorized1"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibSoftMaxInst"2"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibSoftMaxInst"Threaded2"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibSoftMaxInst<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Vectorized"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Vectorized"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Vectorized"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Vectorized"<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Vectorized"<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Vectorized"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded1"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded1"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded1"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded1"<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded1"<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded1"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Vectorized1"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Vectorized1"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Vectorized1"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Vectorized1"<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Vectorized1"<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Vectorized1"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"2"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"2"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"2"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"2"<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"2"<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"2"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded2"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded2"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded2"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded2"<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded2"<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSoftMaxInst"Threaded2"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* SparseLengthsWeightedSum implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibSparseLengthsWeightedSumInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibSparseLengthsWeightedSumInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibSparseLengthsWeightedSumInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseLengthsWeightedSumInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* SparseToDense implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibSparseToDenseInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibSparseToDenseInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibSparseToDenseInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibSparseToDenseInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst"Vectorized"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst"Vectorized"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst"Vectorized"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst"Vectorized"<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst"Vectorized"<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseInst"Vectorized"<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* SparseToDenseMask implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibSparseToDenseMaskInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibSparseToDenseMaskInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibSparseToDenseMaskInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseMaskInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseMaskInst<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseMaskInst<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseMaskInst<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseMaskInst<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseMaskInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseMaskInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseMaskInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseMaskInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseMaskInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSparseToDenseMaskInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, std::array<uint64_t, default_mask_size> Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* Splat implementations
****************************************************************************/
// declarations
template <ElemKind out0Type>
void fwdLibSplatInst(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type>
void fwdLibSplatInst"Threaded"(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind out0Type>
void fwdLibSplatInst"Vectorized"(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibSplatInst<FloatTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst<Float16Ty>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst<Int8QTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst<Int64ITy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst<Int32ITy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst<Int16QTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst"Threaded"<FloatTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst"Threaded"<Float16Ty>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst"Threaded"<Int8QTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst"Threaded"<Int64ITy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst"Threaded"<Int32ITy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst"Threaded"<Int16QTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst"Vectorized"<FloatTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst"Vectorized"<Float16Ty>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst"Vectorized"<Int8QTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst"Vectorized"<Int64ITy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst"Vectorized"<Int32ITy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSplatInst"Vectorized"<Int16QTy>(LibTensor* out0, float Value, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* Syncopy implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibSyncopyInst(LibTensor* out0, LibTensor* out0, uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibSyncopyInst<FloatTy>(LibTensor* out0, LibTensor* out0, uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSyncopyInst<Float16Ty>(LibTensor* out0, LibTensor* out0, uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSyncopyInst<Int8QTy>(LibTensor* out0, LibTensor* out0, uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSyncopyInst<Int64ITy>(LibTensor* out0, LibTensor* out0, uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSyncopyInst<Int32ITy>(LibTensor* out0, LibTensor* out0, uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibSyncopyInst<Int16QTy>(LibTensor* out0, LibTensor* out0, uint32_t SyncOffset, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* Tanh implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibTanhInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibTanhInst"Threaded"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibTanhInst<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTanhInst<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTanhInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTanhInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* TensorView implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibTensorViewInst(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibTensorViewInst"Threaded"(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibTensorViewInst"Vectorized"(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibTensorViewInst<FloatTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst<Float16Ty>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst<Int8QTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst<Int64ITy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst<Int32ITy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst<Int16QTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst"Threaded"<FloatTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst"Threaded"<Float16Ty>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst"Threaded"<Int8QTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst"Threaded"<Int64ITy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst"Threaded"<Int32ITy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst"Threaded"<Int16QTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst"Vectorized"<FloatTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst"Vectorized"<Float16Ty>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst"Vectorized"<Int8QTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst"Vectorized"<Int64ITy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst"Vectorized"<Int32ITy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTensorViewInst"Vectorized"<Int16QTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* TopK implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibTopKInst(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibTopKInst"Threaded_all"(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibTopKInst"Threaded_k4"(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibTopKInst"Threaded_k8"(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibTopKInst<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst<Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst<Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst<Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst<Int16QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_all"<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_all"<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_all"<Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_all"<Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_all"<Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_all"<Int16QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k4"<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k4"<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k4"<Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k4"<Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k4"<Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k4"<Int16QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k4"<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k4"<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k4"<Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k4"<Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k4"<Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k4"<Int16QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k8"<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k8"<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k8"<Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k8"<Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k8"<Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTopKInst"Threaded_k8"<Int16QTy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, float K, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

/****************************************************************************
/* Transpose implementations
****************************************************************************/
// declarations
template <ElemKind in0Type>
void fwdLibTransposeInst(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibTransposeInst"Threaded"(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibTransposeInst"Vectorized"(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
template <ElemKind in0Type>
void fwdLibTransposeInst"Aligned32Bytes"(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);

// extern template declarations
extern template void fwdLibTransposeInst<FloatTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst<Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst<Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst<Int64ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst<Int32ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst<Int16QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Vectorized"<FloatTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Vectorized"<Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Vectorized"<Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Vectorized"<Int64ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Vectorized"<Int32ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Vectorized"<Int16QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Aligned32Bytes"<FloatTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Aligned32Bytes"<Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Aligned32Bytes"<Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Aligned32Bytes"<Int64ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Aligned32Bytes"<Int32ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
extern template void fwdLibTransposeInst"Aligned32Bytes"<Int16QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // namespace dnn_lib

#endif /* LIBNODES_H_ */
