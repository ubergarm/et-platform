/*-------------------------------------------------------------------------
 * Copyright (C) 2019, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef LIB_TENSOR_H
#define LIB_TENSOR_H

#include <numeric>
#include <cassert>
#include <cstring>

#include "LibTypes.h"
#include "LibUtils.h"
#include "cacheops.h"

namespace dnn_lib {

 
struct Type final {
   
   /*@brief contains the dimensions (sizes) of the tensor.
    */
   const dim_array_t sizes_;

   /*@brief contains the strides for each dimension (in elements) same order
    * as in sizes_.
    */
   const dim_array_t strides_;
  
   /*@brief Specifies the element type of the tensor.
    */
   const dnn_lib::ElemKind elementType_{dnn_lib::ElemKind::Int64ITy};

   /*@brief contains the number of dimensions used by the tensor.
    */
   const dim_t numSizes_;

   /*@brief On quantized tensors, this represents the scale of the values.
    */
  const float scale_ {};

   /*@brief On quantized tensors, this represents the offset of the values.
    */
  const int32_t offset_ {};
 
  /*@brief Initialize a new quantized type with \p scale an \p offset.
   */
  template<size_t numSizes>
  constexpr Type(dnn_lib::ElemKind elk, const std::array<dim_t, numSizes> &dims,  float scale, int32_t offset) :
    sizes_(make_dims(dims)),
    elementType_(elk), numSizes_(numSizes),
    scale_(scale), offset_(offset)
  {
    assert( isQuantizedElemKind(elk));
  }
  
  /*@brief Initialize a new non-quantized type.
   */
  template<size_t numSizes>
  constexpr Type(dnn_lib::ElemKind elk, const std::array<dim_t, numSizes> &dims) :
    sizes_(make_dims(dims)),
    elementType_(elk), numSizes_(numSizes)
  {
    assert( !isQuantizedElemKind(elk));
  }
  
  /*@brief Initialize a new quantized type with \p scale an \p offset.
   */
  template<size_t numSizes>
  constexpr Type(dnn_lib::ElemKind elk,  const std::array<dim_t, numSizes> &dims, const std::array<dim_t, numSizes> &strides,
       float scale, int32_t offset) :
    sizes_(make_dims(dims)),
    strides_(make_strides(strides)),
    elementType_(elk),
    numSizes_(numSizes),
    scale_(scale), offset_(offset)
  {
    assert(isQuantizedElemKind(elk));
  } 
  
  /*@brief Initialize a new non-quantized type.
   */
  template<size_t numSizes>
  constexpr Type(dnn_lib::ElemKind elk,  const std::array<dim_t, numSizes> &dims, const std::array<dim_t, numSizes> &strides) :
    sizes_(make_dims(dims)),
    strides_(make_strides(strides)),
    elementType_(elk),
    numSizes_(numSizes)
  {
    assert(  !isQuantizedElemKind(elk));
  }

  /*@brief non templated version of the previous constructors (receiving dimensions/strides with max_tensor_dimensions,
    and an extra parameter to set the actual number of dimensions
   */
  /*@brief Initialize a new quantized type with \p scale an \p offset.
   */
  constexpr Type(dnn_lib::ElemKind elk, size_t numSizes, const dim_array_t &dims, const dim_array_t &strides,
       float scale, int32_t offset) :
    sizes_(dims),
    strides_(strides),
    elementType_(elk),
    numSizes_(numSizes),
    scale_(scale), offset_(offset)
  {
    assert(isQuantizedElemKind(elk));
  } 
  
  /*@brief Initialize a new non-quantized type.
   */
  constexpr Type(dnn_lib::ElemKind elk, size_t numSizes, const dim_array_t &dims, const dim_array_t &strides) :
    sizes_(dims),
    strides_(strides),
    elementType_(elk),
    numSizes_(numSizes)
  {
    assert(  !isQuantizedElemKind(elk));
  }

    
  /*@brief Reshape existing type this takes care of quantized types.
   */
  template<size_t numSizes>
  static Type newShape(const Type &T, const std::array<dim_t, numSizes> &dims) {
    if(T.isQuantizedType()) return Type(T.elementType_, dims, T.scale_, T.offset_);
    else return Type(T.elementType_, dims);
  }

  /*@brief Reshape existing type and change alignments.
    */
  template<size_t numSizes>
  static Type newShape(const Type &T, const std::array<dim_t, numSizes> &dims, const std::array<dim_t, numSizes> &pitches){
    if (T.isQuantizedType()) return Type(T.elementType_, dims, pitches, T.scale_, T.offset_);
    else return Type(T.elementType_, dims, pitches);
  }

  static Type newShape(const Type &T, size_t numSizes, const dim_array_t &dims, const dim_array_t &pitches){
    if (T.isQuantizedType()) return Type(T.elementType_, numSizes, dims, pitches, T.scale_, T.offset_);
    else return Type(T.elementType_, numSizes, dims, pitches);
  }

   /*@brief Reshape existing type by taking shapes and strides of \p shapeType.
    */
   static Type newShape(const Type &kindType, const Type shapeType) {
     //@TODO  T.getElementType() == shapeType->getelementSize() Size should be the same
     if (kindType.isQuantizedType())
       return Type(kindType.elementType_, shapeType.sizes_, shapeType.strides_, kindType.scale_, kindType.offset_);
     else 
       return Type(kindType.elementType_, shapeType.sizes_, shapeType.strides_);

     //TODO: the numSizes_is set wrong => because of dimension and strides extension. Either set properly (e.g. separate extended
     // and non extended arrays... or maybe just delete this newShape, in case it is not needed)
   }
   
  /* brief returns true if \p other is the same type.
   */
  // bool isEqual(TypeRef other) const { return isEqual(*other); }

  /*@brief returns the scale of a quantized type.
   */
  float getScale() const {
    //@TODO assert(isQuantizedType() && "Can't get the scale of non-quantized type");
    return scale_;
  }

  /*@brief returns the offset of quantized type.
   */
  int32_t getOffset() const {
    //@TODO assert(isQuantizedType() && "Can't get the offset of a non-quantized type");
    return offset_;
  }

   /*@brief returns the Tensor dimension.
    */
   unsigned char getNumDims() const { return numSizes_;}


   /*@brief returns the elemet type
    */
   ElemKind getElementType() const { return elementType_; }

   /*@brief return the number of elements in the tensor.
    */
   const dim_t size() const {
     dim_t acum = 1;
     for(auto i: sizes_) acum*=i;
     return acum;
   }
   
   // /*@brief Calculate the size of the slice starting at \p StartDim. Returns the
   //  *number of elements in a slice in the tensor.
   //  */
   // dim_t getSliceSize(unsigned char startDim) const {

   //   assert(startDim <= numSizes_ && " Invalid start dim");
   //   dim_t s = 1;
   //   for(unsigned char i = startDim; i < numSizes_; i++) {
   //     s *= dim_t(sizes_[i]);
   //   }
   //   return s;
   // }

   /*@brief returns true if the templated parameter \p Elemkind matches this type.
    */
   template<class ElemTy> bool isType() const {
     return isType<ElemTy>(elementType_);
   }

   /*@brief returns true if the templated parameter \p ElemKind matches the type
    *that's specified by the parameter \p Ty.
    */
   template<class ElemTy> static bool isType(dnn_lib::ElemKind elk) {
     switch (elk) {
     case dnn_lib::ElemKind::FloatTy:
       return std::is_same<ElemTy, float>::value;
     case dnn_lib::ElemKind::Float16Ty:
       return std::is_same<ElemTy, float16_t>::value;
     case dnn_lib::ElemKind::Int8QTy:
       return std::is_same<ElemTy, int8_t>::value;
     case dnn_lib::ElemKind::UInt8QTy:
       return std::is_same<ElemTy, uint8_t>::value;
     case dnn_lib::ElemKind::Int16QTy:
       return std::is_same<ElemTy, int16_t>::value;
     case dnn_lib::ElemKind::Int32QTy:
       return std::is_same<ElemTy, int32_t>::value;
     case dnn_lib::ElemKind::Int32ITy:
       return std::is_same<ElemTy, int32_t>::value;
     case dnn_lib::ElemKind::Int64ITy:
       return std::is_same<ElemTy, int64_t>::value;
     case dnn_lib::ElemKind::UInt8FusedQTy:
       return std::is_same<ElemTy, uint8_t>::value;
     case dnn_lib::ElemKind::UInt8FusedFP16QTy:
       return std::is_same<ElemTy, uint8_t>::value;
     case dnn_lib::ElemKind::UInt4FusedFP16QTy:
       return std::is_same<ElemTy, uint8_t>::value;
     case dnn_lib::ElemKind::BoolTy:
       return std::is_same<ElemTy, bool>::value;
     }
     assert(true && "Invalid type");
   }

   /*@brief true if the type of this Tensor is one of the quantized
    *types.
    */
   bool isQuantizedType() const { return isQuantizedElemKind(elementType_); }

   // /*@brief returns true if the type of this Tensor is one of the floating point
   //  *types.
   //  */
   // bool isFPType() const {
   //   return (getElementType() == dnn_lib::ElemKind::FloatTy ||
   //           getElementType() == dnn_lib::ElemKind::Float16Ty);
   // }

   /*@brief returns the size of the type element.
    */
   unsigned getElementSize() const { return getElementSize(elementType_); }

   /*@brief returns the size in bytes for this Tensor.
    */
   size_t getSizeInBytes() const {
     return sizes_[0] * strides_[0] * getElementSize();
   }

   /*@brief the actual number of elements in the tensor taking striding into
    * account. Since size() does not take striding into account, size() is
    * always <= actualSize()
    */
   size_t actualSize() const { return (sizes_[0] * strides_[0]); }

   /// \return the size of the element \p Ty.
   static unsigned getElementSize(dnn_lib::ElemKind Ty) {
     switch (Ty) {
     case dnn_lib::ElemKind::FloatTy:
       return sizeof(float);
     case dnn_lib::ElemKind::Float16Ty:
       return sizeof(uint16_t); 
     case dnn_lib::ElemKind::Int8QTy:
       return sizeof(int8_t);
     case dnn_lib::ElemKind::UInt8QTy:
       return sizeof(uint8_t);
     case dnn_lib::ElemKind::Int16QTy:
       return sizeof(int16_t);
     case dnn_lib::ElemKind::Int32QTy:
       return sizeof(int32_t);
     case dnn_lib::ElemKind::Int32ITy:
       return sizeof(int32_t);
     case dnn_lib::ElemKind::Int64ITy:
       return sizeof(int64_t);
     case dnn_lib::ElemKind::UInt8FusedQTy:
       return sizeof(uint8_t);
     case dnn_lib::ElemKind::UInt8FusedFP16QTy:
       return sizeof(uint8_t);
     case dnn_lib::ElemKind::UInt4FusedFP16QTy:
       return sizeof(uint8_t);
     case dnn_lib::ElemKind::BoolTy:
       return sizeof(bool);
     }
     assert(true && "Invalid type");
     return sizeof(bool);
   }


   // /// Given a string \p str containing the name of an ElemKind from
   // /// Type::getElementName, returns the corresponding ElemKind or Error if a
   // /// mapping couldn't be found.
   // static dnn_lib::ElemKind getElementKindFromName(string str) {
   //   if (str == Type::getElementName(dnn_lib::ElemKind::FloatTy)) {
   //     return dnn_lib::ElemKind::FloatTy;
   //   } else if (str == Type::getElementName(dnn_lib::ElemKind::Float16Ty)) {
   //     return dnn_lib::ElemKind::Float16Ty;
   //   } else if (str == Type::getElementName(dnn_lib::ElemKind::Int8QTy)) {
   //     return dnn_lib::ElemKind::Int8QTy;
   //   } else if (str == Type::getElementName(dnn_lib::ElemKind::UInt8QTy)) {
   //     return dnn_lib::ElemKind::UInt8QTy;
   //   } else if (str == Type::getElementName(dnn_lib::ElemKind::Int16QTy)) {
   //     return dnn_lib::ElemKind::Int16QTy;
   //   } else if (str == Type::getElementName(dnn_lib::ElemKind::Int32QTy)) {
   //     return dnn_lib::ElemKind::Int32QTy;
   //   } else if (str == Type::getElementName(dnn_lib::ElemKind::Int32ITy)) {
   //     return dnn_lib::ElemKind::Int32ITy;
   //   } else if (str == Type::getElementName(dnn_lib::ElemKind::Int64ITy)) {
   //     return dnn_lib::ElemKind::Int64ITy;
   //   } else if (str == Type::getElementName(dnn_lib::ElemKind::UInt8FusedQTy)) {
   //     return dnn_lib::ElemKind::UInt8FusedQTy;
   //   } else if (str == Type::getElementName(dnn_lib::ElemKind::UInt8FusedFP16QTy)) {
   //     return dnn_lib::ElemKind::UInt8FusedFP16QTy;
   //   } else if (str == Type::getElementName(dnn_lib::ElemKind::UInt4FusedFP16QTy)) {
   //     return dnn_lib::ElemKind::UInt4FusedFP16QTy;
   //   } else if (str == Type::getElementName(dnn_lib::ElemKind::BoolTy)) {
   //     return dnn_lib::ElemKind::BoolTy;
   //   } else {
   //     //assert(true && "Invalid ElemKind string");
   //     return dnn_lib::ElemKind::FloatTy;
   //   }
   // }

 }; //class Type
  

 
class LibTensor final {

 private:
  
  char* const ptrData_;

  const Type type_;
  
  template <class ElemTy> friend class Handle;

 public:

  /* @brief returns the type of the tensor.
   */
  const Type &getType() const { return type_;}

  /*@brief returns the element type of the tensor.
   */
  const ElemKind getElementType() const { return type_.getElementType(); }

  /*@brief returns True if the coordinate is within the array.
   */
  // template<std::size_t sz>
  // bool isInBounds(std::array<dim_t, sz>& coord) const {
  //   assert(type_.numSizes_ == coord.size() && "Invalid number of indices");
  //   for (size_t i = 0u, e = coord.size(); i < e; i++) {
  //     if (coord[i] >= type_.sizes_[i]) {
  //       return false;
  //     }
  //   }
  //   return true;    
  // }

//   /*@brief set the content of the tenosr to zero. if \p resetFusedScalesOffsets,
//    *then fused scales/offsets will be set to 1.0/0.0 as well.
//    */
//   void zero(bool resetFusedScalesOffsets = false) {
//     size_t size = actualSize();
//     //Quantized tensors should go to their offset.
//     switch (type_.getElementType()) {
//     case dnn_lib::ElemKind::Int8QTy: {
//       auto *data = reinterpret_cast<int8_t *>(getData());
//       std::fill(&data[0], &data[0] + size, (int8_t)type_.getOffset());
//       break;
//     }
//     case dnn_lib::ElemKind::UInt8QTy: {
//       auto *data = reinterpret_cast<uint8_t *>(getData());
//       std::fill(&data[0], &data[0] + size, (uint8_t)type_.getOffset());
//       break;
//     }
//     case dnn_lib::ElemKind::Int16QTy: {
//       auto *data = reinterpret_cast<int16_t *>(getData());
//       std::fill(&data[0], &data[0] + size, (int16_t)type_.getOffset());
//       break;
//     }
//     case dnn_lib::ElemKind::Int32QTy: {
//       auto *data = reinterpret_cast<int32_t *>(getData());
//       std::fill(&data[0], &data[0] + size, (int32_t)type_.getOffset());
//       break;
//     }
// #define FUSED_CASE(ELEM_KIND, DATA_TYPE)  case dnn_lib::ElemKind::ELEM_KIND: break
//     /* FUSED_CASE(dnn_lib::ElemKind::UInt8FusedQTy, float);       */
//     /* FUSED_CASE(dnn_lib::ElemKind::UInt8FusedFP16QTy, float16_t); */
// #undef FUSED_CASE
//      default:
//       // Non-quantized tensors are set to 0.
//        for(dim_t i = 0; i < (size * type_.getElementSize()); i++) {
         
//        }
       
//       break;
//     }     

//   }

  /*@brief Get number of dimensions the tensor has
   */
  const dim_t ndims() const { return type_.numSizes_; }
  
  /*@brief returns the dimensions (padded with 1 until max_tensor_dimensions)
   */
  const dim_array_t &dims() const { return type_.sizes_;}
  
  /*@brief returns the strides (padded with 0 until max_tensor_dimensions)
   */
  const dim_array_t &strides() const { return type_.strides_;}

  /*@brief returns the number of real menaingful elements in the tensor. Does
   *not take strides into account.
   */
  dim_t size() const { return type_.size(); }

  /*@brief returns the actaul number of elements in the tensor taking stridding
   *into account. Since size() does not take striding into account, size() is 
   *always <= actualSize(),
   */
  dim_t actualSize() const { return type_.actualSize(); }

  /*@brief returns the number of bytes required to store the tensor based on its
   *Type. Note that this includes the size required for padding.
   */
  uint64_t getSizeInBytes() const { return type_.getSizeInBytes(); }

  //constructor for quant types
  template<size_t numSizes>
  LibTensor(dnn_lib::ElemKind elk, void* rawdata, const std::array<dim_t, numSizes> &dims,
            const std::array<dim_t, numSizes> &pitches, float scale, int offset)
    : ptrData_(reinterpret_cast<char*>(rawdata)),
      type_(elk, dims, pitches, scale, offset) {}

  // constructor for non quant types
  template<size_t numSizes>
  LibTensor(dnn_lib::ElemKind elk, void* rawdata, const std::array<dim_t, numSizes> &dims,
            const std::array<dim_t, numSizes> &pitches)
    : ptrData_(reinterpret_cast<char*>(rawdata)),
      type_(elk, dims, pitches) {}

  // constructor from type
  LibTensor(const Type &type, void* rawdata)
    : ptrData_(reinterpret_cast<char*>(rawdata)),
      type_(type) {}

  LibTensor(const Type &&type, void* rawdata)
    : ptrData_(reinterpret_cast<char*>(rawdata)),
      type_(std::move(type)) {}


  // LibTensor(const LibTensor &other) = delete;
  // LibTensor &operator=(const LibTensor &other) = delete;

  
  /*@brief return a new handle that points and manages this tensor.
   */
  
  template <class ElemTy> Handle<ElemTy> getHandle() & {
    //@TODO check Elemty type_.isType<ElemType>() handle to wrong type.
    return Handle<ElemTy>(this);
  }

  template <class ElemTy> const Handle<ElemTy> getHandle() const & {
    //@TODO check Elemty type_.isType<ElemType>() handle to wrong type.
    return Handle<ElemTy>(const_cast<LibTensor*>(this));    
  }

  template <class ElemTy = float> Handle<ElemTy> getHandle() && = delete;

  
  /* @brief copy raw data value at ptrData_ buffer tensor given at \p inTensor
   * to the other ptrData_ buffer at (this).
   *
   * @param[in] inTensor tensor from copy data
   */
  void copyRawFrom(LibTensor* inT) {
    //@TODO check both tensor has the same shape!!
    memcpy(ptrData_, inT->ptrData_, inT->getSizeInBytes());
  }

  // /*@brief create a new copy of the current tensor.
  //  */
  // LibTensor clone() const {
  //   LibTensor slice;
  //      slice.assign(this);
  //   return slice;
  // }

  /*@brief return the raw unsafe pointer to the tensor payload.
   */
  //TODO: REMOVE if not used  char* getUnsafePtr() const { return getData(); }


  /*TODO: After re-do sw-2429 (refact operands) are the getters necessary? if not remove them. */
public:  
  float   getScale() const  { return type_.getScale(); }
  int32_t getOffset() const { return type_.getOffset(); }
  size_t getElementSize() const { return type_.getElementSize(); }  


  /*@brief returns a pointer to the raw data, of type \p ElemTy.
   */
  template<class ElemTy>  ElemTy *getRawDataPointer() {
    //@TODO check Elemty is type_.isType<>()
    return reinterpret_cast<ElemTy*>(ptrData_);
  }

  /*@brief returns a const pointer to the raw data, of type \p ElemTy.
   */
  template<class ElemTy> const ElemTy *getRawDataPointer() const {
    //@TODO check Elemty is type_.isType<>()
    return reinterpret_cast<const ElemTy*>(ptrData_);
  }


  // returns offset and maxRead (in number of elements)
  void partitionCL(uint64_t minionId,  unsigned activeMinions,
                   dim_t &offset, dim_t &maxRead) const{
    assume(minionId < 1024);
    size_t elementSize = type_.getElementSize();
    size_t inCL = (type_.getSizeInBytes() + CACHE_LINE_BYTES -1) / CACHE_LINE_BYTES;
    size_t CLperMin = inCL / activeMinions;
    size_t spare = inCL - CLperMin * activeMinions;
    // all minions will have CLperMin cache lines to process
    // and minions with id < spare will have an extra CL
    if (minionId < spare) {
      offset = (CLperMin +1) * minionId;
      maxRead = CLperMin +1;
    } else {
      offset = CLperMin * minionId + spare;
      maxRead = CLperMin;
    }

    offset*=CACHE_LINE_BYTES / elementSize;
    maxRead*=CACHE_LINE_BYTES / elementSize;
    
  }

  dim_array_t offset2Coord(size_t offset) const {
    dim_array_t coords = {0};
    assert (type_.strides_[type_.numSizes_ - 1] == 1);
    uint32_t rm = offset; // operations in uint32_t.. division is faster
    for (size_t i = 0; i < type_.numSizes_ ; i++) {
      coords[i] = rm / static_cast<uint32_t>(type_.strides_[i]);
      rm = rm - static_cast<uint32_t>(coords[i]) * static_cast<uint32_t>(type_.strides_[i]);
    }
    return coords;
  }


  void evict(uint64_t dst, size_t offset, size_t count) const{
    FENCE;
    const size_t typeSize = type_.getElementSize();    
    size_t cl = (count * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
    assert(cl > 0);
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptrData_) + typeSize*offset;
    while(cl > 16) {
      evict_va(0, dst, addr, 15, CACHE_LINE_BYTES);
      addr += (CACHE_LINE_BYTES*16);
      cl -= 16;
    }
    if (cl > 0)
      evict_va(0, dst, addr, cl-1, CACHE_LINE_BYTES);
  }

  void evict(uint64_t dst) const {
    evict(dst, 0, this->actualSize());
  }
  
}; //end LibTensorBase class



/*@brief Convert to flattened 1d offset given \p indices.
 *
 *@param[inout] indices keeps the coords of the element 
 *@return
 */
template<size_t N>
INLINE_ATTR size_t getFlattenedOffset(const std::array<dim_t, N> &indices, const dim_array_t &strides) {
  /*@TODO check indices size isn't bigger than strides*/
  //assert(indices.size() <= strides.size());
  size_t r = 0;
  for (size_t i = 0 ; i < N; i++) r+=indices[i] * strides[i];
  return r;   
}

template<size_t N>
INLINE_ATTR size_t getFlattenedOffset(const std::array<dim_t, N> &indices, 
                                     const dim_array_t &strides, 
                                     const dim_array_t &extStrides, size_t ndx) {

  size_t r = 0;
  for (size_t i = 0; i < N; i++) {
    if (i == ndx)
      r += indices[i] * extStrides[i];
    else 
      r += indices[i] * strides[i];
  }
   return r;
}

#include "LibTensorIterator.h"

template <class ElemTy> class Handle final {
  
   /*brief pointer to the tensor that this handle wraps.
    */
  LibTensor * const tensor_;

   /*@brief It has the mult of the sizes for each position to end.
    */
  const dim_array_t &strides_;

  const dim_array_t &sizes_;

  /*@brief the number of dimensions used in the tensor.
   */
  const dim_t numDims_;

public:
  using iterator = HandleIterator<ElemTy>;
  friend class HandleIterator<ElemTy>;

  const iterator begin() { return iterator(*this);}
  const iterator end() { return iterator(*this, sizes_[0]*strides_[0], sizes_);}
  iterator getIterator(size_t offset) { return iterator(*this, offset);}
  iterator getIterator(const dim_array_t &coords) { return iterator(*this, coords);}
  
 
  /*@brief Calculate the index for a specific element in the tensor.
   *
   *@param[inout] coords indices to access element. It has to have the same
   * dimensions as tensor to be acessed.
   *@return flattened 1D element position.
   */
  template<size_t N>
  size_t getElementPtr(const std::array<dim_t, N> &indices) {
    return getFlattenedOffset(indices, strides_);
  }

  template<size_t N>
  size_t getElementPtr(const std::array<dim_t, N> &indices, 
                      const dim_array_t &extStrides, size_t ndx) {
    return getFlattenedOffset(indices, strides_, extStrides, ndx);
  }

  /*@brief returns the value of the n'th dimension \p dim, for the index \p idx.
   * 0 <= idx < size(), meaning that \p idx addresses a real data elements,
   * not paddings.
   */
  size_t getDimForPtr(size_t dim, size_t idx) const {
    // assert(dim < numDims_ && "Invalid dimension");
    // assert(idx < size() && "Invalid index");
    auto R = idx;
    for (size_t i = dim + 1; i < numDims_; i++) {
      R /= sizes_[i];
    }
    return R % sizes_[dim];
  }
  
   /*@brief returns the type of the tensor.
    */
   const Type& getType() const { return tensor_->getType(); }

   /*@brief returns the element type of the tensor.
    */
   ElemKind getElementType() const { return tensor_->getElementType(); }

   /*@brief Construct a Tensor handle.
    */
   explicit Handle(LibTensor* tensor) :
     tensor_(tensor),
     strides_(tensor->strides()),
     sizes_(tensor->dims()),
     numDims_ (tensor->ndims())
  {
  }

   /*@brief returns the number of elements in the whole tensor.
    */
   dim_t size() const { return tensor_->size(); }

   /*@brief returns the number of elements in the tensor taking striding/pitches
    *into account. Since size() does not take striding into account, size() is
    *always <= actualSize():
    */
   dim_t actualSize() const { return tensor_->actualSize(); }

   /*@brief check if given \p indices is into the dims_ bounds.
    *
    *@param[inout] indices 
    *@return true if indices is into the bounds.
    */
   /* bool isInBounds(dim_t* indices) const { */
   /*   return tensor_->isInBounds(indices); */
   /* } */
  // template<std::size_t sz>
  // bool isInBounds(std::array<dim_t, sz>& indices) const {
  //   return tensor_->isInBounds(indices);
  // }

  
  void clear(ElemTy value = 0) {  
    std::fill(this->begin(), this->end(), value); 
  }

  /*@brief return reference to a meaningful data element. This method skip
   *padding elements.
   */
  template<size_t N>
  ElemTy &at(std::array<dim_t, N> indices) {
    size_t index = getElementPtr(indices);   
    auto *data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  template<size_t N>
  const ElemTy &at(std::array<dim_t, N> indices) const {
    size_t index = getElementPtr(indices);
    auto *data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  template<size_t N>
  ElemTy &at(std::array<dim_t, N> indices, const dim_array_t &extStrides, 
            size_t ndx) {
    size_t index = getElementPtr(indices, extStrides, ndx);
    auto *data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  float getScale(void) {return tensor_->getScale();}
  int32_t getOffset(void) {return tensor_->getOffset();}
  
}; //end Handle class

}

#endif // _LIB_TENSOR_H_
