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

namespace dnn_lib {


/*@brief return whether \p e is a quantized ElemKind.
  */
// inline bool isQuantizedElemKind(ElemKind e) {
//   return (e == ElemKind::Int8QTy || e == ElemKind::UInt8QTy ||
//           e == ElemKind::Int16QTy); //|| e == ElemKind::UInt32QTy ||
//           // e == ElemKind::Int8FusedQTy || e == ElemKind::UInt8FusedFP16QTy ||
//           // e == ElemKind::Int4FusedFP16QTy         
// }
  
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
  Type(dnn_lib::ElemKind elk, const std::array<dim_t, numSizes> &dims,  float scale, int32_t offset) :
    sizes_(make_dims(dims)),
    elementType_(elk), numSizes_(numSizes),
    scale_(scale), offset_(offset)
  {
    assert( isQuantizedElemKind(elk));
  }
  
  /*@brief Initialize a new non-quantized type.
   */
  template<size_t numSizes>
  Type(dnn_lib::ElemKind elk, const std::array<dim_t, numSizes> &dims) :
    sizes_(make_dims(dims)),
    elementType_(elk), numSizes_(numSizes)
  {
    assert( !isQuantizedElemKind(elk));
  }
  
  /*@brief Initialize a new quantized type with \p scale an \p offset.
   */
  template<size_t numSizes>
  Type(dnn_lib::ElemKind elk,  const std::array<dim_t, numSizes> &dims, const std::array<dim_t, numSizes> &strides,
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
  Type(dnn_lib::ElemKind elk,  const std::array<dim_t, numSizes> &dims, const std::array<dim_t, numSizes> &strides) :
    sizes_(make_dims(dims)),
    strides_(make_strides(strides)),
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
   size_t actualSize() const { return (getSizeInBytes() / getElementSize()); }

   /// \return the size of the element \p Ty.
   static unsigned getElementSize(dnn_lib::ElemKind Ty) {
     switch (Ty) {
     case dnn_lib::ElemKind::FloatTy:
       return sizeof(float);
     case dnn_lib::ElemKind::Float16Ty:
       return sizeof(float16_t);
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
//TODO: REMOVE if not used  dim_t dbgDims(unsigned int ndx) const {
//TODO: REMOVE if not used    return sizes_[ndx];
//TODO: REMOVE if not used  }
//TODO: REMOVE if not used  
//TODO: REMOVE if not used  dim_t dbgSize(unsigned int ndx) const {
//TODO: REMOVE if not used    return strides_[ndx];
//TODO: REMOVE if not used  }

 }; //class Type
  

 
class LibTensor final {

 private:
  
  char* const ptrData_;

  const Type type_;

//TODO: REMOVE if not used  bool isUnowned_ {false};
//TODO: REMOVE if not used  
//TODO: REMOVE if not used  size_t unpaddedSize_{0};
  
  /* std::array<uint64_t, max_tensor_dimensions> dims_; */
  /* std::array<uint64_t, max_tensor_dimensions> pitches_; */
  /* std::array<uint64_t, max_tensor_dimensions> coord_; */
  /* unsigned int numDims_; */
  /* float scale_; */
  /* int8_t offset; */
  //add more info if it is need....
  
  template <class ElemTy> friend class Handle;

 public:
  /* @brief returns a pointer to the tensor data buffer.
   */
  const char* dbgData() const { return ptrData_; }

//TODO: REMOVE if unused  /*@brief returns true if it is an unowned.
//TODO: REMOVE if unused   */
//TODO: REMOVE if unused  bool isUnowned() const { return isUnowned_; }
  
  /* @brief returns the type of the tensor.
   */
  const Type &getType() const { return type_;}

  /* @brief set type of the Tensor to \p t.
   */
//TODO: REMOVE if unused => assuming type does not change  void setType(const TypeRef t) {
//TODO: REMOVE if unused => assuming type does not change    //@TODO assert(type_.dims() == t->dims() && "New type must retain the same shape.");
//TODO: REMOVE if unused => assuming type does not change    //@TOODassert(((type_.getElementType() == t->getElementType() &&
//TODO: REMOVE if unused => assuming type does not change           //   type_.size() == t->size()) ||
//TODO: REMOVE if unused => assuming type does not change           //  type_.getSizeInBytes() == t->getSizeInBytes()) &&
//TODO: REMOVE if unused => assuming type does not change           // "New type must retain the same size in bytes.");
//TODO: REMOVE if unused => assuming type does not change    type_ = *t;
//TODO: REMOVE if unused => assuming type does not change  }

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
  const dim_array_t & dims() const { return type_.sizes_;}
  

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

  template<size_t numSizes>
  LibTensor(dnn_lib::ElemKind elk, void* rawdata, std::array<dim_t, numSizes> &dims,
            std::array<dim_t, numSizes> &pitches, float scale, int offset)
    : ptrData_(reinterpret_cast<char*>(rawdata)),
      type_(elk, dims, pitches, scale, offset) {}

  // LibTensor(const LibTensor &other) = delete;
  // LibTensor &operator=(const LibTensor &other) = delete;

  
  /*@brief return a new handle that points and manages this tensor.
   */
  template <class ElemTy = float> Handle<ElemTy> getHandle() &;

  template <class ElemTy = float> const Handle<ElemTy> getHandle() const &;

  template <class ElemTy = float> Handle<ElemTy> getHandle() && = delete;

//TODO: REMOVE if not used  /*@brief returns an unowned tensor with the exact same dimensions as this.
//TODO: REMOVE if not used   */
//TODO: REMOVE if not used  LibTensor getUnowned() const {
//TODO: REMOVE if not used    dim_t cpydims[max_tensor_dimensions] = {0,};
//TODO: REMOVE if not used    uint8_t numdim = this->type_.dims(cpydims);
//TODO: REMOVE if not used    return getUnowned(cpydims, numdim);
//TODO: REMOVE if not used  }
//TODO: REMOVE if not used    
//TODO: REMOVE if not used  /*@brief Create a Tensor using the data buffer in \p dims as the current tensor
//TODO: REMOVE if not used   *but having different dimensions \p dims. \p offsets represents an optional
//TODO: REMOVE if not used   *offset into the tensor representing th elocation of the first element to start
//TODO: REMOVE if not used   *a subview from. The returned tensor is essentially a different view or subview
//TODO: REMOVE if not used   *on the same data.
//TODO: REMOVE if not used   *
//TODO: REMOVE if not used   *@param[in] dims keep the current tensor size.
//TODO: REMOVE if not used   *@param[in] optional indices It keeps the position subview.
//TODO: REMOVE if not used   *@return a Tensor.
//TODO: REMOVE if not used   */
//TODO: REMOVE if not used  LibTensor getUnowned(dim_t *dims, uint8_t numDims, bool useSameStrides = false,
//TODO: REMOVE if not used                       dim_t *indices = {nullptr}, uint8_t indNumDims = 0) const {
//TODO: REMOVE if not used
//TODO: REMOVE if not used    LibTensor unownedTensor;
//TODO: REMOVE if not used    auto* ptrToData = getData();
//TODO: REMOVE if not used
//TODO: REMOVE if not used    if (indNumDims) {
//TODO: REMOVE if not used      //@TODO check indNumDims == numDims
//TODO: REMOVE if not used      dim_t strides[max_tensor_dimensions] = {0,};
//TODO: REMOVE if not used      uint8_t numSize = type_.strides(strides);
//TODO: REMOVE if not used      size_t index = 0 ;
//TODO: REMOVE if not used      
//TODO: REMOVE if not used      for (size_t i = 0; i < numSize; i++) {
//TODO: REMOVE if not used        index += strides[i] * indices[i];
//TODO: REMOVE if not used      }
//TODO: REMOVE if not used      ptrToData = &ptrToData[index * type_.getElementSize()];
//TODO: REMOVE if not used    }
//TODO: REMOVE if not used
//TODO: REMOVE if not used    unownedTensor.ptrData_ = ptrToData;
//TODO: REMOVE if not used    unownedTensor.isUnowned_ = true;
//TODO: REMOVE if not used
//TODO: REMOVE if not used    unownedTensor.type_ = Type::newShape(getType(), dims, numDims);
//TODO: REMOVE if not used
//TODO: REMOVE if not used    if(useSameStrides) {
//TODO: REMOVE if not used      dim_t strides[max_tensor_dimensions] = {0,};
//TODO: REMOVE if not used      uint8_t numStrides = this->type_.strides(strides);
//TODO: REMOVE if not used
//TODO: REMOVE if not used      for (unsigned int i = 0; i < numStrides; i++) {
//TODO: REMOVE if not used        unownedTensor.type_.strides_[i] = strides[i];
//TODO: REMOVE if not used      }
//TODO: REMOVE if not used    }
//TODO: REMOVE if not used
//TODO: REMOVE if not used
//TODO: REMOVE if not used    if(indNumDims) {
//TODO: REMOVE if not used      unownedTensor.unpaddedSize_ = unpaddedSize_;
//TODO: REMOVE if not used      //@TODO check actualSize() == unownedTensor.actualSize()
//TODO: REMOVE if not used    }
//TODO: REMOVE if not used    else {
//TODO: REMOVE if not used      unownedTensor.unpaddedSize_ = unownedTensor.type_.getSizeInBytes();
//TODO: REMOVE if not used      //@TODO check getSizeInBytes() == getUnpaddedSizeInBytes()
//TODO: REMOVE if not used      //@TODO check actualSize() >= unownedensor.actualSize()
//TODO: REMOVE if not used    }
//TODO: REMOVE if not used
//TODO: REMOVE if not used    return unownedTensor;
//TODO: REMOVE if not used  }
  
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


  /*debug purpose*/
  const dim_array_t &strides() const { return type_.strides_;}
  float    dbggetscale() { return type_.getScale(); }
  int32_t dbggetoffset() { return type_.getOffset(); }
  
public:

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


}; //end LibTensorBase class



/*@brief Convert to flattened 1d offset given \p indices.
 *
 *@param[inout] indices keeps the coords of the element 
 *@return
 */
  template<size_t N>
  INLINE_ATTR size_t getFlattenedOffset(std::array<dim_t, N> &indices, dim_array_t &sizeIntegral) {
    /*@TODO check indices size isn't bigger than strides*/
    //assert(indices.size() <= sizeIntegral.size());
    return  static_cast<size_t>(std::inner_product(indices.cbegin(), indices.cend(), sizeIntegral.cbegin(), 0));

  }


template <class ElemTy, bool IsConst>
class HandleIterator
    : public std::iterator<std::random_access_iterator_tag, ElemTy> {
  using HandleTy = typename dnn_lib::conditional_t<IsConst, const Handle<ElemTy> *,
                                          Handle<ElemTy> *>;
  using ElemTyRef =
    typename dnn_lib::conditional_t<IsConst, const ElemTy &, ElemTy &>;

  HandleTy const handle_;
  const dim_array_t &sizes_;
  const dim_t nSizes_;
  const bool isAligned_;
  dim_t idx_;

  HandleIterator(HandleTy handle) :
    handle_(handle),
    sizes_(handle->sizes_),
    nSizes_(handle->numDims_),
    isAligned_(handle->size() < handle->actualSize())
  {
  }

  static HandleIterator begin(HandleTy handle) {
    auto res = HandleIterator(handle);
    res.idx_ = 0;
    return res;
  }

  static HandleIterator end(HandleTy handle) {
    auto res = HandleIterator(handle);
    res.idx_ = res.handle_->size();
    return res;
  }

  friend class Handle<ElemTy>;

public:
  HandleIterator &operator++() {
    if (*this != handle_>end()) {
        idx_++;
    }
  }

  HandleIterator &operator--() {
    if (idx_) {
      idx_--;
    }
    return *this;
  }

  HandleIterator operator+(int n) const {
    auto res = HandleIterator(handle_);
    res.idx_ = std::max(static_cast<int>(idx_) + n, 0);
    res.idx_ = std::min(res.idx_, res.handle_->size());
    return res;
  }

  HandleIterator operator-(int n) const {return *this + (-n);}

  operator int() const { return idx_; }

  ElemTyRef operator*() {
    if(!isAligned_) {
      return handle_->raw(idx_);
    }

    dim_array_t indices{};
    size_t rem = idx_;
    size_t idx = nSizes_ -1;

    for (int64_t i = nSizes_ -1; i >= 0; i--) {
      indices[i] = rem % sizes_[i];
      rem /= sizes_[i];
    }
    return handle_->at(indices);
  }

  bool operator==(const HandleIterator<ElemTy, IsConst> &other) const {
    return idx_ == other.idx_;
  }

  bool operator!=(const HandleIterator<ElemTy, IsConst> &other) const {
    return !(*this == other);
  }
};
  
template <class ElemTy> class Handle final {

   /*brief pointer to the tensor that this handle wraps.
    */
   LibTensor *tensor_{nullptr};

   /*@brief It has the mult of the sizes for each position to end.
    */
  const dim_array_t &sizeIntegral_;

  const dim_array_t &sizes_;

  /*@brief the number of dimensions ussed in the tensor.
   */
  const dim_t numDims_;


  // TODO: REMOVE => assuming we won't use invalid Handles , always from a tensor
  // Handle() = default;
  //TODO: END REMOVE
public:
  
  
  // TODO: REMOVE => assuming we won't use invalid Handles , always from a tensor
  //   /*@brief Allocate anew invalid handle.
  //    */
  //   static Handle createInvalidHandle() { return Handle(); }
  //
  //
  ///*@brief returns true if this Handle points to a valid tensor.
  // */
  //bool isValid() const { return tensor_; }

  //TODO: END REMOVE
  
   /*@brief Calculate the index for a specific element in the tensor.
    *
    *@param[inout] coords indices to access element. It has to have the same
    * dimensions as tensor to be acessed.
    *@return flattened 1D element position.
    */
  template<size_t N>
  size_t getElementPtr(std::array<dim_t, N> &indices) {
    return getFlattenedOffset(indices, sizeIntegral_);
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
     sizeIntegral_(tensor->strides()),
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
//TODO: remove if unused 
//TODO: remove if unused /*@brief get a copy of dims_ internal dimension Tensor.
//TODO: remove if unused   *
//TODO: remove if unused   *@param[inout] dim array pointer to copy the tensor sizes_.
//TODO: remove if unused   *@return the numDims_ value. (the number of dims be copied).
//TODO: remove if unused   */
//TODO: remove if unused  uint8_t cpydims(dim_t* cpydim) {
//TODO: remove if unused    for(uint8_t i = 0; i < numDims_; i++) {
//TODO: remove if unused      cpydim[i] = sizes_[i];
//TODO: remove if unused    }
//TODO: remove if unused    return numDims_;
//TODO: remove if unused  }
//TODO: remove if unused
//TODO: remove if unused
  
  char* getPtrdbg(void) const {return tensor_->dbgData();}
  const dim_array_t & getSizeIntdbg(void) {return sizeIntegral_;}
  const dim_array_t & getSizesdbg(void) {return sizes_;}
  dim_t getNumDimsdbg(void) {return numDims_;}
  LibTensor* getTensordbg(void) {return tensor_;}
  //TODO: REMOVE if not used  char* getUnsafePtrdbg(void) {return tensor_->getUnsafePtr();}
  float getScaledbg(void) {return tensor_->dbggetscale();}
  int32_t getOffsetdbg(void) {return tensor_->dbggetoffset();}
//TODO: remove if unused  uint8_t cpypitchesdbg(dim_t* cpypitch) { return tensor_->dbgcpypitches(cpypitch); }
//TODO: remove if unused  uint8_t cpydimsdbg(dim_t* cpydims) { return tensor_->dims(cpydims); }
}; //end Handle class

  template <class ElemTy> Handle<ElemTy> LibTensor::getHandle() & {
    //@TODO check Elemty type_.isType<ElemType>() handle to wrong type.
    return Handle<ElemTy>(this);
  }

  template <class ElemTy> const Handle<ElemTy> LibTensor::getHandle() const & {
    //@TODO check Elemty type_.isType<ElemType>() handle to wrong type.
    return Handle<ElemTy>(const_cast<LibTensor*>(this));    
  }
}

#endif // _LIB_TENSOR_H_
