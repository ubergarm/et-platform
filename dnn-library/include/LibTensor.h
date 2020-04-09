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

#include <array>
#include <cassert>

#include "LibTypes.h"
#include "Float16.h"

namespace dnn_lib {

constexpr unsigned max_tensor_dimensions = 6;
constexpr unsigned max_scale_dimensions = 2;

constexpr ElemKind IndexElemKind =
    (sizeof(dim_t) == 4) ? ElemKind::Int32ITy : ElemKind::Int64ITy;
  
 struct Type;

 using TypeRef = const Type *;

 using float16_t = float16;

  
template<ElemKind elK>
struct elemKind2elemTy {
  using type =
    typename std::conditional<elK == ElemKind::FloatTy, float,
     typename std::conditional<elK == ElemKind::Float16Ty, float16,
      typename std::conditional<elK == ElemKind::Int8QTy, int8_t,
       typename std::conditional<elK == ElemKind::UInt8QTy, uint8_t,
        typename std::conditional<elK == ElemKind::Int16QTy, int16_t,
         typename std::conditional<elK == ElemKind::Int32QTy, int32_t,
          typename std::conditional<elK == ElemKind::Int32ITy, int32_t,
           typename std::conditional<elK == ElemKind::Int64ITy, int64_t,
            typename std::conditional<elK == ElemKind::UInt8FusedQTy, uint8_t,
             typename std::conditional<elK == ElemKind::UInt8FusedFP16QTy, uint8_t,
              typename std::conditional<elK == ElemKind::UInt4FusedFP16QTy, uint8_t,
               typename std::conditional<elK == ElemKind::BoolTy, bool,
                    void // void is the default value, if no elKind matches
          >::type >::type >::type >::type >::type> ::type
    >::type >::type >::type >::type >::type>::type;
    
  //@TODO static_assert(!std::is_same<type, void>::value);
};
 
template <class ElemTy> class Handle;

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
   dim_t sizes_[max_tensor_dimensions] = {0,};

   /*@brief contains the strides for each dimension (in elements) same order
    * as in sizes_.
    */
   dim_t strides_[max_tensor_dimensions] = {0,};
  
   /*@brief Specifies the element type of the tensor.
    */
   dnn_lib::ElemKind elementType_{dnn_lib::ElemKind::Int64ITy};

   /*@brief contains the number of dimensions used by the tensor.
    */
   unsigned char numSizes_{0};

   /*@brief On quantized tensors, this represents the scale of the values.
    */
   float scale_{0};

   /*@brief On quantized tensors, this represents the offset of the values.
    */
   int32_t offset_{0};
 
  /*@brief Initialize a new quantized type with \p scale an \p offset.
   */
  Type(dnn_lib::ElemKind elk, dim_t *dims, uint8_t numSizes,
       float scale, int32_t offset) : elementType_(elk), numSizes_(numSizes),
                                      scale_(scale), offset_(offset) { initDims(dims); }

  /*@brief Initialize a new non-quantized type.
   */
  Type(dnn_lib::ElemKind elk, dim_t *dims, uint8_t numSizes)
    : elementType_(elk), numSizes_(numSizes) { initDims(dims); }
  
  /*@brief Initialize a new quantized type with \p scale an \p offset.
   */
  Type(dnn_lib::ElemKind elk, dim_t* dims, uint8_t numSizes,
       dim_t* pitches, float scale, int32_t offset)
    : elementType_(elk), numSizes_(numSizes), scale_(scale), offset_(offset) {
    initDims(dims, pitches);
  } 
  
  /*@brief Initialize a new non-quantized type.
   */
  Type(dnn_lib::ElemKind elk, dim_t *dims, uint8_t numSizes,
       uint64_t *pitches) : elementType_(elk), numSizes_(numSizes) {
    initDims(dims, pitches);
  }

  /*@brief Reshape existing type this takes care of quantized types.
   */
  static Type newShape(const Type &T, dim_t *dims, uint8_t numDims) {
    if(T.isQuantizedType()) {
      return Type(T.getElementType(), dims, numDims, T.getScale(),
                  T.getOffset());
    }
     else {
         return Type(T.getElementType(), dims, numDims);
     }
   }
 
   /*@brief Reshape existing type and change alignments.
    */
   static Type newShape(const Type &T, dim_t* dims, uint8_t numDims,
                        dim_t* pitches) {
     if (T.isQuantizedType()) {
       return Type(T.getElementType(), dims, numDims, pitches, T.getScale(),
                   T.getOffset());
     }
     else {
       return Type(T.getElementType(), dims, numDims, pitches);
     }
   }

   /*@brief Reshape existing type by taking shapes and strides of \p shapeType.
    */
   static Type newShape(const Type &T, TypeRef shapeType) {
     //@TODO  T.getElementType() == shapeType->getelementSize() Size should be the same
     Type ty;
     dim_t shapeTypeDims[max_tensor_dimensions];
     uint8_t shapeNumDims = shapeType->dims(shapeTypeDims);
     
     if (T.isQuantizedType()) {
       ty = Type(T.getElementType(), shapeTypeDims, shapeNumDims, T.getScale(),
                 T.getOffset());
     }
     else {
       ty = Type(T.getElementType(), shapeTypeDims, shapeNumDims);
     }

     for (uint8_t i = 0; i < ty.numSizes_; i++) {
       ty.strides_[i] = shapeType->strides_[i];
     }
     
     return ty;
   }
   
  /*brief an empty type.
   */
  Type() = default;

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

   
   // /*@brief returns true if \p other is the same type. If \p allowDifferentShape
   //  * then shapes will not be considered as part of the equal. comparision.
   //  */
   // bool isEqual(const Type &other, bool allowDifferentShape = false,
   //              bool allowDifferentStrides = false) const  {

   //  // Element type must be the same
   //   if (elementType_ != other.elementType_) {
   //     return false;
   //   }
   //   // Must have the same number of sizes.
   //   if (numSizes_ != other.numSizes_) {
   //     return false;
   //   }
   //   // Sizes must be the same.
   //   if (!allowDifferentShape) {
   //     for (size_t i = 0; i < numSizes_; i++) {
   //       if (sizes_[i] != other.sizes_[i]) {
   //         return false;
   //       }
   //     }
   //   }
   //   if (!allowDifferentStrides) {
   //     // Strides must be the same.
   //     for (size_t i = 0; i < numSizes_; i++) {
   //       if(strides_[i] != other.strides_[i]) {
   //         return false;
   //       }
   //     }
   //   }

   //   // check the scale an offset
   //   if (isQuantizedType()) {
   //     if (scale_ != other.scale_ || offset_ != other.offset_) {
   //       return false;
   //     }
   //   }

   //   return true;
   // }

   /*@brief returns the elemet type
    */
   ElemKind getElementType() const { return elementType_; }

   /*@brief Copy the shape of the tensor in the given \p array.
    *
    *@param[inout] cpydims pointer array where left the copy.
    *@return the numdims copied.
    */
   uint8_t dims(dim_t* cpydims) const {
     for (unsigned int i = 0; i < numSizes_; i++) {
       cpydims[i] = sizes_[i];
     }
     return numSizes_;
   }

   /*@brief Copy the strides of the tensor in the given \p array.
    *
    *@param[inout] cpystrides pointer array where left the copy.
    *@return the numSizes_.
    */ 
   uint8_t strides(dim_t* cpystrides) const {
     for (unsigned int i = 0; i < numSizes_; i++) {
       cpystrides[i] = strides_[i];
     }

     return getNumDims();
   }

   /*@brief return the number of elements in the tensor.
    */
   dim_t size() const {
     dim_t acum = 1;
     for (unsigned char i = 0; i < numSizes_; i++) {
       acum *= dim_t(sizes_[i]);
     }
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
     size_t s = getElementSize();
     size_t acc = 0;
     for (unsigned char i = 0; i < numSizes_; i++) {
       acc += size_t(sizes_[i]) * s * strides_[i];
     }
     return acc;
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

   // /// \return the textual name of the element.
   // string getElementName() const {
   //   return getElementName(elementType_);
   // }

   // /// \return the textual name of the element \p elk.
   // static string getElementName(dnn_lib::ElemKind elk) {
   //   static const char *names[] = {
   //      "float",    "float16",      "i8",           "ui8",
   //      "i16",      "i32",          "index32",      "index64",
   //      "ui8fused", "ui8fusedfp16", "ui4fusedfp16", "bool",
   //   };
   //   return names[(int)elk];
   // }

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
  dim_t dbgDims(unsigned int ndx) const {
    return sizes_[ndx];
  }
  
  dim_t dbgSize(unsigned int ndx) const {
    return strides_[ndx];
  }

private:

   void initDims(dim_t* dims) {
     //@TODO assert(numSizes_ <= max_tensor_dimensions && "Too many dimensions.");
     for (unsigned char i = 0; i < numSizes_; i++) {
       sizes_[i] = dims[i];
     }
   }
   
   void initDims(dim_t* dims, dim_t* pitches) {
     //@TODO assert(numSizes_ <= max_tensor_dimensions && "Too many dimensions.");     
     for (unsigned int i = 0; i < numSizes_; i++) {
       sizes_[i] = dims[i];
       strides_[i] = pitches[i];
     }
   }
 }; //class Type
  

 
class LibTensor final {

 private:
  
  char* ptrData_{nullptr};

  Type type_;

  bool isUnowned_ {false};
  
  size_t unpaddedSize_{0};
  
  /* std::array<uint64_t, max_tensor_dimensions> dims_; */
  /* std::array<uint64_t, max_tensor_dimensions> pitches_; */
  /* std::array<uint64_t, max_tensor_dimensions> coord_; */
  /* unsigned int numDims_; */
  /* float scale_; */
  /* int8_t offset; */
  //add more info if it is need....
  
  template <class ElemTy> friend class Handle;
  

  /* @brief returns a pointer to the tensor data buffer.
   */
  char* getData() const { return ptrData_; }

 public:
  char* dbgData()  {return ptrData_;}

  /*@brief returns true if it is an unowned.
   */
  bool isUnowned() const { return isUnowned_; }
  
  /* @brief returns the type of the tensor.
   */
  const Type &getType() const { return type_;}

  /* @brief set type of the Tensor to \p t.
   */
  void setType(const TypeRef t) {
    //@TODO assert(type_.dims() == t->dims() && "New type must retain the same shape.");
    //@TOODassert(((type_.getElementType() == t->getElementType() &&
           //   type_.size() == t->size()) ||
           //  type_.getSizeInBytes() == t->getSizeInBytes()) &&
           // "New type must retain the same size in bytes.");
    type_ = *t;
  }

  /*@brief returns the element type of the tensor.
   */
  ElemKind getElementType() const { return type_.getElementType(); }

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

  /*@brief Get a copy of the shape of tensor.
   *
   *@param[inout] cpydims to make a copy at.
   *@return the number of dimensions copied.
   */
  uint8_t dims(dim_t* cpydims) { return type_.dims(cpydims); }

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

  LibTensor() = default;
  
  LibTensor(dnn_lib::ElemKind elk, void* rawdata, dim_t* dims,
            dim_t* pitches, unsigned int numDims, float scale, int offset)
            : ptrData_(reinterpret_cast<char*>(rawdata)),
              type_(elk, dims, numDims, pitches, scale, offset) {}

  // LibTensor(const LibTensor &other) = delete;
  // LibTensor &operator=(const LibTensor &other) = delete;

  
  /*@brief return a new handle that points and manages this tensor.
   */
  template <class ElemTy = float> Handle<ElemTy> getHandle() &;

  template <class ElemTy = float> const Handle<ElemTy> getHandle() const &;

  template <class ElemTy = float> Handle<ElemTy> getHandle() && = delete;

  /*@brief returns an unowned tensor with the exact same dimensions as this.
   */
  LibTensor getUnowned() const {
    dim_t cpydims[max_tensor_dimensions] = {0,};
    uint8_t numdim = this->type_.dims(cpydims);
    return getUnowned(cpydims, numdim);
  }
    
  /*@brief Create a Tensor using the data buffer in \p dims as the current tensor
   *but having different dimensions \p dims. \p offsets represents an optional
   *offset into the tensor representing th elocation of the first element to start
   *a subview from. The returned tensor is essentially a different view or subview
   *on the same data.
   *
   *@param[in] dims keep the current tensor size.
   *@param[in] optional indices It keeps the position subview.
   *@return a Tensor.
   */
  LibTensor getUnowned(dim_t *dims, uint8_t numDims, bool useSameStrides = false,
                       dim_t *indices = {nullptr}, uint8_t indNumDims = 0) const {

    LibTensor unownedTensor;
    auto* ptrToData = getData();

    if (indNumDims) {
      //@TODO check indNumDims == numDims
      dim_t strides[max_tensor_dimensions] = {0,};
      uint8_t numSize = type_.strides(strides);
      size_t index = 0 ;
      
      for (size_t i = 0; i < numSize; i++) {
        index += strides[i] * indices[i];
      }
      ptrToData = &ptrToData[index * type_.getElementSize()];
    }

    unownedTensor.ptrData_ = ptrToData;
    unownedTensor.isUnowned_ = true;

    unownedTensor.type_ = Type::newShape(getType(), dims, numDims);

    if(useSameStrides) {
      dim_t strides[max_tensor_dimensions] = {0,};
      uint8_t numStrides = this->type_.strides(strides);

      for (unsigned int i = 0; i < numStrides; i++) {
        unownedTensor.type_.strides_[i] = strides[i];
      }
    }


    if(indNumDims) {
      unownedTensor.unpaddedSize_ = unpaddedSize_;
      //@TODO check actualSize() == unownedTensor.actualSize()
    }
    else {
      unownedTensor.unpaddedSize_ = unownedTensor.type_.getSizeInBytes();
      //@TODO check getSizeInBytes() == getUnpaddedSizeInBytes()
      //@TODO check actualSize() >= unownedensor.actualSize()
    }

    return unownedTensor;
  }
  
  /* @brief copy raw data value at ptrData_ buffer tensor given at \p inTensor
   * to the other ptrData_ buffer at (this).
   *
   * @param[in] inTensor tensor from copy data
   */
  void copyRawFrom(LibTensor* inT) {
    //@TODO check both tensor has the same shape!!
    char* dstBuff = this->getData();
    char* startBuff = inT->getData();
    //std::copy(&inTensor->ptrdata_[0], &inTensor->ptrdata_[bufferSize], this->getHandleData());    
    for (unsigned long int i = 0; i <= inT->getSizeInBytes(); i++) {
      *(dstBuff++)= *(startBuff++);
    }
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
  char* getUnsafePtr() const { return getData(); }

  /*@brief Updates contents of a deice resident Tensor with the data from \p t
   *without copying its contents.
   */
  void copyRawToDevice(const LibTensor *t);


  /*debug purpose*/
  uint8_t pitches(dim_t* cpydims) { return type_.strides(cpydims); }
  
private:

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
  INLINE_ATTR size_t getFlattenedOffset(dim_t* indices, uint8_t indDim,
                                        dim_t* sizeIntegral, uint8_t numDims) {
    /*@TODO check indices size isn't bigger than strides*/
    size_t ndx = 0;
    for (size_t i = 0; i < indDim; i++) {
      ndx += size_t(sizeIntegral[i]) * size_t(indices[i]);
    }
    
    return ndx;
  }


template <class ElemTy, bool IsConst>
class HandleIterator
    : public std::iterator<std::random_access_iterator_tag, ElemTy> {
  using HandleTy = typename dnn_lib::conditional_t<IsConst, const Handle<ElemTy> *,
                                          Handle<ElemTy> *>;
  using ElemTyRef =
    typename dnn_lib::conditional_t<IsConst, const ElemTy &, ElemTy &>;

  HandleTy handle_;
  dim_t sizes_[max_tensor_dimensions] = {0,};
  uint8_t nSizes_;
  dim_t idx_;
  bool isAligned_;

  HandleIterator() = default;

  HandleIterator(HandleTy handle) : handle_(handle) {
    nSizes_ = handle->dims(sizes_);
    isAligned_ = handle->size() < handle->actualSize();
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
    dim_t indices[nSizes_] = {0,};
    size_t rem = idx_;
    for (int i = static_cast<int>(nSizes_) - 1; i >= 0; i--) {
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
   dim_t sizeIntegral_[max_tensor_dimensions] = { 0, };

   dim_t sizes_[max_tensor_dimensions] = { 0, };

  /*@brief the number of dimensions ussed in the tensor.
   */
   uint8_t numDims_{0};

 
   Handle() = default;

 public:


   /*@brief Allocate anew invalid handle.
    */
   static Handle createInvalidHandle() { return Handle(); }
  
   /*@brief returns true if this Handle points to a valid tensor.
    */
   bool isValid() const { return tensor_; }

   /*@brief Calculate the index for a specific element in the tensor.
    *
    *@param[inout] coords indices to access element. It has to have the same
    * dimensions as tensor to be acessed.
    *@return flattened 1D element position.
    */
  size_t getElementPtr(dim_t *indices, uint8_t indDims) {
    return getFlattenedOffset(indices, indDims, sizeIntegral_, numDims_);
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
   explicit Handle(LibTensor* tensor) : tensor_(tensor) {
      numDims_ = tensor->dims(sizes_);

      if (numDims_) {
        for(uint8_t i = 0; i < tensor_->type_.getNumDims(); i++) {
          sizes_[i] = tensor_->type_.sizes_[i];
          sizeIntegral_[i] = tensor_->type_.strides_[i];
        }
        //@TODO check numDims_ <= max_tensor_dimesnions
      }
   }

   /*@brief get a copy of dims_ internal dimension Tensor.
    *
    *@param[inout] dim array pointer to copy the tensor sizes_.
    *@return the numDims_ value. (the number of dims be copied).
    */
   uint8_t cpydims(dim_t* cpydim) {
     for(uint8_t i = 0; i < numDims_; i++) {
       cpydim[i] = sizes_[i];
     }
     return numDims_;
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
  ElemTy &at(dim_t *indices, uint8_t indDims) {
    size_t index = getElementPtr(indices, indDims);   
    auto *data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  const ElemTy &at(dim_t *indices, uint8_t indDims) const {
    size_t index = getElementPtr(indices, indDims);
    auto *data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  char* getPtrdbg(void) const {return tensor_->dbgData();}
  dim_t* getSizeIntdbg(void) {return sizeIntegral_;}
  dim_t* getSizesdbg(void) {return sizes_;}
  uint8_t getNumDimsdbg(void) {return numDims_;}
  LibTensor* getTensordbg(void) {return tensor_;}
  
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
