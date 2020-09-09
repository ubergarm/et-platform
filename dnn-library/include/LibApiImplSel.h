#ifndef _LIB_API_IMPL_SEL_H_
#define _LIB_API_IMPL_SEL_H_

#include "LibTensor.h"

namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // default implementation selector
  ////////////////////////////////////////////////////////////////////////////////
  class implSel{
  public:
    template<int implCount>
    static size_t defaultSel(std::vector<LibTensor*>&, std::vector<LibTensor*>&) {
      return implCount -1 ;
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // custom implementation selectors
    ////////////////////////////////////////////////////////////////////////////////
    
    // Best implementation selector for operator ConvertTo. Return values are:
    //   0: base implementation (threaded)
    //   1: "Vectorized" 
    static size_t ConvertTo(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){

      LibTensor *inT = inTensors[0];
      ElemKind dstElK = outTensors[0]->getElementType();
      ElemKind srcElK = inTensors[0]->getElementType();
      if ((inT->dims()[inT->ndims()-1] == 1 && inT->strides()[0] != inT->stridesNoPadding()[0]) ||
          (srcElK == FloatTy  && dstElK == Int64ITy ) ||
          (srcElK == FloatTy  && dstElK == Int32ITy)  || 
          (srcElK == Int32ITy && dstElK == FloatTy)   ||
          (srcElK == Int32ITy && dstElK == Int64ITy)  ||
          (srcElK == Int32ITy && dstElK == Float16Ty) ||
          (srcElK == Int64ITy && dstElK == Float16Ty) ||
          (srcElK == Int64ITy && dstElK == Int32ITy)  || 
          (srcElK == BoolTy   && dstElK == Float16Ty) ||
          (srcElK == BoolTy   && dstElK == FloatTy)) {
        // check for SW-3726
        return 0;
      }
      else{
        return 1;
      }
    }
  
  
    // Best implementation selector for operator Convolution. Return values are:
    //   0: base implementation
    //   1: "Threaded"
    //   2: "Vectorized" 
    static size_t Convolution(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      // check for SW-3816
      LibTensor *filterT = inTensors[1];
      if (filterT->dims()[filterT->ndims()-1] < 4 ) return 1;
      else return 2;
    }
  
  
  
    // Best implementation selector for operator Copy. Return values are:
    //   0: base implementation
    //   1: "Threaded"
    //   2: "Vectorized"
    //   3: "Tensorized" 
    static size_t Copy(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      // Tensorized only works with same shape in-out and CL aligment
      if (inTensors[0]->getType().hasSameShape(outTensors[0]->getType()) and
          (((uintptr_t) inTensors[0]->getAddress()  & 0x3F) == 0) and ((inTensors[0]->getType().getSizeInBytes()  & 0x3F) == 0) and 
          (((uintptr_t) outTensors[0]->getAddress() & 0x3F) == 0) and ((outTensors[0]->getType().getSizeInBytes() & 0x3F) == 0)) {
        return 3;
      } else if (!outTensors[0]->getUntouchable()) {
        return 2;
      } else {
        return 0;
      }
    }
  

    // Best implementation selector for operator generic ElementBinary instructions. Return values are:
    //   0: base implementation (threaded)
    //   1: "Vectorized" 
    static size_t ElementBinary(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      ElemKind dstElK = outTensors[0]->getElementType();
      LibTensor *in1T = inTensors[0];
      LibTensor *in2T = inTensors[1];
      if ( (dstElK == FloatTy || dstElK == Float16Ty || dstElK == Int8QTy) &&
           in1T->strides()[0] == in2T->strides()[0] )
        return 1;
      else
        return 0;
    }

    static size_t ElementAdd(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors) {
      return ElementBinary(outTensors, inTensors);
    }
    static size_t ElementSub(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors) {
      return ElementBinary(outTensors, inTensors);
    }
    static size_t ElementDiv(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors) {
      return ElementBinary(outTensors, inTensors);
    }
    static size_t ElementMax(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors) {
      return ElementBinary(outTensors, inTensors);
    }
    static size_t ElementMin(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors) {
      return ElementBinary(outTensors, inTensors);
    }
    static size_t ElementMul(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors) {
      return ElementBinary(outTensors, inTensors);
    }
    static size_t ElementPow(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors) {
      return ElementBinary(outTensors, inTensors);
    }




    // Best implementation selector for operator ElementBool instructions. Return values are:
    //   0: base implementation (threaded)
    //   1: "Vectorized" 
    static size_t ElementBool(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      ElemKind src1ElK = inTensors[0]->getElementType();
      LibTensor *in1T = inTensors[0];
      LibTensor *in2T = inTensors[1];
      if ( (src1ElK == FloatTy || src1ElK == Float16Ty || src1ElK == Int8QTy) && 
           in1T->strides()[0] == in2T->strides()[0] )
        return 1;
      else
        return 0;
    }
    
    
    static size_t ElementCmpEQ(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors) {
      return ElementBool(outTensors, inTensors);
    }
    static size_t ElementCmpLTE(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors) {
      return ElementBool(outTensors, inTensors);
    }
    static size_t ElementCmpLT(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors) {
      return ElementBool(outTensors, inTensors);
    }



    // Best implementation selector for operator EmbeddingBag. Return values are:
    //   0: base implementation
    //   1: "Vectorized" 
    static size_t EmbeddingBag(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
#ifdef  SW_3755
      if (outTensors[0]->getUntouchable())
        return 0;
      else
#endif
        return 1;
    }



    // Best implementation selector for operator MaxSplat. Return values are:
    //   0: base implementation
    //   1: "Threaded"
    //   2: "Vectorized"
    //   3: "Aligned32Bytes" 
    static size_t MaxSplat(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      LibTensor *outT = outTensors[0];
      LibTensor *inT = inTensors[0];
      const static size_t batchDim = inT->ndims() - 2;
      if (inT->ndims() >= 2 &&
          ( outT->strides()[batchDim] % 32 == 0 ||  32 % outT->strides()[batchDim] == 0 ) &&
          (  inT->strides()[batchDim] % 32 == 0 ||  32 %  inT->strides()[batchDim] == 0 ))
        return 3;
      else
        return 2;
    }



    // Best implementation selector for operator RowwiseQuantizedFullyConnected. Return values are:
    //   0: base implementation
    //   1: "Threaded"
    //   2: "Vectorized"
    //   3: "Aligned32Bytes" 
    static size_t RowwiseQuantizedFullyConnected(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      LibTensor *outT = outTensors[0];
      LibTensor *in1T = inTensors[0];
      LibTensor *in2T = inTensors[1];
    
      const static size_t batchDim = in1T->ndims() - 2;
      if ( outT->strides()[batchDim] % 32 == 0 && 
           in1T->strides()[batchDim] % 32 == 0 &&
           in2T->strides()[batchDim] % 32 == 0 )
        return 3;
      else 
        return 2;
    
    }

  
    // Best implementation selector for operator RowwiseQuantizedSparseLengthsWeightedSum. Return values are:
    //   0: base implementation
    //   1: "Threaded"
    //   2: "Vectorized" 
    static size_t RowwiseQuantizedSparseLengthsWeightedSum(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      LibTensor *dataT = inTensors[0];
      // check for SW-3119
      if (dataT->dims()[dataT->ndims()-1] < 4)
        return 0;
      else
        return 2;
    }

    
    // Best implementation selector for operator Transpose. Return values are:
    //   0: base implementation
    //   1: "Threaded"
    //   2: "Vectorized"
    //   3: "Aligned32Bytes"
    static size_t Transpose(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      LibTensor *outT = outTensors[0];
      LibTensor *inT = inTensors[0];
      ElemKind elK = inTensors[0]->getElementType();
      const size_t batchDim = inT->ndims() - 2;
      if ( inT->ndims() >= 2 &&
           (outT->strides()[batchDim] * outT->getElementSize() )% 32 == 0  &&
           (inT->strides()[batchDim] * inT->getElementSize() ) % 32 == 0  &&
           (elK == FloatTy || elK == Float16Ty || elK==Int8QTy ) )
        return 3;
      else
        return 2;
  }
    
  };
  
}

#endif
