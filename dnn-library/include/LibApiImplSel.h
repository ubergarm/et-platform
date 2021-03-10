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

    // Best implementation selector for operator Copy. Return values are:
    //   0: base implementation (Vectorized)
    //   1: Tensorized 
    static size_t Copy(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      // Tensorized only works with same shape in-out and CL aligment
      if (inTensors[0]->getType().hasSameShape(outTensors[0]->getType()) and
          (((uintptr_t) inTensors[0]->getAddress()  & 0x3F) == 0) and ((inTensors[0]->getType().getSizeInBytes()  & 0x3F) == 0) and 
          (((uintptr_t) outTensors[0]->getAddress() & 0x3F) == 0) and ((outTensors[0]->getType().getSizeInBytes() & 0x3F) == 0)) {
        return 1;
      } else  {
	return 0;
      }
      
    }

    // Best implementation selector for operator ElementBool instructions. Return values are:
    //   0: base implementation (threaded)
    //   1: Vectorized 
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

    // Best implementation selector for operator MaxSplat. Return values are:
    //   0: base implementation (vectorized)
    //   1: Aligned32Bytes 
    static size_t MaxSplat(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      LibTensor *outT = outTensors[0];
      LibTensor *inT = inTensors[0];
      const static size_t batchDim = inT->ndims() - 2;
      if (inT->ndims() >= 2 && (uintptr_t)inT->getAddress() % 32 == 0 && (uintptr_t)outT->getAddress() % 32 == 0 &&
          (outT->strides()[batchDim] % 32 == 0 || 32 % outT->strides()[batchDim] == 0) &&
          (inT->strides()[batchDim] % 32 == 0 || 32 % inT->strides()[batchDim] == 0))
        return 1;
      else
        return 0;
    }



    // Best implementation selector for operator RowwiseQuantizedFullyConnected. Return values are:
    //   0: base implementation (Vectorized)
    //   1: Aligned32Bytes 
    static size_t RowwiseQuantizedFullyConnected(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      LibTensor *outT = outTensors[0];
      LibTensor *in1T = inTensors[0];
      LibTensor *in2T = inTensors[1];
    
      const static size_t batchDim = in1T->ndims() - 2;
      if ((uintptr_t)outT->getAddress() % 32 == 0 && (uintptr_t)in1T->getAddress() % 32 == 0 &&
          (uintptr_t)in2T->getAddress() % 32 == 0 && outT->strides()[batchDim] % 32 == 0 &&
          in1T->strides()[batchDim] % 32 == 0 && in2T->strides()[batchDim] % 32 == 0)
        return 1;
      else 
        return 0;
    
    }

  
    // Best implementation selector for operator RowwiseQuantizedSparseLengthsWeightedSum. Return values are:
    //   0: base implementation
    //   1: Vectorized 
    static size_t RowwiseQuantizedSparseLengthsWeightedSum(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      LibTensor *dataT = inTensors[0];
      // check for SW-3119
      if (dataT->dims()[dataT->ndims()-1] < 4)
        return 0;
      else
        return 1;
    }

    
    // Best implementation selector for operator Transpose. Return values are:
    //   0: base implementation (Vectorized)
    //   1: Aligned32Bytes
    static size_t Transpose(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      LibTensor *outT = outTensors[0];
      LibTensor *inT = inTensors[0];
      ElemKind elK = inTensors[0]->getElementType();
      const size_t batchDim = inT->ndims() - 2;
      if (inT->ndims() >= 2 && (uintptr_t)outT->getAddress() % 32 == 0 && (uintptr_t)inT->getAddress() % 32 == 0 &&
          (outT->strides()[batchDim] * outT->getElementSize()) % 32 == 0 &&
          (inT->strides()[batchDim] * inT->getElementSize()) % 32 == 0 &&
          (elK == FloatTy || elK == Float16Ty || elK == Int8QTy))
        return 1;
      else
        return 0;
    }
    
    // Best implementation selector for operator SoftMax. Return values are:
    //   0: base implementation
    //   1: Vectorized 
    static size_t SoftMax(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      LibTensor* outT = outTensors[0];
      unsigned cll = CACHE_LINE_BYTES / outT->getElementSize();
      const size_t numDims = outT->ndims();
      if ((uintptr_t)outT->getAddress() % CACHE_LINE_BYTES == 0 and numDims >= 2 and
          outT->strides()[numDims - 2] % cll == 0) {
        return 1;
      }
      return 0;
    }

    // Best implementation selector for operator LocalResponseNormalization. Return values are:
    //   0: base implementation (threaded)
    //   1: Vectorized 
    static size_t LocalResponseNormalization(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      return 1;
    }

    
    // Best implementation selector for operator SparseLengthsWeightedSum. Return values are:
    //   0: base implementation
    //   1: Threaded 
    static size_t SparseLengthsWeightedSum(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      return 0;
    }

    // Best implementation selector for operator InsertTensor. Return values are:
    //   0: base implementation
    //   1: Threaded 
    static size_t InsertTensor(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      // threaded version only works for CL aligned output tensors
      // otherwise, call the single thread version
      // checking size is multiple of 64B => assuming start is CL aligned
      if ((uintptr_t)outTensors[0]->getAddress() % 64 == 0 && outTensors[0]->getSizeInBytes() % CACHE_LINE_BYTES != 0) {
        return 0;
      } else {
        auto typeSize = outTensors[0]->getElementSize();
        auto cll = CACHE_LINE_BYTES/typeSize;
        auto &dstPitch = outTensors[0]->strides();
        auto dstDimNum = outTensors[0]->ndims();
        return ((dstDimNum >= 2) && (dstPitch[dstDimNum - 2]%cll != 0)) ? 0 : 1;
      }

      return 0;
    }
    
    // Best implementation selector for operator AvgPool. Return values are:
    //   0: base implementation
    //   1: Threaded 
    static size_t AvgPool(std::vector<LibTensor*> &outTensors, std::vector<LibTensor*> &inTensors){
      return (inTensors[0]->ndims() == 5) ? 0 : 1;
    }
    

  };
  
}

#endif
