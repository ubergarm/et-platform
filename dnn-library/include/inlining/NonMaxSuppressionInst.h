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

#ifndef _NON_MAX_SUPPRESSION_H_
#define _NON_MAX_SUPPRESSION_H_

#include <limits>
#include <assert.h>
#include <queue>
#include <utility>

#include "utils.h"
#include "LibTypes.h"
#include "LibTensor.h"
#include "LibCommon.h"

namespace dnn_lib {

namespace inlining {

//numBoxes set to const number 64
#define MAX_CUSTOM_ARRAY_SIZE 64

using ClassBox = std::pair<float, size_t>;

struct Box {
  float classValue{0.0f};
  size_t batchIndex{0};
  size_t classIndex{0};
  size_t boxIndex{0};
};

template <typename T, size_t MAX>
 class CustomFifo {
 private:
  size_t rd_ptr_ = 0;
  size_t wr_ptr_ = 0;
  size_t count_ = 0;
  std::array<T, MAX> data_;

  template<typename func_t>
  void makeRoom(int &pos, ClassBox nbox, func_t fnc) {
    
    if((fnc(nbox, data_[pos]) == true) && (static_cast<unsigned int>(pos) < data_.size()) && (static_cast<unsigned int>(pos) < count_)) {
      this->makeRoom(++pos, nbox, fnc);
    }
    else {
      if ((count_ +1) < MAX) {
        //move element of array one position to the right
	for(int i = count_; i > pos; i--) {
	    data_[i] = data_[i-1];
	}
	wr_ptr_++;
      } 
      else
	assert(true && "Max CustomArray capacity reached.");
    } 
  }

  void fill(T element) {
    data_.fill(element);
  }    

 public:

  CustomFifo() {
    rd_ptr_ = 0;
    wr_ptr_ = 0;
    count_ = 0;
  }

  void push(T &v) {
    assert(!full());
    count_++;
    data_[wr_ptr_++] = v;
  }

  void pushAt(size_t ndx, T &v) {
    assert(!full());
    count_++;
    data_[ndx] = v;
    wr_ptr_++;
  }

  T pop(void) {
    assert(!empty());
    count_--;
    return data_[rd_ptr_++];
  }

  void clear(T element) {
    this->fill(element);
    count_ = 0;
    rd_ptr_ = 0;
    wr_ptr_ = 0;
  }

  bool empty(void) { return count_ == 0; } 
  bool full(void) { return count_ == MAX; }
  const T front() { return data_.front();}
  const T back() {return data_.back();}
  const auto begin() { return data_.begin();}
  const auto end() { return data_.end();}

  std::array<T, MAX> &data() { return data_;}
  size_t countElem() { return count_;}
 
  template<typename func_t>
  void pushConditional(float p1, int p2, func_t fnc) {

    if (this->empty()) {
      ClassBox nbx = {p1,p2};
      this->push(nbx);
    }
    else {
      ClassBox newbox = {p1,p2};
      int tmp_ndx = 0;
      this->makeRoom(tmp_ndx, newbox, fnc);
      this->pushAt(tmp_ndx, newbox);
    }
  }
};


static void maxMin(float lhs, float rhs, float &min, float &max) {
  if (lhs >= rhs) {
    min = rhs;
    max = lhs;
  } else {
    min = lhs;
    max = rhs;
  }
}

template <typename ElemTy>
static bool doIOU(Handle<ElemTy> &boxes, dim_t batchIndex,
                  dim_t selectedBoxIndex, dim_t candidateBoxIndex,
                  int centerPointBox, float iouThreshold, bool isV4) {

  float sx[] = {0.0f, 0.0f, 0.0f, 0.0f};
  float cx[] = {0.0f, 0.0f, 0.0f, 0.0f};
  
  if (isV4) {
    for (size_t i = 0; i < 4; i++) {
      sx[i] = boxes.at(std::array<size_t, 2>{selectedBoxIndex, i});
      cx[i] = boxes.at(std::array<size_t, 2>{candidateBoxIndex, i});
    }
  }
  else {
    for (size_t i = 0; i < 4; i++) {
      sx[i] = boxes.at(std::array<size_t, 3>{batchIndex, selectedBoxIndex, i});
      cx[i] = boxes.at(std::array<size_t, 3>{batchIndex, candidateBoxIndex, i});
    }
  }
 
  float xSMin = 0.0f;
  float ySMin = 0.0f;
  float xSMax = 0.0f;
  float ySMax = 0.0f;

  float xCMin = 0.0f;
  float yCMin = 0.0f;
  float xCMax = 0.0f;
  float yCMax = 0.0f;
 
  // Standardizing coordinates so that (xmin, ymin) is upper left corner of a
  // box and (xmax, ymax) is lower right corner of the box.
  if (!centerPointBox) {
    // 0 means coordinates for diagonal ends of a box.
    // Coordinates can either be absolute or normalized.
    maxMin(sx[0], sx[2], xSMin, xSMax);
    maxMin(sx[1], sx[3], ySMin, ySMax);

    maxMin(cx[0], cx[2], xCMin, xCMax);
    maxMin(cx[1], cx[3], yCMin, yCMax);
  } else {
    float onedbytwo;
    fpReciprocalSingleElement(2.0, onedbytwo);

    float halfWidthS = sx[2]  * onedbytwo;
    float halfHeightS = sx[3] * onedbytwo;
    float halfWidthC = cx[2]  * onedbytwo;
    float halfHeightC = cx[3] * onedbytwo;

    xSMin = sx[0] - halfWidthS;
    ySMin = sx[1] - halfHeightS;
    xSMax = sx[0] + halfWidthS;
    ySMax = sx[1] + halfHeightS;

    xCMin = cx[0] - halfWidthC;
    yCMin = cx[1] - halfHeightC;
    xCMax = cx[0] + halfWidthC;
    yCMax = cx[1] + halfHeightC;
  }

  // finding upper left and lower right corner of a box formed by intersection.
  float xMin = std::max(xSMin, xCMin);
  float yMin = std::max(ySMin, yCMin);
  float xMax = std::min(xSMax, xCMax);
  float yMax = std::min(ySMax, yCMax);

  float intersectionArea =
      std::max(0.0f, xMax - xMin) * std::max(0.0f, yMax - yMin);

  if (intersectionArea == 0.0f) {
    return false;
  }

  float sArea = (xSMax - xSMin) * (ySMax - ySMin);
  float cArea = (xCMax - xCMin) * (yCMax - yCMin);
  float unionArea = sArea + cArea - intersectionArea;

  return intersectionArea > iouThreshold * unionArea;
}


/**
 * @brief pruning away boxes that have high intersection-over-union (IOU) 
 * overlap with previously selected boxes. Bounding boxes are supplied 
 * as [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates 
 * of any diagonal pair of box corners and the coordinates can be provided 
 * as normalized (i.e., lying in the interval [0, 1]) or absolute. 
 * Note that this algorithm is agnostic to where the origin is in the 
 * coordinate system.
 * The output of this operation is a set of integers indexing into the 
 * input collection of bounding boxes representing the selected boxes.
 *
 * @tparam Elemkind the kind of the element which hast to be resolved.
 * @param[out] indicesT LibTensor destination. It holds the expected.
 * @param[out] numOfSelIndT LibTensor destination. It holds the expected.
 * @param[in] boxesT LibTensor input. It keeps the inputs to being handle.
 * @param[in] scoresT LibTensor input. It keeps the inputs to being handle.
 * @param[in] CenterPointBox 
 * @param[in] MaxOutputBoxesPerClass
 * @param[in] IouThreshold
 * @param[in] isTFVersion
 * @param[flags] flags Gives the information of the Active Shires and the
 * type of evict required.
 */
 template <ElemKind srcElk>
void fwdLibNonMaxSuppressionInst(LibTensor* indicesT, LibTensor* numOfSelIndT, 
                                 LibTensor* boxesT, LibTensor* scoresT, 
                                 const int64_t centerPointBox, 
                                 const int64_t maxOutputBoxesPerClass, 
                                 const float iouThreshold, 
                                 const float scoreThreshold, 
                                 const bool isTFVersion4, uint64_t flags, 
                                 const uint32_t minionOffset = 0, 
                                 const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(indicesT->getElementType() == numOfSelIndT->getElementType());
  assert(srcElk == Int32ITy || srcElk == Int64ITy);
  assert(boxesT->getElementType() == scoresT->getElementType());
  assert(boxesT->getElementType() == FloatTy);

  using srcType = typename elemKind2elemTy<srcElk>::type; 

  auto indicesH = indicesT->getHandle<srcType>();
  auto numOfSelIndH = numOfSelIndT->getHandle<srcType>();
  auto boxesH = boxesT->getHandle<float>();
  auto scoresH = scoresT->getHandle<float>();

  int boxesBoxDim = boxesT->ndims() - 2;

  size_t numBatches = 1;
  size_t numClasses = 1;
  size_t numBoxes = boxesT->dims()[boxesBoxDim];

  size_t maxOutputPerBatch = 0;

  if (!isTFVersion4) {
    ssize_t boxesBatchDim = boxesT->ndims() - 3;

    ssize_t scoresBatchDim = scoresT->ndims() - 3;
    ssize_t scoresBoxDim = scoresT->dims().size() -1;
    ssize_t scoresClassDim = scoresT->ndims() - 2;
    assert(scoresT->dims()[scoresBoxDim] == boxesT->dims()[boxesBoxDim]);
    assert(scoresT->dims()[scoresBatchDim] == boxesT->dims()[boxesBatchDim]);

    (void)boxesBatchDim;
    (void)scoresBoxDim;
    numBatches = scoresT->dims()[scoresBatchDim];
    numClasses = scoresT->dims()[scoresClassDim];
    numBoxes = boxesT->dims()[boxesBoxDim];
    maxOutputPerBatch = indicesT->dims()[(indicesT->ndims() - 2)] / numBatches;
  }
  else {
    maxOutputPerBatch = indicesT->dims()[(indicesT->ndims() - 1)] / numBatches;
  }

  ClassBox cbZero(0.0f,0); //for init std::array

  CustomFifo<ClassBox, MAX_CUSTOM_ARRAY_SIZE> selectedIndices;

  //std::vector<ClassBox> selectedIndices;
  size_t outPutBoxIndex = 0;

  for (size_t batchIndex = 0; batchIndex < numBatches; ++batchIndex) {
    Box minBox{scoresH.raw(batchIndex * numClasses * numBoxes), batchIndex, 0, 0};
    int32_t detectedPerBatch = 0;
    for (size_t classIndex = 0; classIndex < numClasses; ++classIndex) {
      /* selectedIndices.fill(cbZero);  */
      selectedIndices.clear(cbZero);
      ssize_t detectedPerClass = 0;
      //std::priority_queue<ClassBox,std::vector<ClassBox>, 64>, decltype(cmpFunc)> queue(cmpFunc);
      CustomFifo<ClassBox, MAX_CUSTOM_ARRAY_SIZE> fifoArray;
      
      for (size_t boxIndex = 0; boxIndex < numBoxes; ++boxIndex) {
	
	size_t position = ((batchIndex * numClasses + classIndex) * numBoxes + boxIndex);

	size_t pos0stride = (position / (scoresT->dims()[1] * scoresT->dims()[2])) * scoresT->strides()[0];
	size_t pos1stride = (position % (scoresT->dims()[1] * scoresT->dims()[2]))/scoresT->dims()[2] * scoresT->strides()[1];
	size_t pos2stride = (position % scoresT->dims()[2]) * scoresT->strides()[2];

	float classValue = scoresH.raw(pos0stride + pos1stride + pos2stride);

        if (classValue > scoreThreshold) {
          fifoArray.pushConditional(classValue, boxIndex, [](const ClassBox &a, const ClassBox &b) {
              return a.first <= b.first;});
	}
      }

      float tScore = minBox.classValue;
      while (!fifoArray.empty()) {
        auto priorBox = fifoArray.pop();
        bool selected = true;
	//        for (auto &sBox : selectedIndices) {
	for(size_t i = 0; i < selectedIndices.countElem(); i++) {
	  auto sBox = selectedIndices.data()[i];

          if (doIOU(boxesH, batchIndex, sBox.second, priorBox.second,
                    centerPointBox, iouThreshold, isTFVersion4)) {
            selected = false;
            break;
          }
	}

        if (selected) {
          selectedIndices.push(priorBox);  

          if (isTFVersion4) {
            indicesH.at(std::array<size_t,1>{outPutBoxIndex}) = priorBox.second;
            tScore = scoresH.at(std::array<size_t,1>{priorBox.second});
          }
          else {
            indicesH.at(std::array<size_t,2>{outPutBoxIndex, 0}) = static_cast<uint32_t>(batchIndex);
            /* convert from float to uint64 has to pas through 32bit conversion*/
            indicesH.at(std::array<size_t,2>{outPutBoxIndex, 1}) = static_cast<uint32_t>(classIndex);
            indicesH.at(std::array<size_t,2>{outPutBoxIndex, 2}) = static_cast<uint32_t>(priorBox.second);
            tScore = scoresH.at(std::array<size_t,3>{batchIndex, classIndex, priorBox.second});
          }
          
          ++outPutBoxIndex;
          ++detectedPerClass;
          ++detectedPerBatch;
        }
        if (maxOutputBoxesPerClass == detectedPerClass) {
          break;
        }
      }
    
      if (tScore < minBox.classValue) {
        minBox.classValue = tScore;
        minBox.classIndex = classIndex;
        if (isTFVersion4) {
          minBox.boxIndex = indicesH.at(std::array<size_t, 1>{outPutBoxIndex -1});
        }
        else {
          minBox.boxIndex = static_cast<uint32_t>(indicesH.at(std::array<size_t, 2>{outPutBoxIndex -1,2}));
        }
      }
    }
  
    for (size_t i = detectedPerBatch; i < maxOutputPerBatch; i++) {
      if (isTFVersion4) {
        indicesH.at(std::array<size_t,1>{outPutBoxIndex}) = minBox.boxIndex;
      }
      else {
        indicesH.at(std::array<size_t,2>{outPutBoxIndex, 0}) = minBox.batchIndex;
        indicesH.at(std::array<size_t,2>{outPutBoxIndex, 1}) = minBox.classIndex;
        indicesH.at(std::array<size_t,2>{outPutBoxIndex, 2}) = minBox.boxIndex;
      }
    
      ++outPutBoxIndex;
    }

    for(ssize_t i = 0; i < maxOutputBoxesPerClass; ++i) {
      numOfSelIndH.at(std::array<size_t,1>{batchIndex * maxOutputBoxesPerClass + i}) = detectedPerBatch;
    }
  }

  // and evict if need be
  indicesT->evict(DO_EVICTS);
  numOfSelIndT->evict(DO_EVICTS);
 }

} //inlining
} //dnn_lib

#endif
