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

#include "utils.h"
#include "LibTypes.h"
#include "LibTensor.h"
#include "LibCommon.h"

namespace dnn_lib {

namespace inlining {

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
 * @param[out] outT LibTensor destination. It holds the expected.
 * @param[out] outT LibTensor destination. It holds the expected.
 * @param[in] inT LibTensor input. It keeps the inputs to being handle.
 * @param[in] inT LibTensor input. It keeps the inputs to being handle.
 * @param[in] CenterPointBox 
 * @param[in] MaxOutputBoxesPerClass
 * @param[in] IouThreshold
 * @param[in] isTFVersion
 * @param[flags] flags Gives the information of the Active Shires and the
 * type of evict required.
 */


static void maxMin(float lhs, float rhs, float &min, float &max) {
  if (lhs >= rhs) {
    min = rhs;
    max = lhs;
  } else {
    min = lhs;
    max = rhs;
  }
}

using ClassBox = std::pair<float, size_t>;

struct Box {
  float classValue{0.0f};
  size_t batchIndex{0};
  size_t classIndex{0};
  size_t boxIndex{0};
};

template <typename ElemTy>
static bool doIOU(Handle<ElemTy> &boxes, dim_t batchIndex,
		  dim_t selectedBoxIndex, dim_t candidateBoxIndex,
		  int centerPointBox, float iouThreshold, bool isV4) {

  float sx[] = {0.0f, 0.0f, 0.0f, 0.0f};
  float cx[] = {0.0f, 0.0f, 0.0f, 0.0f};
  
  if (isV4) {
    for (size_t i = 0; i < 4; i++) {
      sx[i] = boxes.at(std::array<size_t, 2>{selectedBoxIndex, i});
      cx[i] = boxes.at(std::array<size_t, 2>{candidatedBoxIndex, i});
    }
  }
  else {
    for (size_t i = 0; i < 4; i++) {
      sx[i] = boxes.at(std::array<size_t, 3>{batchIndex, selectedBoxIndex, i});
      cx[i] = boxes.at(std::array<size_t, 3>{batchIndex, candidatedBoxIndex, i});
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
    float halfWidthS = sx[2] / 2.0f;
    float halfHeightS = sx[3] / 2.0f;
    float halfWidthC = cx[2] / 2.0f;
    float halfHeightC = cx[3] / 2.0f;

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


template <ElemKind elKout>
void fwdLibNonMaxSuppressionInst(LibTensor* indicesT, LibTensor* numOfSelIndT, LibTensor* boxesT, 
			      LibTensor* scoresT, const int64_t CenterPointBox, 
			      const int64_t MaxOutputBoxesPerClass, 
			      const float IouThreshold, 
			      const float ScoreThreshold, 
			      const bool IsTFVersion4, uint64_t flags, 
			      const uint32_t minionOffset = 0, 
			      const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(indicesT->getElementType() == numOfSelIndT->getElementType());
  assert(elKout == Int32ITy || elKout == Int64ITy);
  assert(boxesT->getElementType() == scoresT->getElementType());
  assert(boxesT->getElementType() == FloatTy);

  using elkTypeOut = typename elemKind2elemTy<elKout>::type; 

  auto indicesH = indicesT->getHandle<elkTypeOut>();
  auto numOfSelIndH = numOfSelIndT->getHandle<elkTypeOut>();
  auto boxesH = boxesT->getHandle<float>();
  auto scoresH = scoresT->getHandle<float>();

  int boxesBoxDim = boxesT->ndims() - 2;

  size_t numBatches = 1;
  size_t numClasses = 1;
  size_t numBoxes = boxes->dims()[boxesBoxDim];

  if (!isTFVersion4) {
    ssize_t boxesBatchDim = boxes->ndims() - 3;
    ssize_t scoresBatchDim = scores->ndims() - 1;
    ssize_t socresClassDim = scores->ndims() - 2;

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

  auto cmpFunc = [](const ClassBox &a, const ClassBox &b) {
    return a.first < b.first;
  };

  std::array<ClassBox,numBoxes> selectedIndices;
  size_t outPutBoxIndex = 0;

  for (size_t batchIndex = 0; batchIndex < numBatches; ++batchIndex) {
    Box minBox{scoresH.raw(batchIndex * numClasses * numBoxes), batchIndex, 0, 0};
    int32_t detectedPerBatch = 0;
    for (size_t classIndex = 0; classIndex < numClasses; ++classIndex) {
      selectedIndices.clear();
      size_t detectedPerClass = 0;
      //std::priority_queue<ClassBox, std

      for (size_t boxIndex = 0; boxIndex < numBoxes; ++boxIndex) {
	float classValue = scoresH.raw((batchIndex * numClasses + classIndes) * numBoxes + boxIndex);
	if (classValue > scoreThreshold) 
	  queue.emplace(classValue, boxIndex);
      }

      float tScore = minBox.classValue;
      while (!queue.empty()) {
	auto priorBox = queue.top();
	queue.pop();

	bool selected = true;
	for (auto &sBox : selectedIndices) {
	  if (doIOU(boxesH, batchIndex, sBox.second, priorBox.second,
		    centerPointBox, iouThreshold, isTFVersion4)) {
	    selected = false;
	    break;
	  }
	}
	
	if (selected) {
	  selectedIndices.emplace_back(priorBox);
	  if (isTFVersion4) {
	    indicesH.at(std::array<size_t,1>{priorBox.second}) = priorBox.second;
	    tScore = scoresH.at(std::array<size_t,1>{priorBox.second});
	  }
	  else {
	    indicesH.at(std::array<size_t,2>{outPutBoxIndex, 0}) = batchIndex;
	    indicesH.at(std::array<size_t,2>(outPutBoxIndex, 1}) = classIndex;
	    indicesH.at(std::array<size_t,2>(outPutBoxIndex, 2}) = priorBox.second;
	    tScore = scoresH.at(std::array<size_t,3>{batchIndex, classIndex, priorBox.second});
	  }
	  
	  ++outPutBoxIndex;
	  ++detectedPerClass;
	  ++detectedperBatch;
	}
	if (maxBoxesPerClass == detectedPerClass) {
	  break;
	}
      }

      if (tScore < minBox.classValue) {
	minBox.classValue = tScore;
	minBox.classIndex = classIndex;
	if (isTFVersion4) {
	  indicesH.at(std::array<size_t,1>{outPutBoxIndex}) = minBox.boxIndex;
	}
	else {
	  indicesH.at(std::array<size_t,1>{outPutBoxIndex, 0}) = minBox.batchIndex;
	  indicesH.at(std::array<size_t,1>{outPutBoxIndex, 1}) = minBox.classIndex;
	  indicesH.at(std::array<size_t,1>{outPutBoxIndex, 2}) = minBox.boxIndex;
	}
	
	++outPutBoxIndex;
      }

      for(dim_t i = 0; i < maxBoxesPerClass; ++i) {
	numDetectedH.at(std::array<size_t,1>{batchIndex * maxBoxesPerClass + i}) = detectePerBatch;
      }
    }
  }

}

} //inlining
} //dnn_lib

#endif
