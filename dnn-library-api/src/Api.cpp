/*-------------------------------------------------------------------------
 * Copyright (C) 2021, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#include "dnnLibraryApi/LibApi.h"

#include <cctype>
#include <cmath>

namespace dnn_lib {

operandState instrConfig::getOperandStateL1(size_t implIdx, size_t operand) {
  assert(operand < nrOutputTensors + nrInputTensors);
  return stateL1[implIdx][operand];
}
operandState instrConfig::getOperandStateL2(size_t implIdx, size_t operand) {
  assert(operand < nrOutputTensors + nrInputTensors);
  return stateL2[implIdx][operand];
}
operandState instrConfig::getOperandStateCB(size_t implIdx, size_t operand) {
  assert(operand < nrOutputTensors + nrInputTensors);
  return stateCB[implIdx][operand];
}
bool instrConfig::getOperandAutoEvict(size_t implIdx, size_t operand) {
  assert(operand < nrOutputTensors + nrInputTensors);
  return ((evictAvailableMask[implIdx] >> operand) & 1);
}
operandState instrConfig::getSrcStateL1(size_t implIdx, size_t idx) {
  return getOperandStateL1(implIdx, idx + nrOutputTensors);
}
operandState instrConfig::getSrcStateL2(size_t implIdx, size_t idx) {
  return getOperandStateL2(implIdx, idx + nrOutputTensors);
}
operandState instrConfig::getSrcStateCB(size_t implIdx, size_t idx) {
  return getOperandStateCB(implIdx, idx + nrOutputTensors);
}
bool instrConfig::getSrcAutoEvict(size_t implIdx, size_t idx) {
  return getOperandAutoEvict(implIdx, idx + nrOutputTensors);
}
operandState instrConfig::getDstStateL1(size_t implIdx, size_t idx) {
  assert(idx < nrOutputTensors);
  return getOperandStateL1(implIdx, idx);
}
operandState instrConfig::getDstStateL2(size_t implIdx, size_t idx) {
  assert(idx < nrOutputTensors);
  return getOperandStateL2(implIdx, idx);
}
operandState instrConfig::getDstStateCB(size_t implIdx, size_t idx) {
  assert(idx < nrOutputTensors);
  return getOperandStateCB(implIdx, idx);
}
bool instrConfig::getDstAutoEvict(size_t implIdx, size_t idx) {
  assert(idx < nrOutputTensors);
  return getOperandAutoEvict(implIdx, idx);
}

//////////////

bool caseInsCharCompare(char a, char b) {
  return (std::toupper(a) == std::toupper(b));
}

bool caseInsCompare(const std::string& s1, const std::string& s2) {
  return ((s1.size() == s2.size()) && equal(s1.begin(), s1.end(), s2.begin(), caseInsCharCompare));
}

bool getInstrConfig(const std::string& operatorName, instrConfig& instConfig) {
  for (auto it = instrConfigTable.begin(); it != instrConfigTable.end(); ++it) {
    if (caseInsCompare(it->name, operatorName)) {
      instConfig = *it;
      return true;
    }
  }
  return false;
}

size_t getInstrNumCycles(const std::string& operatorName, size_t assignedMinions,
                         const std::vector<LibTensor*>& operands) {
  size_t result = 2000;
  if (caseInsCompare("FusedRowwiseQuantizedSparseLengthsWeightedSum", operatorName)) {
    // Computes lines to read from DDR
    size_t lineSize = operands[0]->strides()[0];
    size_t lineCL = (size_t)ceilf((float)lineSize / 64.0f); // Converts to cachelines
    size_t lookups = operands[1]->dims()[0];
    // Data read from the embedding
    size_t totalBytes = lineCL * lookups * 64;
    // Data read to get lookup indices and the weight
    // TODO: need to get the lookup index size and weight size
    totalBytes += lookups * 8 + lookups * 4;

    // Cycles is amount of CL to read by cycles per CL
    // TODO: gather this info from device
    float ddrBytesPerCycle = 133.0f * 0.55f;
    float ddrBytesPerMinion = ddrBytesPerCycle / 1024.0f;
    float minions = (float)assignedMinions;
    float ddrCycles = (float)totalBytes / ddrBytesPerMinion / minions;

    // Total amount of time is fixed time plus ddr time
    result = 4000 + (size_t)ddrCycles;
  } else if (caseInsCompare("FusedRowwiseQuantizedSparseLengthsSum", operatorName)) {
    // Computes lines to read from DDR
    size_t lineSize = operands[0]->strides()[0];
    size_t lineCL = (size_t)ceilf((float)lineSize / 64.0f); // Converts to cachelines
    size_t lookups = operands[1]->dims()[0];
    // Data read from the embedding
    size_t totalBytes = lineCL * lookups * 64;
    // Data read to get lookup indices and the weight
    // TODO: need to get the lookup index size
    totalBytes += lookups * 8;

    // Cycles is amount of CL to read by cycles per CL
    // TODO: gather this info from device
    float ddrBytesPerCycle = 133.0f * 0.55f;
    float ddrBytesPerMinion = ddrBytesPerCycle / 1024.0f;
    float minions = (float)assignedMinions;
    float ddrCycles = (float)totalBytes / ddrBytesPerMinion / minions;

    // Total amount of time is fixed time plus ddr time
    result = 4000 + (size_t)ddrCycles;
  }

  return result;
}

} // end namespace dnn_lib
