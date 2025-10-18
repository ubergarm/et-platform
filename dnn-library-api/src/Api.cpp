/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

// Local
#include "LibApiImplSel.h"
#include "LibTensor.h"
#include "dnnLibraryApi/LibApi.h"

// STD
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dnn_lib {

// Enum with list of known members
#define SCALAR_MB_DEF(NAME, TYPE) mb##NAME,
#define VECTOR_MB_DEF(NAME, TYPE) mb##NAME,
enum class instrMembers {
  mbInvalid = 0,
#include "LibApiMembers.def"
  mbMaxMembers
};

// Instruction properties. This is the internal implementation. This needs to match
// dnn_lib::libManager.py automatically generated initialization.
struct InstrConfigInt {
  using implStateVector = std::vector<std::vector<operandState>>;
  using selFnc = size_t (*)(std::vector<LibTensor*>&, std::vector<LibTensor*>&);

  std::string name;
  size_t nrOutputTensors; // number of output and in/out tensor operands
  size_t nrInputTensors;  // number of input tensor operands
  std::vector<instrMembers> members;
  uint64_t templateMask;
  std::vector<std::string> versions;
  selFnc implSel;

  implStateVector stateL1;
  implStateVector stateL2;
  implStateVector stateCB;
  std::vector<uint64_t> evictAvailableMask;
  std::vector<uint64_t> dstGlobalStore;
};

// The instruction table contains all the instructions available in the DNN Library
#include <dnn_lib/InstrTableGenerated.h>

operandState instrConfig::getOperandStateL1(size_t implIdx, size_t operand) {
  assert(operand < (nrOutputTensors + nrInputTensors));
  assert(implIdx < stateL1.size());
  return stateL1[implIdx][operand];
}

operandState instrConfig::getOperandStateL2(size_t implIdx, size_t operand) {
  assert(operand < (nrOutputTensors + nrInputTensors));
  assert(implIdx < stateL2.size());
  return stateL2[implIdx][operand];
}

operandState instrConfig::getOperandStateCB(size_t implIdx, size_t operand) {
  assert(operand < (nrOutputTensors + nrInputTensors));
  assert(implIdx < stateCB.size());
  return stateCB[implIdx][operand];
}

bool instrConfig::getOperandAutoEvict(size_t implIdx, size_t operand) {
  assert(operand < (nrOutputTensors + nrInputTensors));
  assert(implIdx < evictAvailableMask.size());
  return ((evictAvailableMask[implIdx] >> operand) & 1);
}

operandState instrConfig::getSrcStateL1(size_t implIdx, size_t idx) {
  assert(idx < nrInputTensors);
  return getOperandStateL1(implIdx, idx + nrOutputTensors);
}

operandState instrConfig::getSrcStateL2(size_t implIdx, size_t idx) {
  assert(idx < nrInputTensors);
  return getOperandStateL2(implIdx, idx + nrOutputTensors);
}

operandState instrConfig::getSrcStateCB(size_t implIdx, size_t idx) {
  assert(idx < nrInputTensors);
  return getOperandStateCB(implIdx, idx + nrOutputTensors);
}

bool instrConfig::getSrcAutoEvict(size_t implIdx, size_t idx) {
  assert(idx < nrInputTensors);
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

bool instrConfig::getDstGlobalStore(size_t implIdx, size_t idx) {
  assert(idx < nrOutputTensors);
  assert(implIdx < dstGlobalStore.size());
  return ((dstGlobalStore[implIdx] >> idx) & 1);
}

uint64_t instrConfig::getDstGlobalStore(size_t implIdx) {
  assert(implIdx < dstGlobalStore.size());
  return dstGlobalStore[implIdx];
}

bool caseInsCharCompare(char a, char b) {
  return (std::toupper(a) == std::toupper(b));
}

bool caseInsCompare(const std::string& s1, const std::string& s2) {
  return ((s1.size() == s2.size()) && equal(s1.begin(), s1.end(), s2.begin(), caseInsCharCompare));
}

bool getInstrConfig(const std::string& operatorName, InstrConfigInt& instConfig) {
  // Linear access through all elements
  auto it = std::find_if(instrConfigTable.begin(), instrConfigTable.end(),
                         [&](auto& e) { return caseInsCompare(e.name, operatorName); });
  if (it != instrConfigTable.end()) {
    instConfig = *it;
    return true;
  }
  return false;
}

bool getInstrConfig(const std::string& operatorName, instrConfig& instConfig) {
  InstrConfigInt instConfigInt;

  // Looks for the internal config
  if (!getInstrConfig(operatorName, instConfigInt)) {
    return false;
  }

  // Converts to the API config
  instConfig.name = instConfigInt.name;
  instConfig.nrOutputTensors = instConfigInt.nrOutputTensors;
  instConfig.nrInputTensors = instConfigInt.nrInputTensors;
  for (auto member : instConfigInt.members) {
    instConfig.members.push_back((InstrMembers)member);
  }
  instConfig.templateMask = instConfigInt.templateMask;
  instConfig.versions = instConfigInt.versions;
  instConfig.stateL1 = instConfigInt.stateL1;
  instConfig.stateL2 = instConfigInt.stateL2;
  instConfig.stateCB = instConfigInt.stateCB;
  instConfig.evictAvailableMask = instConfigInt.evictAvailableMask;
  instConfig.dstGlobalStore = instConfigInt.dstGlobalStore;

  return true;
}

size_t getInstrNumCycles(const std::string& operatorName, size_t assignedMinions, const std::vector<Tensor>& operands) {
  size_t result = 2000;
  if (caseInsCompare("FusedRowwiseQuantizedSparseLengthsWeightedSum", operatorName)) {
    // Computes lines to read from DDR
    size_t lineSize = operands[0].strides[0];
    size_t lineCL = (size_t)ceilf((float)lineSize / 64.0f); // Converts to cachelines
    size_t lookups = operands[1].sizes[0];
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
    size_t lineSize = operands[0].strides[0];
    size_t lineCL = (size_t)ceilf((float)lineSize / 64.0f); // Converts to cachelines
    size_t lookups = operands[1].sizes[0];
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

size_t getImplementation(const std::string& operatorName, const std::vector<Tensor>& outOperands,
                         const std::vector<Tensor>& inOperands) {
  InstrConfigInt instConfig;
  // If the operator doesn't exist, return a negative
  if (!getInstrConfig(operatorName, instConfig)) {
    return (size_t)-1;
  }

  // Converts the Tensor to LibTensor
  std::vector<std::unique_ptr<LibTensor>> outOperandsConverted, inOperandsConverted;
  std::vector<LibTensor*> outOperandsConvertedPtr, inOperandsConvertedPtr;

  for (auto& operand : outOperands) {
    auto libTensor = std::make_unique<dnn_lib::LibTensor>(operand);
    outOperandsConvertedPtr.push_back(libTensor.get());
    outOperandsConverted.push_back(std::move(libTensor));
  }
  for (auto& operand : inOperands) {
    auto libTensor = std::make_unique<dnn_lib::LibTensor>(operand);
    inOperandsConvertedPtr.push_back(libTensor.get());
    inOperandsConverted.push_back(std::move(libTensor));
  }
  return instConfig.implSel(outOperandsConvertedPtr, inOperandsConvertedPtr);
}

std::string getMemberName(InstrMembers mb) {
// clang-format off
#define SCALAR_MB_DEF(NAME, TYPE) \
  case instrMembers::mb##NAME:    \
    ret = #NAME;                  \
    break;
#define VECTOR_MB_DEF(NAME, TYPE) \
  case instrMembers::mb##NAME:    \
    ret = #NAME;                  \
    break;
  // clang-format on

  std::string ret = "";
  auto mbInt = (instrMembers)mb;
  switch (mbInt) {
#include "LibApiMembers.def"

  default:
    assert(false && "invalid member");
  }
  return ret;
}

std::string getMemberType(InstrMembers mb) {
// clang-format off
#define SCALAR_MB_DEF(NAME, TYPE) \
  case instrMembers::mb##NAME:    \
    ret = #TYPE;                  \
    break;
#define VECTOR_MB_DEF(NAME, TYPE) \
  case instrMembers::mb##NAME:    \
    ret = #TYPE;                  \
    break;
  // clang-format on

  std::string ret = "";
  auto mbInt = (instrMembers)mb;
  switch (mbInt) {
#include "LibApiMembers.def"

  default:
    assert(false && "invalid member");
  }
  return ret;
}

bool getMemberScalar(InstrMembers mb) {
// clang-format off
#define SCALAR_MB_DEF(NAME, TYPE) \
  case instrMembers::mb##NAME:    \
    ret = true;                   \
    break;
#define VECTOR_MB_DEF(NAME, TYPE) \
  case instrMembers::mb##NAME:    \
    ret = false;                  \
    break;
  // clang-format on

  bool ret = false;
  auto mbInt = (instrMembers)mb;
  switch (mbInt) {
#include "LibApiMembers.def"

  default:
    assert(false && "invalid member");
  }
  return ret;
}

// Granular include selection:
// General idea for now is inlining.h has "almost everything" with the exception of the ops
// pulling huge static data structures challenging the compiler.
// TODO: avoid inlining.h and implement full-granular include mapping through automatic-code-gen.
std::vector<std::string> getGenericOperatorIncludes(const std::string& operatorName) {
  (void)operatorName;
  return {"inlining.h", "LibCommon.h", "LibTypes.h", "LibTensor.h", "Float16.h"};
}

std::vector<std::string> getSpecificOperatorIncludes(const std::string& operatorName) {
  static std::map<std::string, std::vector<std::string>> op2Headers{{"ETSOCGenericOp", {"ETSOCGenericOpInst.h"}}};

  dnn_lib::instrConfig conf;
  auto found = getInstrConfig(operatorName, conf);

  if (not found or op2Headers.count(conf.name) == 0) {
    return {};
  }
  return op2Headers[conf.name];
}

} // end namespace dnn_lib
