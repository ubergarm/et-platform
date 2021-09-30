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

#ifndef _LIB_API_H_
#define _LIB_API_H_

#include "LibTensor.h"
#include "LibTypes.h"
#include <experimental/array> // not available on macos - if we ever need dnnLibraryApi available there
                              // we should remove the std::experimental::make_array
#include <string>
#include <vector>

namespace dnn_lib {

////////////////////////////////////////////////////////////////////////////////
// definitions for non-tensor operands (aka members)
////////////////////////////////////////////////////////////////////////////////

// enum with list of known members
#define SCALAR_MB_DEF(NAME, TYPE, GETTER) mb##NAME,
#define VECTOR_MB_DEF(NAME, TYPE, GETTER) mb##NAME,
enum class instrMembers {
  mbInvalid = 0,
#include "LibApiMembers.def"
  mbMaxMembers
};

// type and name maps
template <instrMembers mb> struct memberMap;

// clang-format off
#define SCALAR_MB_DEF(NAME, TYPE, GETTER)                                                                              \
  template <> struct memberMap<instrMembers::mb##NAME> {                                                                    \
    using type = TYPE;                                                                                                 \
    static const std::string name() {                                                                                  \
      return #NAME;                                                                                                    \
    }                                                                                                                  \
  };

#define VECTOR_MB_DEF(NAME, TYPE, GETTER)                                                                              \
  template <> struct memberMap<instrMembers::mb##NAME> {                                                                    \
    using type = std::vector<TYPE>;                                                                                    \
    static const std::string name() {                                                                                  \
      return #NAME;                                                                                                    \
    }                                                                                                                  \
  };
// clang-format on

#include "LibApiMembers.def"

////////////////////////////////////////////////////////////////////////////////
// INSTRUCTION PROPERTIES CLASS
////////////////////////////////////////////////////////////////////////////////
static constexpr size_t maxImplVersions = 4;
static constexpr size_t maxInstrConfigStrLen = 256;
static constexpr size_t maxNrOperands = 12;

enum class operandState { dirty, clean, untouched };

struct instrConfig {
  using operandStateArray = std::array<operandState, maxNrOperands>;
  using implStateArray = std::array<operandStateArray, maxImplVersions + 1>;
  using sel_fnc_t = size_t (*)(std::vector<LibTensor*>&, std::vector<LibTensor*>&);

  char name[maxInstrConfigStrLen];
  size_t nrOutputTensors; // number of output and in/out tensor operands
  size_t nrInputTensors;  // number of input tensor operands
  std::array<instrMembers, (long unsigned int)instrMembers::mbMaxMembers> members;
  uint64_t templateMask;
  std::array<char[maxInstrConfigStrLen], maxImplVersions> versions;
  sel_fnc_t implSel;

  implStateArray stateL1;
  implStateArray stateL2;
  implStateArray stateCB;
  std::array<uint64_t, maxImplVersions + 1> evictAvailableMask;

  // functions to retrieve operand information
  operandState getOperandStateL1(size_t implIdx, size_t operand);
  operandState getOperandStateL2(size_t implIdx, size_t operand);
  operandState getOperandStateCB(size_t implIdx, size_t operand);
  bool getOperandAutoEvict(size_t implIdx, size_t operand);

  // and same as before, but index is either input or output
  operandState getSrcStateL1(size_t implIdx, size_t idx);
  operandState getSrcStateL2(size_t implIdx, size_t idx);
  operandState getSrcStateCB(size_t implIdx, size_t idx);
  bool getSrcAutoEvict(size_t implIdx, size_t idx);

  operandState getDstStateL1(size_t implIdx, size_t idx);
  operandState getDstStateL2(size_t implIdx, size_t idx);
  operandState getDstStateCB(size_t implIdx, size_t idx);
  bool getDstAutoEvict(size_t implIdx, size_t idx);
};

bool getInstrConfig(const std::string& operatorName, instrConfig& instConfig);
size_t getInstrNumCycles(const std::string& operatorName, size_t assignedMinions,
                         const std::vector<LibTensor*>& operands);

} // end namespace dnn_lib

#endif
