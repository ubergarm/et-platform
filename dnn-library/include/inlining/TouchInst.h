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

#ifndef _TOUCH_INST_H_
#define _TOUCH_INST_H_

namespace dnn_lib {

namespace inlining {

template <ElemKind elK>
inline void fwdLibTouchInst(LibTensor* outT, uint64_t flags, const uint32_t minionOffset = 0,
                            const uint32_t assignedMinions = 0) {

  // nop as a glow does.

  if (!DO_EVICTS)
    return;
}

} // namespace inlining

} // namespace dnn_lib

#endif //_TOUCH_INST_H_
