/*-------------------------------------------------------------------------
 * Copyright (C) 2023, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#include <etsoc/common/utils.h>
#include <etsoc/isa/hart.h>

#include "entryPoint.h"
#include "exhaustive_cast_arguments.h"

int entryPoint_0(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, nullptr);

void int64ToFloat(dataContainer* out, dataContainer* in) {
  for (uint64_t i = 0; i < numElements; i++) {
    out->a[i] = (float)in->b[i];
  }
  return;
}

void uint64ToFloat(dataContainer* out, dataContainer* in) {
  for (uint64_t i = 0; i < numElements; i++) {
    out->a[i] = (float)in->c[i];
  }
  return;
}

void int32ToFloat(dataContainer* out, dataContainer* in) {
  for (uint64_t i = 0; i < numElements; i++) {
    out->a[i] = (float)in->d[i];
  }
  return;
}

void uint32ToFloat(dataContainer* out, dataContainer* in) {
  for (uint64_t i = 0; i < numElements; i++) {
    out->a[i] = (float)in->e[i];
  }
  return;
}

void floatToInt64(dataContainer* out, dataContainer* in) {
  for (uint64_t i = 0; i < numElements; i++) {
    out->b[i] = (int64_t)in->a[i];
  }
  return;
}

void floatToUint64(dataContainer* out, dataContainer* in) {
  for (uint64_t i = 0; i < numElements; i++) {
    out->c[i] = (uint64_t)in->a[i];
  }
  return;
}

void floatToInt32(dataContainer* out, dataContainer* in) {
  for (uint64_t i = 0; i < numElements; i++) {
    out->d[i] = (int32_t)in->a[i];
  }
  return;
}

void floatToUint32(dataContainer* out, dataContainer* in) {
  for (uint64_t i = 0; i < numElements; i++) {
    out->e[i] = (uint32_t)in->a[i];
  }
  return;
}

int entryPoint_0(KernelArguments* vectors) {
  if (get_relative_thread_id() == 0) {
    switch (vectors->cast_type) {
    case floatToInt64_t: // Float to int64_t
      floatToInt64(vectors->out, vectors->in);
      break;
    case floatToUint64_t: // Float to uint64_t
      floatToUint64(vectors->out, vectors->in);
      break;
    case floatToInt32_t: // Float to int32_t
      floatToInt32(vectors->out, vectors->in);
      break;
    case floatToUint32_t: // Float to uint32_t
      floatToUint32(vectors->out, vectors->in);
      break;
    case int64_tToFloat: // int64_t to float
      int64ToFloat(vectors->out, vectors->in);
      break;
    case uint64_tToFloat: // uint64_t to float
      uint64ToFloat(vectors->out, vectors->in);
      break;
    case int32_tToFloat: // int32_t to float
      int32ToFloat(vectors->out, vectors->in);
      break;
    case uint32_tToFloat: // uint32_t to float
      uint32ToFloat(vectors->out, vectors->in);
      break;
    default:
      break;
    }
  }
  return 0;
}