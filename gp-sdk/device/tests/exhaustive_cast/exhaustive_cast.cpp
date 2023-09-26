#include <etsoc/common/utils.h>
#include <etsoc/isa/hart.h>

#include "entryPoint.h"
#include "exhaustive_cast_arguments.h"

int entryPoint_0(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, nullptr);

void int64ToFloat(outputContainer* out, inputContainer* in) {
  for (uint64_t i = 0; i < numElements; i++) {
    out->f[i] = (float)in->b[i];
  }
  return;
}

void uint64ToFloat(outputContainer* out, inputContainer* in) {
  for (uint64_t i = 0; i < numElements; i++) {
    out->f[i] = (float)in->c[i];
  }
  return;
}

void int32ToFloat(outputContainer* out, inputContainer* in) {
  for (uint64_t i = 0; i < numElements; i++) {
    out->f[i] = (float)in->d[i];
  }
  return;
}

void uint32ToFloat(outputContainer* out, inputContainer* in) {
  for (uint64_t i = 0; i < numElements; i++) {
    out->f[i] = (float)in->e[i];
  }
  return;
}

void floatToInt64(outputContainer* out, inputContainer* in) {
  for (uint64_t i = 0; i < numElements; i++) {
    out->g[i] = (int64_t)in->a[i];
  }
  return;
}

void floatToUint64(outputContainer* out, inputContainer* in) {
  for (uint64_t i = 0; i < numElements; i++) {
    out->h[i] = (uint64_t)in->a[i];
  }
  return;
}

void floatToInt32(outputContainer* out, inputContainer* in) {
  for (uint64_t i = 0; i < numElements; i++) {
    out->i[i] = (int32_t)in->a[i];
  }
  return;
}

void floatToUint32(outputContainer* out, inputContainer* in) {
  for (uint64_t i = 0; i < numElements; i++) {
    out->j[i] = (uint32_t)in->a[i];
  }
  return;
}

int entryPoint_0(KernelArguments* vectors) {
  if (get_relative_thread_id() == 0) {
    switch (vectors->cast_type) {
    case 1: // Float to int64_t
      floatToInt64(vectors->out, vectors->in);
      break;
    case 2: // Float to uint64_t
      floatToUint64(vectors->out, vectors->in);
      break;
    case 3: // Float to int32_t
      floatToInt32(vectors->out, vectors->in);
      break;
    case 4: // Float to uint32_t
      floatToUint32(vectors->out, vectors->in);
      break;
    case 5: // int64_t to float
      int64ToFloat(vectors->out, vectors->in);
      break;
    case 6: // uint64_t to float
      uint64ToFloat(vectors->out, vectors->in);
      break;
    case 7: // int32_t to float
      int32ToFloat(vectors->out, vectors->in);
      break;
    case 8: // uint32_t to float
      uint32ToFloat(vectors->out, vectors->in);
      break;
    default:
      break;
    }
  }
  return 0;
}