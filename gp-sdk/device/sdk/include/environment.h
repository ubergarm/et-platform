#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <cstdint>

/* @brief This struct information is filled by the runtime when the kernel is launched */
struct Environment {
  uint64_t shireMask;
  int32_t numThreads;
  int8_t numEntryPoints;
  int32_t freqMHz;
} __attribute__ ((packed));

struct Arguments {
  Environment env;
} __attribute__ ((packed));

#endif
