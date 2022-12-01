// clang-format off


#include <stdio.h>

#include <etsoc/isa/hart.h>
#include <etsoc/isa/barriers.h>
#include <etsoc/isa/cacheops-umode.h>
#include <etsoc/common/utils.h>
#include "CommonCode.h"
#include "entryPoint.h"

/* Linker labels to global .bss and .data sections */
extern uint32_t _bss_end;
extern uint32_t _bss_start;
extern uint32_t _data_end;
extern uint32_t _data_start;
extern uint32_t _data_ro_copy_end;
extern uint32_t _data_ro_copy_start;

/* Number of times the kernel has been launched */
uint64_t numberOfBoots __attribute__ ((section("persistentData"))) = { 1 };

#define _UNIQUE_CALL(function, ...)\
 do { \
  function( __VA_ARGS__ ); \
  __asm__ __volatile__ (\
    ".global " #function "_return_point\n"\
    #function "_return_point:" : : : );\
} while(0);

#define _UNIQUE_CALL_PARALLEL(function, param) {\
  function( param ); \
}

#define _UNIQUE_CALL_PARALLEL_RETURN(function_name) {\
  __asm__ __volatile__ (\
    ".global " #function_name "_return_point\n"\
    #function_name "_return_point:" : : : );\
}

#define _UNIQUE_LBL(function, param) {\
  __asm__ __volatile__ (\
    ".global " #function "_return_point\n"\
    #function "_return_point:" : : : );\
}

void resetBSS();
void resetData();
extern "C" int deviceGpSdkEntry(KernelArguments * args);

/* resets global memory region .bss to zero */
void resetBSS() {
  uint8_t *bss_end = (uint8_t *) &_bss_end;
  uint8_t *bss_start = (uint8_t *) &_bss_start;
  global_memset(bss_start, 0, bss_end - bss_start);
}

/* initialized global memory region .bss to zero */
void resetData() {
  uint8_t *data_end = (uint8_t *) &_data_end;
  uint8_t *data_start =  (uint8_t *) &_data_start;
  uint8_t *data_ro_copy_start =  (uint8_t *) &_data_ro_copy_start;

  if (numberOfBoots == 1) {
    // backup .data section
    global_memcpy(data_ro_copy_start, data_start, data_end - data_start);
  }
  else {
    // restore .data section
    global_memcpy(data_start, data_ro_copy_start, data_end - data_start);
  }
}

extern "C" int deviceGpSdkEntry(KernelArguments * args) {
  uint32_t hart = get_hart_id();
  uint32_t threadId = hart & 1;
  uint32_t minionId = hart >> 1;
  uint32_t shireId = minionId >> 5;
  minionId = minionId & 0x1F;
  uint32_t globalMinionId = shireId * 32 + minionId;

  if (globalMinionId == 0 && threadId == 0) {
    // Reset .bss and .data sections on each kernel launch
    resetBSS();
    resetData();

    // increase number of boots atomically
    const uint32_t increment = 1;
    uint32_t result;
    __asm__ __volatile__("amoaddg.w %[result], %[increase], (%[dst])\n"
                         : [ result ] "=r" (result)
                         : [ increase ] "r" (increment), [ dst ] "r"(&numberOfBoots)
                         :);
    
    // barrier 0 - send credits to THREAD_0 of all minions
    for (uint32_t sId = 0; sId < 32; sId++) {
      fcc_send(sId, THREAD_0, FCC_0, 0xFFFFFFFF);
    }
  }

  if ((shireId < 32) && (threadId == 0)) {
    // barrier 0 - wait for credits
    fcc_consume(FCC_0);

    _UNIQUE_CALL(entryPoint, args);
  }
  
  return 0;
}

