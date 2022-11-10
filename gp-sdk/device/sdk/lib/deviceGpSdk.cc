// clang-format off


#include <stdio.h>

#include <etsoc/isa/hart.h>
#include <etsoc/isa/barriers.h>
#include <etsoc/isa/cacheops-umode.h>
#include <etsoc/common/utils.h>
#include "SyncComputeNode.h"
#include "kernel_arguments.h"
#include "entryPoint.h"
#include "inst_pref_decls.h"

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

extern "C" int deviceGpSdkEntry(kernelArguments * layer_dyn_info) {
  uint32_t hart = get_hart_id();
  uint32_t threadId = hart & 1;
  uint32_t minionId = hart >> 1;
  uint32_t shireId = minionId >> 5;
  minionId = minionId & 0x1F;
  uint32_t globalMinionId = shireId * 32 + minionId;

  if ((shireId < 32) && (threadId == 0)) {
    _UNIQUE_CALL(entryPoint, layer_dyn_info);
    // SyncComputeNode(minionId, shireId, 32, false, true, 0);
    //try to flush caches transparent to the user not called them from entryPoint user program.
  }

  return 0;
}

extern "C" __attribute((noclone , noinline)) void SyncMinionsCode(uint32_t minionId, uint32_t threadId, uint32_t nComputeShires, uint32_t syncThread0Mask, uint32_t syncThread1Mask, kernelArguments * layer_dyn_info) {
  if (threadId & 0x1) {
    if (((syncThread1Mask >> minionId) & 0x1) == 0) return;
  } else {
    if (((syncThread0Mask >> minionId) & 0x1) == 0) return;
  }


  //
  // Sync for node testOperator_Sync_5
  //

  global_barrier_receiver(
      FCC_0,                    // Which FCC to wait for
      0,                        // FLB to be used in the source shire for the barrier
      minionId,                 // Sync minion id
      threadId,                 // Sync thread id
      THREAD_1,                 // Thread of the FCC dest
      FCC_1,                    // FCC for dest
      HELPER_ACTIVATION_THREADS // Mask of minions in dest shire to receive FCC
    | HELPER_WEIGHTS_THREADS
    | HELPER_CODE_THREADS
    | HELPER_EVICT_W_THREADS
    ,nComputeShires
    ,syncThread0Mask //mask of threads 0 of the sync Minions that will receive the credit
    ,syncThread1Mask //mask of threads 1 of the sync Minions that will receive the credit
    ,32
    ,0
    ,0
    );
}
