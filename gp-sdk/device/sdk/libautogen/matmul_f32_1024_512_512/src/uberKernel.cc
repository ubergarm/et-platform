#include <stdio.h>

#include <etsoc/isa/hart.h>
#include <etsoc/isa/barriers.h>
#include <etsoc/isa/cacheops-umode.h>
#include <etsoc/common/utils.h>
#include "SyncComputeNode.h"
#include "kernel_arguments.h"
#include "neuralizer_device_types.h"
#include "MatmulCmd_compute.h"
#include "MatmulCmd_w_pref.h"
#include "MatmulCmd_act_pref.h"
#include "inst_pref_decls.h"


extern "C" void startUberKernelComputes(uint64_t shireId, uint32_t minionId, kernelArguments * layer_dyn_info);
extern "C" void startUberKernelPrefActivations(uint64_t shireId, uint32_t minionId, kernelArguments * layer_dyn_info);
extern "C" void startUberKernelPrefWeights(uint64_t shireId, uint32_t minionId, kernelArguments * layer_dyn_info);
extern "C" void startUberKernelPrefInst(uint64_t shireId, uint32_t minionId, kernelArguments * layer_dyn_info);
extern "C" void startUberKernelCBDrainers(kernelArguments * layer_dyn_info);
extern "C" void SyncMinionsCode(uint32_t minionId, uint32_t threadId, uint32_t nComputeShires, uint32_t syncThread0Mask, uint32_t syncThread1Mask, kernelArguments * layer_dyn_info);


extern "C" int uberKernel_RAWKERNEL_entry_point(kernelArguments * layer_dyn_info) {
  uint32_t hart = get_hart_id();
  uint32_t threadId = hart & 1;
  uint32_t minionId = hart >> 1;
  uint32_t shireId = minionId >> 5;
  minionId = minionId & 0x1F;
  if (shireId == 32) {
    et_printf("Entering\n");
    if (minionId >= 16) {
      SyncMinionsCode(minionId, threadId, 32, 0xffff0000, 0xffff0000, layer_dyn_info);
      et_printf("S:M:T - %u:%u:%u\n", shireId, minionId, threadId);
    }
  } else {
    if (shireId >= 32) {
      return 0;
    } else if (threadId == 0) {
      sendInitCreditsToActPref(minionId, shireId, 0);
      sendInitCreditsToWeightPref(minionId, shireId, 0);
      startUberKernelComputes(shireId,minionId,layer_dyn_info);
    } else {
      if (minionId == 30) {
        startUberKernelPrefInst(shireId, minionId, layer_dyn_info);
      } else if (minionId <16) {
        startUberKernelPrefActivations(shireId, minionId, layer_dyn_info);
      } else if ((minionId < 20) || ((minionId >= 24) && minionId < 28)) {
        startUberKernelPrefWeights(shireId, minionId, layer_dyn_info);
      } else if (minionId == 20) {
        //CBDrainersCode(minionId, shireId, 1);
      }
    }
  } //finish thread 1
  // et_printf("S:M:T - %u:%u:%u\n", shireId, minionId, threadId);
  return 0;
} //uberKernelStart Finish

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

extern "C" __attribute((noclone , noinline)) void startUberKernelComputes(uint64_t shireId, uint32_t minionId, kernelArguments * layer_dyn_info){

  uint32_t globalMinionId = shireId * 32 + minionId;

  _UNIQUE_CALL(MatmulCmd_compute, layer_dyn_info)

  SyncComputeNode(minionId, shireId, 32, false, false, 0);

}


extern "C" __attribute((noclone , noinline)) void startUberKernelPrefActivations(uint64_t shireId, uint32_t minionId, kernelArguments * layer_dyn_info){

  _UNIQUE_CALL(MatmulCmd_act_pref, layer_dyn_info)
  
  actPrefWaitsSyncCredit(0);

}


extern "C" __attribute((noclone , noinline)) void startUberKernelPrefWeights(uint64_t shireId, uint32_t minionId, kernelArguments * layer_dyn_info){

  _UNIQUE_CALL(MatmulCmd_w_pref, layer_dyn_info)


}


extern "C" __attribute((noclone , noinline)) void startUberKernelPrefInst(uint64_t shireId, uint32_t minionId, kernelArguments * layer_dyn_info){

  _UNIQUE_CALL(MatmulCmd_inst_pref, layer_dyn_info)

  actPrefWaitsSyncCredit(0);

}


void startUberKernelCBDrainers(kernelArguments * layer_dyn_info) {
  //TODOMatmulCmd
}
extern "C" __attribute((noclone , noinline)) void SyncMinionsCode(uint32_t minionId, uint32_t threadId, uint32_t nComputeShires, uint32_t syncThread0Mask, uint32_t syncThread1Mask, kernelArguments * layer_dyn_info) {
  if (threadId & 0x1) {
    if (((syncThread1Mask >> minionId) & 0x1) == 0) return;
  } else {
    if (((syncThread0Mask >> minionId) & 0x1) == 0) return;
  }


  //
  // Sync for node MatmulCmd_Sync_8
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
