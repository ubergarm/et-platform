#if MINION_LOGGING
#include "logging.h"
#endif

#include <dnn_lib/utils.h>

#include "CommonCode.h"
#include "SyncComputeNode.h"
#include "SysEmuControl.h"
#include "common_arguments.h"
#include <etsoc/isa/cacheops-umode.h>

void SendCreditsToActPref(uint32_t minionId, uint32_t shireID, [[maybe_unused]] uint64_t logHartHeaderAddress = 0) {
  if ( minionId < N_CREDITS_TO_ACT_PREF){
    fcc_send(shireID, THREAD_1, FCC_0, HELPER_ACTIVATION_THREADS);
  }
}

void ActPrefWaitsSyncCredit(uint32_t minionId, uint32_t shireID, [[maybe_unused]] uint64_t logHartHeaderAddress = 0) {
  fcc(FCC_1);
}

static inline void fast_flush_l1_cache(uint32_t shireId) {
  for (uint32_t cacheLine = 0; cacheLine < L1_FLUSH_RESERVATION_LINES; cacheLine++) {
    uint64_t cache_line_address =
      (uint64_t)(2U << 30U) + (shireId << 23) + (L1_FLUSH_RESERVATION_START_LINE + cacheLine) * CACHE_LINE_BYTES;
    __asm__ __volatile__("ld x0, %[cache_line_address]\n"
                         :
                         : [ cache_line_address ] "m"(*(volatile uint64_t*)cache_line_address)
                         : "x0");
  }
  riscv_fence();
  __asm__ __volatile__("slti x0, x0, " TOSTRING(SYSEMU_FLUSH_L1) "\n" ::: "x0", "memory");
}

void __attribute__((section("sync_compute_node")))
SyncComputeNode(uint32_t minionId, uint32_t shireId, uint32_t syncShireId, bool flushCB, bool evictL2,
                [[maybe_unused]] uint64_t logHartHeaderAddress) {

  // At this point we are not using the flush booleans, but we will

  // We need a separate barrier for the completely done barrier, as some
  // minions might be done at some point before the other ones. If we use
  // the regular 2 thread 0 barriers, the minion will get boggus data

  // Fence other stores and loads
  riscv_fence();

  // Fast flush L1 cache
  fast_flush_l1_cache(shireId);

  // This fence waits for Write Arounds to finish!!
  // Note: the other waits are not really needed, but prevent false failures in the l1 scp checker
  // in sysemu. As they don't have a performance impact, we keep all of them
  tensorwait(7);  // TensorFMA
  tensorwait(10); // TensorQuant
  tensorwait(9);  // TensorReduce
  tensorwait(6);  // CacheOp
  tensorwait(8);  // TensorStore

  // This is the best option until the moment
  // FLB at barrier 27 and with 32 (minions)

  // Waits for all the minions to be done with the L1 flush
  if (flbarrier(27, 32 - 1)) {
    fcc_send(shireId, 0, FCC_0, 0xF);
  }

  // For L2 and CB draining, we need one minion taking care of a different bank
  if (minionId < L2_CACHE_BANKS) {
    // First we wait until the last minion is done with the L1 evict. Adding a SW hint instruction
    fcc(FCC_0);
    __asm__ __volatile__ ("slti x0, x0, 0xab\n" ::: "memory");

    if (evictL2) {
      // Evicts L2 to L3
      ecall_shire_cache_bank_op(SHIRE_OWN, minionId, SC_CACHEOP_L2_EVICT);
    } else if (flushCB) {
      // Drains CB to L3
      cache_ops_cb_drain(shireId, minionId);
    }

    // Waits until the last of the evict/drain is done and sends credit to sync minion reporting shire is done
    if (flbarrier(26, L2_CACHE_BANKS - 1)) {
      uint64_t thread = shireId & 1;
      uint64_t minionMask = 1ULL << ((shireId >> 1) + 16);
      fcc_send(syncShireId, thread, FCC_0, minionMask);
    }
  }
}

#define CB_DRAINERS_FIRST_DRINER 20
void CBDrainersCode(uint32_t minionId, uint32_t shireId, uint32_t syncShireId, uint32_t nNodes,
                    [[maybe_unused]] uint64_t logHartHeaderAddress) {
  uint64_t thread = shireId & 1;
  uint64_t minionMask = 1ULL << ((shireId >> 1) + 16);

  for (int i = 0; i < nNodes; i++) {
    fcc(FCC_0);
    drain_cbs_no_broad();
    fcc_send(syncShireId, thread, FCC_0, minionMask);
  }
}
