/*-------------------------------------------------------------------------
* Copyright (c) 2025 Ainekko, Co.
* SPDX-License-Identifier: Apache-2.0
*-------------------------------------------------------------------------*/

/*
* Test: Basic PLIC register access
* Expected: PASS
*
* This test verifies that PLIC registers are accessible at the expected
* address (0xA000_0000) and basic read/write operations work correctly.
*
* Test sequence:
* 1. Write to priority register and read back
* 2. Write to enable register and read back
* 3. Write to threshold register and read back
* 4. Read claim register (should return 0 when no interrupts pending)
*/

#include "test.h"
#include <stdint.h>

/* PLIC base address for Erbium */
#define PLIC_BASE               0xA0000000UL

/* PLIC register offsets (from RISC-V PLIC 1.0.0 spec) */
#define PLIC_PRIORITY_BASE      0x000000UL
#define PLIC_PENDING_BASE       0x001000UL
#define PLIC_ENABLE_BASE        0x002000UL
#define PLIC_THRESHOLD_BASE     0x200000UL
#define PLIC_CLAIM_BASE         0x200004UL

/* Register access macros */
#define PLIC_PRIORITY(src)      (*(volatile uint32_t *)(PLIC_BASE + PLIC_PRIORITY_BASE + (src) * 4))
#define PLIC_PENDING(word)      (*(volatile uint32_t *)(PLIC_BASE + PLIC_PENDING_BASE + (word) * 4))
#define PLIC_ENABLE(ctx, word)  (*(volatile uint32_t *)(PLIC_BASE + PLIC_ENABLE_BASE + (ctx) * 0x80 + (word) * 4))
#define PLIC_THRESHOLD(ctx)     (*(volatile uint32_t *)(PLIC_BASE + PLIC_THRESHOLD_BASE + (ctx) * 0x1000))
#define PLIC_CLAIM(ctx)         (*(volatile uint32_t *)(PLIC_BASE + PLIC_CLAIM_BASE + (ctx) * 0x1000))

/* Test constants */
#define TEST_SOURCE_ID          1
#define TEST_CONTEXT_ID         0
#define INVALID_CONTEXT_ID      31

/* Validation1 diagnostics */
#define ET_DIAG_IRQ_INJ         0x5ULL
#define ET_DIAG_IRQ_INJ_PLIC    0x3ULL

static inline void plic_diag_set_pending(uint32_t source_id, uint32_t raise) {
    uint64_t cmd = (ET_DIAG_IRQ_INJ << 56)
                 | ((uint64_t)(raise & 1) << 55)
                 | (ET_DIAG_IRQ_INJ_PLIC << 53)
                 | source_id;
    asm volatile("csrw validation1, %0" :: "r"(cmd));
}

int main() {
    uint32_t val;

    /*
     * Test 1: Priority register write/read
     * Priority for source 0 is reserved (always 0), use source 1
     */
    PLIC_PRIORITY(TEST_SOURCE_ID) = 5;
    val = PLIC_PRIORITY(TEST_SOURCE_ID);
    /* Priority is typically masked to 3 bits (0-7), so 5 should read back as 5 */
    if (val != 5) {
        TEST_FAIL;
    }

    /* Write 0 to disable */
    PLIC_PRIORITY(TEST_SOURCE_ID) = 0;
    val = PLIC_PRIORITY(TEST_SOURCE_ID);
    if (val != 0) {
        TEST_FAIL;
    }

    /*
     * Test 1b: Source 0 priority is hardwired to 0.
     */
    PLIC_PRIORITY(0) = 7;
    val = PLIC_PRIORITY(0);
    if (val != 0) {
        TEST_FAIL;
    }

    /*
     * Test 2: Enable register write/read
     * Enable source 1 for context 0
     */
    PLIC_ENABLE(TEST_CONTEXT_ID, 0) = (1U << TEST_SOURCE_ID);
    val = PLIC_ENABLE(TEST_CONTEXT_ID, 0);
    if (val != (1U << TEST_SOURCE_ID)) {
        TEST_FAIL;
    }

    /* Clear enables */
    PLIC_ENABLE(TEST_CONTEXT_ID, 0) = 0;
    val = PLIC_ENABLE(TEST_CONTEXT_ID, 0);
    if (val != 0) {
        TEST_FAIL;
    }

    /*
     * Test 2a: Out-of-range enable word access must be ignored.
     * Erbium has 32 sources => only word 0 is valid.
     */
    PLIC_ENABLE(TEST_CONTEXT_ID, 1) = 0xFFFFFFFFu;
    val = PLIC_ENABLE(TEST_CONTEXT_ID, 1);
    if (val != 0) {
        TEST_FAIL;
    }
    val = PLIC_ENABLE(TEST_CONTEXT_ID, 0);
    if (val != 0) {
        TEST_FAIL;
    }

    /*
     * Test 2b: Writes to unmapped context must not alias context 0.
     * Erbium only has context address IDs 0 and 1.
     */
    PLIC_ENABLE(INVALID_CONTEXT_ID, 0) = (1U << TEST_SOURCE_ID);
    val = PLIC_ENABLE(TEST_CONTEXT_ID, 0);
    if (val != 0) {
        TEST_FAIL;
    }

    /*
     * Test 3: Threshold register write/read
     */
    PLIC_THRESHOLD(TEST_CONTEXT_ID) = 3;
    val = PLIC_THRESHOLD(TEST_CONTEXT_ID);
    /* Threshold is typically masked to 3 bits (0-7) */
    if (val != 3) {
        TEST_FAIL;
    }

    PLIC_THRESHOLD(TEST_CONTEXT_ID) = 0;
    val = PLIC_THRESHOLD(TEST_CONTEXT_ID);
    if (val != 0) {
        TEST_FAIL;
    }

    /*
     * Test 4: Claim clears pending for claimed source.
     */
    PLIC_PRIORITY(TEST_SOURCE_ID) = 1;
    PLIC_ENABLE(TEST_CONTEXT_ID, 0) = (1U << TEST_SOURCE_ID);
    plic_diag_set_pending(TEST_SOURCE_ID, 1);

    val = PLIC_CLAIM(TEST_CONTEXT_ID);
    if (val != TEST_SOURCE_ID) {
        TEST_FAIL;
    }
    /* While source is in-flight, no second claim is allowed. */
    val = PLIC_CLAIM(TEST_CONTEXT_ID);
    if (val != 0) {
        TEST_FAIL;
    }
    /* Complete claimed source. If claim did not clear pending, source reappears. */
    PLIC_CLAIM(TEST_CONTEXT_ID) = TEST_SOURCE_ID;
    val = PLIC_CLAIM(TEST_CONTEXT_ID);
    if (val != 0) {
        TEST_FAIL;
    }
    plic_diag_set_pending(TEST_SOURCE_ID, 0);

    /*
     * Test 4b: Out-of-range completion ID write must be ignored.
     */
    plic_diag_set_pending(TEST_SOURCE_ID, 1);
    val = PLIC_CLAIM(TEST_CONTEXT_ID);
    if (val != TEST_SOURCE_ID) {
        TEST_FAIL;
    }
    PLIC_CLAIM(TEST_CONTEXT_ID) = 0xFFFFFFFFu;
    val = PLIC_CLAIM(TEST_CONTEXT_ID);
    if (val != 0) {
        TEST_FAIL;
    }
    PLIC_CLAIM(TEST_CONTEXT_ID) = TEST_SOURCE_ID;
    val = PLIC_CLAIM(TEST_CONTEXT_ID);
    if (val != 0) {
        TEST_FAIL;
    }
    plic_diag_set_pending(TEST_SOURCE_ID, 0);

    /* Restore defaults used by the rest of this test. */
    PLIC_ENABLE(TEST_CONTEXT_ID, 0) = 0;
    PLIC_PRIORITY(TEST_SOURCE_ID) = 0;

    /*
     * Test 5: Pending register is read-only
     * Read should succeed, initial value should be 0
     */
    val = PLIC_PENDING(0);
    if (val != 0) {
        TEST_FAIL;
    }

    /*
     * Test 5b: Erbium has 32 sources => only one pending word is valid.
     * Reading word 1 must return zero.
     */
    val = PLIC_PENDING(1);
    if (val != 0) {
        TEST_FAIL;
    }

    /* All tests passed */
    TEST_PASS;
    return 0;
}
