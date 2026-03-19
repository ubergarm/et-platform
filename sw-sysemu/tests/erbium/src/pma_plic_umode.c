/*-------------------------------------------------------------------------
* Copyright (c) 2026 AIFoundry
* SPDX-License-Identifier: Apache-2.0
*-------------------------------------------------------------------------*/

/*
* Test: U-mode PLIC access triggers load access fault
* Expected: Load access fault (cause=5)
*/

#include "test.h"
#include "trap.h"
#include "priv.h"

#define PLIC_BASE  0xA0000000ull

static void umode_test(void) {
    volatile uint32_t *plic = (volatile uint32_t *)PLIC_BASE;
    (void)*plic;
    TEST_FAIL;
}

int main(void) {
    expect_exception(CAUSE_LOAD_ACCESS_FAULT);
    run_in_umode(umode_test);
    return 0;
}
