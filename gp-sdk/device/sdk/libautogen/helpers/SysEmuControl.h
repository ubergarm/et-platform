/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef SYS_EMU_CONTROL_H
#define SYS_EMU_CONTROL_H

// Flushes L1
#define SYSEMU_FLUSH_L1 0x602
// Disables read coherency checking
#define SYSEMU_DISABLE_READ_COHERENCY_CHECKING 0x603
// Enables read coherency checking
#define SYSEMU_ENABLE_READ_COHERENCY_CHECKING 0x604

#endif // SYS_EMU_CONTROL_H