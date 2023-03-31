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

extern const thread_local uint32_t  * testTlsPtr;

void testExternalTlsAccess();
void testExternalTlsAccess() {
  et_printf("%x %x ", testTlsPtr[0], testTlsPtr[1]);
}
