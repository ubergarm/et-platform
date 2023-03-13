
/*-------------------------------------------------------------------------
 * Copyright (C) 2022, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#include <etsoc/common/utils.h>
#include <etsoc/isa/fcc.h>
#include <etsoc/isa/hart.h>
#include <etsoc/isa/tensors.h>
#include <etsoc/isa/utils.h>

#include "entryPoint.h"

#include "CommonCode.h"

#include <math.h>
#include "entryPoint.h"

int entryPoint_0(KernelArguments* args);
extern DeviceConfig config {1, entryPoint_0, nullptr};

union uif {
    float f;
    unsigned int i;
};

int entryPoint_0([[maybe_unused]] KernelArguments* args) {
    if (get_relative_thread_id() == 0) {
        union uif d, a, b, c;
        union uif pos_inf, neg_inf, pos_nan, neg_zero, pos_zero;

        a.f = 22.0f;
        b.f = -7.0f;
        c.f =  2.0f;

        pos_inf.i  = 0x7f800000;
        neg_inf.i  = 0xff800000;
        pos_nan.i  = 0x7ff00000;
        neg_zero.i = 0x80000000;
        pos_zero.i = 0;

        et_printf("\n");

        d.f = a.f        / b.f;             et_printf("0x%x / 0x%x --> 0x%x\n", a.i, b.i, d.i);
        d.f = pos_zero.f / pos_zero.f;      et_printf("0x%x / 0x%x --> 0x%x\n", pos_zero.i, pos_zero.i, d.i);
        d.f = pos_zero.f / b.f;             et_printf("0x%x / 0x%x --> 0x%x\n", pos_zero.i, b.i, d.i);
        d.f = a.f        / pos_zero.f;      et_printf("0x%x / 0x%x --> 0x%x\n", a.i, pos_zero.i, d.i);
        d.f = pos_inf.f  / b.f;             et_printf("0x%x / 0x%x --> 0x%x\n", pos_inf.i, b.i, d.i);
        d.f = a.f        / pos_inf.f;       et_printf("0x%x / 0x%x --> 0x%x\n", a.i, pos_inf.i, d.i);
        d.f = pos_nan.f  / b.f;             et_printf("0x%x / 0x%x --> 0x%x\n", pos_nan.i, b.i, d.i);

        et_printf("\n");

        d.f = sqrtf(a.f);           et_printf("sqrt(%f) --> %f\n", static_cast<double>(a.f), static_cast<double>(d.f));
        d.f = sqrtf(b.f);           et_printf("sqrt(%f) --> %f\n", static_cast<double>(b.f), static_cast<double>(d.f));
        d.f = sqrtf(c.f);           et_printf("sqrt(%f) --> %f\n", static_cast<double>(c.f), static_cast<double>(d.f));
        d.f = sqrtf(pos_zero.f);    et_printf("sqrt(0x%f) --> 0x%f\n", static_cast<double>(pos_zero.f), static_cast<double>(d.f));
        d.f = sqrtf(neg_zero.f);    et_printf("sqrt(0x%f) --> 0x%f\n", static_cast<double>(neg_zero.f), static_cast<double>(d.f));
        d.f = sqrtf(pos_inf.f);     et_printf("sqrt(0x%f) --> 0x%f\n", static_cast<double>(pos_inf.f), static_cast<double>(d.f));
        d.f = sqrtf(neg_inf.f);     et_printf("sqrt(0x%f) --> 0x%f\n", static_cast<double>(neg_inf.f), static_cast<double>(d.f));
        d.f = sqrtf(pos_nan.f);     et_printf("sqrt(0x%f) --> 0x%f\n", static_cast<double>(pos_nan.f), static_cast<double>(d.f));

        et_printf("\n");

        et_printf("%f\n", static_cast<double>(d.f));
        // d.f = a.f / b.f;    print_u  f3(a, b, d);
        // printf("%ld\n", et_flt_to_i64(d.f));
    }

  return 0;
}
