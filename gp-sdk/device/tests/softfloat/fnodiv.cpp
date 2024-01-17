
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

#include "CommonCode.h"

#include <math.h>
#include <errno.h>
#include "entryPoint.h"


class KernelArguments;
int entryPoint_0(KernelArguments* args);
bool isEqualWithPrecision(double value1, double value2, double precision);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, nullptr);

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

        double dd;
        const double da = static_cast<double>(a.f);
        const double dc = static_cast<double>(c.f);

        pos_inf.i  = 0x7f800000;
        neg_inf.i  = 0xff800000;
        pos_nan.i  = 0x7ff00000;
        neg_zero.i = 0x80000000;
        pos_zero.i = 0;
        
        et_printf("\n");
        auto expected = -3.142857;
        double precision = 0.001; // Set your desired precision here
        d.f = a.f        / b.f;             et_printf("0x%x / 0x%x --> 0x%x\n", a.i, b.i, d.i);
        et_assert(isEqualWithPrecision(static_cast<double>(d.f), expected, precision));
        d.f = pos_zero.f / pos_zero.f;      et_printf("0x%x / 0x%x --> 0x%x\n", pos_zero.i, pos_zero.i, d.i);
        et_assert(std::isnan(d.f));
        d.f = pos_zero.f / b.f;             et_printf("0x%x / 0x%x --> 0x%x\n", pos_zero.i, b.i, d.i);
        expected = 0.000000;
        et_assert(isEqualWithPrecision(static_cast<double>(d.f), expected, precision));
        d.f = a.f        / pos_zero.f;      et_printf("0x%x / 0x%x --> 0x%x\n", a.i, pos_zero.i, d.i);
        et_assert(std::isinf(d.f));
        d.f = pos_inf.f  / b.f;             et_printf("0x%x / 0x%x --> 0x%x\n", pos_inf.i, b.i, d.i);
        et_assert(std::isinf(d.f));
        d.f = a.f        / pos_inf.f;       et_printf("0x%x / 0x%x --> 0x%x\n", a.i, pos_inf.i, d.i);
        et_assert(isEqualWithPrecision(static_cast<double>(d.f), expected, precision));
        d.f = pos_nan.f  / b.f;             et_printf("0x%x / 0x%x --> 0x%x\n", pos_nan.i, b.i, d.i);
        et_assert(std::isnan(d.f));

        et_printf("\n");

        d.f = sqrtf(a.f);           et_printf("sqrt(%f) --> %f\n", static_cast<double>(a.f), static_cast<double>(d.f));
        expected = 4.690416;
        et_assert(isEqualWithPrecision(static_cast<double>(d.f), expected, precision));
        d.f = sqrtf(b.f);           et_printf("sqrt(%f) --> %f\n", static_cast<double>(b.f), static_cast<double>(d.f));
        et_assert(std::isnan(d.f));
        d.f = sqrtf(c.f);           et_printf("sqrt(%f) --> %f\n", static_cast<double>(c.f), static_cast<double>(d.f));
        expected = 1.414214;
        et_assert(isEqualWithPrecision(static_cast<double>(d.f), expected, precision));
        d.f = sqrtf(pos_zero.f);    et_printf("sqrt(0x%f) --> 0x%f\n", static_cast<double>(pos_zero.f), static_cast<double>(d.f));
        expected = 0.000000;
        et_assert(isEqualWithPrecision(static_cast<double>(d.f), expected, precision));
        d.f = sqrtf(neg_zero.f);    et_printf("sqrt(0x%f) --> 0x%f\n", static_cast<double>(neg_zero.f), static_cast<double>(d.f));
        et_assert(isEqualWithPrecision(static_cast<double>(d.f), expected, precision));
        d.f = sqrtf(pos_inf.f);     et_printf("sqrt(0x%f) --> 0x%f\n", static_cast<double>(pos_inf.f), static_cast<double>(d.f));
        et_assert(std::isinf(d.f));
        d.f = sqrtf(neg_inf.f);     et_printf("sqrt(0x%f) --> 0x%f\n", static_cast<double>(neg_inf.f), static_cast<double>(d.f));
        et_assert(std::isnan(d.f));
        d.f = sqrtf(pos_nan.f);     et_printf("sqrt(0x%f) --> 0x%f\n", static_cast<double>(pos_nan.f), static_cast<double>(d.f));
        et_assert(std::isnan(d.f));

        auto r = sqrtf(-1);
        et_assert(std::isnan(r));
        et_printf("sqrtf(-1) = %f, errno: %d \n", double(r), errno);
        auto val = 15689.3256f;
        r = sqrt(val);
        et_printf("sqrtf(%f) = %f, errno: %d \n", double(val), double(r),  errno);
        expected = 125.257034;
        et_assert(isEqualWithPrecision(static_cast<double>(r), expected, precision));
        et_printf("\n");

        et_printf("%f\n", static_cast<double>(d.f));
        
        // pow - f32
        d.f = powf(a.f, c.f);             et_printf("powf(%f, %f) --> %f\n", da,  dc, static_cast<double>(d.f));
        expected = 484.000000;
        et_assert(isEqualWithPrecision(static_cast<double>(d.f), expected, precision));
        // pow - f64
        dd = pow(da, dc);                 et_printf("pow(%f, %f) --> %f\n", da,  dc, dd);
        et_assert(isEqualWithPrecision(static_cast<double>(d.f), expected, precision));
        // atanh - f32
        d.f = atanhf(a.f);                et_printf("atanhf(%f) --> %f\n", da, static_cast<double>(d.f));
        et_assert(std::isnan(d.f));
        // atanh - f64
        dd = atanh(da);                   et_printf("atanh(%f) --> %f\n", da, dd);
        et_assert(std::isnan(d.f));

        // asin - f32
        d.f = asinf(0.7f);                et_printf("asinf(0.7) --> %f\n", static_cast<double>(d.f));
        expected = 0.775397;
        et_assert(isEqualWithPrecision(static_cast<double>(d.f), expected, precision));
        d.f = asinf(a.f);                 et_printf("asinf(%f) --> %f\n", da, static_cast<double>(d.f));
        et_assert(std::isnan(d.f));
        // asin - f64
        dd = asin(0.7);
        et_printf("asin(0.7) --> %f\n", dd);
        expected = 0.775397;
        et_assert(isEqualWithPrecision(static_cast<double>(dd), expected, precision));
        dd = asin(da);                    et_printf("asin(%f) --> %f\n", da, static_cast<double>(d.f));
        et_assert(std::isnan(dd));
        // acos - f32
        d.f = acosf(0.7f);                et_printf("acosf(0.7) --> %f\n", static_cast<double>(d.f));
        expected = 0.795399;
        et_assert(isEqualWithPrecision(static_cast<double>(d.f), expected, precision));
        d.f = acosf(a.f);                 et_printf("acosf(%f) --> %f\n", da, static_cast<double>(d.f));
        et_assert(std::isnan(d.f));
        // acos - f64
        dd = acos(0.7);                   et_printf("acos(0.7) --> %f\n", dd);
        et_assert(isEqualWithPrecision(static_cast<double>(dd), expected, precision));
        dd = acos(da);                    et_printf("acos(%f) --> %f\n", da, static_cast<double>(d.f));
        et_assert(std::isnan(dd));
        // acos - f32
        d.f = acoshf(0.7f);                et_printf("acoshf(0.7) --> %f\n", static_cast<double>(d.f));
        et_assert(std::isnan(d.f));
        d.f = acoshf(a.f);                 et_printf("acoshf(%f) --> %f\n", da, static_cast<double>(d.f));
        expected = 3.783673;
        et_assert(isEqualWithPrecision(static_cast<double>(d.f), expected, precision));
        // acos - f64
        dd = acosh(0.7);                   et_printf("acosh(0.7) --> %f\n", dd);
        et_assert(std::isnan(dd));
        dd = acosh(da);                    et_printf("acosh(%f) --> %f\n", da, static_cast<double>(d.f));
        et_assert(isEqualWithPrecision(static_cast<double>(dd), expected, precision));

        expected = 484.0f;
        float res = powf(22,2);
        et_printf("powf(22,2): %f vs %f\n", double(res), double(expected));
        et_assert(isEqualWithPrecision(static_cast<double>(res), expected, precision));

        expected = 3.091043f;
        res = logf(22);
        et_printf("logf(22): %f vs %f\n", double(res), double(expected));
        et_assert(isEqualWithPrecision(static_cast<double>(res), expected, precision));

        res = logf(-1);
        et_printf("logf(-1): %f vs %s\n", double(res), "nan");
        et_assert(std::isnan(res));

        res = log1pf(0.3f);
        expected = 0.26236f;
        et_printf("log: %f vs %f\n", double(res), double(expected));
        et_assert(isEqualWithPrecision(static_cast<double>(res), expected, precision));

        res = log1pf(-2.0f);
        et_printf("log1pf(-2.0): %f vs %s\n", double(res), "nan");
        et_assert(std::isnan(res));

        res = log1pf(-1.0f);
        et_printf("log1p(-1.0): %f vs %s\n", double(res), "-inf");
        et_assert(std::isinf(res));

        expected = 3.091043f;
        auto resf = log(22);
        et_printf("log(22): %f vs %f\n", double(resf), double(expected));
        et_assert(isEqualWithPrecision(static_cast<double>(resf), expected, precision));

        resf = log(-1);
        et_printf("log(-1): %f vs %s\n", double(resf), "nan");
        et_assert(std::isnan(resf));

        resf = log1p(0.3);
        expected = 0.26236f;
        et_printf("log1p(0.3): %f vs %f\n", double(resf), double(expected));
        et_assert(isEqualWithPrecision(static_cast<double>(resf), expected, precision));

        resf = log1p(-2.0);
        et_printf("log10(-2.0): %f vs %s\n", double(resf), "nan");
        et_assert(std::isnan(resf));

        resf = log1p(-1.0);
        et_printf("log1p(-1.0): %f vs %s\n", double(resf), "-inf");
        et_assert(std::isinf(resf));
    }

  return 0;
}

bool isEqualWithPrecision(double value1, double value2, double precision) {
  double diff = std::abs(value1 - value2);
  return diff <= precision;
}
