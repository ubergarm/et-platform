/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

#include <stdint.h>

float __wrap___ieee754_sqrtf(float x);

union bitcast {
    float f;
    uint32_t u;
};


float __wrap___ieee754_sqrtf(float x)
{
    float z;
    int32_t ix, s, q, m, t, i;
    uint32_t r;

    do {
        union bitcast f2u;
        f2u.f = x;
        ix = f2u.u;
    } while (0);

    /* take care of Inf and NaN */
    if ((ix & 0x7f800000) == 0x7f800000) {
        /* sqrt(NaN)=NaN, sqrt(+inf)=+inf, sqrt(-inf)=sNaN */
        return x*x + x;
    }

    /* take care of zero */
    if (ix <= 0) {
        if ((ix & 0x7fffffff) == 0) {
            /* sqrt(+-0) = +-0 */
            return x;
        }
        if (ix < 0) {
            /* sqrt(-ve) = sNaN */
            return (x - x) / (x - x);
        }
    }

    /* normalize x */
    m = (ix >> 23);
    if (m == 0) {
        /* subnormal x */
        for (i=0; (ix & 0x00800000) == 0; ++i) {
            ix <<= 1;
        }
        m -= i - 1;
    }
    m -= 127; /* unbias exponent */
    ix = (ix & 0x007fffff) | 0x00800000;
    if ((m & 1) != 0) {
        /* odd m, double x to make it even */
        ix += ix;
    }
    m >>= 1;    /* m = [m/2] */

    /* generate sqrt(x) bit by bit */
    ix += ix;
    q = s = 0;          /* q = sqrt(x) */
    r = 0x01000000;     /* r = moving bit from right to left */

    while (r!=0) {
        t = s + r;
        if (t <= ix) {
            s   = t + r;
            ix -= t;
            q  += r;
        }
        ix += ix;
        r >>= 1;
    }

    /* use floating add to find out rounding direction */
    if (ix != 0) {
        z = 0x1p0 - 0x1.4484cp-100; /* trigger inexact flag. */
        if (z >= 0x1p0) {
            z = 0x1p0 + 0x1.4484cp-100;
            q += (z > 0x1p0) ? 2 : (q & 1); 
        }
    }
    ix = (q >> 1) + 0x3f000000;
    ix += (m << 23);
    do {
        union bitcast u2f;
        u2f.u = ix;
        z = u2f.f;
    } while (0);
    return z;
}

