/* wf_log.c -- float version of w_log.c.
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */

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

/*
 * wrapper logf(x)
 */

#include "etmath_private.h"
#include <math.h>

float __wrap_logf(float x);

#if __OBSOLETE_MATH
#include <errno.h>

float __wrap_logf(float x)
{
#ifdef _IEEE_LIBM
	return __ieee754_logf(x);
#else
	float z;
	z = __ieee754_logf(x);
	if(_LIB_VERSION == _IEEE_ || isnan(x) || x > 0.0f) return z;
	if(x==0.0f) {
	    /* logf(0) */
	    // errno = ERANGE; disabled error management due to difficulties with function pointers in et-soc-1
	    return -HUGE_VALF;
	} else { 
	    /* logf(x<0) */
	    // errno = EDOM; disabled error management due to difficulties with function pointers in et-soc-1
        SET_FLOAT_WORD(x,0x7fc00000);
        return x;
    }
#endif
}

#endif /* __OBSOLETE_MATH */

