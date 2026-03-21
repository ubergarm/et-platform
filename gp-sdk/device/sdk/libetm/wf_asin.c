/* wf_asin.c -- float version of w_asin.c.
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
 *
 */

/* 
 * wrapper asinf(x)
 */


#include "etmath_private.h"
#include <math.h>

float __wrap_asinf(float x);

float __wrap_asinf(float x)
{
#ifdef _IEEE_LIBM
	return __ieee754_asinf(x);
#else
	float z;
	z = __ieee754_asinf(x);
	if(_LIB_VERSION == _IEEE_ || isnan(x)) return z;
	if(fabsf(x)>1.0f) {
	    /* asinf(|x|>1) */
	    // errno = EDOM; disabled error management due to difficulties with function pointers in et-soc-1
        SET_FLOAT_WORD(x,0x7fc00000);
	} else
	    return z;
#endif
}

