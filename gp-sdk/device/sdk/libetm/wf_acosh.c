/* wf_acosh.c -- float version of w_acosh.c.
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
 * wrapper acoshf(x)
 */

#include "etmath_private.h"
#include <math.h>

float __wrap_acoshf(float x);

float __wrap_acoshf(float x)
{
#ifdef _IEEE_LIBM
	return __ieee754_acoshf(x);
#else
	float z;
	z = __ieee754_acoshf(x);
	if(_LIB_VERSION == _IEEE_ || isnan(x)) return z;
	if(x<1.0f) {
	    /* acoshf(x<1) */
	    // errno = EDOM;
	    return 0.0f/0.0f;
	} else
	    return z;
#endif
}

