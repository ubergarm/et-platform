/* wf_pow.c -- float version of w_pow.c.
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
 * wrapper powf(x,y) return x**y
 */

#include "etmath_private.h"
#include <math.h>

float __wrap_powf(float x, float y);

#if __OBSOLETE_MATH
#include <errno.h>

float __wrap_powf(float x, float y)	/* wrapper powf */
{
#ifdef _IEEE_LIBM
	return  __ieee754_powf(x,y);
#else
	float z;
	z=__ieee754_powf(x,y);
	if(_LIB_VERSION == _IEEE_|| isnan(y)) return z;
	if(isnan(x)) {
	    if(y==0.0f) {
		/* powf(NaN,0.0) */
		/* Not an error.  */
		return 1.0f;
	    } else 
		return z;
	}
	if(x==0.0f){
	    if(y==0.0f) {
		/* powf(0.0,0.0) */
		/* Not an error.  */
		return 1.0f;
	    }
	    if(finitef(y)&&y<0.0f) {
		/* 0**neg */
		// errno = EDOM;
		return -HUGE_VALF;
	    }
	    return z;
	}
	if(!finitef(z)) {
	    if(finitef(x)&&finitef(y)) {
		if(isnan(z)) {
		    /* neg**non-integral */
		    // errno = EDOM;
		    /* Use a float divide, to avoid a soft-float double
		       divide call on single-float only targets.  */
		    return 0.0f/0.0f;
		} else {
		    /* powf(x,y) overflow */
		    // errno = ERANGE;
		    if(x<0.0f&&rintf(y)!=y)
		      return -HUGE_VALF;
		    return HUGE_VALF;
		}
	    }
	} 
	if(z==0.0f&&finitef(x)&&finitef(y)) {
	    /* powf(x,y) underflow */
	    // errno = ERANGE;
	    return 0.0f;
        }
	return z;
#endif
}

#endif /* __OBSOLETE_MATH */


