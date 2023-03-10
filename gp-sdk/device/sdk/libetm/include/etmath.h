#ifndef _ETMATH_H_
#define _ETMATH_H_

#include <stdint.h>

uint32_t __errno;


float __wrap___ieee754_sqrtf(float x);

extern double __extendsfdf2(float);
static inline unsigned long et_flt_to_ui64(float x)
{
    return (unsigned long)__extendsfdf2(x);
}

extern double __extendsfdf2(float);
static inline long et_flt_to_i64(float x)
{
    return (long)__extendsfdf2(x);
}


#endif 
