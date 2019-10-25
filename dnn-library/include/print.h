/*-------------------------------------------------------------------------
* Copyright (C) 2018, Esperanto Technologies Inc.
* The copyright to the computer program(s) herein is the 
* property of Esperanto Technologies.
* The program(s) may be used and/or copied only with 
* the written permission of Esperanto Technologies or 
* in accordance with the terms and conditions stipulated in the
* agreement/contract under which the program(s) have been supplied.
*-------------------------------------------------------------------------
*/

/**
* @file $Id$
* @version $Release$
* @date $Date$
* @author 
*
* @brief tcMain.c Main function for all TCs
*
* Setup SoC to enable TC run
*/

/**
 *  @Component      Print
 *
 *  @Filename       print.h
 *
 *  @Description    print Macro definitions
 *
 *//*======================================================================== */

#ifndef __PRINT_H
#define __PRINT_H

#ifdef __cplusplus
extern "C"
{
#endif

extern int tb_printf(char*, ...); 

// RR: Fix mess
// printDbg Macro

#define printDbg(...)


#ifdef __cplusplus
}
#endif

#endif	/* __PRINT_H */


/*     <EOF>     */
