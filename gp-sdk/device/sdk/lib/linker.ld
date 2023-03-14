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


SECTIONS
{
  . = BASE_ADDRESS;
  .text.init : 
  {
    _text_init_start = .;
     *(.text.init)
  }
  _before_align = .;
  . = ALIGN(64);
  .text : { *(.text) *(.text.*) }
  . = ALIGN(64);


   /* init and fini array for global constructors and destructors */
  .preinit_array : ALIGN(64) {
        PROVIDE_HIDDEN (__preinit_array_start = .);
        KEEP (*(.preinit_array))
        PROVIDE_HIDDEN (__preinit_array_end = .);
   } 

  .init_array : ALIGN(64) {
     PROVIDE_HIDDEN (__init_array_start = .);
     KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*) SORT_BY_INIT_PRIORITY(.init_array)))
     PROVIDE_HIDDEN (__init_array_end = .);
  }

  .fini_array : ALIGN(64) {
     PROVIDE_HIDDEN (__fini_array_start = .);
     KEEP (*(SORT_BY_INIT_PRIORITY(.fini_array.*) SORT_BY_INIT_PRIORITY(.fini_array)))
     PROVIDE_HIDDEN (__fini_array_end = .);
  }

 
  .persistentData :
  {
    *(.persistentData);
  }
  . = ALIGN(64);

  .data :
  {
    /* This section must be restored every time a kernel is relaunched in the
       device. We reserve a region of the same size as _data
       labeled (_data_ro_copy*) where we will place a read-only copy of _data
       the first time a kernel is launched. Subsequent launches will restore  
      _data_ro_copy into _data */
    _data_start = .;
    . = ALIGN(64);
    *(.sdata .sdata.*)
     PROVIDE( __global_pointer$ = . + 0x800 );
    *(.data .data.*)
    . = ALIGN(64);
    _data_end = .;
    _data_ro_copy_start = .;
    . +=(_data_end - _data_start);
    _data_ro_copy_end = .;
  }
  . = ALIGN(64);
  .sdata :
  {
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
    *(.gnu.linkonce.s.* )
  }
  . = ALIGN(64);
  .bss(NOLOAD) :
  {
    _bss_start = .;
    *(.sbss*);
    *(.bss*);
    . = ALIGN(64);
    _bss_end = .;
  }
  _end = .;
}
