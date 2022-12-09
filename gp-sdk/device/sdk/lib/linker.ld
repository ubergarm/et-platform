SECTIONS
{
  . = 0x00000000; /* BASE_ADDRESS_PLACEHOLDER */ 
  .text.init : { *(.text.init) }
  . = ALIGN(64);
  .text : { *(.text) *(.text.*) }
  . = ALIGN(64);
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
    *(.data* .sdata .sdata.*)  ;
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
    *(.bss*);
    . = ALIGN(64);
    _bss_end = .;
  }
  _end = .;
}
