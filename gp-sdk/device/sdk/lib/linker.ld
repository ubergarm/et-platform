SECTIONS
{
  . = 0x00000001;
  .text.init : { *(.text.init) }
  . = ALIGN(0x00001000);
  .text : { *(.text.init) *(.text) *(.text.*) }
  /* Code prefetching helper section */
  .sync_compute_node : { *(.sync_compute_node) }
  _sync_compute_node_start = LOADADDR(.sync_compute_node);
  _sync_compute_node_end = LOADADDR(.sync_compute_node) + SIZEOF(.sync_compute_node);
  . = ALIGN(0x00001000);
  .persistentData :
  {
    *(.persistentData);
  }
  .data :
  {
    /* This section must be restored every time a kernel is relaunched in the
       device. We reserve a region of the same size as _data
       labeled (_data_ro_copy*) where we will place a read-only copy of _data
       the first time a kernel is launched. Subsequent launches will restore  
      _data_ro_copy into _data */
    . = ALIGN(64);
    _data_start = .;
    *(.data) ;
    _data_end = .;
    _data_ro_copy_start = .;
    . +=(_data_end - _data_start);
    _data_ro_copy_end = .;
    . = ALIGN(64);
  }
  .sdata          :
  {
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
  }
  .bss(NOLOAD) :
  {
    . = ALIGN(64);
    _bss_start = .;
    *(.bss*);
    _bss_end = .;
    . = ALIGN(64);
  }
  _end = .;
}
