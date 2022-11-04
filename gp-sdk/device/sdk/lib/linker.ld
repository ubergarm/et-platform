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
  .data : { *(.data) }
  .sdata          :
  {
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
  }
  .bss : { *(.bss) }
  _end = .;
}
