#-------------------------------------------------------------------------
#
# Copyright (c) 2025 Ainekko, Co.
# SPDX-License-Identifier: Apache-2.0
# 
#
# utility to parse a Sysemu.log file and provide c-code semantics to the instruction-stream
# parses lines of the following format: 
#
# 13471862: DEBUG EMU: [H1 S0:N0:C0:T1] I(U): 0x8005b36484 (0x00813083) ld x1,8(x2)
# 


usage() {
  printf "usage: \n\n"
  printf "       $0 <debug_elf> <sysemu_log> <hart_id> [<mode>]\n\n"
  printf "where mode can be:\n\n"
  printf "       full            Find full source code information (default)\n"
  printf "       inline_unwind   Also unwind inlined function calls\n"
  printf "       fast            Only display source code location\n\n"
  printf "examples: \n\n"
  printf "      ./sysemu_explorer.sh device/build/tests/fftKernel.elf_dbg sysemu.log0  0 \n"
  printf "      ./sysemu_explorer.sh device/build/tests/fftKernel.elf_dbg sysemu.log0  1  fast\n\n"
}

getCode() {
  if [[ "$OPTIONS" == "inline_unwind" ]]; then
    opt="-i";
  fi
  riscv64-unknown-elf-addr2line -e $1 $2   -f -p --demangle  ${opt}
}

getCLine() {
  #"fn at file:line" so we need to parse $3
  #split file and line:
  if [[ $3 != "??:?" && $3 != *":?" ]]; then
      sed "${3##*:}q;d" "${3%%:*}"
  fi
}

parseSysemuLog() {
  # hart, mode, logFile -> address, asmInst, asmCode
  awk "\$4 ~ /^.H$1\$/ && \$6 ~ /^I.$2.:\$/ { print \$7,\$8,\$9,\$10 }" $3
}

if [ "$#" -lt 3 ]; then
  usage
  exit
fi

ELF=$1 
LOG=$2
HART=$3
OPTIONS="full"
if [ "$#" -gt 3 ]; then
  OPTIONS=$4
fi
case "$OPTIONS" in
  full|fast|inline_unwind)
    ;;
  *)
    usage
    exit
esac 


parseSysemuLog $HART U $LOG | while read -r address asmInst asmCode; do
  code=$(getCode ${ELF} ${address})
  
  if [[ "$OPTIONS" != "fast" ]]; then
    codeSep=" --- "
    cLine=$(getCLine ${code})
  fi

  printf "%s \t%s \t%s \t%s \t%s\n" ${address}  "${asmCode}" "${code}" "$codeSep" "${cLine}"
done

