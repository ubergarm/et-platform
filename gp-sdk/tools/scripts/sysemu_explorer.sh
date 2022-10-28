#
# utility to parse a Sysemu.log file and provide c-code semantics to the instruction-stream
# 
ELF=$1 
LOG=$2
HART=$3

#TODO: 
# 1. pass multiple-addressses to addr2line to avoid loading the elf every time.
# 2. print the atual asm instruction
# 3. print the actual c/c++/ line. (only when it changes). 

#13471862: DEBUG EMU: [H1 S0:N0:C0:T1] I(U): 0x8005b36484 (0x00813083) ld x1,8(x2)

inst_stream=/tmp/inst_stream_tmp_${HART}.log

$(cat $LOG | grep "I(U)" | grep  "H${HART} " |cut -d " " -f 7 > ${inst_stream};)

while read -r address; do 
  code=$(riscv64-unknown-elf-addr2line -e ${ELF} $address -i);
  printf "%s %s \n" ${address} ${code}
done < ${inst_stream}

rm ${inst_stream}

