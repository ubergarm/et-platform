#!/usr/bin/python3
import io
import argparse
import re
import subprocess
import os

def getAddr2line(addr_dict, elf, opt):
    lineno_re = re.compile(r'.*:([0-9]*)', ) # match line number after :
    filepath_re = re.compile(r'.* at (.*):') # match linux file path

    # Join all unique addr in a single space-separated string
    txt = ' '.join(addr_dict.keys())

    # Compose addr2line command
    cmd = 'riscv64-unknown-elf-addr2line -e {0} -f -p --demangle {1} {2}'.format(elf, opt, txt)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    results = result.stdout.decode("utf-8").splitlines()

    # Add each unique source code file in results add it to a dict
    source_files = [ filepath_re.match(line)[1] for line in results ]
    source_codes = dict.fromkeys(source_files)

    # Delete keys from files that do not exist
    for path in list(source_codes.keys()):
        if (os.path.isfile(path)):
            with open(path, 'r') as f:
                source_codes[path] = f.readlines()
        else:
            source_codes.pop(path, None)

    # Fill each address in the dictionary with:
    #  1- the addr2line output (instruction address, assembly, function, file:lineno)
    #  2- the line of source code corresponding to file:lineno
    for k, r in zip(addr_dict.keys(), results):
        # get source code file and line number
        path = filepath_re.match(r)[1]
        
        code = ''
        if path in source_codes:
            if lineno_re.match(r)[1]:
                src_line = int(lineno_re.match(r)[1])
                if len(source_codes[path]) > (src_line-1):
                    code = source_codes[path][src_line-1].strip()
                    
        addr_dict[k] = [r, code]

def parseSysemuLog(filepath, hartid):
    pattern = r'DEBUG EMU: \[H' + str(hartid) + '.*I\(U\): (?P<addr>0x[0-9a-f]*) (?P<instrhex>\(0x[0-9a-f]*\)) (?P<instr>.* .*)' 
    
    with open(filepath, 'r') as f:
      log = f.read()

    matches = re.findall(pattern, log)

    list_of_addresses = []
    list_of_asm = []
    for match in matches:
        # get the list of addresses
        list_of_addresses.append(match[0])
        # get the asm
        list_of_asm.append(match[2])

    return list_of_addresses, list_of_asm


def main(args):
    # options (todo)
    opt = ""

    # Parse sysemu log
    addr_list, asm_list = parseSysemuLog(args.sysemu_log, args.hart_id)
    
    # Create a dictionary with unique addr
    address_dictionary = dict.fromkeys(addr_list)

    # Addr2line
    getAddr2line(address_dictionary, args.debug_elf, opt)

    # Print output
    for addr, asm in zip(addr_list,asm_list):
        p = '{0:15} {1:20} {2:50}     ---    {3}'.format(addr, asm, address_dictionary[addr][0], address_dictionary[addr][1])
        print(p)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('debug_elf', type=str, help='Path to ELF file with debug info')
    parser.add_argument('sysemu_log', type=str, help='Path to sysemu.log')
    parser.add_argument('hart_id', type=int, help='hart id to inspect')
    args = parser.parse_args()

    main(args)