#!/usr/bin/env python3
from ast import List, Tuple
import io
import argparse
import re
import subprocess
import os
from pathlib import Path
from subprocess import PIPE


def getAddr2line(addr_dict: dict, elf: Path, opt: str):
    lineno_re = re.compile(r'.*:([0-9]*)', ) # match line number after :
    filepath_re = re.compile(r'.* at (?P<filename>.*):') # match linux file path

    # Join all unique addr in a single space-separated string
    txt = '\n'.join(addr_dict.keys())

    # Compose addr2line command
    cmd = f'riscv64-unknown-elf-addr2line -e {elf} -f -p --demangle {opt}'
    result = subprocess.run(cmd, stdout=subprocess.PIPE, input=txt, shell=True, timeout=360, text=True, check=True)
    results = result.stdout.splitlines()
    if result.stderr:
        raise RuntimeError(result.stderr)
    
    source_files = []
    for line in results:
        match = filepath_re.match(line)
        if match:
            source_files.append(match.group('filename'))

    # Add each unique source code file in results add it to a dict
    source_codes = dict.fromkeys(source_files)

    print(f'Found {len(source_codes)} source files.')
    
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
        match = filepath_re.match(r)
        if match:
            path = match.group(1)
        
        code = ''
        if path in source_codes:
            if lineno_re.match(r)[1]:
                src_line = int(lineno_re.match(r)[1])
                if len(source_codes[path]) > (src_line-1):
                    code = source_codes[path][src_line-1].strip()
                    
        addr_dict[k] = [r, code]

def parseSysemuLog(filepath: Path, hartid: int):
    
    pattern = r'DEBUG EMU: \[H' + str(hartid) + '\\b' + '.*I\(U\): (?P<addr>0x[0-9a-f]*) (?P<instrhex>\(0x[0-9a-f]*\)) (?P<instr>.* .*)' 
    with open(filepath, 'r') as f:
      log = f.read()

    matches = re.findall(pattern, log)
    num_matches = len(matches)
    
    print(f'{num_matches} logged instructions for hart<{hartid}> found.')

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
    addr_list, asm_list = parseSysemuLog(Path(args.sysemu_log), args.hart_id)

    # Exit if no instruction addresses have been found
    if len(addr_list) == 0:
        exit(0)
    
    # Create a dictionary with unique addresses
    address_dictionary = dict.fromkeys(addr_list)
    
    # Addr2line
    getAddr2line(address_dictionary, Path(args.debug_elf), opt)

    # print(address_dictionary)
    # Print output
    for addr, asm in zip(addr_list,asm_list):
        info1 = address_dictionary[addr][0]
        info2 = address_dictionary[addr][1]
        p = f'{addr:15} {asm:20} {info1:50}  ---  {info2}'
        print(p)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('debug_elf', type=Path, help='Path to ELF file with debug info')
    parser.add_argument('sysemu_log', type=Path, help='Path to sysemu.log')
    parser.add_argument('hart_id', type=int, help='hart id to inspect')
    args = parser.parse_args()

    # check args


    main(args)
