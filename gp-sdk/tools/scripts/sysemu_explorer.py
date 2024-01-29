#!/usr/bin/env python3
from ast import List, Tuple
import io
import argparse
import logging
import re
import subprocess
import os
from pathlib import Path
from subprocess import PIPE

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def getAddr2line(addr_list: list, elves: list, opt: str):
    """
    Add the source code line for each instruction to the instruction information
    """

    # match linux file path
    src_path_re = re.compile(
        r'(?P<function>.*) at (?P<filename>.*):(?P<line>[\d?]+)')

    # Join all unique addr in a single newline string to pass to addr2line
    txt = '\n'.join([item['address'] for item in addr_list])

    # Run addr2line on each ELF file to resolve all addresses
    source_codes = {}
    for elf in elves:
        # Call addr2line and get output
        cmd = f'riscv64-unknown-elf-addr2line -e {elf} -f -p --demangle {opt}'
        lines = subprocess.run(cmd, stdout=subprocess.PIPE, input=txt,
                               shell=True, timeout=360, text=True, check=True)
        if lines.stderr:
            raise RuntimeError(lines.stderr)
        lines = lines.stdout.splitlines()

        # Get function name, file name and line number addr2line output
        source_info = [list(m.groups()) if m else None
                       for m in map(src_path_re.match, lines)]

        # Addr source information to address list dictionary
        for i, item in enumerate(addr_list):
            if item['file'] is not None:
                continue

            # If source hasn't been set yet set it
            # (assumes 1 ELF contains the address)
            if source_info[i] is not None:
                if source_info[i][2].isnumeric():
                    item['file'] = source_info[i][1]
                    item['line'] = int(source_info[i][2])
                else:
                    item['file'] = source_info[i][0]
                    item['line'] = None

        # Get full source code for each source file identified
        source_codes.update(dict.fromkeys(
            [x[1] for x in source_info if x is not None]))

    # Add each unique source code file in results add it to a dict
 
    logger.info(f'Found {len(source_codes)} source files.')

    # Delete keys from files that do not exist
    for path in list(source_codes.keys()):
        if os.path.isfile(path):
            with open(path, 'r') as f:
                source_codes[path] = f.readlines()
        else:
            logging.warning(f"Could not open {path}!")
            source_codes.pop(path, None)

    # Fill each address in the dictionary with:
    #  1- addr2line output (instr address, assembly, function, file:lineno)
    #  2- the line of source code corresponding to file:lineno
    for item in addr_list:
        # get source code file and line number
        path = item['file']
        code = None
        
        if source_codes and path in source_codes:
            line_num = item['line'] or 0
            if len(source_codes[path]) > (line_num-1):
                code = source_codes[path][line_num-1].strip()

        item['code'] = code


def parseSysemuLog(filepath: Path, hartid: int):
    """
    Parses a sysemu log file and creates a list of information per instruction
    """
    
    sysemu_re = (
        rf'DEBUG EMU: \[H{hartid} \b'
        rf'.*I\((?P<mode>U|S|M)\): '
        rf'(?P<addr>0x[0-9a-f]*) '
        rf'(?P<instrhex>\(0x[0-9a-f]*\)) '
        rf'(?P<instr>.*)\n'
    )

    # Read sysemu log
    with open(filepath, 'r') as f:
        log = f.read()

    # Holds information per instruction
    addr_list = []

    # Create a list entry to hold info about each address
    for match in re.finditer(sysemu_re, log):
        # match addr2line address format
        addr = f"0x{int(match.group('addr'), 16):016X}"
        addr_list.append({
            'address': addr,
            'asm': match.group('instr'),
            'mode': match.group('mode'),
            'file': None,
            'line': None,
            'code': None
        })

    logger.info(
        f'{len(addr_list)} logged instructions for hart<{hartid}> found.')

    return addr_list


def main(args):
    # options (todo)
    opt = ""

    # Parse sysemu log
    addr_list = parseSysemuLog(Path(args.sysemu_log), args.hart_id)

    getAddr2line(addr_list, args.debug_elf, opt)

    # Exit if no instruction addresses have been found
    if len(addr_list) == 0:
        logger.warning("No instruction addresses found. Exiting.")
        exit(0)

    max_asm_len = max([len(e['asm']) for e in addr_list])

    src_width = 100
    # Print output
    for item in addr_list:
        addr = item['address']
        mode = item['mode']
        asm = item['asm']

        line = f":{item['line'] or '':<5}"
        src = item['file'] + line

        if len(src) > src_width:
            w = src_width - 3
            src = f"...{src[-w:]}"

        code = item['code'] or ''

        print('{{}} {{:18}}  {{:{}}} {{:{}}} --- {{}}'
              .format(max_asm_len, src_width)
              .format(mode, addr, asm, src, code)
              )


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--debug-elf', type=Path, nargs='+',
                        help='Path to ELF file(s) with debug info')
    parser.add_argument('--sysemu-log', type=Path, help='Path to sysemu.log')
    parser.add_argument('--hart-id', type=int, help='hart id to inspect')
    args = parser.parse_args()

    # check args

    main(args)
