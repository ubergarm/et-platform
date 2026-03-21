#!/usr/bin/python3
import argparse
import shutil
from pathlib import Path
import re

def getOperandPrefix(source: Path) -> str:
    # Get prefix from source
    prefix_regex = r'src\/(.*)Cmd_w_pref[.]cc'
    filename = str(list(source.glob('src/*Cmd_w_pref.cc'))[0])
    print(f'Found file {filename}')
    m = re.search(prefix_regex, filename)
    if m:
        prefix = m.group(1)
        print(f'Prefix is {prefix}')
    else:
        print(f'Could not find prefix pattern')
        exit(-1)

    return prefix


def copyFilesToDest(origin: Path, dest: Path, prefix: str):
    # List of files to copy 
    srcFiles =  ['uberKernel.cc','SyncComputeNode.cc', f'{prefix}Cmd_w_pref.cc',
                f'{prefix}Cmd_inst_pref.cc', f'{prefix}Cmd_compute.cc', f'{prefix}Cmd_act_pref.cc']
    
    includeFiles = ['CommonCode.h', 'common_arguments.h', 'inst_pref_decls.h',
                'kernel_arguments.h', f'{prefix}Cmd_act_pref.h', f'{prefix}Cmd_compute.h',
                f'{prefix}Cmd_w_pref.h', 'SyncComputeNode.h', 'UberKernelCommon.h']

    # Create src destination folder
    destFolder = dest.joinpath('src')
    destFolder.mkdir()

    # Copy src files
    for f in srcFiles:
        o = origin.joinpath('src', f)
        d = destFolder.joinpath(f)

        if o.is_file():
            shutil.copy(o, d)
            print(f'Copied {o.as_posix()} into {d.as_posix()}')
        else:
            print(f'Could not find required file: {o.as_posix()}')
            exit(-1)
    
    # Create include destination folder
    destFolder = dest.joinpath('include')
    destFolder.mkdir()

    # Copy include files
    for f in includeFiles:
        o = origin.joinpath('inc', f)
        d = destFolder.joinpath(f)
        
        if o.is_file():
            shutil.copy(o, d)
            print(f'Copied {o.as_posix()} into {d.as_posix()}')
        else:
            print(f'Could not find required file: {o.as_posix()}')
            exit(-1)

def updateCMakeLists(cmakelistPath, lib_name, prefix):
    template = (f'\nadd_library({lib_name} STATIC\n'
    f'    {lib_name}/src/SyncComputeNode.cc\n'
    f'    {lib_name}/src/uberKernel.cc\n'
    f'    {lib_name}/src/{prefix}Cmd_compute.cc\n'
    f'    {lib_name}/src/{prefix}Cmd_w_pref.cc\n'
    f'    {lib_name}/src/{prefix}Cmd_act_pref.cc\n'
    f'    {lib_name}/src/{prefix}Cmd_inst_pref.cc)\n\n'
    f'target_link_libraries({lib_name} etsoc_crt0 dnnLibrary::dnn_lib)\n'
    f'target_include_directories({lib_name} PRIVATE {lib_name}/include helpers)\n'
    f'target_include_directories({lib_name} PUBLIC ../sdk/)\n'
    f'target_compile_options({lib_name} PRIVATE -fno-exceptions -falign-functions=64 -O3 -g3)\n')

    with open(cmakelistPath, "a") as f:
        f.write(template)

def main(args):
    dest = Path(args.dest)
    origin = Path(args.origin)

    # Get the prefix
    prefix = getOperandPrefix(origin)

    # Create new operand library folder
    newFolder = dest.joinpath(args.lib_name)
    newFolder.mkdir()  

    # Copy files
    copyFilesToDest(origin, newFolder, prefix)

    # Update libautogen CMakeList.txt to build new operand library (i.e: <args.lib_name>.a)
    cmakelistPath = dest.joinpath('CMakeLists.txt')

    if cmakelistPath.is_file():
        print(f'Writing File: {cmakelistPath}')
        updateCMakeLists(cmakelistPath, args.lib_name, prefix)
    else:
        print(f'File not found error: {cmakelistPath}\n')

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Import Codegen Kernel Arguments')
    parser.add_argument('origin', type=str, help='Path to the CompiledNetwork folder to import')
    parser.add_argument('dest', type=str, help='Path to the GP-SDK folder where it will be integrated')
    parser.add_argument('lib_name', type=str, help='Arbitrary name of the new operand added. Will be used as a folder and library name')
    args = parser.parse_args()

    main(args)