#!/usr/bin/env python3

import os
import argparse
import sys
import subprocess
import mmap

KERNELS = [
    "print",
    "print2",
    "bss",
    "data",
    "fftKernel",
    "saxpy_scalar",
    "saxpy_vector",
    "saxpy_profiling",
    "syncDeviceBasic",
    "syncMinion",
    "txfma",
    "hang",
]

LAUNCHERS = [
    "basic_launcher",
    "barrier_launcher",
    "saxpy_launcher",
    "multiKernel_launcher",
    "hello_world_launcher",
    "fft_launcher",
    "txfma_launcher",
]

KERNEL_LAUNCHERS = {
    "print": "basic_launcher",
    "print2": "basic_launcher",
    "bss": "basic_launcher",
    "data": "basic_launcher",
    "fftKernel": "fft_launcher",
    "saxpy_scalar": "saxpy_launcher",
    "saxpy_vector": "saxpy_launcher",
    "saxpy_profiling": "saxpy_launcher",
    "syncDeviceBasic": "barrier_launcher",
    "syncMinion": "barrier_launcher",
    "txfma": "txfma_launcher",
    "hang": "basic_launcher",
    "exception": "basic_launcher",
}


if __name__ == "__main__":
    """ launcher gpSdk test for CI-gitlab and check de results.
    return an error if some test fail."""

    parser = argparse.ArgumentParser("")
    parser.add_argument("--path", help="", required = True)
    args = parser.parse_args()
    
    error = False
    launcher_base_path = ""
    kernels_base_path = ""
    device_type = ""
    try:  
        launcher_base_path = os.environ["LAUNCHERS_BASE_PATH"]
    except KeyError: 
        print ("Please set the environment variable LAUNCHERS_BASE_PATH")

    try:  
        kernels_base_path = os.environ["KERNELS_BASE_PATH"]
    except KeyError: 
        print ("Please set the environment variable KERNELS_BASE_PATH")

    try:  
        device_type = os.environ["DEVICE_TYPE"]
    except KeyError: 
        print ("Please set the environment variable DEVICE_TYPE")


    cwd = os.getcwd()
    print(f'Current directory {cwd}')
    os.chdir(f'{args.path}')
    cwd = os.getcwd()
    print(f'Change Current directory to {cwd}')

    print(f'launcher_base_path --> {launcher_base_path}')
    print(f'kernels_base_path -> {kernels_base_path}')
    
    result= subprocess.run([f'{launcher_base_path}/{KERNEL_LAUNCHERS["print"]} -kernel_path={kernels_base_path}/print.elf -device-type={device_type}'], shell=True, capture_output=True, text=True)

    print(f'test: print')
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    if result.returncode != 0:
        error = True

    result= subprocess.run([f'{launcher_base_path}/{KERNEL_LAUNCHERS["print2"]} -kernel_path={kernels_base_path}/print2.elf -device-type={device_type}'], shell=True, capture_output=True, text=True)
    print(f'test: print2')
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    if result.returncode != 0:
        error = True

    result= subprocess.run([f'{launcher_base_path}/{KERNEL_LAUNCHERS["bss"]} -kernel_path={kernels_base_path}/bss.elf -device-type={device_type}'], shell=True, capture_output=True, text=True)
    print(f'test: bss')
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    if result.returncode != 0:
        error = True

    result= subprocess.run([f'{launcher_base_path}/{KERNEL_LAUNCHERS["data"]} -kernel_path={kernels_base_path}/data.elf -device-type={device_type}'], shell=True, capture_output=True, text=True)
    print(f'test: data')
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    if result.returncode != 0:
        error = True

    result= subprocess.run([f'{launcher_base_path}/{KERNEL_LAUNCHERS["fftKernel"]} -kernel_path={kernels_base_path}/fftKernel.elf -device-type={device_type}'], shell=True, capture_output=True, text=True)
    print(f'test: fftKernel')
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    if result.returncode != 0:
        error = True
        
    result= subprocess.run([f'{launcher_base_path}/{KERNEL_LAUNCHERS["saxpy_scalar"]} -kernel_path={kernels_base_path}/saxpy_scalar.elf -device-type={device_type}'], shell=True, capture_output=True, text=True)
    print(f'test: saxpy_scalar')
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    if result.returncode != 0:
        error = True

    result= subprocess.run([f'{launcher_base_path}/{KERNEL_LAUNCHERS["saxpy_vector"]} -kernel_path={kernels_base_path}/saxpy_vector.elf -device-type={device_type}'], shell=True, capture_output=True, text=True)
    print(f'test: saxpy_vector')
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    if result.returncode != 0:
        error = True

    result= subprocess.run([f'{launcher_base_path}/{KERNEL_LAUNCHERS["txfma"]} -kernel_path={kernels_base_path}/txfma.elf -device-type={device_type}'], shell=True, capture_output=True, text=True)
    print(f'test: txfma')
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    if result.returncode != 0:
        error = True
        
    result= subprocess.run([f'{launcher_base_path}/{KERNEL_LAUNCHERS["hang"]} -kernel_path={kernels_base_path}/hang.elf -enableCoreDump -device-type={device_type}'], shell=True, capture_output=True, text=True)
    print(f'test: hang')
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    #For this test a Fail it is expected so 
    if result.returncode != 1:
        print("expected hang test fails.")

    corefound = False
    #check coredump file has been created
    for file in os.listdir(os.getcwd()):
        if file.startswith("core"):
            corefound = True
            print(f'Core was {file}')
    if not corefound:
        error = True

    result= subprocess.run([f'{launcher_base_path}/{KERNEL_LAUNCHERS["exception"]} -kernel_path={kernels_base_path}/exception.elf -device-type={device_type}'], shell=True, capture_output=True, text=True)
    print(f'test: exception')
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    #For this test a Fail it is expected so
    if result.returncode != 1:
        print("expected exception test fails.")

    if device_type == "sysemu":
        #find exception interrupt in sysemu.log0 file
        with open('sysemu.log0', 'rb', 0) as file, \
             mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as s:
            if s.find(b'raise_device_interrupt(type = PU)') == -1:
                error = True
    elif device_type == "silicon":
        if ("Error code: KernelLaunchException" not in result.stdout):
            error = True            
    
    if error is True:
        print("error was Found")
        exit(1)
    else:
        print("All TEST PASSED")
