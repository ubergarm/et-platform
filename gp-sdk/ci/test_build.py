"""
Build the GP-SDK exmamples.

Tests:
    - test_build_host: Build host-side launchers
    - test_build_device: Build device-side kernels
"""

# pylint: disable=fixme # notes about current GP-SDK issues

import logging
from dataclasses import dataclass
import os
from typing import Optional
from pathlib import Path
import pytest

LAUNCHERS_GCC = [
    "basic_launcher",
    "barrier_launcher",
    "saxpy_launcher",
    "multiKernel_launcher",
    "hello_world_launcher",
    "fft_launcher",
    "txfma_launcher",
]

LAUNCHERS_CLANG = [
    "basic_launcher",
    "barrier_launcher",
    "saxpy_launcher",
    "multiKernel_launcher",
    "hello_world_launcher",
    "fft_launcher",
    "txfma_launcher",
    "exhaustive_cast_launcher",
]

KERNELS_GCC = [
    "print",
    "print2",
    "bss",
    "sysemu_fatal",
    "data",
    "fftKernel",
    "saxpy_scalar",
    "saxpy_vector",
    "saxpy_profiling",
    "syncDeviceBasic",
    "syncAll",
    "syncMinion",
    "txfma",
    "hang",
    "exception",
]

KERNELS_CLANG = [
    "print",
    "print2",
    "bss",
    "sysemu_fatal",
    "data",
    "fftKernel",
    "saxpy_scalar",
    "saxpy_vector",
    "saxpy_profiling",
    "syncDeviceBasic",
    "syncAll",
    "syncMinion",
    "txfma",
    "hang",
    "exception",
    "exhaustive_cast",
]


@dataclass
class Symbol:
    """An ELF symbol"""

    name: str
    type: str
    addr: Optional[list] = None


def collect_nm(shell, elf: Path):
    """Extract all the symbols of a given ELF"""
    assert elf.exists()
    cmd = shell.run(f"nm -S {elf}", quiet=True)
    stdout = cmd.stdout.decode("utf-8")
    for line in stdout.splitlines():
        words = line.split()
        if len(words) == 2:
            yield Symbol(name=words[-1], type=words[-2])
        elif len(words) == 3:
            yield Symbol(name=words[-1], type=words[-2], addr=[words[0]])
        elif len(words) == 4:
            yield Symbol(name=words[-1], type=words[-2], addr=[words[0], words[1]])


def collect_ldd(shell, elf: Path):
    """Extract all the linked libraries of a given ELF"""
    assert elf.exists()
    cmd = shell.run(f"ldd {elf}", quiet=True)
    stdout = cmd.stdout.decode("utf-8")
    for line in stdout.splitlines():
        words = line.split()
        if len(words) < 2:
            continue
        if words[1] == "=>":
            yield words[0]


def check_symbols(shell, elf: Path, symbols: list):
    """Check whether the ELF contains a given set of symbols"""
    found_symbols = list(collect_nm(shell, elf))

    def is_defined(target):
        for sym in found_symbols:
            if sym.name == target.name and sym.type in target.type:
                return True
        return False

    for sym in symbols:
        assert is_defined(sym), f"Symbol {sym.name} is not defined in {elf}"


def check_linked_libraries(shell, elf: Path, libs: list):
    """Check whether the ELF is linked with a given set of libraries"""
    found_libs = list(collect_ldd(shell, elf))

    for lib in libs:
        assert lib in found_libs, f"Library {lib} is not linked with {elf}"


def test_build_host(gp_sdk, shell, build_dir):
    """Build host-side launchers"""
    logging.info("Building host-side launchers")
    shell.mkdir(Path("build"))
    shell.cmake(
        source_dir=gp_sdk.path / "host",
        build_dir=Path("build"),
        generator="make",
        cmake_toolchain_file="$ET_SDK_HOME/.builds/host/conan_toolchain.cmake",
        cmake_build_type="Release",
        use_conan=True,
    )
    shell.make("build")
    build_dir.save_host(shell.tmp_path / "build")


def test_build_device(gp_sdk, shell, build_dir):
    """Build device-side kernels"""
    logging.info("Building device-side kernels")
    shell.mkdir(Path("build"))
    shell.cmake(
        source_dir=gp_sdk.path / "device",
        build_dir=Path("build"),
        generator="make",
        cmake_toolchain_file="$ET_SDK_HOME/.builds/device/$DEV_COMPILER/conan_toolchain.cmake",
        cmake_build_type="Release",
        address=gp_sdk.kernel_address,
        use_conan=True,
    )
    shell.make("build")
    build_dir.save_device(shell.tmp_path / "build")

@pytest.mark.skipif(os.environ["DEV_COMPILER"]=="clang11", reason="Skipping GCC tests as DEV_COMPILER = clang11")
@pytest.mark.parametrize("launcher", LAUNCHERS_GCC)
def test_host_artifacts_gcc(launcher, build_dir, shell):
    """Check the linkage of the GCC launchers"""
    logging.info("Checking host/sdk/%s", launcher)
    check_linked_libraries(
        shell,
        build_dir.host / "sdk" / launcher,
        [
            "libetrt.so",
            "libdeviceLayer.so",
        ],
    )

@pytest.mark.skipif(os.environ["DEV_COMPILER"]=="gcc8.2", reason="Skipping GCC tests as DEV_COMPILER = gcc8.2")
@pytest.mark.parametrize("launcher", LAUNCHERS_CLANG)
def test_host_artifacts_clang(launcher, build_dir, shell):
    """Check the linkage of the clang launchers"""
    logging.info("Checking host/sdk/%s", launcher)
    check_linked_libraries(
        shell,
        build_dir.host / "sdk" / launcher,
        [
            "libetrt.so",
            "libdeviceLayer.so",
        ],
    )

@pytest.mark.skipif(os.environ["DEV_COMPILER"]=="clang11", reason="Skipping GCC tests as DEV_COMPILER = clang11")
@pytest.mark.parametrize("kernel", KERNELS_GCC)
def test_device_kernel_artifacts_gcc(kernel, build_dir, shell):
    """Check the symbols defined in the GCC device kernels"""
    logging.info("Checking device/tests/%s.elf", kernel)
    symbols = [
        Symbol(name="_start", type="TtWw"),
    ]
    check_symbols(shell, build_dir.device / "tests" / f"{kernel}.elf", symbols)
    check_symbols(shell, build_dir.device / "tests" / f"{kernel}.elf_dbg", symbols)

@pytest.mark.skipif(os.environ["DEV_COMPILER"]=="gcc8.2", reason="Skipping Clang tests as DEV_COMPILER = gcc8.2")
@pytest.mark.parametrize("kernel", KERNELS_CLANG)
def test_device_kernel_artifacts_clang(kernel, build_dir, shell):
    """Check the symbols defined in the clang device kernels"""
    logging.info("Checking device/tests/%s.elf", kernel)
    symbols = [
        Symbol(name="_start", type="TtWw"),
    ]
    check_symbols(shell, build_dir.device / "tests" / f"{kernel}.elf", symbols)
    check_symbols(shell, build_dir.device / "tests" / f"{kernel}.elf_dbg", symbols)

def test_device_sdk_artifacts(build_dir, shell):
    """Check the device crt0"""
    check_symbols(
        shell,
        build_dir.device / "sdk/libetsoc_crt0.a",
        [
            Symbol(name="deviceGpSdkEntry", type="TtWw"),
            Symbol(name="_start", type="TtWw"),
        ],
    )
