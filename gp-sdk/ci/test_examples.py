"""
Test the GP-SDK exmamples.

Tests:
    - test_build_examples: Build both device-side and host-side artifacts
    - test_run_example: Run each of the provided examples
"""

# pylint: disable=fixme # notes about current GP-SDK issues

from collections import defaultdict
import csv
from dataclasses import dataclass
import logging
from typing import Optional
from pathlib import Path
import pytest


@dataclass
class Symbol:
    """An ELF symbol"""

    name: str
    type: str
    addr: Optional[list] = None


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
}


EXTRA_ARGS = defaultdict(list)
EXTRA_ARGS["saxpy_profiling"] = ["--launch_mult=2"]


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


def check_device_trace(shell, path: Path):
    """Check whether the device trace exists and is well formatted"""
    assert path.exists()
    dt2json = shell.run(f"dt2json -t {path}")
    trace = list(csv.reader(dt2json.stdout.decode("utf-8").splitlines(), delimiter=";"))
    assert trace[0][0] == "TIMESTAMP"
    assert trace[0][1] == "STATS_TYPE"


def check_run_artifacts(shell, device_type: str, nkernels: int = 1):
    """Default checks for run artifacts"""
    if device_type == "sysemu":
        logging.info("Checking UART logs")
        for log in [
            "pu_uart0_tx.log",
            "pu_uart1_tx.log",
            "spio_uart0_tx.log",
            "spio_uart1_tx.log",
        ]:
            assert Path(log).exists()

        uart_log = Path("spio_uart0_tx.log").read_text()
        assert "Initialize Minion Shire" in uart_log

        logging.info("Checking sysemu logs")
        sysemu_log_path = Path("sysemu.log0")
        assert sysemu_log_path.exists()
        sysemu_log = sysemu_log_path.read_text()
        assert "ERROR" not in sysemu_log
        assert "FATAL" not in sysemu_log

    logging.info("Checking device trace")
    if nkernels == 1:
        check_device_trace(
            shell,
            Path("traceKernels_dev0_0.bin"),
        )
    else:
        for i in range(nkernels):
            check_device_trace(
                shell,
                Path(f"traceKernels_dev0_0_{i}.bin"),
            )


@pytest.mark.parametrize(
    "generator",
    [
        "make",
        # "ninja",  # FIXME: Currently not working, see CS-49
    ],
)
def test_build_examples(gp_sdk, shell, generator, build_dir):
    """Build both device-side and host-side artifacts"""
    logging.info("Building device-side kernels")
    shell.mkdir(shell.tmp_path / "device")
    shell.cmake(
        source_dir=gp_sdk.path / "device",
        build_dir="device",
        generator=generator,
        cmake_toolchain_file="$ET_SDK_HOME/.builds/device/conan_toolchain.cmake",
        cmake_build_type="Release",
        address=gp_sdk.kernel_address,
        use_conan=True,
    )
    shell.make("device", generator=generator)
    check_symbols(
        shell,
        Path("device/sdk/libetsoc_crt0.a"),
        [
            Symbol(name="deviceGpSdkEntry", type="TtWw"),
            Symbol(name="_start", type="TtWw"),
        ],
    )
    for kernel in KERNELS:
        symbols = [
            Symbol(name="_start", type="TtWw"),
            Symbol(name="entryPoint_0", type="TtWw"),
            Symbol(name="entryPoint_1", type="TtWw"),
        ]
        logging.info("Checking device/tests/%s.elf", kernel)
        check_symbols(shell, Path(f"device/tests/{kernel}.elf"), symbols)
        check_symbols(shell, Path(f"device/tests/{kernel}.elf_dbg"), symbols)
    logging.info("Building host-side launchers")
    shell.mkdir(Path("host"))
    shell.cmake(
        source_dir=gp_sdk.path / "host",
        build_dir=Path("host"),
        generator=generator,
        cmake_toolchain_file="$ET_SDK_HOME/.builds/host/conan_toolchain.cmake",
        cmake_build_type="Release",
        use_conan=True,
    )
    shell.make("host", generator=generator)
    for launcher in LAUNCHERS:
        logging.info("Checking host/sdk/%s", launcher)
        check_linked_libraries(
            shell,
            Path(f"host/sdk/{launcher}"),
            [
                "libetrt.so",
                "libdeviceLayer.so",
            ],
        )
    build_dir.save_device(shell.tmp_path / "device")
    build_dir.save_host(shell.tmp_path / "host")


@pytest.mark.parametrize("device_type", ["sysemu", "silicon"])
@pytest.mark.parametrize("kernel", KERNELS)
def test_run_example(shell, device_type, kernel, build_dir):
    """Run one of the provided examples"""
    if not build_dir.exists():
        pytest.skip("the examples have not been built")
    logging.info("Running %s on %s", kernel, device_type)
    kernel_path = build_dir.device / "tests" / f"{kernel}.elf"
    launch_cmd = " ".join(
        [
            str(build_dir.host / "sdk" / KERNEL_LAUNCHERS[kernel]),
            f"--kernel_path={kernel_path}",
            f"--device_type={device_type}",
        ]
        + EXTRA_ARGS[kernel]
    )
    shell.run(launch_cmd)
    check_run_artifacts(shell, device_type)


@pytest.mark.parametrize("device_type", ["sysemu", "silicon"])
def test_run_multi_kernel(shell, device_type, build_dir):
    """Run multi kernel example"""
    if not build_dir.exists():
        pytest.skip("the examples have not been built")
    kernels = ["bss", "saxpy_scalar"]
    logging.info("Running %s on %s", kernels, device_type)
    kernel_paths = [build_dir.device / "tests" / f"{kernel}.elf" for kernel in kernels]
    launch_cmd = " ".join(
        [
            str(build_dir.host / "sdk/multiKernel_launcher"),
            f"--kernel_path1={kernel_paths[0]}",
            f"--kernel_path2={kernel_paths[1]}",
            f"--device_type={device_type}",
        ]
    )
    shell.run(f"( cd {shell.tmp_path} ; {launch_cmd} )")
    check_run_artifacts(shell, device_type, nkernels=2)
