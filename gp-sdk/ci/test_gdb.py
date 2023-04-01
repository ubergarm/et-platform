"""
Test the GDB support.

Tests:
    - test_gdb_sysemu: Test GDB on sysemu
"""

# pylint: disable=redefined-outer-name # this is needed for local fixtures

from collections import namedtuple
import logging
from pathlib import Path
import time
import pytest

Command = namedtuple("Command", ["input", "expected_output"])


def gdb_script(entry_pc: int):
    """Generate the GDB script for testing"""
    return [
        Command(
            "print entryPoint_0",
            [
                f"*{entry_pc:#x}*entryPoint_0*KernelArguments*",
            ],
        ),
        Command(
            "target remote :1337",
            [
                "Remote debugging using :1337",
                "0x* in*",
                "at*",
                "*",
            ],
        ),
        Command(
            "break entryPoint_0 thread 1",
            [
                f"Breakpoint 1 at {entry_pc:#x}:*",
            ],
        ),
        Command(
            "continue",
            [
                "Continuing.",
                "",
                'Thread 1 "S0:M0:T0" hit Breakpoint 1, entryPoint_0 *',
                "*",
            ],
        ),
        Command(
            "print *vectors",
            [
                "*{numElements = 256, x = 0x800*, y = 0x800*, a = 3}",
            ],
        ),
        Command(
            "break saxpy_vector thread 1",
            [
                "Breakpoint 2 at 0x800*: *saxpy.cpp*",
            ],
        ),
        Command(
            "continue",
            [
                "Continuing.",
                "",
                'Thread 1 "S0:M0:T0" hit Breakpoint 2, saxpy_vector *',
                "*",
            ],
        ),
        Command("set disassemble-next-line on", []),
        Command("set width unlimited", []),
        Command(
            "next",
            [
                "*for*vlen*",
                "=> 0x*800*entryPoint_0*slli*",
            ],
        ),
        Command(
            "next",
            [
                "*asm*volatile*flw.ps*",
                "=> 0x*800*entryPoint_0*flw.ps*fa4*",
                "0x*800*entryPoint_0*flw.ps*fa5*",
            ],
        ),
        Command(
            "next",
            [
                "*asm*volatile*",
                "=> 0x*800*entryPoint_0*fmadd.ps*fa5,fa4,fa3,fa5*",
            ],
        ),
        Command(
            "next",
            [
                "*asm*volatile*fsw.ps*",
                "=> 0x*800*entryPoint_0*fsw.ps*fa5*",
            ],
        ),
        Command(
            "p $fa3",
            [
                "*float =*, int32 =*, int8 =*",
            ],
        ),
        Command(
            "p $fa4",
            [
                "*float =*, int32 =*, int8 =*",
            ],
        ),
        Command(
            "p $fa5",
            [
                "*float =*, int32 =*, int8 =*",
            ],
        ),
        Command("delete", []),
        Command("continue", []),
    ]


def launch_kernel(shell, launcher: Path, kernel: Path):
    """Launch a test kernel"""
    entry_pc = shell.find_symbol(Path(f"{kernel}_dbg"), "entryPoint_0")
    logging.debug("PC of entryPoint_0: %#x", entry_pc)

    return entry_pc, shell.popen(
        " ".join(
            [
                f"{launcher}",
                f"--kernel_path={kernel}",
                "--device_type=sysemu",
                f"--simulator_params='-gdb_at_pc={entry_pc:#x}'",
                "--kernel_launch_timeout=10000",
            ]
        ),
    )


def test_gdb_sysemu(request, build_dir, shell, gdb):
    """Execute a saxpy kernel and debug with GDB"""
    if not build_dir.exists():
        pytest.skip("the examples have not been built")
    if request.config.getoption("--skip-gdb"):
        pytest.skip("gdb tests are is disabled")
    logging.info("Starting kernel launcher")
    entry_pc, launcher = launch_kernel(
        shell,
        launcher=build_dir.host / "sdk/saxpy_launcher",
        kernel=build_dir.device / "tests/saxpy_vector.elf",
    )

    time.sleep(2)  # Wait some time for the launcher to start

    if request.config.getoption("--gdb-custom"):
        logging.info("Waiting for gdb connection")
    else:
        gdb = gdb(str(build_dir.device / "tests/saxpy_vector.elf_dbg"))
        gdb.read_until("Reading symbols")
        logging.info("Running gdb commands")
        for cmd in gdb_script(entry_pc):
            time.sleep(0.4)
            gdb.eval(cmd.input, cmd.expected_output)
        gdb.close()

    err = launcher.wait()
    assert err == 0, "launcher returned non-zero exit-code"
