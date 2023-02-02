"""
Test the GDB support.

Tests:
    - test_gdb_sysemu: Test GDB on sysemu
"""

from collections import namedtuple
import fnmatch
import logging
from pathlib import Path
import subprocess
import shutil
import time

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
                "*in deviceGpSdkEntry*",
                "*deviceGpSdk.cc*",
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
                "   0x*800*entryPoint_0*flw.ps*fa5*",
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
                f"-kernel-path={kernel}",
                "-device-type=sysemu",
                f"-simulator-params='-gdb_at_pc={entry_pc:#x}'",
                "-kernel-launch-timeout=10000",
            ]
        ),
    )


def launch_gdb(shell, kernel: Path):
    """Launch GDB and advance past the license blob"""
    gdb = shell.popen(
        f"riscv64-unknown-elf-gdb {kernel}",
        stdin=subprocess.PIPE,
    )

    while gdb.poll() is None:
        line = gdb.stdout.readline().decode("utf-8")
        if "Reading symbols" in line:
            break

    return gdb


def check_commands(gdb, commands: list, timeout: float):
    """Execute a sequence of commands and check the expected output"""
    script = []
    for command in commands:
        time.sleep(timeout)
        script.append(command.input)
        gdb.stdin.write(command.input.encode("utf-8"))
        gdb.stdin.write(b"\n")
        gdb.stdin.flush()
        logging.debug("(gdb) %s", command.input)
        for expected_output in command.expected_output:
            output = gdb.stdout.readline().decode("utf-8").rstrip()
            if output.startswith("(gdb) "):
                output = output.replace("(gdb) ", "", 1)
            logging.debug(output)
            match = fnmatch.fnmatch(output, expected_output)
            if not match:
                logging.error("(gdb) %s", command.input)
                logging.error("  expected: %s", expected_output)
                logging.error("  found:    %s", output)
                return False, script
    return True, script


def test_gdb_sysemu(request, gp_sdk, shell):
    """Execute a saxpy kernel and debug with GDB"""
    build_cache = request.config.cache.get("build-make", None)
    if build_cache is None or not Path(build_cache).exists():
        logging.info("Building example for gdb")
        build_dir = Path("build")
        shell.mkdir(build_dir / "device")
        shell.cmake(
            source_dir=gp_sdk.path / "device",
            build_dir=build_dir / "device",
            cmake_toolchain_file="$ET_SDK_HOME/.builds/device/conan_toolchain.cmake",
            cmake_build_type="Release",
            address=gp_sdk.kernel_address,
            use_conan=True,
        )
        shell.make(
            build_dir / "device", target=["saxpy_vector.elf", "saxpy_vector.elf_dbg"]
        )
        shell.mkdir(build_dir / "host")
        shell.cmake(
            source_dir=gp_sdk.path / "host",
            build_dir=build_dir / "host",
            cmake_toolchain_file="$ET_SDK_HOME/.builds/host/conan_toolchain.cmake",
            cmake_build_type="Release",
            use_conan=True,
        )
        shell.make(build_dir / "host", target="saxpy_launcher")
    else:
        logging.info("Reusing build cache")
        build_dir = Path(build_cache)

    logging.info("Starting kernel launcher")
    entry_pc, launcher = launch_kernel(
        shell,
        launcher=build_dir / "host/sdk/saxpy_launcher",
        kernel=build_dir / "device/tests/saxpy_vector.elf",
    )

    time.sleep(2)  # Wait some time for the launcher to start

    if request.config.getoption("--gdb-custom"):
        logging.info("Waiting for gdb connection")
    else:
        logging.info("Starting gdb session")
        gdb = launch_gdb(
            shell,
            kernel=build_dir / "device/tests/saxpy_vector.elf_dbg",
        )
        logging.info("Running gdb commands")
        res, script = check_commands(
            gdb,
            commands=gdb_script(entry_pc),
            timeout=0.4,
        )
        logging.debug("Saving gdb commands")
        Path("script.gdb").write_text("\n".join(script))
        assert res, "gdb mismatch"
        gdb.stdin.close()
        err = gdb.wait()
        assert err == 0, "gdb returned non-zero exit-code"

    err = launcher.wait()
    assert err == 0, "launcher returned non-zero exit-code"
