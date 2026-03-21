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
import fnmatch
import glob
import os

Command = namedtuple("Command", ["input", "expected_output"])


def gdb_script_gcc(entry_pc: int):
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
                "*asm*volatile*fbcx*",
                "=> 0x*800*entryPoint_0*",
            ],
        ),
        Command(
            "next",
            [
                "*42*for*vlen*",
                "=> 0x*800*entryPoint_0*KernelArguments*",
            ],
        ),
        Command(
            "next",
            [
                "*asm*volatile*",
                "=> 0x*800*entryPoint_0*KernelArguments*",
            ],
        ),
        Command(
            "next",
            [
                "*800*entryPoint_0*KernelArguments*",
            ],
        ),
        Command(
            "p $fa3",
            [
                "61*asm*volatile*",
            ],
        ),
        Command(
            "p $fa4",
            [
                "*800*entryPoint*KernelArguments*fmadd*",
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

def gdb_script_clang(entry_pc: int):
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
                f"Breakpoint 1 at 0x800*:*",
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
                "*42*for*vlen*",
                "=> 0x*800*entryPoint_0*KernelArguments*",
            ],
        ),
        
        Command(
            "next",
            [
                "*asm*volatile*fbcx*",
                "=> 0x*800*entryPoint_0*",
            ],
        ),
        Command(
            "next",
            [
                "42*for*vlen*",
                "=> 0x*800*entryPoint_0*KernelArguments*bgeu*",
                "*slli*",
                "*add*",
                "*add*",
                "*li*",
                "*mov*",
            ],
        ),
        Command(
            "next",
            [
                "49*asm*volatile*flw.ps*",
                "=> 0x*800*entryPoint_0*KernelArguments*flw.ps*",
                "*0x*800*entryPoint_0*KernelArguments*flw.ps*",
            ],
        ),
        Command(
            "next",
            [
                "61*asm*volatile*",
                "=> 0x*800*entryPoint_0*fmadd.ps*",
            ],
        ),
        Command(
            "p $ft1",
            [
                "*float*3, 3, 3,*, int32 =*, int8 =*",
            ],
        ),
        Command(
            "p $ft2",
            [
                "*float*0, 1, 2, 3,*7*, int32 = *, int8 =*",
            ],
        ),
        Command(
            "p $ft3",
            [
                "*float =*100, 101, 102*107*, int32 =*, int8 =*",
            ],
        ),
        
        Command("delete", []),
        Command("continue", []),
    ]


def gdb_tls_script(entry_pc: int):
    """Generate the GDB script for testing"""
    return [
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
            "break testTlsVarSimple thread 1",
            [
                'Breakpoint 1 at*',
            ],
        ),
        Command(
            "continue",
            [
                'Continuing.',
                '*',
                'Thread 1 "S0:M0:T0" hit Breakpoint 1, testTlsVarSimple*',
                '30*testTls1++;',
            ],
        ),
        Command(
            "print testTls1",
            [
                '$1 = 0',
            ],
        ),
        Command("delete", []),
        Command("continue", []),
    ]

def gdb_script_silicon(entry_pc: int):
    """Generate the GDB script for testing"""
    return [
        Command(
            "target remote :51000",
            [
                "Remote debugging using :51000",
                "0x* in*",
            ],
        ),
        Command(
            "b saxpy.cpp:86",
            [
                'Breakpoint 1*',
            ],
        ),
        Command(
            "continue",
            [
                'Continuing.',
                '',
                'Breakpoint 1, entryPoint_0*',
                '*',
            ],
        ),
        Command(
            "print *vectors",
            [
                "*{numElements = 256, x = 0x800*, y = 0x800*, a = 3}",
            ],
        ),
        Command("delete", []),
        Command("continue", []),
    ]

def launch_gdb_server(shell):
    """Launch GDB server for silicon debugging"""
    return shell.popen(
        " ".join(
            [
                "debug-server -n 0 -s 0 -m 0x1 -t 200000",
                "&"
                "sleep 10 && kill $!"
            ]
        ),
    )


def find_coredump_file(path: Path):
    # List all files in the specified path
    files = os.listdir(path)
    
    # Filter files that start with "core" and contain "etsoc"
    filtered_files = [filename for filename in files if filename.startswith("core") and "etsoc" in filename]

    # Get absolute paths for filtered files
    absolute_paths = [str(path.joinpath(filename)) for filename in filtered_files]
    
    return absolute_paths
    #return filtered_files


def validate_core_dump(file_path):
    try:
        found = True

        with open(file_path, 'r') as file:
            contents = file.readlines()

            # Check if the lines with incremental numbers exist in the file
            line_to_check = ""
            for num in range(1, 1024):
                # Generate the line pattern to match
                if num != 1:
                    line_to_check = f"*{num}*LWP*{(num-1) * 2} *"
                else:
                    line_to_check = f"*{num}*process*{num} *"
                line_found = False
                for line in contents:
                    # Skip lines that don't match the expected format
                    if not fnmatch.fnmatch(line.strip(), line_to_check):
                        continue
                    
                    if fnmatch.fnmatch(line.strip(), line_to_check):
                        line_found = True
                        break

                if not line_found:
                    found = False
                    break

        return found

    except Exception as e:
        logging.info(f"An error occurred: {e}")
        return False


def launch_kernel(shell, launcher: Path, kernel: Path, entry_point: str, device: str, core_dump: bool = False):
    """Launch a test kernel"""
    entry_pc = shell.find_symbol(Path(f"{kernel}_dbg"), entry_point)
    logging.debug(f"PC of {entry_point}: %#{entry_pc}")

    command_parts = [
        str(launcher),
        f"--kernel_path={kernel}",
        f"--device_type={device}",
        "--kernel_launch_timeout=10000",
    ]

    if core_dump == True:
        command_parts.append("--enableCoreDump > test_coredump.log")

    if device == "sysemu":
        command_parts.append(f"--simulator_params='-gdb_at_pc={entry_pc:#x}'")
    elif device == "silicon" and not core_dump:
        command_parts.insert(0, "sleep 3 &&")
        command_parts.append("&")

    command = " ".join(command_parts)
    return entry_pc, shell.popen(command)



@pytest.mark.parametrize("kernel_info, script", [
    pytest.param(["saxpy_vector", "saxpy_launcher", "entryPoint_0"], gdb_script_gcc if os.environ["DEV_COMPILER"]=="gcc8.2" else  gdb_script_clang),
    pytest.param(["c_tls", "basic_launcher", "entryPoint"], gdb_tls_script),
])
def test_gdb_sysemu(kernel_info, script, request, build_dir, shell, gdb):
    """Execute a saxpy kernel and debug with GDB"""
    if not build_dir.exists():
        pytest.skip("the examples have not been built")
    if request.config.getoption("--skip-gdb"):
        pytest.skip("gdb tests are is disabled")
    logging.info("Starting kernel launcher")
    entry_pc, launcher = launch_kernel(
        shell,
        launcher=build_dir.host / f"sdk/{kernel_info[1]}",
        kernel=build_dir.device / f"tests/{kernel_info[0]}.elf",
        entry_point=kernel_info[2],
        device="sysemu"
    )

    time.sleep(2)  # Wait some time for the launcher to start

    if request.config.getoption("--gdb-custom"):
        logging.info("Waiting for gdb connection")
    else:
        gdb = gdb(str(build_dir.device / f"tests/{kernel_info[0]}.elf_dbg"))
        gdb.read_until("Reading symbols")
        logging.info("Running gdb commands")
        for cmd in script(entry_pc):
            time.sleep(0.4)
            gdb.eval(cmd.input, cmd.expected_output)
        gdb.close()

    err = launcher.wait()
    assert err == 0, "launcher returned non-zero exit-code"


@pytest.mark.parametrize("kernel_info, script", [
    pytest.param(["saxpy_vector", "saxpy_launcher", "entryPoint_0"], gdb_script_silicon),
])
def test_gdb_silicon(kernel_info, script, request, build_dir, shell, gdb):
    """Execute a saxpy kernel and debug with GDB"""
    if not build_dir.exists():
        pytest.skip("the examples have not been built")
    if request.config.getoption("--skip-gdb"):
        pytest.skip("gdb tests are is disabled")
    logging.info("Starting GDB server")
    launch_gdb_server(shell)
    time.sleep(2)  # Wait some time for the server to start
    logging.info("Starting kernel launcher")
    entry_pc, launcher = launch_kernel(
        shell,
        launcher=build_dir.host / f"sdk/{kernel_info[1]}",
        kernel=build_dir.device / f"tests/{kernel_info[0]}.elf",
        entry_point=kernel_info[2],
        device="silicon",
    )

    time.sleep(1)  # Wait some time for the launcher to start

    if request.config.getoption("--gdb-custom"):
        logging.info("Waiting for gdb connection")
    else:
        gdb = gdb(str(build_dir.device / f"tests/{kernel_info[0]}.elf_dbg"))
        gdb.read_until("Reading symbols")
        logging.info("Running gdb commands")
        for cmd in script(entry_pc):
            time.sleep(0.4)
            gdb.eval(cmd.input, cmd.expected_output)
        gdb.close()

    err = launcher.wait()
    assert err == 0, "launcher returned non-zero exit-code"


@pytest.mark.parametrize("kernel_info", [
    pytest.param(["exception", "basic_launcher", "entryPoint_0"]),
])
def test_gdb_coredump_silicon(kernel_info, request, build_dir, shell, gdb):
    """Execute a saxpy kernel and debug with GDB"""
    if not build_dir.exists():
        pytest.skip("the examples have not been built")
    if request.config.getoption("--skip-gdb"):
        pytest.skip("gdb tests are is disabled")

    logging.info("Starting kernel launcher")
    entry_pc, launcher = launch_kernel(
        shell,
        launcher=build_dir.host / f"sdk/{kernel_info[1]}",
        kernel=build_dir.device / f"tests/{kernel_info[0]}.elf",
        entry_point=kernel_info[2],
        device="silicon",
        core_dump = True,
    )

    time.sleep(2)  # Wait some time for the launcher to start
    launcher.wait()

    if request.config.getoption("--gdb-custom"):
        logging.info("Waiting for gdb connection")
    else:
        coredump_file = find_coredump_file(Path(os.getcwd()))
        gdb_cmd = " ".join([f"""riscv64-unknown-elf-gdb -batch -ex "set logging enabled on" -ex "info threads" -ex "set logging enabled off" -ex "quit" {str(build_dir.device / f"tests/{kernel_info[0]}.elf_dbg") + " " + coredump_file[0] + f" > {kernel_info[0]}.txt"}""",])
        shell.run(gdb_cmd)
        gdb_log_files = glob.glob(str(f"{os.getcwd()}/{kernel_info[0]}.txt"))
        for file in gdb_log_files:
            result = validate_core_dump(file)
            assert result == True, "Core dump file not matched"
