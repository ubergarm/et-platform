"""
Test the GP-SDK examples.

Tests:
    - test_run_example: Run each of the provided examples
    - test_run_multi_kernel: Run multiple kernels
    - test_run_arguments: Run kernel with different optional arguments
"""

# pylint: disable=fixme # notes about current GP-SDK issues

from collections import defaultdict
import csv
import logging
import glob
from pathlib import Path
import re
import subprocess
import pytest
from collections import namedtuple

KERNEL_LAUNCHERS = {
    "bss": "basic_launcher",
    "busy10sec": "basic_launcher",
    "c_contructors": "basic_launcher",
    "c_tls": "basic_launcher",
    "cacheops_flush": "basic_launcher",
    "check_pmc": "basic_launcher",
    "cpp_contructors": "basic_launcher",
    "cpp_tls": "basic_launcher",
    "data": "basic_launcher",
    "exception": "basic_launcher",
    "external_tls": "basic_launcher",
    "fail_abort": "basic_launcher",
    "fail_assert": "basic_launcher",
    "fftKernel": "fft_launcher",
    "fnodiv": "basic_launcher",
    "gp": "basic_launcher",
    "hang": "basic_launcher",
    "OneTrapOnSync": "basic_launcher",
    "print": "basic_launcher",
    "print2": "basic_launcher",
    "profiling_simple": "basic_launcher",
    "profiling_stress": "basic_launcher",
    "saxpy_profiling": "saxpy_launcher",
    "saxpy_scalar": "saxpy_launcher",
    "saxpy_vector": "saxpy_launcher",
    "syncAll": "basic_launcher",
    "syncDeviceBasic": "barrier_launcher",
    "syncShire2EP": "barrier_launcher",
    "syncAll": "basic_launcher",
    "syncMinion": "barrier_launcher",
    "txfma": "txfma_launcher",
    "variableStrings": "basic_launcher"
}


KERNELS = list(KERNEL_LAUNCHERS.keys())

SKIP_SYSEMU = ["check_pmc", "busy10sec"]
SKIP_ANY = ["variableStrings"]

EXTRA_ARGS = defaultdict(list)
EXTRA_ARGS["saxpy_profiling"] = ["--launch_mult=2"]
EXTRA_ARGS["hang"] = ["--enableCoreDump"]
EXTRA_ARGS["exception"] = ["--enableCoreDump"]
EXTRA_ARGS["bss"] = ["--num_launches=5"]
EXTRA_ARGS["data"] = ["--num_launches=5"]
EXTRA_ARGS["check_pmc"] = ["--kernel_launch_timeout=12"]
EXTRA_ARGS["busy10sec"] = ["--kernel_launch_timeout=12"]
EXTRA_ARGS["OneTrapOnSync"] = ["--enableCoreDump"]

# only needed for device_type = sysemu
EXTRA_ARGS["profiling_stress"] = ["--kernel_launch_timeout=200"]

SHOULD_FAIL = ["hang", "exception", "OneTrapOnSync", "fail_abort", "fail_assert"]

ERROR_COMMENT = {
    "hang": "Generate code hang",
    "exception": "Generate code exception",
}

MASK_SWEEP_KERNELS = ["syncAll", "syncDeviceBasic", "syncMinion"]
MASK_SWEEP = ["0x1", "0xF", "0xFF", "0xFFFF", "0xFFFFFFFF"]

CmdLineArg = namedtuple("CmdLineArg", ["param", "valid"])

KERNEL_PATH = [CmdLineArg("--kernel_path", True), CmdLineArg("-k", True)]
DEVICE_TYPE = [CmdLineArg("--device_type", True), CmdLineArg("--device-type",
                                                             False), CmdLineArg("-device_type", True), CmdLineArg("-d", True)]
SIMULATOR_PARAMS = [CmdLineArg("--simulator_params", True), CmdLineArg(
    "--simulator-params", False), CmdLineArg("-simulator_params", False)]
SIMULATOR_OPT_ARG = [CmdLineArg("-l -lm 0", True), CmdLineArg(
    "-l -lm 0 -mem_check", True), CmdLineArg("-l -lm 0 -mem-check", False)]
ENABLE_CORE = [CmdLineArg("--enableCoreDump", True)]

str_example_not_built = "the examples have not been built"


def check_variable_strings(trace):
    """ This function validates the trace which contains et_printf output.
    It contains a total of 23 lines, and the first line contains the information of the file name "variableStrings". 
    It is a hardcoded token used to identify the correct traces. It also contains the number of characters for the next et_printf call. 
    For a better understanding, refer to the print loop in device\tests\variableStrings\variableStrings.cc. 
    The trace looks like below:
    1575921352789;string;{plain_string};{"variableStrings,2\n"}
    1575921355817;string;{plain_string};{"01"}
    1575921360652;string;{plain_string};{"variableStrings,4\n"}
    1575921362805;string;{plain_string};{"0123"}
    1575921367638;string;{plain_string};{"variableStrings,8\n"}
    1575921370124;string;{plain_string};{"01234567"}"""

    for i in range(1, len(trace), 2):
        if i + 1 <= len(trace):
            trace_identifier = re.findall(
                r'\b\w+\b', trace[i][3].split(',')[-2])
            number_of_chars = re.findall(r'\d+', trace[i][3].split(',')[-1])
            actual_string = re.findall(r'\d+', trace[i + 1][3])
            if trace_identifier[0] == "variableStrings":
                logging.info(f"Checking srting size of {number_of_chars}")
                assert int(number_of_chars[0]) == len(actual_string[0])


def trace_contains_token(trace, token: str) -> bool:
    for substr in trace:
        if any(token in item for item in substr):
            return True
    return False


def check_device_trace(shell, path: Path):
    """Check whether the device trace exists and is well formatted"""
    assert path.exists()
    dt2json = shell.run(f"dt2json -t {path}")
    #some traceKernels_dev0_0.bin contains a null character which can't be readed by csv as fail_assert and fail_abort
    trace = list(csv.reader(dt2json.stdout.decode("utf-8").replace('\0','').splitlines(), delimiter=";"))
    assert trace[0][0] == "TIMESTAMP"
    assert trace[0][1] == "STATS_TYPE"

    if trace_contains_token(trace, 'variableStrings'):
        check_variable_strings(trace)


def check_run_artifacts(shell, device_type: str, multikernel: bool = False, kernelId1: int = 0, kernelId2: int = 0):
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

        uart_log = Path("spio_uart0_tx.log").read_text(encoding="utf-8")
        assert "Initialize Minion Shire" in uart_log

        logging.info("Checking sysemu logs")
        sysemu_log_path = Path("sysemu.log0")
        assert sysemu_log_path.exists()
        sysemu_log = sysemu_log_path.read_text(encoding="utf-8")
        assert "ERROR" not in sysemu_log
        assert "FATAL" not in sysemu_log

    logging.info("Checking device trace")
    if multikernel == False:
        check_device_trace(
            shell,
            Path("traceKernels_dev0_0.bin"),
        )
    else:
        #currently only 2 kernerl are allowed to run on multikernel
        check_device_trace(shell, Path(f'traceKernels_dev0_0_{kernelId1}.bin'))
        check_device_trace(shell, Path(f'traceKernels_dev0_0_{kernelId2}.bin'))

def check_core_dump(gdb, elf: Path, comment: str, skip_gdb: bool):
    """Check whether the core dump exists and is debuggable"""
    logging.info("Checking core dump")
    core_dumps = glob.glob("core.*.etsoc.*")
    assert len(core_dumps) == 1, "Should create exactly one core dump file"
    if skip_gdb:
        logging.info("Skipping gdb checks")
        return
    gdb = gdb(f"{elf} {core_dumps[0]}")
    gdb.eval("c", [])
    gdb.read_until("Core was generated by")
    gdb.eval("set pagination off", [])
    gdb.eval("set width unlimited", [])
    gdb.eval("set disassemble-next-line on", [])
    gdb.eval(
        "info threads",
        [
            "*entryPoint_0*args*",
            "*at*.cc:*",
            "*",
            "*Current thread is 1*",
            "*Id*Target*Id*Frame*",
        ],
    )
    gdb.read_until("1023")
    gdb.eval("list")
    gdb.read_until(comment)
    gdb.close()


@pytest.mark.parametrize("device_type", ["sysemu", "silicon"])
@pytest.mark.parametrize("kernel", KERNEL_LAUNCHERS.keys())
def test_run_example(shell, device_type, kernel, build_dir, gdb, request):
    """Run one of the provided examples"""
    if not build_dir.exists():
        pytest.skip("the examples have not been built")
    if device_type == "sysemu" and kernel in SKIP_SYSEMU:
        pytest.skip(f"do not run {kernel} on {device_type}")
    if kernel in SKIP_ANY:
        pytest.skip(f"Skipping {kernel} on {device_type}")
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
    if kernel in SHOULD_FAIL:
        with pytest.raises(subprocess.CalledProcessError):
            shell.run(launch_cmd)

        if ((kernel=="hang") or (kernel=="exception")) :
            check_core_dump(
                gdb,
                kernel_path.parent / (kernel_path.name + "_dbg"),
                ERROR_COMMENT[kernel],
                skip_gdb = request.config.getoption("--skip-gdb"),
            )
             
    elif kernel in MASK_SWEEP_KERNELS:
        for mask in MASK_SWEEP:
            mask_cmd = launch_cmd + " --shire_mask=" + mask
            shell.run(mask_cmd)
    else:
        shell.run(launch_cmd)

    if kernel not in SHOULD_FAIL:
      check_run_artifacts(shell, device_type)


@pytest.mark.parametrize("kernel", ["print"])
@pytest.mark.parametrize("kernel_pth_param", KERNEL_PATH)
@pytest.mark.parametrize("device_type", DEVICE_TYPE)
@pytest.mark.parametrize("device_type_optarg", ["sysemu", "silicon"])
@pytest.mark.parametrize("simulator_param", SIMULATOR_PARAMS)
@pytest.mark.parametrize("simulator_optargs", SIMULATOR_OPT_ARG)
@pytest.mark.parametrize("enable_core", ENABLE_CORE)
def test_run_optional_arguments(shell, kernel, kernel_pth_param, device_type,
                                device_type_optarg, enable_core, simulator_param,
                                simulator_optargs, build_dir):
    """Run a single kernel sweeping optional arguments"""
    if not build_dir.exists():
        pytest.skip(f'{str_examplesNotBuilt}')
    if device_type_optarg == "sysemu" and kernel in SKIP_SYSEMU:
        pytest.skip(f"do not run {kernel} on {device_type_optarg}")
    logging.info("Running %s on %s", kernel, device_type_optarg)
    kernel_path = build_dir.device / "tests" / "print.elf"

    launch_cmd = " ".join(
        [
            str(build_dir.host / "sdk" / KERNEL_LAUNCHERS[kernel]),
            f'{kernel_pth_param.param} "{kernel_path}"',
            f'{device_type.param} "{device_type_optarg}"',
            f'{simulator_param.param} "{simulator_optargs.param}"' if (
                "sysemu" in {device_type_optarg}) else f'{""}',
            f'{enable_core.param}',
        ]
        + EXTRA_ARGS[kernel]
    )

    if (("simulator" in launch_cmd) and
        (kernel_pth_param.valid and device_type.valid and
         simulator_param.valid and simulator_optargs.valid)):
        shell.run(launch_cmd)
    elif (("simulator" not in launch_cmd) and
          (kernel_pth_param.valid and device_type.valid)):
        shell.run(launch_cmd)
    else:
        with pytest.raises(subprocess.CalledProcessError):
            shell.run(launch_cmd)


@pytest.mark.parametrize("device_type", ["sysemu", "silicon"])
def test_run_multi_kernel(shell, device_type, build_dir):
    """Run multi kernel example"""
    if not build_dir.exists():
        pytest.skip(f'{str_examplesNotBuilt}')
    kernels = ["bss", "saxpy_scalar"]
    logging.info("Running %s on %s", kernels, device_type)
    kernel_paths = [build_dir.device / "tests" /
                    f"{kernel}.elf" for kernel in kernels]
    launch_cmd = " ".join(
        [
            str(build_dir.host / "sdk/multiKernel_launcher"),
            f"--kernel_path1={kernel_paths[0]}",
            f"--kernel_path2={kernel_paths[1]}",
            f"--device_type={device_type}",
        ]
    )
    cmd_out = shell.run(f"( cd {shell.tmp_path} ; {launch_cmd} )")

    regexp = re.compile(r'.*kernel_id=([0-9a-fA-F]+).*kernel_id=([0-9a-fA-F]+).*')
    m = regexp.match(str(cmd_out))
    if not m:
        logging.info(f'not found KernelId used on execution')

    check_run_artifacts(shell, device_type, True, int(m.group(1),16), int(m.group(2),16))
