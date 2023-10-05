"""
Configuration for GP-SDK tests

Important fixtures:
    - gp_sdk: Setup a checkout of the GP-SDK
    - shell: Wrappers for a subset of shell commands
"""

# pylint: disable=redefined-outer-name # this is needed for local fixtures

import logging
from collections import namedtuple
import fnmatch
import os
from pathlib import Path
import subprocess
import shlex
import shutil
import stat
from typing import Optional
import re
import pytest


KERNEL_ADDRESS = "0x8006335000"

Sdk = namedtuple("Sdk", ["path", "kernel_address"])

GENERATOR_TARGET = {
    "make": "Unix Makefiles",
    "ninja": "Ninja",
}


class BuildCache:
    """
    Cache build directories for host launchers and device kernels.
    """

    def __init__(self, config):
        self._config = config

    def exists(self):
        """Check whether host/device build directories exist"""
        device = self.device
        host = self.host
        return (
            device is not None
            and device.exists()
            and host is not None
            and host.exists()
        )

    @property
    def device(self) -> Optional[Path]:
        """Path to the device build"""
        build = self._config.getoption("--device-build")
        if build is None:
            build = self._config.cache.get("device-build", None)
        if build is None:
            return None
        path = Path(build)
        if not path.is_absolute():
            path = Path(self._config.rootdir / path)
        return path

    @property
    def host(self) -> Optional[Path]:
        """Path to the host build"""
        build = self._config.getoption("--host-build")
        if build is None:
            build = self._config.cache.get("host-build", None)
        if build is None:
            return None
        path = Path(build)
        if not path.is_absolute():
            path = Path(self._config.rootdir / path)
        return path

    def save_device(self, path: Path):
        """Cache the device build"""
        self._config.cache.set("device-build", str(path))

    def save_host(self, path: Path):
        """Cache the host build"""
        self._config.cache.set("host-build", str(path))


class GdbSession:
    """
    GDB session fixture
    """

    def __init__(self, process):
        self.process = process
        self.commands = []

    def __del__(self):
        if self.process.returncode is None:
            self.close()

    def _readline(self):
        line = self.process.stdout.readline().decode("utf-8")
        if line.startswith("(gdb) "):
            line = line.replace("(gdb) ", "")
        line = line.strip()
        logging.debug(line)
        return line

    @classmethod
    def launch(cls, shell, cmd):
        """Start a GDB session"""
        logging.info("Starting gdb session")
        if not cmd.startswith("riscv64-unknown-elf-gdb"):
            cmd = f"riscv64-unknown-elf-gdb {cmd}"
        return cls(shell.popen(cmd, stdin=subprocess.PIPE))

    def close(self):
        """Close stdin and exit the process"""
        self.process.stdin.close()
        returncode = self.process.wait()
        if returncode != 0:
            raise RuntimeError("gdb returned non-zero exit-code")

    def read_until(self, marker: str):
        """Process GDB output until a marker is read"""
        while self.process.poll() is None:
            line = self._readline()
            if marker in line:
                break

    def eval(self, cmd: str, expected_output: Optional[list] = None):
        """Evaluate and check a single GDB command"""
        if expected_output is None:
            expected_output = []
        self.commands.append(cmd)
        self.process.stdin.write(cmd.encode("utf-8"))
        self.process.stdin.write(b"\n")
        self.process.stdin.flush()
        logging.debug("(gdb) %s", cmd)
        for expected_line in expected_output:
            line = self._readline()
            match = fnmatch.fnmatch(line, expected_line)
            if not match:
                logging.error("(gdb) %s", cmd)
                logging.error("  expected: %s", expected_line)
                logging.error("  found:    %s", line)
                raise RuntimeError("gdb script mismatch")

    def save(self, path: Path):
        """Save GDB commands to a file"""
        path.write_text("\n".join(self.commands) + "\n")


class ShellSession:
    """
    Shell environment fixture
    """

    def __init__(self, tmp_path):
        self._commands = []
        self._should_keep = False
        self.tmp_path = tmp_path

    def _prerun(self, cmd: str, quiet: bool = False, **kwargs):
        if not quiet:
            logging.debug("> %s", cmd)
            self._commands.append(cmd)

        if "stdout" not in kwargs:
            kwargs["stdout"] = subprocess.PIPE

        kwargs["args"] = cmd
        kwargs["shell"] = True

        return kwargs

    def clean_tmp(self):
        """Clean temporary files"""
        if not self._should_keep:
            shutil.rmtree(self.tmp_path, ignore_errors=True)

    def save_commands(self):
        """Create a script of all shell commands that were run"""
        script = (
            [
                "#!/bin/bash",
                "",
                "set -eux",
                "",
            ]
            + self._commands
            + ["\n"]
        )
        commands = self.tmp_path / "commands.sh"
        commands.write_text("\n".join(script))
        os.chmod(commands, os.stat(commands).st_mode | stat.S_IEXEC)

    def keep_tmp(self, should_keep: bool = True):
        """Keep temporary files even on pass"""
        self._should_keep = should_keep

    def run(self, cmd: str, quiet: bool = False, **kwargs):
        """Run a single shell command"""
        return subprocess.run(**self._prerun(cmd, quiet=quiet, **kwargs), check=True)

    def popen(self, cmd: str, quiet: bool = False, **kwargs):
        """Run a single shell command in a separate process"""
        return subprocess.Popen(**self._prerun(cmd, quiet=quiet, **kwargs))

    def mkdir(self, path: Path):
        """Create a directory"""
        return self.run(f"mkdir -p {path}")

    def cmake(self, source_dir, build_dir, generator=None, **kwargs):
        """Configure a cmake build"""
        cmd = [
            "/usr/local/bin/cmake",
            f"-B{build_dir}",
        ]
        if generator:
            cmd.append(f"-G{shlex.quote(GENERATOR_TARGET[generator])}")

        def fmtval(val):
            if isinstance(val, bool):
                return "ON" if val else "OFF"
            if isinstance(val, str):
                return val if " " not in val else f'"{val}"'
            return str(val)

        for key, value in kwargs.items():
            cmd.append(f"-D{key.upper()}={fmtval(value)}")
        cmd.append(str(source_dir))

        return self.run(" ".join(cmd))

    def make(self, build_dir, target=None, generator="make", jobs=1):
        """Run build command"""
        cmd = f"{generator} -C {build_dir}"
        if jobs > 1:
            cmd += f" -j {jobs}"
        if target is not None:
            if isinstance(target, list):
                target = " ".join(target)
            cmd += f" {target}"
        return self.run(cmd)

    def find_symbol(self, elf: Path, symbol: str):
        """Find an ELF symbol using nm"""
        cmd = self.run(f"riscv64-unknown-elf-nm -SC {elf} | awk '$4 ~ \"{symbol}\" {{ print 0x$1 }}'")
        return int(cmd.stdout.decode("utf-8"), 16)


def pytest_addoption(parser):
    """Add custom pytest options"""
    parser.addoption(
        "--with-gp-sdk",
        help="Path to the GP-SDK installation",
    )
    parser.addoption(
        "--gdb-custom",
        default=False,
        action="store_true",
        help="Enable manual GDB testing (for debug)",
    )
    parser.addoption(
        "--keep-on-pass",
        default=False,
        action="store_true",
        help="Keep build artifacts on pass",
    )
    parser.addoption(
        "--device-build",
        help="Path to device builds",
    )
    parser.addoption(
        "--host-build",
        help="Path to device builds",
    )
    parser.addoption(
        "--skip-gdb",
        action="store_true",
        help="Skip GDB checks",
    )


@pytest.fixture(scope="session")
def shell_env(request):
    """Keep track of all shell sessions"""
    keep_on_pass = request.config.getoption("--keep-on-pass")
    env = []
    yield env
    if not keep_on_pass:
        logging.debug("Cleaning up")
        for session in env:
            session.clean_tmp()


@pytest.fixture
def shell(request, tmp_path_factory, shell_env):
    """Per-function shell fixture"""
    name = re.sub(r"[\W]", "_", request.node.name)
    if name[-1] == "_":
        name = name[:-1]
    session = ShellSession(tmp_path_factory.mktemp(name, numbered=False))
    shell_env.append(session)
    os.chdir(session.tmp_path)
    yield session
    session.save_commands()
    if request.session.testsfailed != 0:
        session.keep_tmp()


@pytest.fixture
def gdb(shell):
    """Create a gdb session"""
    sessions = []

    def _gdb(cmd: str):
        sessions.append(GdbSession.launch(shell, cmd))
        return sessions[-1]

    yield _gdb
    for i, session in enumerate(sessions):
        session.save(Path(f"script{i}.gdb"))


@pytest.fixture(scope="session")
def gp_sdk(request):
    """Setup the GP-SDK environment"""
    gp_sdk_opt = request.config.getoption("--with-gp-sdk")
    if gp_sdk_opt is None:
        gp_sdk_opt = (
            os.environ["GP_SDK_HOME"] if "GP_SDK_HOME" in os.environ else os.getcwd()
        )
    gp_sdk_path = Path(gp_sdk_opt)
    assert gp_sdk_path.exists()
    assert (gp_sdk_path / ".git").exists()
    logging.info("Using GP-SDK in %s", gp_sdk_opt)
    return Sdk(path=gp_sdk_path, kernel_address=KERNEL_ADDRESS)


@pytest.fixture
def build_dir(request):
    """Build directory (cmdline or cached)"""
    return BuildCache(request.config)


@pytest.fixture
def devices_list():
    devs = [int(dev[2:-5]) \
        for dev in os.listdir("/dev") if dev.startswith("et") and dev.endswith("_mgmt")]
    devs.sort()
    return devs
