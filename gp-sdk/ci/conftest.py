"""
Configuration for GP-SDK tests

Important fixtures:
    - gp_sdk: Setup a checkout of the GP-SDK
    - shell: Wrappers for a subset of shell commands
"""

# pylint: disable=redefined-outer-name # this is needed for local fixtures

import logging
from collections import namedtuple
import os
from pathlib import Path
import subprocess
import shlex
import pytest


KERNEL_ADDRESS = "0x8006335000"

Sdk = namedtuple("Sdk", ["path", "kernel_address"])

Step = namedtuple("Step", ["name", "commands"])

GENERATOR_TARGET = {
    "make": "Unix Makefiles",
    "ninja": "Ninja",
}


class ShellEnv:
    """
    Shell environment fixture
    """

    def __init__(self, temp):
        self._temp = temp
        self._steps = [Step("setup test session", [])]

    def fixture(self, name):
        """Enter a new test fixture"""
        self._steps.append(Step(name, []))

    def _command(self, command):
        """Record a shell command"""
        self._steps[-1].commands.append(
            command.replace(str(self._temp.absolute()), "$TESTDIR")
        )

    def getscript(self):
        """Create a script of all shell commands that were run"""
        script = [
            "#!/bin/bash",
            "",
            "set -eu",
            "",
            f"TESTDIR={self._temp}",
        ]
        for step in self._steps:
            script.extend(["", f"# {step.name}"])
            script.extend(step.commands)
        script.append("")
        return "\n".join(script)

    def run(self, cmd: str, **kwargs):
        """Run a single shell command"""
        logging.debug("> %s", cmd)
        self._command(cmd)

        if "stdout" not in kwargs:
            kwargs["stdout"] = subprocess.PIPE

        return subprocess.run(
            cmd,
            check=True,
            shell=True,
            **kwargs,
        )

    def popen(self, cmd: str, **kwargs):
        """Run a single shell command in a separate process"""
        logging.debug("> %s &", cmd)
        # logging.debug("* %s", shlex.split(cmd))
        self._command(f"{cmd} &")

        if "stdout" not in kwargs:
            kwargs["stdout"] = subprocess.PIPE

        return subprocess.Popen(
            cmd,
            shell=True,
            **kwargs,
        )

    def mkdir(self, path):
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
        cmd = self.run(f"nm -S {elf} | awk '$4==\"{symbol}\" {{ print 0x$1 }}'")
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


@pytest.fixture(scope="session")
def shell_env(tmp_path_factory):
    """Session-wide instance of the shell environment"""
    env = ShellEnv(tmp_path_factory.getbasetemp())
    yield env
    logging.debug("Saving shell commands")
    commands_path = tmp_path_factory.mktemp("shell", numbered=False) / "commands.sh"
    commands_path.write_text(env.getscript(), encoding="utf-8")


@pytest.fixture
def shell(shell_env, request):
    """Per-function shell fixture"""
    shell_env.fixture(request.node.name)
    return shell_env


@pytest.fixture(scope="session")
def gp_sdk(request):
    """Setup the GP-SDK environment"""
    gp_sdk_opt = request.config.getoption("--with-gp-sdk")
    if gp_sdk_opt is None:
        gp_sdk_opt = os.environ["GP_SDK_HOME"] if "GP_SDK_HOME" in os.environ else os.getcwd()
    gp_sdk_path = Path(gp_sdk_opt)
    assert gp_sdk_path.exists()
    assert (gp_sdk_path / ".git").exists()
    logging.info("Using GP-SDK in %s", gp_sdk_opt)
    return Sdk(path=gp_sdk_path, kernel_address=KERNEL_ADDRESS)
