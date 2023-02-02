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
import shutil
import stat
import re
import pytest


KERNEL_ADDRESS = "0x8006335000"

Sdk = namedtuple("Sdk", ["path", "kernel_address"])

GENERATOR_TARGET = {
    "make": "Unix Makefiles",
    "ninja": "Ninja",
}


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
                "set -eu",
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
    parser.addoption(
        "--keep-on-pass",
        default=False,
        action="store_true",
        help="Keep build artifacts on pass",
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
