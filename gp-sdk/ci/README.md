# CI infrastructure for GP-SDK

This document describes the CI scripts you can use to assess the health of the GP-SDK.

The following table gives you an overview of the current test scripts.

| Script | Description |
|:-------|:------------|
| ci/conftest.py | Configuration file and common test fixtures |
| ci/test_build.py | Builds host launchers and device kernels and verifies the build artifacts |
| ci/test_examples.py | Runs the basic examples on both sysemu and silicon |
| ci/test_gdb.py | Runs the interactive GDB example (sysemu-only) |


## How to run the scripts

All the tests need to be executed inside the docker prompt.

```bash
$ et_docker --develop prompt
[Docker] $ pytest ci --basetemp=citemp
```

Alternatively you can also run the commands directly.

```bash
$ et_docker --develop run "python3 -m pytest ci --basetemp=citemp"
$ et_docker --develop run "python3 -m pytest ci -k 'not silicon' --basetemp=citemp"
```

Here are some of the more useful pytest options:

```
## Standard pytest options:
--collect-only         # Do not actually run tests, only show which ones would run
-k <expr>              # Filter tests based on a regular expression, ex:
-k 'not silicon'       #   do not run tests on silicon
--log-cli-level=DEBUG  # Add extra logging information to tests
--basetemp <path>      # Store temporary files at the given path

## Extra GP-SDK options:
--keep-on-pass         # Keep temporary files, even if the test passes
--with-gp-sdk <path>   # Specify the path to the GP-SDK
                       # Normally the GP-SDK is expected at $PWD or $GP_SDK_HOME
--device-build <path>  # Path to the device build
--host-build <path>    # Path to the host build
--skip-gdb             # Skip checks with GDB
```


## Some implementation details

The test frameworks makes heavy use of pytest fixtures. To learn more about this, see here: https://docs.pytest.org/en/6.2.x/fixture.html.

To use any of the test fixtures, you simply need to specify them as an argument in your test function. For example, this is how you would use the `shell` fixture:

```py
def test_foo(shell):
    shell.run("ls -lah")
```

In this section we will go over the main three fixtures used throughout the tests.

### `gp_sdk`

This fixture holds information of the GP-SDK. It returns a tuple with the path to the root directory and the value of the kernel offset.

```py
def test_foo(gp_sdk):
    print(gp_sdk.path)            # /path/to/gp-sdk
    print(gp_sdk.kernel_address)  # ex: 0x8006335000


```

### `build_dir` cache

This fixture holds information about where the build artifacts (kernel launchers and ELFs) are located.
If you have already built the artifacts outside of pytest, for example by using cmake, you can specify these paths with the `--device-build` and `--host-build` paramters.
Otherwise, pytest will build the artifacts itself and save their path in the build cache.
If you want to conserver the build artifacts from pytest, remember to use `--keep-on-pass`.

### `shell` session

Most of the tests are bash scripts. This fixture helps with that. It will generate a temporary directory for each shell session and keep a trace of the shell commands that are executed.
It also has a few helper functions that map to common shell command patterns.

```py
def test_foo(shell):
    shell.mkdir("build")
    shell.cmake(build_dir="build", ...)
    shell.make("build", jobs=4)

```

This would generate all temporary artifacts under `$BASETEMP/test_foo` as well as a `commands.sh` script that has the sequence of bash commands.
If the test passes, all temporary artifacts are deleted by default. This behavior can be changed by adding the `--keep-on-pass` command line argument.


### `gdb` session

A GDB session simulates an interactive debugger session.
It will open a GDB prompt and input commands one at a time, comparing the output with an expected string.

```py
gdb = gdb("path/to/debug.elf_dbg")               # Open GDB session
gdb.read_until("Reading symbols")                # Advance stream until marker
time.sleep(0.4)                                  # Necessary to throttle the RE{}
gbd.eval("continue",                             # Execute 'continue'
        [                                        # Expects the following list of lines ('*' are interpreted as wildcards)
            "Continuing.",
            "",
            'Thread 1 "S0:M0:T0" hit Breakpoint 2, saxpy_vector *',
            "*",
        ])
gdb.close()                                      # Close GDB session

```


## Caveats

- The gdb session relies on timeout and does not work super consistently.
- Most of the tests are tightly coupled with the examples.
- From time to time you will need to update the `KERNEL_ADDRESS` parameter in `ci/conftest.py`.


## Running as part of the SQA regression

The SQA regression can be run from the software/system-qa repository.
This will both build the GP-SDK artifacts, as well as run all silicon and sysemu tests.

To launch the regression pipeline: https://gitlab.esperanto.ai/software/system-qa/-/pipelines/new

Here select the following options:

- `TARGET_MACHINE`: Select the machine this runs on. Make sure to reserve a slot in the queue first!
- `GP_SDK_VERSION`: Version tag or branch of the GP-SDK.
- `TAGS`: Set to `gp_sdk` (with an underscore) to only create pipeline jobs targetting the GP-SDK.

Then click `Run pipeline`.

Make sure that the `TARGET_MACHINE` has the correct version of the ET-SDK stack installed.


### Checklist when creating a new GP-SDK release

- [ ] Verify the `KERNEL_ADDRESS` parameter in the CI scripts
- [ ] Update the default `GP_SDK_VERSION` in the software/system-qa repository
- [ ] Launch the gp-sdk regression (using the `gp_sdk` tag)
