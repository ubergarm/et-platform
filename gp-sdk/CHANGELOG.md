# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[[_TOC_]]



## Unreleased
### Added
### Changed
### Deprecated
### Removed
### Fixed

## 1.5.0 (2024/04/19)
### Added
- [SW-20383] Relocation Support. device executables are 100% PIE. (Proper host runtime support is requried. It is added in 0.14.0)
- [SW-19824] Adapted to Ubuntu 22 distro (docker)  and GCC 11.4.0 host compiler
### Changed
- [SW-20362] Update sysemu_explorer to support multiple ELF files
### Deprecated
### Removed
### Fixed
- [SW-18223] Fixed log and logf of negative numbers
- [SW-18719] Handled NaN, -inf and +inf castings in the exhaustive casting test

## 1.4.0 (2023/12/18)
### Added
- [SW-19548] User-Defined stacks support. (See companion documentation for details).
- [SW-18213] Use Runtime for generating coredump.
- [SW-18615] Added SAXPY example with intrinsics (clang11), with new builtins syntax
### Changed
- [SW-19209] dt2json -t now shows the actual user strings for string traces
### Removed
### Fixed
- [SW-19038] Fixed Barrier code bug when synchronizing overlapping groups of threads.
- [SW-19310] Fixed FFT verification
- [SW-18994] Fixed dt2json Decoding of hidden traces after an aborted sysemu session.
- [SW-18606] Fixed Device TLS (Thread local storage)  variable debug issues when compiling with clang.
- [SW-19635] Added -fno-jump-tables compiler option to avoid indirect jumps on large switches.

## 1.3.0 (2023/09/15)
### Added
- [SW-18144] Add SCOPED_TIMED_REGION
- [SW-17597] Dump Trace-buffer on sysemu Fatals.
- [SW-17595] Auto-attach gdbserver on user code.
- [SW-17450] Support for clang compiler on device side along with gcc.  Please, refer to README.md and companion docs for further detail.  In general an envoronment variable DEV_COMPILER (gcc8.2 or clang11) needs to be provided when configuring the build on device side:

```
DEV_COMPILER=clang11 cmake .. -DCMAKE_TOOLCHAIN_FILE=/usr/local/esperanto/.builds/device/${DEV_COMPILER}/conan_toolchain.cmake -DADDRESS:STRING=0x8006335000  -DCMAKE_BUILD_TYPE=Release
```
- [SW-17450] Added saxpy with intrinsics (preplim syntax) test

### Changed
### Deprecated
### Removed
### Fixed
- [SW-17765] Fixed autogenereated matrix-multiplication tests
- [SW-18359] Fixed Support for large Sysemu traces in sysemuExplorer script
- [SW-18330] properly align porsistentData section
- [SW-18331] globally disabled linker-relaxation in search for debug-builds with reproducible layout across base-address linkage.


## 1.2.1 (2023/07/26)
### Added
### Changed
### Deprecated
### Removed
### Fixed
- [SW-17429] fixed hart::barrier() race conditions on flbs in barrier().count < 32
- [SW-17783] fixed assert retrieval on basic_launcher.
- [SW-17923] fixed cmd-line parsing in legacy-code bases using strict=false.

## 1.2.0 (2023/06/13)
### Added
- [SW-17172] Quote arbitrary strings instead of func-name:code in profiling.
- [SYSQA-178] Cover TLS checking on gdb QA / CI tests
- [SW-17293] Fixed bug in barriers(start,count) if count=2
- [SW-17063] Add some fixes to make CI fail on MultiProcess if error callbacks...
- [SYSQA-171] Test various string lengths on CI
- [SW-17371] Autodected Multiprocess feature.
- [SW-17062] Fixed file socket path.
- Remove unused includes.
- [SW-17063] Add test failing to be check on runtimeMP.
- [SW-17062] runtime-MP basic implementation.
- [SW-16559] Added toolchain to add neuralizer code on GP-SDK
- [SW-16832] Add support for address sanitizer ASAN an LSAN.
### Changed
### Deprecated
### Removed
### Fixed

## 1.1.1 (2023/04/27)
### Added
### Changed
### Deprecated
### Removed
### Fixed
- [SW-16830] Initialize UserTrace struct fields properly.
- [SW-16671] Improve error checking on wrong cmd-line parameters. Note that GenericLauncher now by default does strict checking of cmd-line params. 
      it may break compilation of some pre 1.1.1 examples in case the app passes extra args, (e.g pre-existing apps using other libs or parsing strategies). In those cases, relaxed check needs to be requested.
```
class Launcher : public GenericLauncher {
//`...,
};

int main(int argc, char** argv) {
// ...
  Launcher launcher(config, argc, argv, /* strictArgs */ false);
```

## 1.1.0 (2023/04/06)
### Added
- [SW-16515] Added routines for querying time, shire-mask, frequency, ...
- [SW-15636] full support for libmath (patched previously failing routines to use frcp instead of fpdiv)
- [SW-16228] Added thread_local storage through tp register (enabled direct use of __thread or thread_local with hart semantics)
- [SW-16228] Complete Baremetal startup. (C++ dynamic initialization, global ctors, ..).
- [SW-15907] Added explicit per-hart entry-point  registration  [ DECLARE_KERNEL_ENTRY_POINTS(entryPoint, entryPoint); ]
- [SW-16668] Simplified building out-of-sdk tree (using gp-sdk as an external resource)
- [SW-15771] Ability to Change Shire-Cache configuration mode (cache/scp partition ratios).
- [SW-16598] Enabled text-user-interface on RISC-V gdb
- [SW-16598] Faster sysemu_explorer
- [ARCHSIM-696] Support flush_va cache op on sysemu mem-checker.
- [SW-15758] Enabled on-silicon debug experience.

### Changed
- [SW-15907] Improved Barrier primitives
- [SW-16549] Avoid enforcing KernelArguments name & fwd-decl.
### Deprecated
### Removed
### Fixed
- [SW-16132] Fixed potential profiling corruptions when profiling from both threads in the ET-Minion core simultaneously
- [SW-16557] Fixed system-hangs when trace-buffer wraps around. 


## 1.0.3  (2023/03/03)
### Added
### Changed
### Deprecated
### Removed
### Fixed
- Added missing CI/QA test infrastucture

## 1.0.2  (2023/02/03)
### Added
- [SW-115598] Use esr-broadcast feature to wake up threads - after initial .bss and .data intitialization
- [SW-15816] Allow multidevice feature.
### Fixed
- [SW-15600] added __throw__XXX stubs. - Trick is to enable non-trivial stl code to compile with noexcept 
- [SW-15441] Fix treatment of non-debug-code on sysemu-explorer.

## 1.0.1 (2023/01/19)

### Added
- Added TFMA tutorial
- Extra args on cmakelis for building custom kernels
- Extended barrier functionality
- [SW-15785] Added map_feq.py script
### Changed
### Deprecated
### Removed
### Fixed
## 1.0.0
### Initial GP_SDK  version

