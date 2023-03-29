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

## 1.1.0a (2023/03/29)
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

