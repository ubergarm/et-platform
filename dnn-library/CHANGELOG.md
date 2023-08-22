# Changelog

[[_TOC_]]

## Unreleased
### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security

## [0.7.0]
### Added
- Upgrading resizenearest implementation from scalar to theading
- Adapted MaxSplat for clang
- Added ward to use the correct barrier on GPSDK codes
### Changed
- [SW-17446] reduced stack footprint for FFT
- Adapt CI to gitlab.com
- Adapt CI to v15
- Revert package_id changes. Need to keep host packages separated form RISC-V ones to avoid consuming RISC-V dependencies
  in X86 builds.
### Deprecated
### Removed
### Fixed
- [SW-16749] Fixed out of bounds access on AvgPoolInst.h
- [SW-17086] Build fix for ConvertTo with Int32ITy destination
- Removed the unused inlining/GenericHeaders.h
- Multiple fixes to adopt clang in DNN Library
### Security


## [0.6.0]
### Added
- [SW-16352] Add implementation for Neg/Not/Sin/Cos in DNN Library
- [SW-16351] Add empty implementation for Trilu in dnn library
- [SW-11312] Added explicit zero offset zero to {flw|fsw}.ps instances
### Changed
- [SW-16712] Extend CumSum to support multiple dims and axis
- [SW-16750] Make fgh.ps occurrences to follow the official format
- Forked core files from "./internal/" for supporting vector data types
- [SW-17085] Leaving pre-existing asserts alone on ElementBinaryInst.h
### Deprecated
### Removed
### Fixed
- [SW-16407] Fix ElementSin/ElementCos/ElementNot
- Fixed an alignof(void) statement that did not build with LLVM
- [SW-16748] Fixed a typo in FullyConnected operator
- [SW-16876] Fix typo in EmbeddingBag implementation
- [SW-16192] Fix creation of tensor iterator from offset pointing to padding
- [SW-16893] Fix transpose writing out of bounds
- [SW-11425] Declare vector mask register usages for clang in include/Float16.h
- [SW-16749] Fixed out of bounds access on AvgPoolInst.h
### Security

## [0.5.0]
### Added
### Changed
- [SW-16286] Migrate CI pipeline to use pre-release strategy
### Deprecated
### Removed
### Fixed
### Security

## [0.4.0]
### Added
### Changed
- [SW-14877] Fix BatchOneHot operator writing in non-multiple of cacheline
- [SW-14892] Generalize `partitionLoop()` to accept non-consecutive elements.
- [SW-14859] dnnLibrary export public interface. (make it consumable by cmake targets)
- [SW-14005] Added Int8QTy Embedding Bag Fastpath implementation
- [SW-14081] Generalize EmbeddingBag fast-path options
- [SW-15296] Test new barrier primitives in the FFT
### Deprecated
### Removed
### Fixed
### Security


## [0.3.0]
### Added
### Changed
- [SW-14405] Refactor Embedding Bag, Int8 to FP32 and new FP16.
### Deprecated
### Removed
### Fixed
### Security


## [0.2.0]
### Added
### Changed
- [SW-12785] FFT avoid repeated load and stores
- [SW-13180] Prevent evict_va for L2scp addresses in syncopy
- [SW-13174] Parallelize resnet image conditioning
- [SW-13174] Support a tiling of 256 minions on resnet image conditioning
- [SW-13174] Evict to L3 on the resnet image conditioning operator
- [SW-12786] Vectorize reduce() routine
- [SW-13093] support multiple noises in dnnLib
- [SW-13095] 10 different noises for fft freq-dom filter 
- [SW-12786] Fixed wrong ASM param constraint
- [SW-12793] Threading over channel. Using fixed tiling params
- [SW-11367] Add header_only, with_host_headers, with_device_headers options to conanfile options
- [SW-13438] Make denoise filter bank human-radable
- [SW-13532] remove ETSOCGenericOp from inlining
- [SW-13430] Automatic tiling in channels for FFT
- [SW-13430] When not using a full shire, synchronize also with unused minions
- [SW-13430] Clang-format fixes 
- [SW-13621] Support Quantized-weights in EmbeddingBag 
- [SW-13621] dequantie data instead of weights 
- [SW-13621] use global stores in quantized EB
- [SW-13675] Added support for quantized data in the EmbeddingBag vectorized version
- [SW-13675] Fixed quantized version unaligned stores and replaced fgbl loads
- [SW-13963] dnnlib adapters to profiling code
- [SW-13963] enable profiling calls by default
### Deprecated
### Removed
### Fixed
- [SW-13430] Fixed a synchronization corner case 
- [SW-13430] Fixed a bug on the FFT reduce code 
### Security


## [0.1.0] -
Initial version; not tracking changes until 0.2.0
