# Changelog

[[_TOC_]]

## Unreleased
### Added
### Changed
### Deprecated
### Removed
### Fixed
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
