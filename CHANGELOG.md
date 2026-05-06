# Changelog

All notable changes to the project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased - 0.10.0]

### Changed
- `read_rasters` now uses C++ backend and changed API
- `sb.swapRowsValues` is now called `sb.swapValues`
- `sb.maskDataRows` and `sb.maskData` have swapped names for consistency with `sb.maskNan` and `sb.maskNanRows`

### Removed
- `sb.extractArrayRows` - use `sb.selArrayRows` instead
- `sb.extractArrayCols` - use `sb.selArrayCols` instead
- `sb.writeInt16Data`,`sb.writeUInt16Data`,`sb.writeByteData` - use `sb.writeData`
- `sb.warpTile` - use GDAL VRTs

### Deprecated (to be removed in 0.11.0)

- `read_rasters_cpp` Now is `read_rasters`

## [Unreleased - 0.9.3]

### Changed
- `ReadDataCore` now takes a `int band` instead of `std::vector<int> bands_list`, which would segfault on multiple bands.

### Fixed

- `read_rasters_cpp` no longer segfaults on multiple bands selection, but only allows reading a single band from each raster connection.

## [0.9.2] 2026-05-01

### Deprecated (to be removed in 0.10.0)

## [0.9.1] 2025-11-05

### Added
- [documentation](https://feefladder.github.io/scikit-map)

### Deprecated (to be removed in 0.10.0)
- `sb.extractArrayRows` - use `sb.selArrayRows` instead
- `sb.extractArrayCols` - use `sb.selArrayCols` instead
- `sb.writeInt16Data`,`sb.writeUInt16Data`,`sb.writeByteData` - use `sb.writeData`
- `sb.warpTile` - use GDAL VRTs

## [0.8.1] 2025-05-24

## [0.7.3] 2023-10-04

## [0.7.0] 2023-09-19

## [0.6.0] 2023-06-22
