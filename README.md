# AIM

# A lightweight wrapper around ITK ScancoIO to read, convert, and align `.aim` files from Scanco microCT scanners.

## Installation

This package requires Python 3.8 due to ITK-ScancoIO compatibility constraints.

### Installation

```
git clone https://github.com/wallematthias/ScancoAIMreader.git
cd ScancoAIMreader
conda env create -f environment.yml
conda activate aimenv
pip install -e .
```


## Overview

This repository provides Python utilities for:

- Reading `.aim` files into `AIMFile` objects
- Converting Hounsfield units to density (mg/cm^3)
- Writing AIM files back to `.mha`
- Aligning multiple AIM scans to a shared coordinate space

It is designed to support legacy workflows using Scanco AIM files, such as preclinical bone imaging pipelines.

## Usage

# Load an AIM file
```
from aim import load_aim

aim = load_aim("sample.AIM")
print(aim.data.shape)
print(aim.voxelsize)
```
# Write an AIM file
```
from aim import write_aim

write_aim(aim, "output.AIM")
```
# Align multiple AIM files to a common space
```
from aim import pad_to_common_coordinate_system

padded_arrays, common_origin = pad_to_common_coordinate_system([aim1, aim2])
```

## Class: AIMFile

An in-memory representation of an AIM scan:

- data: A 3D pint.Quantity array (e.g., in mg/cm^3)
- voxelsize: A pint.Quantity with voxel size in mm
- position: Origin as voxel index (in scanner coordinates)
- processing_log: Metadata from the AIM header

## Functions

- load_aim(path): Reads and converts `.AIM` or binary mask files.
- write_aim(aim_file, path): Writes an AIMFile object to disk.
- pad_to_common_coordinate_system(aim_files, padding_values=None): Pads multiple AIMFile objects to a shared coordinate system.
- get_aim_calibration_constants_from_processing_log(path): Extracts calibration constants from raw AIM headers.
- convert_hounsfield_to_mgccm(...): Converts Hounsfield units to bone mineral density using scanner-specific calibration.

## Notes

- This repository assumes familiarity with Scanco microCT data and is not intended as a general-purpose AIM file reader.
- The load_aim() function handles both image data and masks.
- You can easily integrate it into preprocessing workflows for morphometry or biomechanical analysis.
