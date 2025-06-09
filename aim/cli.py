import os
import argparse
from glob import glob
import numpy as np
import SimpleITK as sitk
from aim.aim import load_aim

def convert_aim_to_itk_image(aim_path, output_ext=".mha", dtype=np.uint16, verbose=True):
    """
    Convert a Scanco AIM file to a SimpleITK-supported format (e.g., .mha, .nii.gz).

    Parameters:
        aim_path (str): Path to the input .AIM file.
        output_ext (str): Desired output file extension (e.g., .mha, .nii.gz).
        dtype (np.dtype): Output array type.
        verbose (bool): Whether to print info.
    """
    if not output_ext.lower().startswith("."):
        output_ext = "." + output_ext

    if output_ext.lower() not in [".mha", ".nii", ".nii.gz", ".nrrd"]:
        raise ValueError(f"Unsupported extension: {output_ext}")

    data = load_aim(aim_path)
    image_np = np.asarray(data.data).astype(dtype)

    spacing = [data.voxelsize.to('mm').magnitude[0]] * 3
    position = data.position
    origin = [p * s for p, s in zip(position, spacing)]

    sitk_img = sitk.GetImageFromArray(image_np)
    sitk_img.SetSpacing(spacing)
    sitk_img.SetOrigin(origin)

    out_path = os.path.splitext(aim_path)[0] + output_ext
    sitk.WriteImage(sitk_img, out_path)

    if verbose:
        print(f"[✓] {os.path.basename(aim_path)} → {os.path.basename(out_path)}")
        print(f"    Shape: {image_np.shape}, Spacing: {spacing}, Origin: {origin}")

def main():
    parser = argparse.ArgumentParser(description="Convert AIM files to .mha/.nii.gz/etc. in-place.")
    parser.add_argument("input_files", nargs="+", help="Input AIM files (wildcards allowed, e.g., *.AIM)")
    parser.add_argument("--ext", default=".mha", choices=[".mha", ".nii", ".nii.gz", ".nrrd"], help="Output file format")
    parser.add_argument("--dtype", default="float32", choices=["uint8", "uint16", "int16", "float32"], help="Output data type")
    parser.add_argument("--no-verbose", action="store_true", help="Suppress output")

    args = parser.parse_args()

    np_dtype = {
        "uint8": np.uint8,
        "uint16": np.uint16,
        "int16": np.int16,
        "float32": np.float32
    }[args.dtype]

    for pattern in args.input_files:
        for aim_path in sorted(glob(os.path.expanduser(pattern))):
            convert_aim_to_itk_image(
                aim_path,
                output_ext=args.ext,
                dtype=np_dtype,
                verbose=not args.no_verbose
            )

if __name__ == "__main__":
    main()