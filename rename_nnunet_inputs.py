#!/usr/bin/env python3
"""
Batch rename or copy medical images into nnU-Net input naming format.

Default behavior:
- read all .nii.gz files from an input folder
- copy them into an output folder
- rename them to Case_001_0000.nii.gz style names
- write a name_mapping.txt file recording old and new names

Examples
--------
Copy into a new folder:
    python rename_nnunet_inputs.py \
        --input_dir /path/to/raw_images \
        --output_dir /path/to/nnunet_inputs

Rename in place:
    python rename_nnunet_inputs.py \
        --input_dir /path/to/raw_images \
        --inplace
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename .nii.gz files into nnU-Net input format and save a mapping txt file."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help="Folder containing the original .nii.gz files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output folder for renamed files. If omitted, you must use --inplace.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="Case",
        help="Prefix for new file names. Default: Case",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="Starting number for renamed cases. Default: 1",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Minimum number of digits for the case index. Default: 3",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel suffix used by nnU-Net. Single-modality images should use 0. Default: 0",
    )
    parser.add_argument(
        "--mapping_name",
        type=str,
        default="name_mapping.txt",
        help="Name of the mapping txt file. Default: name_mapping.txt",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Rename files directly inside the input folder instead of copying to a new folder.",
    )
    return parser.parse_args()


def collect_nii_gz_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.name.endswith(".nii.gz")
    )


def build_new_name(prefix: str, index: int, width: int, channel: int) -> str:
    return f"{prefix}_{index:0{width}d}_{channel:04d}.nii.gz"


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist or is not a folder: {args.input_dir}")

    if args.start_index < 0:
        raise ValueError("--start_index must be >= 0")

    if args.digits < 1:
        raise ValueError("--digits must be >= 1")

    if args.channel < 0:
        raise ValueError("--channel must be >= 0")

    if not args.inplace and args.output_dir is None:
        raise ValueError("You must provide --output_dir, or use --inplace.")

    if args.inplace and args.output_dir is not None:
        raise ValueError("Please use either --output_dir or --inplace, not both.")


def ensure_no_collisions(
    files: list[Path],
    target_dir: Path,
    prefix: str,
    start_index: int,
    digits: int,
    channel: int,
    mapping_name: str,
) -> int:
    max_index = start_index + len(files) - 1
    width = max(digits, len(str(max_index)))
    planned_names = [build_new_name(prefix, start_index + idx, width, channel) for idx in range(len(files))]

    if len(planned_names) != len(set(planned_names)):
        raise RuntimeError("New file names would collide with each other.")

    for new_name in planned_names:
        target_path = target_dir / new_name
        if target_path.exists():
            raise FileExistsError(f"Target file already exists: {target_path}")

    mapping_path = target_dir / mapping_name
    if mapping_path.exists():
        raise FileExistsError(f"Mapping file already exists: {mapping_path}")

    return width


def write_mapping(mapping_path: Path, mappings: list[tuple[str, str]]) -> None:
    lines = ["original_name\tnew_name"]
    lines.extend(f"{old}\t{new}" for old, new in mappings)
    mapping_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        files = collect_nii_gz_files(args.input_dir)
        if not files:
            raise FileNotFoundError(f"No .nii.gz files were found in: {args.input_dir}")

        target_dir = args.input_dir if args.inplace else args.output_dir
        assert target_dir is not None
        target_dir.mkdir(parents=True, exist_ok=True)

        width = ensure_no_collisions(
            files,
            target_dir,
            args.prefix,
            args.start_index,
            args.digits,
            args.channel,
            args.mapping_name,
        )

        mappings: list[tuple[str, str]] = []

        for offset, src_path in enumerate(files):
            case_index = args.start_index + offset
            new_name = build_new_name(args.prefix, case_index, width, args.channel)
            dst_path = target_dir / new_name

            if args.inplace:
                src_path.rename(dst_path)
            else:
                shutil.copy2(src_path, dst_path)

            mappings.append((src_path.name, new_name))

        mapping_path = target_dir / args.mapping_name
        write_mapping(mapping_path, mappings)

        print(f"Processed {len(mappings)} files.")
        print(f"Output folder: {target_dir}")
        print(f"Mapping file: {mapping_path}")
        return 0
    except Exception as exc:  # pragma: no cover - straightforward CLI error handling
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
