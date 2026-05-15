#!/usr/bin/env python3
"""
Stage GMNER / FMNERG assets into the repository's standard data layout.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.grounding import resolve_vinvl_feature_path


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def read_image_ids(split_file: Path) -> list[str]:
    image_ids: list[str] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("IMGID:"):
                image_ids.append(line.split("IMGID:", maxsplit=1)[-1].strip())
    return image_ids


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        raise FileExistsError(
            f"Destination already exists and is not the expected symlink: {dst}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(os.path.relpath(src, dst.parent))


def ensure_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        if dst.is_dir() and not any(dst.iterdir()):
            dst.rmdir()
        elif dst.is_dir() and mode == "copy":
            return
        raise FileExistsError(
            f"Destination already exists and is not reusable: {dst}"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(os.path.relpath(src, dst.parent))
    else:
        shutil.copytree(src, dst)


def validate_vinvl_dir(vinvl_dir: Path, image_ids: list[str]) -> dict[str, object]:
    npz_files = sorted(vinvl_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {vinvl_dir}")

    with np.load(npz_files[0]) as sample:
        required = {"num_boxes", "box_features", "bounding_boxes"}
        missing = required - set(sample.files)
        if missing:
            raise KeyError(
                f"VinVL sample {npz_files[0]} is missing required arrays: {sorted(missing)}"
            )

    checked = []
    missing = []
    for image_id in image_ids[:50]:
        try:
            resolve_vinvl_feature_path(str(vinvl_dir), image_id)
            checked.append(image_id)
        except FileNotFoundError:
            missing.append(image_id)

    return {
        "num_npz": len(npz_files),
        "checked": checked,
        "missing_examples": missing[:10],
    }


def write_paths_env(
    output_path: Path,
    stage_root: Path,
    vinvl_dir: Path | None,
    image_dir: Path,
) -> None:
    gmner_xml = stage_root / "Twitter10000_v2.0" / "xml"
    fmnerg_xml = stage_root / "Twitter10000v2" / "xml"
    lines = [
        "# Source this file before running GMNER / FMNERG commands.",
        f'export IMAGINE_DATA_DIR="{stage_root}"',
        f'export IMAGINE_IMAGE_DIR="{image_dir}"',
        f'export IMAGINE_GMNER_DATASET="Twitter10000_v2.0"',
        f'export IMAGINE_FMNERG_DATASET="Twitter10000v2"',
        f'export IMAGINE_GMNER_ANNOTATION_DIR="{gmner_xml}"',
        f'export IMAGINE_FMNERG_ANNOTATION_DIR="{fmnerg_xml}"',
        (
            f'export IMAGINE_VINVL_DIR="{vinvl_dir}"'
            if vinvl_dir is not None
            else '# export IMAGINE_VINVL_DIR="/absolute/path/to/Twitter10000 VinVL npz directory"'
        ),
        "",
        "# Example:",
        "# source data/grounded_paths.sh",
        '# python train.py --task gmner --visual_backend vinvl --data_dir "$IMAGINE_DATA_DIR" \\',
        '#   --dataset "$IMAGINE_GMNER_DATASET" --image_dir "$IMAGINE_IMAGE_DIR" \\',
        '#   --vinvl_dir "$IMAGINE_VINVL_DIR" --annotation_dir "$IMAGINE_GMNER_ANNOTATION_DIR"',
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage GMNER / FMNERG assets into ./data.")
    parser.add_argument("--stage-root", type=Path, default=repo_path("data"))
    parser.add_argument(
        "--gmner-src",
        type=Path,
        default=repo_path("thirdparty", "GMNER", "Twitter10000_v2.0"),
        help="Directory containing GMNER Twitter10000_v2.0 txt/xml assets.",
    )
    parser.add_argument(
        "--fmnerg-src",
        type=Path,
        default=repo_path("thirdparty", "FMNERG", "Twitter10000v2"),
        help="Directory containing FMNERG Twitter10000v2 txt_fine/xml assets.",
    )
    parser.add_argument(
        "--vinvl-src",
        type=Path,
        default=None,
        help="Directory containing shared Twitter10000 VinVL .npz features.",
    )
    parser.add_argument(
        "--images-src",
        type=Path,
        default=None,
        help="Optional directory containing original Twitter10000 images.",
    )
    parser.add_argument(
        "--vinvl-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to stage the external VinVL directory into ./data/vinvl.",
    )
    parser.add_argument(
        "--images-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to stage the external image directory into ./data/images.",
    )
    args = parser.parse_args()

    stage_root = args.stage_root.resolve()
    gmner_src = args.gmner_src.resolve()
    fmnerg_src = args.fmnerg_src.resolve()
    if not gmner_src.exists():
        raise FileNotFoundError(
            "GMNER data directory does not exist. Pass --gmner-src /path/to/Twitter10000_v2.0."
        )
    if not fmnerg_src.exists():
        raise FileNotFoundError(
            "FMNERG data directory does not exist. Pass --fmnerg-src /path/to/Twitter10000v2."
        )

    gmner_dst = stage_root / "Twitter10000_v2.0"
    fmnerg_dst = stage_root / "Twitter10000v2"
    image_dst = stage_root / "images" / "Twitter10000"
    vinvl_dst = stage_root / "vinvl" / "Twitter10000_shared"

    ensure_symlink(gmner_src, gmner_dst)
    ensure_symlink(fmnerg_src, fmnerg_dst)

    if args.images_src is not None:
        if not args.images_src.exists():
            raise FileNotFoundError(f"Image directory does not exist: {args.images_src}")
        ensure_link_or_copy(args.images_src.resolve(), image_dst, args.images_mode)
    else:
        image_dst.mkdir(parents=True, exist_ok=True)

    vinvl_summary = None
    if args.vinvl_src is not None:
        if not args.vinvl_src.exists():
            raise FileNotFoundError(f"VinVL directory does not exist: {args.vinvl_src}")
        ensure_link_or_copy(args.vinvl_src.resolve(), vinvl_dst, args.vinvl_mode)

        sample_ids = []
        for split in ("train", "dev", "test"):
            sample_ids.extend(read_image_ids(gmner_src / "txt" / f"{split}.txt"))
        vinvl_summary = validate_vinvl_dir(vinvl_dst, sample_ids)
    else:
        vinvl_dst.mkdir(parents=True, exist_ok=True)

    env_path = stage_root / "grounded_paths.sh"
    write_paths_env(
        env_path,
        stage_root=stage_root,
        vinvl_dir=vinvl_dst if args.vinvl_src is not None else None,
        image_dir=image_dst,
    )

    print(f"[OK] GMNER data staged at: {gmner_dst}")
    print(f"[OK] FMNERG data staged at: {fmnerg_dst}")
    print(f"[OK] Image path ready at: {image_dst}")
    if args.vinvl_src is not None:
        print(f"[OK] VinVL path staged at: {vinvl_dst}")
        print(f"     npz files found: {vinvl_summary['num_npz']}")
        print(f"     sample ids checked: {len(vinvl_summary['checked'])}")
        if vinvl_summary["missing_examples"]:
            print(f"     missing examples: {vinvl_summary['missing_examples']}")
        else:
            print("     first 50 GMNER image ids all resolved successfully")
    else:
        print(f"[WARN] VinVL path placeholder created at: {vinvl_dst}")
        print("       Re-run with --vinvl-src once you finish downloading the shared Twitter10000 .npz features.")
    print(f"[OK] Environment helper written to: {env_path}")
    print("")
    print("Next steps:")
    print(f"  1. source {env_path}")
    print("  2. python scripts/check_assets.py --task gmner --dataset Twitter10000_v2.0 \\")
    print('       --data_dir "$IMAGINE_DATA_DIR" --vinvl_dir "$IMAGINE_VINVL_DIR" \\')
    print('       --annotation_dir "$IMAGINE_GMNER_ANNOTATION_DIR"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
