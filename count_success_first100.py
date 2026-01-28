#!/usr/bin/env python3
import argparse
import os
import pickle
import sys


def _load_results(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Count successful 1-pixel attacks within first 100 images."
    )
    parser.add_argument(
        "--file",
        default="networks/results/results.pkl",
        help="Path to results pickle file.",
    )
    parser.add_argument("--model", default="vit", help="Filter by model name.")
    parser.add_argument("--pixels", type=int, default=1, help="Filter by pixel count.")
    parser.add_argument("--max-image", type=int, default=100, help="Images with id < max-image.")
    args = parser.parse_args()

    try:
        results = _load_results(args.file)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    total = 0
    success = 0
    success_images = set()

    for row in results:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        model = row[0]
        pixels = row[1]
        image_id = row[2]
        is_success = bool(row[5])

        if model != args.model:
            continue
        if pixels != args.pixels:
            continue
        if image_id >= args.max_image:
            continue

        total += 1
        if is_success:
            success += 1
            success_images.add(image_id)

    print(f"model={args.model} pixels={args.pixels} image< {args.max_image}")
    print(f"entries={total} success_entries={success} unique_success_images={len(success_images)}")


if __name__ == "__main__":
    main()
