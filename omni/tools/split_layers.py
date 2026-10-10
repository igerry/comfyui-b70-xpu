#!/usr/bin/env python3
"""Distribute the contents of a directory tree over N staging directories of similar size.

    split_layers.py <root> <stage_dir> <n> [threshold_mb=400]

Each staging directory `<stage_dir>/<nn>/` receives a part of `<root>` under the same absolute
path, so the Dockerfile can rebuild the tree with N `COPY --from=... /stage/<nn>/ /` instructions.
The point is N image layers instead of one: Docker pushes and pulls layers in parallel, and a
broken transfer only repeats the layers that were not finished.

Directories larger than the threshold are descended into; everything else is moved as a unit.
Items are packed largest-first into the currently lightest bucket.
"""
import os
import shutil
import sys

root = os.path.abspath(sys.argv[1])
stage = os.path.abspath(sys.argv[2])
n = int(sys.argv[3])
threshold = (int(sys.argv[4]) if len(sys.argv) > 4 else 400) * 1024 * 1024


def size_of(path):
    if os.path.islink(path) or not os.path.isdir(path):
        return os.lstat(path).st_size
    total = 0
    for base, dirs, files in os.walk(path):
        for name in files + dirs:
            try:
                total += os.lstat(os.path.join(base, name)).st_size
            except OSError:
                pass
    return total


items = []          # (size, path)
empty_dirs = []     # directories that were descended into and must exist in the result


def collect(directory):
    empty_dirs.append(directory)
    for entry in sorted(os.scandir(directory), key=lambda e: e.name):
        size = size_of(entry.path)
        if entry.is_dir(follow_symlinks=False) and size > threshold:
            collect(entry.path)
        else:
            items.append((size, entry.path))


collect(root)
buckets = [[0, []] for _ in range(n)]
for size, path in sorted(items, reverse=True):
    bucket = min(buckets, key=lambda b: b[0])
    bucket[0] += size
    bucket[1].append(path)

for index, (total, paths) in enumerate(buckets):
    base = os.path.join(stage, f"{index:02d}")
    os.makedirs(base, exist_ok=True)
    for path in paths:
        target = base + path
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.move(path, target)
    print(f"layer {index:02d}: {total / 1048576:8.0f} MB, {len(paths)} items")

# Keep the (now empty) directory skeleton in bucket 00 with its original modes.
for directory in empty_dirs:
    target = os.path.join(stage, "00") + directory
    os.makedirs(target, exist_ok=True)
    shutil.copystat(directory, target)
shutil.rmtree(root)
