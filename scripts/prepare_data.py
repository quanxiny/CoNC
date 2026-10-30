#!/usr/bin/env python3
"""Verify supplied assets or unpack the original OverflowGen archive locally."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath

ARCHIVES = {
    'datasets/overflowgen_24455320/data.zip': '70d4785d8c3937055275482e97527a095b0eb28fb1b9deb9ec0b829d582d54fe',
    'datasets/2014-06-08-a-taxonomy-of-buffer-overflows.zip': '79ab9203a8e66271ad623a9feef19c32a6a10e2cebbdf54fd3c4d7c01cad6e80',
    'external_eval/vendor/mit/2073-v2.0.0.zip': 'a969f0c87bc15270d151bec1050b47375ce3e24d4436c70b95e6820295f0ec88',
}
PACKAGE = 'transformer_gru_v76_v2'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def check(root: Path) -> dict:
    checked = []
    for relative, expected in ARCHIVES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f'Missing data archive: {relative}. Extract the data attachment at the repository root.')
        if sha(path) != expected:
            raise ValueError(f'Archive checksum mismatch: {relative}')
        checked.append(relative)
    for dataset in ['mit', 'boundary']:
        directory = root / 'data' / dataset
        manifest = json.loads((directory / 'manifest.json').read_text())
        for relative, expected in manifest['files'].items():
            if sha(directory / relative) != expected:
                raise ValueError(f'Frozen input checksum mismatch: data/{dataset}/{relative}')
            checked.append(f'data/{dataset}/{relative}')
    directory = root / PACKAGE / 'src/research_project/ubuntu_local_research_store/hybrid_numeric_semantic_v59/baseline_dev60'
    manifest = json.loads((directory / 'MANIFEST60.json').read_text())
    for relative, expected in manifest['array_sha256'].items():
        if sha(directory / relative) != expected:
            raise ValueError(f'Selected TRAIN/DEV tensor checksum mismatch: {relative}')
        checked.append(relative)
    return {'status': 'PASS', 'checked_files': len(checked), 'archives': len(ARCHIVES),
            'MIT': {'target_scenes': 1164, 'complete_scenes': 1059},
            'boundary': {'programs': 448, 'pairs': 224}}


def crc(path: Path) -> int:
    value = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            value = zlib.crc32(chunk, value)
    return value & 0xffffffff


def extract_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo, target: Path) -> bool:
    if target.exists():
        if target.is_file() and target.stat().st_size == member.file_size and crc(target) == member.CRC:
            return False
        raise FileExistsError(f'Refusing to overwrite different existing data: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=target.name + '.', suffix='.part', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as output, archive.open(member) as source:
            shutil.copyfileobj(source, output, 2**20)
        Path(name).replace(target)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return True


def unpack(root: Path, mode: str) -> dict:
    archive_path = root / 'datasets/overflowgen_24455320/data.zip'
    expected = ARCHIVES['datasets/overflowgen_24455320/data.zip']
    if sha(archive_path) != expected:
        raise ValueError('OverflowGen archive checksum mismatch')
    destination = root / 'datasets/overflowgen_24455320'
    author_root = root / PACKAGE / 'external/overflowgen/author_data'
    copied = 0
    reused = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in member.filename:
                raise ValueError(f'Unsafe archive member: {member.filename}')
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f'Archive symbolic links are not supported: {member.filename}')
            if member.is_dir() or not path.parts or path.parts[0] != 'data':
                continue
            is_slice_file = len(path.parts) == 3 and path.parts[1] == 'overflow_gen' and path.suffix == '.json'
            if mode == 'slices' and not is_slice_file:
                continue
            targets = [destination / Path(*path.parts)]
            if is_slice_file:
                targets.append(author_root / Path(*path.parts))
            for target in targets:
                if extract_member(archive, member, target):
                    copied += 1
                else:
                    reused += 1
    if mode == 'full':
        alias = root / PACKAGE / 'external/overflowgen/archives/data.zip'
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.exists():
            if sha(alias) != expected:
                raise FileExistsError(f'Refusing to overwrite a different cache-rebuild archive: {alias}')
        else:
            try:
                os.link(archive_path, alias)
            except OSError:
                shutil.copyfile(archive_path, alias)
    return {'status': 'PASS', 'mode': mode, 'created_files': copied, 'verified_existing_files': reused,
            'source_root': 'datasets/overflowgen_24455320/data/overflow_gen',
            'author_input_root': f'{PACKAGE}/external/overflowgen/author_data/data/overflow_gen'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Repository root after extracting the data attachment.')
    parser.add_argument('--check', action='store_true', help='Verify supplied archives and frozen tensors without extracting or modifying data.')
    parser.add_argument('--mode', choices=['slices', 'full'], default='slices',
                        help='slices: extract author JSON inputs; full: also extract original C programs and graph tables for frontend/static-baseline reruns.')
    args = parser.parse_args()
    root = args.root.resolve()
    result = check(root) if args.check else unpack(root, args.mode)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
