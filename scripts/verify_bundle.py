#!/usr/bin/env python3
"""Check the checksums of the source, data and checkpoint release components."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part",choices=["source","data","weights","hybrid_weights","all"],default="source")
    parser.add_argument("--syntax",action="store_true",help="Also parse the released Python source files.")
    args = parser.parse_args()
    parts = ["source","data","weights","hybrid_weights"] if args.part == "all" else [args.part]
    checked = 0
    for part in parts:
        manifest = ROOT/"manifests"/(part+".json")
        if not manifest.is_file():
            raise SystemExit(f"Missing manifest: {manifest.relative_to(ROOT)}")
        for entry in json.loads(manifest.read_text())["files"]:
            path = (ROOT/entry["path"]).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise SystemExit("Missing or invalid path: "+entry["path"])
            if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                raise SystemExit("Checksum mismatch: "+entry["path"])
            if args.syntax and path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8-sig"),filename=entry["path"])
            checked += 1
    print(json.dumps(dict(status="PASS",parts=parts,checked_files=checked)))


if __name__ == "__main__":
    main()
