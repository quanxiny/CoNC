"""Hash-verified portable external datasets with unchanged tensor values."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def load_external(dataset):
    """Load data/<dataset>; OVERFLOW_DATA_DIR can select an unpacked data bundle."""
    directory = Path(os.environ.get("OVERFLOW_DATA_DIR", ROOT / "data")) / dataset
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["dataset"] != dataset:
        raise ValueError("Dataset manifest identity mismatch")
    for name, expected in manifest["files"].items():
        if sha(directory / name) != expected:
            raise ValueError("Dataset file hash mismatch: " + name)
    read_rows = lambda name: [json.loads(line) for line in (directory / name).read_text().splitlines() if line]
    scenes = read_rows("scenes.jsonl")
    slices = read_rows("slices.jsonl")
    if (len(scenes), len(slices)) != (manifest["scene_count"], manifest["slice_count"]):
        raise ValueError("Dataset population changed")
    ids = {row["scene_id"] for row in scenes}
    if len(ids) != len(scenes) or len({row["slice_id"] for row in slices}) != len(slices):
        raise ValueError("Duplicate scene or slice identifiers")
    for row in slices:
        if len(row["owner_scene_ids"]) != 1 or row["owner_scene_ids"][0] not in ids:
            raise ValueError("Invalid slice ownership")
        row["tensor_cache"] = str((directory / "tensors.npz").resolve())
        row["status"] = "encoded"
    for row in scenes:
        row["source_path"] = str((directory / "sources" / Path(row["source_path"]).name).resolve())
        row["record_kind"] = "scene"
        row["task_applicability"] = "target"
        row["frontend_processable"] = row["processable"]
        row["slice_inventory_complete"] = row["processable"]
        row["family_id"] = row["group"]
    identity = {"manifest_sha256": sha(directory / "manifest.json"), **manifest["files"]}
    return {"scenes": scenes, "slices": slices, "manifest": manifest, "identity": identity}


def load_mit():
    data = load_external("mit")
    scenes = {row["scene_id"]: row for row in data["scenes"]}
    strict_ids = {sid for sid, row in scenes.items() if row["strict_eligible"]}
    extended_ids = {sid for sid, row in scenes.items() if row["extended_eligible"]}
    recovered_ids = extended_ids - strict_ids
    strict_slices = [row for row in data["slices"] if row["scene_id"] in strict_ids]
    recovered_slices = [row for row in data["slices"] if row["scene_id"] in recovered_ids]
    if (len(scenes), len(strict_ids), len(extended_ids), len(strict_slices), len(recovered_slices)) != (1164, 796, 1059, 1186, 392):
        raise ValueError("Frozen MIT cohort membership changed")
    return {"protocol": {"protocol_id": "mit_external_numeric_0_32767_v1"},
            "strict": {"identity": data["identity"], "scenes": data["scenes"], "slices": strict_slices},
            "recovered": {"receipt": data["identity"], "scenes": [scenes[sid] for sid in sorted(recovered_ids)], "slices": recovered_slices},
            "scenes": scenes, "strict_ids": strict_ids, "recovered_ids": recovered_ids,
            "extended_ids": extended_ids, "strict_slices": strict_slices,
            "extended_slices": strict_slices + recovered_slices}
