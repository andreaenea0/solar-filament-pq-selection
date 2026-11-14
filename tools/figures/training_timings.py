"""Extract public timing evidence from saved original training-log events."""

import argparse
import csv
import json
from pathlib import Path
import re


def timing_rows(recipe, events):
    """Read printed worker timers, keeping final-epoch and export scopes distinct."""
    text = "\n".join(event.get("data", "") for event in events)
    if recipe == "k17":
        epochs = re.findall(r"\[-1 resnet18\.a1_in1k (\d+)\] epoch 11 loss \S+ (\d+)s", text)
        if {seed for seed, _ in epochs} != {"0", "1", "2"} or len(epochs) != 3:
            raise ValueError("Expected one final-epoch timer for each verifier seed.")
        return [dict(recipe=recipe, worker=f"seed_{seed}", preparation_seconds="", seconds=int(seconds),
                     scope="through_final_training_epoch") for seed, seconds in sorted(epochs)]
    preparation = re.findall(r"(?:^|\n)prep (\d+) s", text)
    workers = re.findall(r"\[([^\]]+)\] done (\d+)s", text)
    if len(preparation) != 1 or len(workers) != 2:
        raise ValueError("Expected one shared preparation timer and two completed pixel workers.")
    return [dict(recipe=recipe, worker=f"{name}_event_{i}", preparation_seconds=int(preparation[0]),
                 seconds=int(seconds), scope="training_and_map_export")
            for i, (name, seconds) in enumerate(workers, 1)]


def main():
    """Convert the five saved logs to a compact table without private service identifiers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True, help="Directory of kNN-training.log JSON event lists.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for recipe in ("k06", "k09", "k11", "k07", "k17"):
        raw = (args.logs / f"{recipe}-training.log").read_bytes()
        encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        rows.extend(timing_rows(recipe, json.loads(raw.decode(encoding))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
