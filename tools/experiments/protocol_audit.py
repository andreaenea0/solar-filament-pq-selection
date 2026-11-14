"""Measure image/fold overlap and pixel duplicates without reading test annotations."""

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from filseg.data import load_records, records_by_image
from filseg.inference import read_jpeg
from training.prepare import read_folds


def split_summary(records, folds):
    """Check complete image-level assignment and measure shared observing days."""
    if records.keys() != folds.keys():
        raise ValueError("Training observations and fold assignments must match exactly.")
    by_day = defaultdict(set)
    for stem in records:
        by_day[stem[:8]].add(folds[stem])
    shared = {day: sorted(groups) for day, groups in by_day.items() if len(groups) > 1}
    groups = {stem: {record.record_id[:2] for record in values} for stem, values in records.items()}
    return dict(images=len(records), records=sum(map(len, records.values())), days=len(by_day),
                fold_images={str(k): sum(fold == k for fold in folds.values()) for k in sorted(set(folds.values()))},
                days_across_folds=shared,
                images_on_cross_fold_days=sum(stem[:8] in shared for stem in records),
                mixed_annotator_group_images=sum(len(values) > 1 for values in groups.values()))


def timestamp_summary(train, test):
    """Measure test proximity to training exposures using filenames for audit only."""
    def stamp(stem):
        return datetime.strptime(stem[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp()

    train_times = np.array([stamp(stem) for stem in train])
    sites = np.array([stem[14:] for stem in train])
    summaries = {}
    for same_site in (False, True):
        distances = []
        for stem in test:
            available = train_times[sites == stem[14:]] if same_site else train_times
            if len(available):
                distances.append(float(np.abs(available - stamp(stem)).min()) / 60)
        summaries["same_site" if same_site else "all_sites"] = dict(
            minimum_minutes=min(distances) if distances else None, eligible_test_images=len(distances),
            within_minutes={str(t): sum(d <= t for d in distances) for t in (2, 4, 6, 10, 60, 1440, 10080)})
    proposed = {(stamp(stem) + offset * 60, stem[14:]) for stem in train for offset in (-6, -4, -2, 2, 4, 6)}
    summaries["test_exposures_equal_proposed_training_neighbour"] = sum((stamp(s), s[14:]) in proposed for s in test)
    return summaries


def image_inventory(directory):
    """Hash exactly the grayscale pixels the release reads, one JPEG at a time."""
    result = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".jpeg", ".jpg"}:
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate JPEG stem: {path.stem}.")
        pixels = read_jpeg(path)
        result[path.stem] = hashlib.sha256(pixels.tobytes()).hexdigest()
    if not result:
        raise ValueError("Expected at least one official JPEG.")
    return result


def pixel_duplicates(train, test):
    """Find equal decoded observations even if JPEG filenames or container bytes differ."""
    by_hash = defaultdict(list)
    for stem, digest in train.items():
        by_hash[digest].append(stem)
    return {stem: by_hash[digest] for stem, digest in test.items() if digest in by_hash}


def summary_rows(result):
    """Yield compact public evidence without individual exposure identifiers."""
    values = {"training_images": result["images"], "training_records": result["records"],
              "observing_days": result["days"], "days_across_folds": len(result["days_across_folds"]),
              "images_on_cross_fold_days": result["images_on_cross_fold_days"], "test_images": result["test_images"],
              "train_test_stem_overlap": len(result["stem_overlap"]),
              "train_test_pixel_duplicates": len(result["pixel_duplicates"]),
              "mixed_annotator_group_images": result["mixed_annotator_group_images"]}
    for site in ("all_sites", "same_site"):
        proximity = result["proximity"][site]
        values[f"test_{site}_minimum_minutes_to_training"] = proximity["minimum_minutes"]
        values[f"test_{site}_within_6_minutes"] = proximity["within_minutes"]["6"]
        values[f"cross_fold_{site}_within_6_minutes"] = sum(
            c["proximity"][site]["within_minutes"]["6"] for c in result["cross_fold_checks"].values())
    values["test_exposures_equal_proposed_training_neighbour"] = result["proximity"][
        "test_exposures_equal_proposed_training_neighbour"]
    values["cross_fold_pixel_duplicates"] = sum(len(c["pixel_duplicates"]) for c in result["cross_fold_checks"].values())
    return [dict(metric=name, value=value) for name, value in values.items()]


def main():
    """Audit official train labels, fold CSV and JPEG directories; accept no test-label input."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-json", "folds", "train-images", "test-images", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="Optional compact public CSV, with no exposure identifiers.")
    args = parser.parse_args()
    records = records_by_image(load_records(args.train_json))
    folds = read_folds(args.folds)
    result = split_summary(records, folds)
    train, test = image_inventory(args.train_images), image_inventory(args.test_images)
    if train.keys() != records.keys():
        raise ValueError("Training JPEGs and annotated observations must match exactly.")
    result.update(test_images=len(test), stem_overlap=sorted(train.keys() & test.keys()),
                  pixel_duplicates=pixel_duplicates(train, test), proximity=timestamp_summary(train, test))
    result["cross_fold_checks"] = {}
    for fold in sorted(set(folds.values())):
        training = {stem: digest for stem, digest in train.items() if folds[stem] != fold}
        held = {stem: digest for stem, digest in train.items() if folds[stem] == fold}
        result["cross_fold_checks"][str(fold)] = dict(
            proximity=timestamp_summary(training, held), pixel_duplicates=pixel_duplicates(training, held))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("metric", "value"), lineterminator="\n")
            writer.writeheader()
            writer.writerows(summary_rows(result))
    print(f"Checked {len(train)} training and {len(test)} test JPEGs; "
          f"{len(result['days_across_folds'])} days cross folds; "
          f"{len(result['pixel_duplicates'])} train/test pixel duplicates. Details: {args.out}")


if __name__ == "__main__":
    main()
