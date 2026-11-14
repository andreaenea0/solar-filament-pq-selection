"""Time the released submitted entry point without changing inference operations.

Run as a module. References are predictions, never test labels.
Stage times are exclusive; GPU stages synchronize CUDA.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager, ExitStack
import csv
from functools import wraps
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ("effv2s_s0_fold-1.pt", "effv2s_s1_fold-1.pt",
           "own_ver3_a05_l025_nometa_lgb.txt", "verifier_resnet18_s0.pt")
STAGES = ("runtime_import", "device_setup", "model_load", "input_io_normalize",
          "unet_seed0_d4", "unet_seed1_d4", "tree_build", "features",
          "lightgbm", "verifier", "dp_selection", "selection_overhead", "rle_csv")


class StageTimer:
    """Measure exclusive wall time, subtracting nested measured work."""
    def __init__(self, clock=perf_counter, synchronize=None):
        self.clock, self.synchronize = clock, synchronize
        self.seconds, self.calls = defaultdict(float), defaultdict(int)
        self.stack = []

    @contextmanager
    def measure(self, stage, gpu=False):
        if gpu and self.synchronize:
            self.synchronize()
        frame = [self.clock(), 0.]
        self.stack.append(frame)
        try:
            yield
        finally:
            if gpu and self.synchronize:
                self.synchronize()
            elapsed = self.clock() - frame[0]
            self.stack.pop()
            self.seconds[stage] += elapsed - frame[1]
            self.calls[stage] += 1
            if self.stack:
                self.stack[-1][1] += elapsed

    def wrap(self, stage, function, gpu=False):
        """Preserve call arguments and return values while measuring a call."""
        @wraps(function)
        def wrapped(*args, **kwargs):
            with self.measure(stage, gpu):
                return function(*args, **kwargs)
        return wrapped


@contextmanager
def instrument(timer, inference, candidates, oof, select, verifier, lgb, submit):
    """Restore every hook on exit, including after an inference exception."""
    model_stages = {}
    original_load, original_predict = inference.load_model, inference.predict_map

    def load_model(*args, **kwargs):
        with timer.measure("model_load", gpu=True):
            model = original_load(*args, **kwargs)
        model_stages[id(model)] = f"unet_seed{len(model_stages)}_d4"
        return model

    def predict_map(model, *args, **kwargs):
        with timer.measure(model_stages[id(model)], gpu=True):
            return original_predict(model, *args, **kwargs)

    hooks = [(inference, "load_model", load_model), (inference, "predict_map", predict_map)]
    for module, name, stage, gpu in (
        (inference, "read_jpeg", "input_io_normalize", False),
        (inference, "normalize", "input_io_normalize", False),
        (candidates, "_feats", "features", False),
        (oof, "build_tree", "tree_build", False),
        (oof, "candidate_record", "features", False),
        (select, "feature_frame", "features", False),
        (select, "learned_masks", "selection_overhead", False),
        (select, "dp_select", "dp_selection", False),
        (lgb.Booster, "__init__", "model_load", False),
        (lgb.Booster, "predict", "lightgbm", False),
        (verifier, "load_verifier", "model_load", True),
        (verifier, "candidate_crops", "features", False),
        (verifier, "verifier_gains", "verifier", True),
        (submit, "write_submission", "rle_csv", False),
    ):
        hooks.append((module, name, timer.wrap(stage, getattr(module, name), gpu)))
    with ExitStack() as stack:
        for module, name, wrapped in hooks:
            stack.enter_context(patch.object(module, name, wrapped))
        yield


def peak_cpu_mib():
    """Return this process's OS high-water resident memory in MiB."""
    if sys.platform != "win32":
        import resource
        scale = 1024 ** 2 if sys.platform == "darwin" else 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in
            ("peak_working", "working", "peak_paged", "paged", "peak_nonpaged",
             "nonpaged", "pagefile", "peak_pagefile")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),
                                      ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.peak_working / 1024 ** 2


def read_rows(path, stems, subset=False):
    """Read prediction RLEs, checking IDs; only the reference may be a superset."""
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != {"filament_id", "segmentation_rle"}:
            raise ValueError(f"unexpected CSV columns in {path}")
        for row in reader:
            stem, separator, index = row["filament_id"].rpartition("_")
            if not separator or not index.isdigit() or int(index) < 1:
                raise ValueError(f"invalid prediction ID: {row['filament_id']}")
            if stem not in stems:
                if subset:
                    continue
                raise ValueError(f"unexpected image: {stem}")
            if row["filament_id"] in result:
                raise ValueError(f"duplicate prediction ID: {row['filament_id']}")
            result[row["filament_id"]] = row["segmentation_rle"]
    return result


def replay_summary(reference, actual, stems):
    """Match masks within each image at strict IoU > .5, independently of IDs."""
    import numpy as np
    from filseg.metric import iou_matrix
    from filseg.rle import counts_to_rle

    shared = set(reference) & set(actual)
    matches, tp, fp, fn = [], 0, 0, 0
    for stem in sorted(stems):
        ref = [counts_to_rle(value) for key, value in reference.items()
               if key.rpartition("_")[0] == stem]
        pred = [counts_to_rle(value) for key, value in actual.items()
                if key.rpartition("_")[0] == stem]
        overlaps = iou_matrix(pred, ref)
        values = overlaps[overlaps > .5]
        tp += len(values)
        fp += len(pred) - len(values)
        fn += len(ref) - len(values)
        matches.extend(values.tolist())
    denominator = tp + .5 * (fp + fn)
    return dict(images=len(stems), reference_masks=len(reference), actual_masks=len(actual),
                shared_ids=len(shared),
                exact_rles=sum(reference[key] == actual[key] for key in shared),
                strict_matches=tp, unmatched_actual=fp, unmatched_reference=fn,
                minimum_iou=min(matches) if matches else None,
                mean_matched_iou=float(np.mean(matches)) if matches else None,
                prediction_pq_agreement=sum(matches) / denominator if denominator else 0.)


def write_csv(path, rows):
    """Write flat, full-precision evidence with stable newline conventions."""
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_weights(directory):
    """Verify all four submitted checkpoints against the public release manifest."""
    expected = {}
    for line in (ROOT / "release_weights.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[Path(name.lstrip("*")).name] = digest
    hashes = {}
    for name in WEIGHTS:
        with (directory / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected[name]:
            raise ValueError(f"release checksum mismatch: {name}")
        hashes[name] = digest
    return hashes


def main():
    """Run the released entry point, then validate and compare predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "docs/reference/submitted_test.csv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tta-batch", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--expected-images", type=int, default=180)
    args = parser.parse_args()
    args.images, args.weights_dir = args.images.resolve(), args.weights_dir.resolve()
    images = sorted(p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"))
    stems = {p.stem for p in images}
    if args.expected_images < 1 or len(images) != args.expected_images or len(stems) != len(images):
        parser.error("image count differs from --expected-images, or duplicate stems")
    if args.out_dir.exists():
        parser.error("--out-dir must be new")
    hashes = verify_weights(args.weights_dir)
    args.out_dir.mkdir(parents=True)
    timer, started = StageTimer(), perf_counter()
    with timer.measure("runtime_import"):
        import torch
        import lightgbm as lgb
        from filseg import inference, candidates, oof, select, verifier
        import submit
    with timer.measure("device_setup"):
        device = torch.device(args.device)
        if device.type not in ("cpu", "cuda"):
            parser.error("only CPU or one CUDA device is supported")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                parser.error("CUDA is unavailable")
            torch.cuda.set_device(device)
            timer.synchronize = lambda: torch.cuda.synchronize(device)
            timer.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            device_name = torch.cuda.get_device_name(device)
        else:
            device_name = platform.processor() or "CPU"
    output = args.out_dir / "submission.csv"
    command = ["submit.py", "--images", str(args.images), "--device", str(device),
               "--tta-batch", str(args.tta_batch), "--final", "submitted",
               "--weights", *(str(args.weights_dir / name) for name in WEIGHTS[:2]),
               "--selector", str(args.weights_dir / WEIGHTS[2]),
               "--verifier-weights", str(args.weights_dir / WEIGHTS[3]),
               "--out", str(output)]
    with instrument(timer, inference, candidates, oof, select, verifier, lgb, submit):
        with patch.object(sys, "argv", command):
            submit.main()
    if timer.synchronize:
        timer.synchronize()
    total = perf_counter() - started
    memory = dict(peak_cpu_mib=peak_cpu_mib(),
                  peak_gpu_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2
                  if device.type == "cuda" else 0.,
                  peak_gpu_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2
                  if device.type == "cuda" else 0.)
    common = dict(images=len(images), device=str(device), device_name=device_name,
                  torch_threads=torch.get_num_threads(), **memory)
    seconds = {stage: timer.seconds[stage] for stage in STAGES}
    seconds["unattributed_overhead"] = total - sum(seconds.values())
    seconds["total"] = total
    records = [dict(stage=stage, calls=timer.calls[stage], seconds=value,
                    seconds_per_image=value / len(images), **common)
               for stage, value in seconds.items()]
    write_csv(args.out_dir / "inference_timings.csv", records)
    from tools.check_submission import check_submission
    check_submission(output, stems, allow_empty_images=True)
    summary = dict(recipe="submitted", **replay_summary(
        read_rows(args.reference, stems, subset=True), read_rows(output, stems), stems))
    write_csv(args.out_dir / "replay_summary.csv", [summary])
    metadata = dict(
        command=command, weights_sha256=hashes,
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        versions={name: importlib.metadata.version(name) for name in
                  ("torch", "numpy", "scipy", "segmentation-models-pytorch", "timm",
                   "lightgbm", "pycocotools")},
        python=sys.version, platform=platform.platform(), **common,
        timing_scope="runtime imports through CSV write; excludes preflight and comparison",
        stage_scope="exclusive wall times; CUDA synchronized around GPU stages",
        cpu_memory_scope="process OS high-water RSS before reference comparison",
        agreement_scope="released predictions, not test ground truth or competition PQ")
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(total_seconds=total, **summary), indent=2))


if __name__ == "__main__":
    main()
