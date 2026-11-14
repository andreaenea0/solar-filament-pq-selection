"""Check that predictions and annotations go through the same geometric summary."""

import numpy as np
import pytest

from filseg.disk import Disk
from filseg.metric import encode
from tools.figures.physics_compare import as_drawings, compare


def test_sources_share_cuts_and_normalize_per_drawing():
    stems = ["20191201000000Bh", "20191202000000Bh"]
    disk = Disk(50, 50, 45, 0, 360)
    mask = np.zeros((101, 101), dtype=bool)
    mask[49:52, 42:59] = True
    rle = {"size": [101, 101], "counts": encode(mask)["counts"]}
    annotations = {stems[0]: {"a": [rle], "b": [rle]}, stems[1]: {"a": []}}
    submission = {stems[0]: [rle["counts"].decode("ascii")]}
    prediction = as_drawings(stems, submission)
    for records in prediction.values():
        for drawing in records.values():
            for item in drawing:
                item["size"] = [101, 101]
    assert prediction[stems[1]] == {"prediction": []}
    rows, instances = compare({"annotations": annotations, "model": prediction},
                              dict.fromkeys(stems, disk), dict.fromkeys(stems, 0.))
    first = {(r["source"], r["section"]): r for r in reversed(rows)}
    assert first["annotations", "radial"]["filaments"] == 2 and first["annotations", "radial"]["records"] == 3
    assert first["model", "radial"]["filaments"] == 1 and first["model", "radial"]["records"] == 2
    assert first["annotations", "tilt_all"]["tilt_deg_median"] == pytest.approx(0, abs=.1)
    assert first["model", "tilt_all"]["tilt_deg_median"] == first["annotations", "tilt_all"]["tilt_deg_median"]
    assert first["model", "size_all"]["eligible"] == 1
    assert first["model", "size_all"]["area_px_median"] == mask.sum()
    assert [(i["source"], i["stem"]) for i in instances] == [("annotations", stems[0])] * 2 + [("model", stems[0])]
    assert instances[0]["rho"] == pytest.approx(instances[2]["rho"])
