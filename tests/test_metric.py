"""The local scorer must reproduce the known annotator-vs-annotator agreement (pooled PQ 0.3399)."""
import os
import pytest
from filseg.data import load_records, gt_rles_by_image
from filseg.metric import Totals, score_record

J = os.environ.get("MAGFILO_TRAIN_JSON")


@pytest.mark.skipif(not J or not os.path.exists(J), reason="set MAGFILO_TRAIN_JSON to the training annotations")
def test_annotator_agreement():
    gt = gt_rles_by_image(load_records(J))
    T = Totals()
    for R in gt.values():
        ids = list(R)
        for a in ids:
            for b in ids:
                if a != b:
                    T += score_record(R[b], R[a])
    assert abs(T.pq - 0.3399) < 5e-4
