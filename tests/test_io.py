import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge_audit import assemble, pairwise, audit


def _records():
    recs = []
    for b in range(4):
        for i in range(30):
            iid = f"anchor-{i}" if i < 8 else f"item-{b}-{i}"
            recs.append({"item_id": iid, "batch": b, "score": 3 + (i % 3),
                         "human": 2 + (i % 3), "length": 100 + i})
            if i % 4 == 0:  # a repeat row
                recs.append({"item_id": iid, "batch": b, "score": 3 + (i % 3),
                             "human": 2 + (i % 3), "length": 100 + i})
    return recs


def test_assemble_builds_expected_keys():
    recs = _records()
    anchors = [f"anchor-{i}" for i in range(8)]
    bench = assemble(recs, scale=(1, 5), anchor_ids=anchors)
    assert "score" in bench and "quality" in bench and "length" in bench
    assert "repeats" in bench  # because some items repeat
    assert "anchor_means" in bench and len(bench["anchor_means"]) == 4
    assert bench["scale"] == (1, 5)


def test_assemble_infers_scale_and_runs_partial_audit():
    # log with score only -> audit still runs, but only what it can
    recs = [{"item_id": f"x{i}", "score": float(1 + i % 5)} for i in range(50)]
    bench = assemble(recs)
    a = audit(bench)
    assert "calibration" not in a  # no human gold
    assert a["n_items"] == 50
    assert bench["scale"][0] <= 1 and bench["scale"][1] >= 5


def test_field_mapping_override():
    recs = [{"id": "a", "rating": 4, "gold": 3, "tok": 120},
            {"id": "b", "rating": 2, "gold": 2, "tok": 80}]
    bench = assemble(recs, fields={"item_id": "id", "score": "rating",
                                   "human": "gold", "length": "tok"},
                     scale=(1, 5))
    assert np.allclose(bench["score"], [4, 2])
    assert np.allclose(bench["quality"], [3, 2])


def test_pairwise_aligns_on_pair_id():
    recs = [
        {"pair_id": "p1", "order": "AB", "winner": 0},
        {"pair_id": "p1", "order": "BA", "winner": 1},
        {"pair_id": "p2", "order": "AB", "winner": 1},
        {"pair_id": "p2", "order": "BA", "winner": 0},
        {"pair_id": "p3", "order": "AB", "winner": 0},  # no BA -> dropped
    ]
    out = pairwise(recs)
    assert len(out["winner_ab"]) == 2 and len(out["winner_ba"]) == 2


def test_assemble_audit_full_axes():
    recs = _records()
    anchors = [f"anchor-{i}" for i in range(8)]
    bench = assemble(recs, scale=(1, 5), anchor_ids=anchors)
    bench.update(pairwise([
        {"pair_id": "p1", "order": "AB", "winner": 0},
        {"pair_id": "p1", "order": "BA", "winner": 0},
    ]))
    a = audit(bench)
    for axis in ("reliability", "bias", "drift", "calibration"):
        assert axis in a
