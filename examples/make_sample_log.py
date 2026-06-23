"""Emit a realistic eval log (JSONL + a pairwise JSONL + CSV) from the synthetic
judge, so you can see what `judge_audit.io` expects and run audit_log.py on it.

    python examples/make_sample_log.py
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from judge_audit import JudgeSpec, make_benchmark

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)


def main():
    spec = JudgeSpec()
    b = make_benchmark(spec)

    # designate the first `anchor_size` distinct item ids (per batch) as anchors:
    # here we just tag a fixed slice of items in every batch as the anchor set.
    n = len(b["score"])
    item_id = []
    counts = {}
    for batch in b["batch"]:
        counts[batch] = counts.get(batch, 0) + 1
        idx_in_batch = counts[batch] - 1
        is_anchor = idx_in_batch < spec.anchor_size
        kind = "anchor" if is_anchor else "item"
        item_id.append(f"{kind}-{batch}-{idx_in_batch}"
                       if not is_anchor else f"anchor-{idx_in_batch}")
    anchor_ids = sorted({iid for iid in item_id if iid.startswith("anchor-")})

    # long log: one row per item (plus a few repeat rows for self-consistency)
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n):
        base = {
            "item_id": item_id[i],
            "batch": int(b["batch"][i]),
            "score": float(b["score"][i]),
            "human": round(float(b["quality"][i]), 2),
            "length": int(b["length"][i]),
            "repeat": 0,
        }
        rows.append(base)
        # extra re-scores for a subset, to enable self-consistency
        if i % 5 == 0:
            for rpt in range(1, spec.repeats):
                rows.append({**base, "repeat": rpt,
                             "score": float(b["repeats"][i, rpt])})

    jsonl_path = os.path.join(OUT, "eval_log.jsonl")
    with open(jsonl_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    csv_path = os.path.join(OUT, "eval_log.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # pairwise log
    pw_path = os.path.join(OUT, "pairwise_log.jsonl")
    with open(pw_path, "w") as f:
        for j in range(len(b["winner_ab"])):
            f.write(json.dumps({"pair_id": f"p{j}", "order": "AB",
                                "winner": int(b["winner_ab"][j])}) + "\n")
            f.write(json.dumps({"pair_id": f"p{j}", "order": "BA",
                                "winner": int(b["winner_ba"][j])}) + "\n")

    with open(os.path.join(OUT, "anchor_ids.json"), "w") as f:
        json.dump(anchor_ids, f)

    print("wrote:", jsonl_path, csv_path, pw_path)
    print(f"{len(rows)} pointwise rows, {len(anchor_ids)} anchor ids")


if __name__ == "__main__":
    main()
