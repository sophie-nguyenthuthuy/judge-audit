"""Adapters: turn real eval logs (CSV / JSONL) into the dict the audit consumes.

Real logs rarely have every axis. `assemble` builds whatever it can from the
columns you have; `audit()` then runs only the axes whose data is present. The
minimum useful log is one row per judgement with a score column.

Canonical "long" record (column names are configurable via `fields`):
    item_id   stable id of the thing being judged (groups repeats)
    score     the judge's score (required)
    human     ground-truth / human label        -> validity + calibration
    length    response length (tokens or chars) -> verbosity bias
    batch     ordered bucket (week, run, date)  -> drift (with anchors)
    repeat    index of a re-score of the same item -> self-consistency

Pairwise position bias uses a separate record list (see `pairwise`):
    pair_id   stable id of the A-vs-B comparison
    order     'AB' or 'BA' (which slot held which content)
    winner    0 = first slot won, 1 = second slot won
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

__all__ = ["read_jsonl", "read_csv", "assemble", "pairwise"]

DEFAULT_FIELDS = {
    "item_id": "item_id",
    "score": "score",
    "human": "human",
    "length": "length",
    "batch": "batch",
    "repeat": "repeat",
}


def read_jsonl(path):
    """Read a JSONL file into a list of dict records."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_csv(path):
    """Read a CSV file into a list of dict records (values stay strings)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _get(rec, key):
    v = rec.get(key, None)
    if v is None or v == "":
        return None
    return v


def _to_repeats(records, f):
    """Group scores by item_id across repeats -> (n_items, max_R), nan-padded."""
    by_item = defaultdict(list)
    order = []
    for r in records:
        iid = _get(r, f["item_id"])
        sc = _get(r, f["score"])
        if iid is None or sc is None:
            continue
        if iid not in by_item:
            order.append(iid)
        by_item[iid].append(float(sc))
    maxR = max((len(v) for v in by_item.values()), default=0)
    if maxR < 2:
        return None  # no item scored more than once -> can't measure consistency
    M = np.full((len(order), maxR), np.nan)
    for i, iid in enumerate(order):
        v = by_item[iid]
        M[i, : len(v)] = v
    return M


def assemble(records, fields=None, scale=None, anchor_ids=None):
    """Build an audit-ready dict from long-format records.

    fields: optional override of column names (see DEFAULT_FIELDS).
    scale:  (lo, hi); inferred from score+human range if omitted.
    anchor_ids: iterable of item_ids forming the fixed anchor set; required to
                compute drift (per-batch mean over those items).

    Returns a dict with only the keys it could build. Pass it straight to
    `audit()`. For position bias, also build `pairwise(...)` and merge.
    """
    f = {**DEFAULT_FIELDS, **(fields or {})}

    item_id, score, human, length, batch = [], [], [], [], []
    have_human = have_len = have_batch = False
    for r in records:
        sc = _get(r, f["score"])
        if sc is None:
            continue
        item_id.append(_get(r, f["item_id"]))
        score.append(float(sc))
        h = _get(r, f["human"])
        human.append(float(h) if h is not None else np.nan)
        have_human |= h is not None
        ln = _get(r, f["length"])
        length.append(float(ln) if ln is not None else np.nan)
        have_len |= ln is not None
        b = _get(r, f["batch"])
        batch.append(b)
        have_batch |= b is not None

    score = np.asarray(score, dtype=float)
    out = {"score": score, "item_id": np.asarray(item_id, dtype=object)}

    if have_human:
        out["quality"] = np.asarray(human, dtype=float)  # "human" plays the gold role
    if have_len:
        out["length"] = np.asarray(length, dtype=float)

    if scale is None:
        pool = score[~np.isnan(score)]
        if have_human:
            hp = out["quality"][~np.isnan(out["quality"])]
            pool = np.concatenate([pool, hp])
        scale = (float(np.floor(pool.min())), float(np.ceil(pool.max())))
    out["scale"] = scale

    reps = _to_repeats(records, f)
    if reps is not None:
        out["repeats"] = reps

    if have_batch and anchor_ids is not None:
        anchor_ids = set(anchor_ids)
        # ordered unique batches (numeric if possible, else by first appearance)
        seen = []
        for b in batch:
            if b is not None and b not in seen:
                seen.append(b)
        try:
            seen = sorted(seen, key=lambda x: float(x))
        except (TypeError, ValueError):
            pass
        means = []
        for b in seen:
            vals = [score[i] for i in range(len(score))
                    if batch[i] == b and item_id[i] in anchor_ids]
            if vals:
                means.append(float(np.mean(vals)))
        if len(means) >= 2:
            out["anchor_means"] = np.asarray(means, dtype=float)

    return out


def pairwise(records, fields=None):
    """Build (winner_ab, winner_ba) aligned by pair_id from swapped judgements.

    Only pairs that were judged in BOTH orders contribute. Returns a dict
    {'winner_ab', 'winner_ba'} to merge into the assemble() output.
    """
    f = {"pair_id": "pair_id", "order": "order", "winner": "winner",
         **(fields or {})}
    ab, ba = {}, {}
    for r in records:
        pid = _get(r, f["pair_id"])
        order = _get(r, f["order"])
        win = _get(r, f["winner"])
        if pid is None or order is None or win is None:
            continue
        w = int(win)
        if str(order).upper() == "AB":
            ab[pid] = w
        elif str(order).upper() == "BA":
            ba[pid] = w
    common = [p for p in ab if p in ba]
    return {
        "winner_ab": np.asarray([ab[p] for p in common], dtype=int),
        "winner_ba": np.asarray([ba[p] for p in common], dtype=int),
    }
