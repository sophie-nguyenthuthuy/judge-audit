"""Audit a REAL eval log (no synthetic ground truth needed).

    python examples/make_sample_log.py     # once, to create examples/data/*
    python examples/audit_log.py [eval_log.jsonl] [pairwise_log.jsonl]

Shows the io adapter path: read logs -> assemble -> audit -> report. Works on
your own logs by passing paths and (if your columns differ) a field mapping.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge_audit import read_jsonl, assemble, pairwise, audit, render_markdown, save_plots

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "eval_log.jsonl")
    pw_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATA, "pairwise_log.jsonl")
    outdir = os.path.join(HERE, "out_log")

    records = read_jsonl(log_path)
    anchor_path = os.path.join(DATA, "anchor_ids.json")
    anchor_ids = json.load(open(anchor_path)) if os.path.exists(anchor_path) else None

    bench = assemble(records, scale=(1, 5), anchor_ids=anchor_ids)
    if os.path.exists(pw_path):
        bench.update(pairwise(read_jsonl(pw_path)))

    a = audit(bench)
    plots = save_plots(a, outdir)
    md = render_markdown(a, plot_paths=plots)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "report.md")
    open(out, "w").write(md)

    print(f"audited {a['n_items']} judgements -> {out}")
    print("axes computed:", [k for k in ("reliability", "bias", "drift", "calibration")
                             if k in a])


if __name__ == "__main__":
    main()
