"""Generate examples/demo.ipynb programmatically (valid JSON guaranteed)."""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = os.path.dirname(os.path.abspath(__file__))

md = new_markdown_cell
code = new_code_cell

cells = [
    md("# judge-audit — demo\n"
       "\n"
       "**Your eval is a measurement instrument. Audit it like one.**\n"
       "\n"
       "This notebook builds a synthetic LLM judge with *known* pathologies, "
       "then shows `judge-audit` recovering them across three axes "
       "(reliability, systematic bias, drift) plus a calibration correction "
       "with honest error bars. Same *inject → recover* idea as "
       "model-collapse-testbed."),

    code("import sys, os\n"
         "sys.path.insert(0, os.path.abspath('..'))\n"
         "import numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "import judge_audit as ja\n"
         "print('judge-audit', ja.__version__)"),

    md("## 1. Build a judge with injected pathologies\n"
       "\n"
       "`JudgeSpec` controls the planted biases; `make_benchmark` simulates "
       "12 batches of judgements plus a fixed anchor set and a pairwise "
       "(order-swapped) subset."),

    code("spec = ja.JudgeSpec()\n"
         "bench = ja.make_benchmark(spec)\n"
         "spec.truth  # the ground-truth magnitudes we'll try to recover"),

    md("## 2. Run the full audit\n"
       "\n"
       "`audit()` runs every axis whose data is present and returns a nested "
       "dict. On a real log some axes may be absent — that's fine."),

    code("a = ja.audit(bench)\n"
         "list(a.keys())"),

    md("### Axis 1 — reliability & validity\n"
       "Self-consistency is the *ceiling*: if the judge disagrees with itself, "
       "nothing downstream can be trusted."),

    code("r = a['reliability']\n"
         "print('self-consistency  alpha :', round(r['self_consistency']['alpha'], 3))\n"
         "print('self-consistency  flip  :', round(r['self_consistency']['flip_rate'], 3))\n"
         "print('validity Spearman rho   :', round(r['spearman_vs_human'], 3))\n"
         "print('validity interval alpha :', round(r['alpha_vs_human'], 3))"),

    md("### Axis 2 — systematic bias (controlled perturbation)\n"
       "Position bias from order-swapped pairwise verdicts; verbosity bias "
       "from an OLS that partials out true quality."),

    code("b = a['bias']\n"
         "print('first-slot rate  :', round(b['position']['first_pos_rate'], 3),\n"
         "      '(injected', spec.truth['first_pos_pref'], ')')\n"
         "print('order flip rate  :', round(b['position']['flip_rate'], 3))\n"
         "print('verbosity coef/SD:', round(b['verbosity']['length_coef_per_sd'], 3),\n"
         "      '(injected', spec.truth['verbosity_coef'], ', t =',\n"
         "      round(b['verbosity']['t_stat'], 1), ')')"),

    md("### Axis 3 — drift on the anchor set\n"
       "The provider silently ships a new model version at batch 7. An EWMA "
       "control chart on the fixed anchor set catches the shift."),

    code("d = a['drift']\n"
         "ew = d['ewma']\n"
         "x = np.arange(len(d['anchor_means']))\n"
         "plt.figure(figsize=(8, 3.3))\n"
         "plt.plot(x, d['anchor_means'], 'o-', color='#888', lw=1, label='anchor mean')\n"
         "plt.plot(x, ew['ewma'], color='#1f77b4', lw=2, label='EWMA')\n"
         "plt.plot(x, ew['ucl'], '--', color='#d62728', lw=1, label='control limits')\n"
         "plt.plot(x, ew['lcl'], '--', color='#d62728', lw=1)\n"
         "if ew['first_alarm'] is not None:\n"
         "    plt.axvline(ew['first_alarm'], color='#d62728', alpha=0.3)\n"
         "plt.title('Anchor-set drift'); plt.xlabel('batch'); plt.ylabel('judge score')\n"
         "plt.legend(fontsize=8); plt.tight_layout(); plt.show()\n"
         "print('EWMA first alarm at batch', ew['first_alarm'],\n"
         "      '| injected changepoint', spec.truth['drift_changepoint'])"),

    md("## 3. Calibration — map judge scores onto the human scale\n"
       "Isotonic regression fit on a calibration split, evaluated on held-out "
       "data. ECE drops sharply; the raw curve bows below the diagonal "
       "(the judge compresses the top of the scale)."),

    code("c = a['calibration']\n"
         "xb, yb = c['curve_before']; xa, ya = c['curve_after']\n"
         "plt.figure(figsize=(4.5, 4.5))\n"
         "plt.plot([0,1],[0,1], ':', color='#aaa', label='perfect')\n"
         "plt.plot(xb, yb, 'o-', color='#d62728', label=f\"raw (ECE {c['before']['ece']:.3f})\")\n"
         "plt.plot(xa, ya, 's-', color='#2ca02c', label=f\"isotonic (ECE {c['after']['ece']:.3f})\")\n"
         "plt.xlabel('predicted (norm)'); plt.ylabel('actual (norm)')\n"
         "plt.title('Reliability diagram'); plt.legend(fontsize=8)\n"
         "plt.tight_layout(); plt.show()"),

    md("## 4. Don't read noise as signal\n"
       "Two models with a 0.05-pt true quality gap. The bootstrap CI on the "
       "judged difference straddles 0 — so the leaderboard **cannot** call a "
       "winner."),

    code("tm = ja.two_model_scores(spec, true_gap=0.05)\n"
         "diff = ja.bootstrap_diff(tm['a'], tm['b'], seed=1)\n"
         "print(f\"A - B = {diff['diff']:+.3f}  95% CI [{diff['lo']:+.3f}, {diff['hi']:+.3f}]\")\n"
         "print('significant:', diff['significant'])"),

    md("## 5. Inject → recover scorecard"),

    code("print(ja.render_markdown(a))"),

    md("## 6. Same audit on a real eval log\n"
       "The `io` adapter turns a CSV/JSONL log into the same dict. Run\n"
       "`python make_sample_log.py` first to create `data/eval_log.jsonl`."),

    code("log_path = 'data/eval_log.jsonl'\n"
         "if os.path.exists(log_path):\n"
         "    import json\n"
         "    recs = ja.read_jsonl(log_path)\n"
         "    anchors = json.load(open('data/anchor_ids.json'))\n"
         "    blog = ja.assemble(recs, scale=(1,5), anchor_ids=anchors)\n"
         "    blog.update(ja.pairwise(ja.read_jsonl('data/pairwise_log.jsonl')))\n"
         "    alog = ja.audit(blog)\n"
         "    print('axes from real log:',\n"
         "          [k for k in ('reliability','bias','drift','calibration') if k in alog])\n"
         "else:\n"
         "    print('run: python make_sample_log.py')"),
]

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})

out = os.path.join(HERE, "demo.ipynb")
nbf.write(nb, out)
print("wrote", out)
