"""judge-audit — measure the reliability of your LLM-as-judge eval pipeline.

Your eval is a measurement instrument. Before you trust its numbers, prove the
instrument is *reliable* (repeatable) and *valid* (measures what you think).
This package audits a judge on three axes — agreement, systematic bias,
temporal drift — and fits a calibration correction with honest error bars.

    from judge_audit import make_benchmark, audit, render_markdown
    bench = make_benchmark()
    report = render_markdown(audit(bench))
"""
from .agreement import krippendorff_alpha, spearman, self_consistency
from .bias import position_bias, length_bias
from .drift import ewma_chart, cusum
from .calibrate import (
    IsotonicCalibrator, isotonic_fit, calibration_error, reliability_curve,
    bootstrap_ci, bootstrap_diff,
)
from .synthetic import JudgeSpec, make_benchmark, two_model_scores
from .report import audit, render_markdown, save_plots
from .io import read_jsonl, read_csv, assemble, pairwise

__version__ = "0.1.0"

__all__ = [
    "krippendorff_alpha", "spearman", "self_consistency",
    "position_bias", "length_bias",
    "ewma_chart", "cusum",
    "IsotonicCalibrator", "isotonic_fit", "calibration_error",
    "reliability_curve", "bootstrap_ci", "bootstrap_diff",
    "JudgeSpec", "make_benchmark", "two_model_scores",
    "audit", "render_markdown", "save_plots",
    "read_jsonl", "read_csv", "assemble", "pairwise",
]
