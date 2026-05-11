"""eval ハーネス (Issue #5)。

メトリクス / 安定性 / runner を提供する。
"""

from backend.evals.metrics import (
    PairwiseAccuracyReport,
    kendall_tau,
    pairwise_accuracy,
    spearman,
    top_k_jaccard,
)
from backend.evals.protocol import EvaluationResult, Evaluator
from backend.evals.runner import EvalReport, MeetingMetrics, run_eval
from backend.evals.schema import (
    ANNOTATION_TAGS,
    PairwiseAnnotation,
    TagAnnotation,
    TopBottomAnnotation,
    load_pairwise_annotations,
    load_tag_annotations,
    load_top_bottom_annotations,
)
from backend.evals.stability import (
    AXES,
    MeetingStability,
    UtteranceStability,
    evaluate_stability,
)

__all__ = [
    "ANNOTATION_TAGS",
    "AXES",
    "EvalReport",
    "EvaluationResult",
    "Evaluator",
    "MeetingMetrics",
    "MeetingStability",
    "PairwiseAccuracyReport",
    "PairwiseAnnotation",
    "TagAnnotation",
    "TopBottomAnnotation",
    "UtteranceStability",
    "evaluate_stability",
    "kendall_tau",
    "load_pairwise_annotations",
    "load_tag_annotations",
    "load_top_bottom_annotations",
    "pairwise_accuracy",
    "run_eval",
    "spearman",
    "top_k_jaccard",
]
