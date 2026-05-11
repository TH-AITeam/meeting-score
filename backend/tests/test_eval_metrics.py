"""evals.metrics のテスト (Issue #5)。"""

from evals.metrics import (
    kendall_tau,
    pairwise_accuracy,
    spearman,
    top_k_jaccard,
)
from evals.schema import PairwiseAnnotation


def test_spearman_perfect_correlation():
    assert abs(spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) - 1.0) < 1e-9


def test_spearman_perfect_inverse():
    assert abs(spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) - (-1.0)) < 1e-9


def test_spearman_with_ties_uses_average_rank():
    rho = spearman([1, 2, 2, 3], [1, 5, 5, 9])
    assert abs(rho - 1.0) < 1e-9


def test_spearman_short_input_returns_zero():
    assert spearman([1], [2]) == 0.0


def test_kendall_tau_perfect():
    assert kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0


def test_kendall_tau_inverse():
    assert kendall_tau([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_kendall_tau_partial():
    # 1ペアが逆順
    tau = kendall_tau([1, 2, 3, 4], [10, 30, 20, 40])
    # ペア数=6, concordant=5, discordant=1, tau = (5-1)/6 ≈ 0.667
    assert abs(tau - 4 / 6) < 1e-9


def test_top_k_jaccard_full_match():
    assert top_k_jaccard(["a", "b", "c"], ["c", "b", "a"], k=3) == 1.0


def test_top_k_jaccard_partial():
    j = top_k_jaccard(["a", "b", "c", "d", "e"], ["a", "b", "x", "y", "z"], k=5)
    # 共通=2, 和=8, j=0.25
    assert j == 0.25


def test_top_k_jaccard_both_empty_returns_one():
    assert top_k_jaccard([], [], k=5) == 1.0


def test_pairwise_accuracy_all_correct():
    pairs = [
        PairwiseAnnotation(meeting_id="m", utt_a="u1", utt_b="u2", winner="A_better"),
        PairwiseAnnotation(meeting_id="m", utt_a="u3", utt_b="u4", winner="B_better"),
    ]
    scores = {"u1": 10.0, "u2": 5.0, "u3": 3.0, "u4": 9.0}
    rep = pairwise_accuracy(pairs, scores)
    assert rep.accuracy == 1.0
    assert rep.n == 2
    assert rep.n_skipped == 0


def test_pairwise_accuracy_tie_threshold():
    pairs = [
        PairwiseAnnotation(meeting_id="m", utt_a="u1", utt_b="u2", winner="tie"),
    ]
    scores = {"u1": 5.0, "u2": 5.4}  # 差 0.4 < 0.5 → tie
    rep = pairwise_accuracy(pairs, scores, tie_threshold=0.5)
    assert rep.accuracy == 1.0
    assert rep.by_class["tie"]["correct"] == 1


def test_pairwise_accuracy_skips_missing():
    pairs = [
        PairwiseAnnotation(meeting_id="m", utt_a="u1", utt_b="u9", winner="A_better"),
    ]
    rep = pairwise_accuracy(pairs, {"u1": 1.0})
    assert rep.n == 0
    assert rep.n_skipped == 1
    assert rep.accuracy == 0.0


def test_pairwise_accuracy_precision_recall():
    pairs = [
        PairwiseAnnotation(meeting_id="m", utt_a="u1", utt_b="u2", winner="A_better"),
        PairwiseAnnotation(meeting_id="m", utt_a="u3", utt_b="u4", winner="A_better"),
        PairwiseAnnotation(meeting_id="m", utt_a="u5", utt_b="u6", winner="B_better"),
    ]
    # システムは 2/3 正解（最後を A_better と誤判定）
    scores = {"u1": 9, "u2": 1, "u3": 8, "u4": 2, "u5": 7, "u6": 3}
    rep = pairwise_accuracy(pairs, scores)
    assert rep.n == 3
    assert abs(rep.accuracy - 2 / 3) < 1e-9
    a = rep.by_class["A_better"]
    assert a["correct"] == 2
    assert a["n_in_human"] == 2
    assert a["n_in_system"] == 3
    assert a["recall"] == 1.0
    assert abs(a["precision"] - 2 / 3) < 1e-9
