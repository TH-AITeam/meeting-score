"""scripts/regress_weights.py のテスト (Issue #16)。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.regress_weights import (  # noqa: E402
    SCORE_KEYS,
    build_arg_parser,
    default_weight_vector,
    fit_pairwise_logistic,
    load_pairs_with_scores,
    normalize_weights,
    pairwise_accuracy,
    run,
)

D = len(SCORE_KEYS)


# ---------- 回帰アルゴリズム ----------


def test_fit_recovers_discriminating_axis():
    """1 軸だけが勝敗を決めるデータ → その軸の重みが最大、他は ~0、全て ≥ 0。"""
    axis = 1  # decision_progress
    diffs = np.zeros((50, D))
    diffs[:, axis] = 2.0  # 勝者は常にこの軸で +2、他は同点
    w = fit_pairwise_logistic(diffs, l2=0.1, lr=0.2, iters=2000, nonneg=True)
    assert np.all(w >= 0)  # 加点軸 ≥ 0 制約
    assert int(np.argmax(w)) == axis
    others = np.delete(w, axis)
    assert np.all(others < 1e-3)


def test_fit_nonneg_clamps_negative_direction():
    """勝者がある軸で常に低い → その軸の重みは 0 に射影される（負にならない）。"""
    diffs = np.zeros((40, D))
    diffs[:, 0] = 1.0  # 軸0 は勝者が高い
    diffs[:, 2] = -1.0  # 軸2 は勝者が低い（負方向）
    w = fit_pairwise_logistic(diffs, l2=0.1, lr=0.2, iters=2000, nonneg=True)
    assert w[2] == 0.0
    assert w[0] > 0.0


def test_pairwise_accuracy():
    w = np.zeros(D)
    w[1] = 1.0
    good = np.zeros((10, D))
    good[:, 1] = 1.0  # 勝者が軸1で高い → 全て正解
    assert pairwise_accuracy(good, w) == 1.0
    bad = np.zeros((10, D))
    bad[:, 1] = -1.0  # 勝者が軸1で低い → 全て不正解
    assert pairwise_accuracy(bad, w) == 0.0


def test_normalize_weights_mean():
    w = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    nw = normalize_weights(w, target_mean=1.0)
    assert pytest.approx(float(np.mean(nw))) == 1.0


def test_default_weight_vector_matches_config_axes():
    d = default_weight_vector()
    assert len(d) == D
    assert d[SCORE_KEYS.index("decision_progress")] == 1.5  # config 既定


# ---------- 入力パース ----------


def _scores(**kw) -> dict:
    base = {k: 0 for k in SCORE_KEYS}
    base.update(kw)
    return base


def test_load_pairs_orients_winner(tmp_path):
    path = tmp_path / "pairs.jsonl"
    rows = [
        # A_better → (winner=scores_a, loser=scores_b)
        {
            "winner": "A_better",
            "scores_a": _scores(decision_progress=3),
            "scores_b": _scores(decision_progress=0),
        },
        # B_better → winner=scores_b
        {"winner": "B_better", "scores_a": _scores(novelty=0), "scores_b": _scores(novelty=3)},
        # tie → 除外
        {"winner": "tie", "scores_a": _scores(), "scores_b": _scores()},
        # scores 欠落 → 除外
        {"winner": "A_better"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    pairs = load_pairs_with_scores(path)
    assert len(pairs) == 2
    win0, lose0 = pairs[0]
    assert win0[SCORE_KEYS.index("decision_progress")] == 3
    assert lose0[SCORE_KEYS.index("decision_progress")] == 0
    win1, _ = pairs[1]
    assert win1[SCORE_KEYS.index("novelty")] == 3  # B 側が winner


# ---------- エンドツーエンド ----------


def test_run_pairs_file_end_to_end(tmp_path):
    # 軸1(decision_progress)が勝敗を決めるペアを作る → 回帰 acc が固定 acc 以上
    pairs_path = tmp_path / "pairs.jsonl"
    rows = []
    for _ in range(40):
        rows.append(
            {
                "winner": "A_better",
                "scores_a": _scores(decision_progress=3, summarization=0),
                "scores_b": _scores(decision_progress=0, summarization=3),
            }
        )
    pairs_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    out_yaml = tmp_path / "w.yaml"
    report = tmp_path / "report.md"
    args = build_arg_parser().parse_args(
        [
            "--pairs",
            str(pairs_path),
            "--out-weights",
            str(out_yaml),
            "--report",
            str(report),
            "--val-ratio",
            "0.25",
        ]
    )
    summary = run(args)
    assert summary["n_pairs"] == 40
    assert out_yaml.exists() and report.exists()
    # decision_progress を上げ summarization を下げる方向に学習しているはず
    assert summary["weights"]["decision_progress"] > summary["weights"]["summarization"]
    # 回帰 acc は固定 acc 以上（この分離データでは固定でも 1.0 になり得るので >=）
    assert summary["acc_regressed"] >= summary["acc_fixed"]
    # yaml に weights セクションがある
    assert "weights:" in out_yaml.read_text(encoding="utf-8")


def test_run_no_input_returns_empty(tmp_path):
    args = build_arg_parser().parse_args(
        [
            "--out-weights",
            str(tmp_path / "w.yaml"),
            "--report",
            str(tmp_path / "r.md"),
        ]
    )
    assert run(args) == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
