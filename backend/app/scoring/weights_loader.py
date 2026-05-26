"""組織別重みプロファイルの読み込み。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from app.scoring.weights import AppConfig, PenaltyWeights, ScoringWeights, _parse_scoring_weights

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROFILE_DIR = _REPO_ROOT / "config" / "weights_profile"


def sanitize_org_id(org_id: str) -> str:
    """ファイルパス用に組織 ID を安全な 1 セグメントへ正規化する。"""
    safe_org_id = org_id.replace("/", "_").replace("\\", "_")
    if safe_org_id in {"", ".", ".."}:
        return "_"
    return safe_org_id


def profile_path(org_id: str, profile_dir: Path | None = None) -> Path:
    """組織 ID からプロファイルパスを返す。"""
    return (profile_dir or _PROFILE_DIR) / f"{sanitize_org_id(org_id)}.yaml"


def load_org_profile(
    org_id: str,
    profile_dir: Path | None = None,
    default_penalty_weights: PenaltyWeights | None = None,
) -> tuple[ScoringWeights, PenaltyWeights] | None:
    """組織別プロファイルを読み込む。存在しなければ None。"""
    path = profile_path(org_id, profile_dir)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    weights = _parse_scoring_weights(raw.get("weights", {}))
    d = default_penalty_weights or PenaltyWeights()
    p = raw.get("penalties", {})
    penalties = PenaltyWeights(
        duplication=p.get("duplication", d.duplication),
        verbosity=p.get("verbosity", d.verbosity),
        off_topic=p.get("off_topic", d.off_topic),
        unsupported_assertion=p.get("unsupported_assertion", d.unsupported_assertion),
        override=p.get("override", d.override),
    )
    return weights, penalties


def load_weights(
    config: AppConfig,
    org_id: str | None = None,
    meeting_type: str | None = None,
    profile_dir: Path | None = None,
) -> ScoringWeights:
    """推論時に使う軸重みを選ぶ。

    優先順位:
    1. org_id があり、組織別プロファイルが存在する場合はそれを使用
    2. 会議タイプ別重み
    3. config.yaml のデフォルト重み
    """
    if org_id:
        profile = load_org_profile(org_id, profile_dir)
        if profile is not None:
            return profile[0]
    if meeting_type and meeting_type in config.meeting_type_weights:
        return config.meeting_type_weights[meeting_type]
    return config.weights


def load_penalty_weights(
    config: AppConfig,
    org_id: str | None = None,
    profile_dir: Path | None = None,
) -> PenaltyWeights:
    """推論時に使う減点重みを選ぶ。"""
    if org_id:
        profile = load_org_profile(
            org_id, profile_dir, default_penalty_weights=config.penalty_weights
        )
        if profile is not None:
            return profile[1]
    return config.penalty_weights


def weights_to_dict(weights: ScoringWeights) -> dict[str, float]:
    return {k: float(v) for k, v in asdict(weights).items()}


def penalties_to_dict(penalties: PenaltyWeights) -> dict[str, float]:
    return {k: float(v) for k, v in asdict(penalties).items()}
