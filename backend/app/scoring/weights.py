"""スコアリング重み設定の読み込み"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"
DEFAULT_LOCAL_LLM_MODEL = "qwen3.6-35b-nvfp4"
DEFAULT_OPENAI_LLM_MODEL = "gpt-4o-mini"
_LOCAL_ONLY_LLM_MODEL_PREFIXES = (
    "deepseek",
    "gemma",
    "llama",
    "meta-llama/",
    "mistral",
    "mixtral",
    "phi",
    "qwen",
    "unsloth/",
)


def is_local_only_llm_model(model: str | None) -> bool:
    """OpenAI Responses API へ直接渡せないローカル向けモデル名かを判定する。"""
    name = (model or "").strip().lower()
    return bool(name) and name.startswith(_LOCAL_ONLY_LLM_MODEL_PREFIXES)


def resolve_llm_model_for_backend(backend: str | None, model: str | None) -> str:
    """backend と model の組み合わせを実行可能な既定値へ正規化する。"""
    backend_name = (backend or "local").strip().lower()
    model_name = (model or "").strip()
    if backend_name == "openai" and (not model_name or is_local_only_llm_model(model_name)):
        return DEFAULT_OPENAI_LLM_MODEL
    return model_name or DEFAULT_LOCAL_LLM_MODEL


def _load_dotenv_near_config(path: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(path.parent / ".env", override=False)


def _env_value(name: str, default: str | None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


@dataclass
class ScoringWeights:
    """評価軸の重み"""

    # 主評価軸
    issue_clarification: float = 1.3
    decision_progress: float = 1.5
    risk_detection: float = 1.2
    actionability: float = 1.3
    # 補助評価軸
    groundedness: float = 0.8
    novelty: float = 0.9
    summarization: float = 0.8


@dataclass
class PenaltyWeights:
    """減点軸の重み (Issue #3)。

    penalty 値そのものは 0〜-3 の負値。ここの重みは正の倍率として掛ける
    （weight=1.0 で従来挙動、weight=2.0 で減点が倍になる）。
    """

    duplication: float = 1.0
    verbosity: float = 1.0
    off_topic: float = 1.0
    unsupported_assertion: float = 1.0


@dataclass
class AppConfig:
    """アプリケーション全体設定"""

    weights: ScoringWeights = field(default_factory=ScoringWeights)
    penalty_weights: PenaltyWeights = field(default_factory=PenaltyWeights)
    meeting_type_weights: dict[str, ScoringWeights] = field(default_factory=dict)
    context_before: int = 3
    context_after: int = 3
    # LLM 推論バックエンド (Issue #12 + Issue #17)
    # 既定はローカル推論 (vLLM 等の OpenAI 互換サーバ)。
    # OpenAI クラウド (Responses API) は蒸留・ベンチマーク用途のみ。
    llm_backend: str = "local"  # "local" | "openai"
    llm_endpoint: str | None = None  # backend=local 時に必須 (例: http://localhost:8001/v1)
    llm_api_key: str | None = None  # OpenAI 互換サーバが要求する場合
    llm_model: str = DEFAULT_LOCAL_LLM_MODEL
    llm_max_tokens: int = 1024
    llm_max_retries: int = 3
    llm_timeout: float = 30.0
    top_utterances_count: int = 5
    top_per_axis_count: int = 3


def _parse_scoring_weights(w: dict, default: ScoringWeights | None = None) -> ScoringWeights:
    d = default or ScoringWeights()
    return ScoringWeights(
        issue_clarification=w.get("issue_clarification", d.issue_clarification),
        decision_progress=w.get("decision_progress", d.decision_progress),
        risk_detection=w.get("risk_detection", d.risk_detection),
        actionability=w.get("actionability", d.actionability),
        groundedness=w.get("groundedness", d.groundedness),
        novelty=w.get("novelty", d.novelty),
        summarization=w.get("summarization", d.summarization),
    )


def get_weights_for_type(config: AppConfig, meeting_type: str | None) -> ScoringWeights:
    """会議タイプに対応する重みを返す。未指定・未定義の場合はデフォルト重みを返す。"""
    if meeting_type and meeting_type in config.meeting_type_weights:
        return config.meeting_type_weights[meeting_type]
    return config.weights


def max_total_score(weights: ScoringWeights) -> float:
    """与えられた重みセットにおける最大総合スコアを返す（全軸 3 点満点時）。"""
    return (
        weights.issue_clarification
        + weights.decision_progress
        + weights.risk_detection
        + weights.actionability
        + weights.groundedness
        + weights.novelty
        + weights.summarization
    ) * 3


def load_config(path: str | Path | None = None) -> AppConfig:
    """config.yaml から設定を読み込む"""
    path = Path(path) if path else _CONFIG_PATH
    _load_dotenv_near_config(path)

    if not path.exists():
        llm_backend = _env_value("LLM_BACKEND", "local") or "local"
        llm_model = _env_value("LLM_MODEL", None)
        return AppConfig(
            llm_backend=llm_backend,
            llm_endpoint=_env_value("LLM_ENDPOINT", None),
            llm_api_key=_env_value("LLM_API_KEY", None),
            llm_model=resolve_llm_model_for_backend(llm_backend, llm_model),
        )

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    weights = _parse_scoring_weights(raw.get("weights", {}))

    p = raw.get("penalties", {})
    penalty_weights = PenaltyWeights(
        duplication=p.get("duplication", 1.0),
        verbosity=p.get("verbosity", 1.0),
        off_topic=p.get("off_topic", 1.0),
        unsupported_assertion=p.get("unsupported_assertion", 1.0),
    )

    mtw_raw = raw.get("meeting_type_weights", {})
    meeting_type_weights = {mt: _parse_scoring_weights(w, weights) for mt, w in mtw_raw.items()}

    ctx = raw.get("context", {})
    llm = raw.get("llm", {})
    agg = raw.get("aggregation", {})
    llm_backend = _env_value("LLM_BACKEND", llm.get("backend", "local")) or "local"
    llm_endpoint = _env_value("LLM_ENDPOINT", llm.get("endpoint"))
    llm_api_key = _env_value("LLM_API_KEY", llm.get("api_key"))
    llm_model = _env_value("LLM_MODEL", llm.get("model"))

    return AppConfig(
        weights=weights,
        penalty_weights=penalty_weights,
        meeting_type_weights=meeting_type_weights,
        context_before=ctx.get("before_count", 3),
        context_after=ctx.get("after_count", 3),
        llm_backend=llm_backend,
        llm_endpoint=llm_endpoint,
        llm_api_key=llm_api_key,
        llm_model=resolve_llm_model_for_backend(llm_backend, llm_model),
        llm_max_tokens=llm.get("max_tokens", 1024),
        llm_max_retries=llm.get("max_retries", 3),
        llm_timeout=llm.get("timeout", 30.0),
        top_utterances_count=agg.get("top_utterances_count", 5),
        top_per_axis_count=agg.get("top_per_axis_count", 3),
    )
