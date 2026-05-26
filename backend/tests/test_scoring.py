"""スコアリングモジュールのテスト"""

import os
import textwrap
from pathlib import Path

from app.schemas.models import Penalties, Scores
from app.scoring.calculator import calculate_total_score
from app.scoring.weights import DEFAULT_OPENAI_LLM_MODEL, PenaltyWeights, load_config


def test_calculate_total_score_basic():
    """基本的なスコア計算"""
    scores = Scores(
        issue_clarification=3,
        decision_progress=2,
        risk_detection=0,
        actionability=0,
        groundedness=1,
        novelty=2,
        summarization=1,
    )
    penalties = Penalties(
        duplication=0,
        verbosity=0,
        off_topic=0,
        unsupported_assertion=0,
    )
    total = calculate_total_score(scores, penalties)
    # 3*1.3 + 2*1.5 + 0 + 0 + 1*0.8 + 2*0.9 + 1*0.8 = 3.9+3.0+0.8+1.8+0.8 = 10.3
    assert total == 10.3


def test_calculate_total_score_with_penalties():
    """減点ありのスコア計算"""
    scores = Scores(
        issue_clarification=0,
        decision_progress=0,
        risk_detection=0,
        actionability=0,
        groundedness=0,
        novelty=0,
        summarization=0,
    )
    penalties = Penalties(
        duplication=-2,
        verbosity=-1,
        off_topic=-3,
        unsupported_assertion=0,
    )
    total = calculate_total_score(scores, penalties)
    assert total == -6.0


def test_calculate_total_score_mixed():
    """加点と減点が混在するケース"""
    scores = Scores(
        issue_clarification=2,
        decision_progress=1,
        risk_detection=1,
        actionability=0,
        groundedness=0,
        novelty=1,
        summarization=0,
    )
    penalties = Penalties(
        duplication=-1,
        verbosity=-1,
        off_topic=0,
        unsupported_assertion=0,
    )
    total = calculate_total_score(scores, penalties)
    # 2*1.3 + 1*1.5 + 1*1.2 + 0 + 0 + 1*0.9 + 0 + (-1) + (-1) = 2.6+1.5+1.2+0.9-2 = 4.2
    assert total == 4.2


def test_all_zeros():
    """全ゼロの場合"""
    total = calculate_total_score(Scores(), Penalties())
    assert total == 0.0


def test_max_scores():
    """全て最高点の場合"""
    scores = Scores(
        issue_clarification=3,
        decision_progress=3,
        risk_detection=3,
        actionability=3,
        groundedness=3,
        novelty=3,
        summarization=3,
    )
    total = calculate_total_score(scores, Penalties())
    # 3*(1.3+1.5+1.2+1.3+0.8+0.9+0.8) = 3*7.8 = 23.4
    assert total == 23.4


# ---------------------------------------------------------------------------
# Issue #3: penalty 重み
# ---------------------------------------------------------------------------


def test_penalty_weights_default_keeps_legacy_behavior():
    """penalty_weights を渡さない場合は従来挙動（係数 1.0）と一致する"""
    scores = Scores()
    penalties = Penalties(duplication=-2, verbosity=-1)
    legacy = calculate_total_score(scores, penalties)
    explicit = calculate_total_score(scores, penalties, penalty_weights=PenaltyWeights())
    assert legacy == explicit == -3.0


def test_penalty_weight_doubles_duplication():
    """duplication 重みを 2.0 にすると、重複減点が従来の倍になる"""
    scores = Scores()
    penalties = Penalties(duplication=-2)

    base = calculate_total_score(scores, penalties)
    doubled = calculate_total_score(
        scores,
        penalties,
        penalty_weights=PenaltyWeights(duplication=2.0),
    )
    assert base == -2.0
    assert doubled == -4.0  # -2 × 2.0


def test_penalty_weights_apply_per_axis():
    """各軸の重みが独立して掛かる"""
    scores = Scores()
    penalties = Penalties(
        duplication=-1, verbosity=-2, off_topic=-1, unsupported_assertion=-3, override=-1
    )
    pw = PenaltyWeights(
        duplication=2.0, verbosity=0.5, off_topic=1.0, unsupported_assertion=1.5, override=2.0
    )
    total = calculate_total_score(scores, penalties, penalty_weights=pw)
    # -1*2.0 + -2*0.5 + -1*1.0 + -3*1.5 + -1*2.0 = -10.5
    assert total == -10.5


def test_penalty_weights_with_scores_mixed():
    """加点・減点・重みが全て混在しても破綻しない"""
    scores = Scores(issue_clarification=2, decision_progress=1)  # 2*1.3 + 1*1.5 = 4.1
    penalties = Penalties(duplication=-1, verbosity=-1)
    pw = PenaltyWeights(duplication=3.0, verbosity=1.0)
    total = calculate_total_score(scores, penalties, penalty_weights=pw)
    # 4.1 + (-1*3.0) + (-1*1.0) = 0.1
    assert total == 0.1


def test_load_config_reads_penalties_section(tmp_path: Path):
    """config.yaml の penalties: セクションが PenaltyWeights に反映される"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """\
            penalties:
              duplication: 2.0
              verbosity: 0.5
              off_topic: 1.0
              unsupported_assertion: 1.5
              override: 2.0
            """
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.penalty_weights.duplication == 2.0
    assert cfg.penalty_weights.verbosity == 0.5
    assert cfg.penalty_weights.off_topic == 1.0
    assert cfg.penalty_weights.unsupported_assertion == 1.5
    assert cfg.penalty_weights.override == 2.0


def test_load_config_defaults_when_penalties_missing(tmp_path: Path):
    """penalties: 節が無い場合は全て 1.0 で既定値が入る"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("weights:\n  issue_clarification: 2.0\n", encoding="utf-8")

    cfg = load_config(cfg_path)
    assert cfg.penalty_weights.duplication == 1.0
    assert cfg.penalty_weights.verbosity == 1.0
    assert cfg.penalty_weights.off_topic == 1.0
    assert cfg.penalty_weights.unsupported_assertion == 1.0
    assert cfg.penalty_weights.override == 0.5


def test_load_config_env_overrides_llm_section(tmp_path: Path, monkeypatch):
    """LLM_* 環境変数は config.yaml の llm セクションより優先される"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """\
            llm:
              backend: "local"
              endpoint: "http://yaml.example/v1"
              api_key: "yaml-key"
              model: "qwen3.6-35b-nvfp4"
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_BACKEND", "local")
    monkeypatch.setenv("LLM_ENDPOINT", "http://env.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    monkeypatch.setenv("LLM_MODEL", "env-model")

    cfg = load_config(cfg_path)

    assert cfg.llm_backend == "local"
    assert cfg.llm_endpoint == "http://env.example/v1"
    assert cfg.llm_api_key == "env-key"
    assert cfg.llm_model == "env-model"


def test_load_config_openai_env_replaces_local_model(tmp_path: Path, monkeypatch):
    """LLM_BACKEND=openai で local 用 model だけが残っても OpenAI 既定値に倒す"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """\
            llm:
              backend: "local"
              model: "qwen3.6-35b-nvfp4"
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_BACKEND", "openai")

    cfg = load_config(cfg_path)

    assert cfg.llm_backend == "openai"
    assert cfg.llm_model == DEFAULT_OPENAI_LLM_MODEL


def test_load_config_reads_dotenv_next_to_config(tmp_path: Path):
    """config.yaml と同じディレクトリの .env から LLM_* を読み込む"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """\
            llm:
              backend: "local"
              endpoint: "http://yaml.example/v1"
              model: "yaml-model"
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        textwrap.dedent(
            """\
            LLM_ENDPOINT=http://dotenv.example/v1
            LLM_MODEL=dotenv-model
            """
        ),
        encoding="utf-8",
    )
    old_endpoint = os.environ.pop("LLM_ENDPOINT", None)
    old_model = os.environ.pop("LLM_MODEL", None)

    try:
        cfg = load_config(cfg_path)
        assert cfg.llm_endpoint == "http://dotenv.example/v1"
        assert cfg.llm_model == "dotenv-model"
    finally:
        os.environ.pop("LLM_ENDPOINT", None)
        os.environ.pop("LLM_MODEL", None)
        if old_endpoint is not None:
            os.environ["LLM_ENDPOINT"] = old_endpoint
        if old_model is not None:
            os.environ["LLM_MODEL"] = old_model


def test_completion_criterion_duplication_doubled(tmp_path: Path):
    """Issue #3 完了条件: penalties.duplication: 2.0 で重複検出時に倍の減点になる"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("penalties:\n  duplication: 2.0\n", encoding="utf-8")
    cfg = load_config(cfg_path)

    scores = Scores()
    penalties = Penalties(duplication=-1)
    legacy = calculate_total_score(scores, penalties)
    weighted = calculate_total_score(scores, penalties, penalty_weights=cfg.penalty_weights)
    assert legacy == -1.0
    assert weighted == -2.0  # 従来比2倍
