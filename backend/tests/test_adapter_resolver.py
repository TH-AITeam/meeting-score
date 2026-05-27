"""adapter_resolver と LocalEvaluator のアダプタ・フォールバックのテスト (Issue #83)。"""

from __future__ import annotations

import yaml

from app.context_builder.builder import EvaluationContext
from app.evaluators.adapter_resolver import (
    KIND_ADAPTER,
    KIND_DEFAULT,
    KIND_WEIGHTS_PROFILE,
    AdapterResolver,
)
from app.evaluators.local_evaluator import LocalEvaluator
from app.schemas.models import Utterance

BASE_MODEL = "Qwen/Qwen3.5-9B"


def _registry(tmp_path, orgs: dict) -> str:
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({"organizations": orgs}), encoding="utf-8")
    return str(path)


def _resolver(tmp_path, orgs: dict) -> AdapterResolver:
    return AdapterResolver(
        base_model=BASE_MODEL,
        registry_path=_registry(tmp_path, orgs),
        adapters_root=tmp_path / "adapters",
        weights_profile_dir=tmp_path / "weights_profile",
    )


def _make_adapter(tmp_path, org_id: str, version: str) -> None:
    d = tmp_path / "adapters" / org_id / version
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")


# ---------- resolver: 3 ケース ----------


def test_resolve_adapter_present(tmp_path):
    _make_adapter(tmp_path, "org_001", "v3")
    r = _resolver(tmp_path, {"org_001": {"active_version": "v3"}})
    choice = r.resolve("org_001")
    assert choice.kind == KIND_ADAPTER
    assert choice.model == "org_001"  # vLLM の lora-modules 名
    assert choice.version == "v3"
    assert choice.adapter_path is not None


def test_resolve_weights_profile_when_no_adapter(tmp_path):
    (tmp_path / "weights_profile").mkdir(parents=True)
    (tmp_path / "weights_profile" / "org_002.yaml").write_text("weights: {}", encoding="utf-8")
    r = _resolver(tmp_path, {})
    choice = r.resolve("org_002")
    assert choice.kind == KIND_WEIGHTS_PROFILE
    assert choice.model == BASE_MODEL
    assert choice.weights_profile_path is not None


def test_resolve_default_when_unregistered(tmp_path):
    r = _resolver(tmp_path, {})
    choice = r.resolve("unknown_org")
    assert choice.kind == KIND_DEFAULT
    assert choice.model == BASE_MODEL


def test_registered_but_missing_path_falls_back(tmp_path):
    # registry にあるが adapters/org_003/v1 が無い → adapter にならない
    r = _resolver(tmp_path, {"org_003": {"active_version": "v1"}})
    choice = r.resolve("org_003")
    assert choice.kind == KIND_DEFAULT


# ---------- 組織分離（完了条件: A のアダプタが B に使われない） ----------


def test_org_isolation(tmp_path):
    _make_adapter(tmp_path, "org_a", "v1")
    r = _resolver(tmp_path, {"org_a": {"active_version": "v1"}})
    a = r.resolve("org_a")
    b = r.resolve("org_b")
    assert a.kind == KIND_ADAPTER and a.model == "org_a"
    # org_b は org_a のアダプタを受け取らない
    assert b.kind == KIND_DEFAULT and b.model == BASE_MODEL


# ---------- ホットリロード ----------


def test_hot_reload_on_registry_change(tmp_path):
    import os
    import time

    _make_adapter(tmp_path, "org_x", "v1")
    reg = _registry(tmp_path, {})
    r = AdapterResolver(
        base_model=BASE_MODEL,
        registry_path=reg,
        adapters_root=tmp_path / "adapters",
        weights_profile_dir=tmp_path / "weights_profile",
    )
    assert r.resolve("org_x").kind == KIND_DEFAULT  # まだ未登録

    # registry を更新（mtime を確実に進める）
    from pathlib import Path

    Path(reg).write_text(
        yaml.safe_dump({"organizations": {"org_x": {"active_version": "v1"}}}), encoding="utf-8"
    )
    os.utime(reg, (time.time() + 1, time.time() + 1))

    assert r.resolve("org_x").kind == KIND_ADAPTER  # 再ロードされて反映


# ---------- path traversal 安全化 ----------


def test_path_traversal_rejected(tmp_path):
    r = _resolver(tmp_path, {"../../etc": {"active_version": "v1"}})
    choice = r.resolve("../../etc")
    assert choice.kind == KIND_DEFAULT  # 不正パスは adapter にしない


def test_weights_profile_path_traversal_rejected(tmp_path):
    """org_id に ../ を含めても weights_profile_dir 外の YAML は読まない (R0Q)。"""
    # weights_profile_dir の外（兄弟ディレクトリ）に既存の YAML を置く
    (tmp_path / "weights_profile").mkdir(parents=True)
    (tmp_path / "outside.yaml").write_text("weights: {}", encoding="utf-8")
    r = _resolver(tmp_path, {})
    # ../outside で dir 外の outside.yaml を狙う
    choice = r.resolve("../outside")
    assert choice.kind == KIND_DEFAULT
    assert choice.weights_profile_path is None


# ---------- lora_modules 生成 ----------


def test_lora_modules_lists_active_adapters(tmp_path):
    _make_adapter(tmp_path, "org_a", "v2")
    _make_adapter(tmp_path, "org_b", "v1")
    r = _resolver(
        tmp_path,
        {"org_a": {"active_version": "v2"}, "org_b": {"active_version": "v1"}, "org_c": {}},
    )
    mods = r.lora_modules()
    assert any(m.startswith("org_a=") and m.endswith("org_a/v2") for m in mods)
    assert any(m.startswith("org_b=") for m in mods)
    assert not any(m.startswith("org_c=") for m in mods)  # active_version 無しは除外


# ---------- LocalEvaluator: アダプタ失敗 → ベースモデルへフォールバック ----------

_VALID_JSON = (
    '{"speech_type": "情報共有", '
    '"scores": {"issue_clarification": 0, "decision_progress": 0, '
    '"risk_detection": 0, "actionability": 0, "groundedness": 0, '
    '"novelty": 0, "summarization": 0}, '
    '"penalties": {"duplication": 0, "verbosity": 0, "off_topic": 0, '
    '"unsupported_assertion": 0}, "reason": "テスト"}'
)


class _FakeClient:
    """指定モデルでは例外、それ以外は valid JSON を返す chat client。"""

    def __init__(self, fail_models: set[str]) -> None:
        self.fail_models = set(fail_models)
        self.calls: list[str] = []
        self.chat = self
        self.completions = self

    def create(self, *, model: str, **_kw):
        self.calls.append(model)
        if model in self.fail_models:
            raise RuntimeError(f"adapter {model} not loaded")
        return {"choices": [{"message": {"content": _VALID_JSON}}]}


def _ctx() -> EvaluationContext:
    return EvaluationContext(
        meeting_goal="g",
        agenda=["a"],
        decision_points=[],
        current_topic="a",
        before_utterances=[],
        target_utterance=Utterance(
            utterance_id="u1", speaker="話者", timestamp="00:01", text="発言"
        ),
        after_utterances=[],
    )


def test_evaluate_uses_primary_model():
    client = _FakeClient(fail_models=set())
    ev = LocalEvaluator(model="org_001", endpoint="http://x/v1", client=client, max_retries=1)
    result = ev.evaluate(_ctx())
    assert not result.evaluation_failed
    assert client.calls == ["org_001"]


def test_evaluate_falls_back_to_base_on_adapter_failure():
    client = _FakeClient(fail_models={"org_001"})
    ev = LocalEvaluator(
        model="org_001",
        endpoint="http://x/v1",
        client=client,
        fallback_model=BASE_MODEL,
        max_retries=1,
    )
    result = ev.evaluate(_ctx())
    assert not result.evaluation_failed  # ベースモデルで応答が返る
    assert client.calls == ["org_001", BASE_MODEL]


def test_evaluate_fails_without_fallback():
    client = _FakeClient(fail_models={"org_001"})
    ev = LocalEvaluator(model="org_001", endpoint="http://x/v1", client=client, max_retries=1)
    result = ev.evaluate(_ctx())
    assert result.evaluation_failed
    assert client.calls == ["org_001"]  # フォールバック先が無い


# ---------- ensure_lora_loaded: 選択前に vLLM へロード (Issue #83 P1) ----------


def test_ensure_lora_loaded_success_and_cache():
    from app.evaluators.lora_loader import ensure_lora_loaded, reset_loaded_cache

    reset_loaded_cache()
    calls: list[tuple[str, dict]] = []

    def post(url, payload, timeout):
        calls.append((url, payload))
        return 200, "ok"

    assert ensure_lora_loaded("http://x/v1", "org_a", "/p/org_a/v1", post=post) is True
    assert calls[0][0] == "http://x/v1/load_lora_adapter"
    assert calls[0][1] == {"lora_name": "org_a", "lora_path": "/p/org_a/v1"}
    # 2 回目はキャッシュされ POST されない
    assert ensure_lora_loaded("http://x/v1", "org_a", "/p/org_a/v1", post=post) is True
    assert len(calls) == 1


def test_ensure_lora_loaded_already_loaded_is_success():
    from app.evaluators.lora_loader import ensure_lora_loaded, reset_loaded_cache

    reset_loaded_cache()
    assert (
        ensure_lora_loaded("http://x/v1", "o", "/p", post=lambda *_a: (400, "LoRA already loaded"))
        is True
    )


def test_ensure_lora_loaded_failure_and_exception():
    from app.evaluators.lora_loader import ensure_lora_loaded, reset_loaded_cache

    reset_loaded_cache()
    assert ensure_lora_loaded("http://x/v1", "o", "/p", post=lambda *_a: (500, "err")) is False

    def boom(*_a):
        raise RuntimeError("connection refused")

    assert ensure_lora_loaded("http://x/v1", "o", "/p", post=boom) is False


def test_ensure_lora_loaded_no_path():
    from app.evaluators.lora_loader import ensure_lora_loaded

    assert ensure_lora_loaded("http://x/v1", "o", None) is False


# ---------- /api/analyze ルーティング (Issue #83 P1/P2 のレビュー反映) ----------


class _StubResolver:
    def __init__(self, choice):
        self._choice = choice

    def resolve(self, org_id):
        return self._choice


class _StubEvaluator:
    def evaluate(self, ctx):
        from app.evaluators.base import EvaluationResult
        from app.schemas.models import Penalties, Scores

        return EvaluationResult(
            speech_type="情報共有",
            scores=Scores(decision_progress=2),
            penalties=Penalties(),
            reason="ok",
            evaluation_failed=False,
        )


def _analyze_payload(org_id: str) -> dict:
    return {
        "meeting_id": "m1",
        "title": "t",
        "goal": "g",
        "org_id": org_id,
        "utterances": [
            {"utterance_id": "u1", "speaker": "A", "timestamp": "00:01", "text": "発言1"},
            {"utterance_id": "u2", "speaker": "B", "timestamp": "00:02", "text": "発言2"},
        ],
    }


def test_route_adapter_used_only_after_successful_load(monkeypatch):
    """P1: アダプタはロード成功時のみ採用。失敗時はベース (model_override=None)。"""
    from fastapi.testclient import TestClient

    from app.api import routes
    from app.api.main import app
    from app.evaluators.adapter_resolver import AdapterChoice

    choice = AdapterChoice(
        org_id="org_a", kind="adapter", model="org_a", version="v1", adapter_path="/p/org_a/v1"
    )
    monkeypatch.setattr(routes, "_get_adapter_resolver", lambda config: _StubResolver(choice))

    captured: dict = {}

    def fake_create(config, *, model_override=None, fallback_model=None):
        captured["model_override"] = model_override
        return _StubEvaluator()

    monkeypatch.setattr(routes, "create_evaluator", fake_create)

    # ロード失敗 → アダプタ不採用 (ベース)
    monkeypatch.setattr(routes, "ensure_lora_loaded", lambda *a, **k: False)
    with TestClient(app) as c:
        assert c.post("/api/analyze", json=_analyze_payload("org_a")).status_code == 200
    assert captured["model_override"] is None

    # ロード成功 → アダプタ採用
    monkeypatch.setattr(routes, "ensure_lora_loaded", lambda *a, **k: True)
    with TestClient(app) as c:
        assert c.post("/api/analyze", json=_analyze_payload("org_a")).status_code == 200
    assert captured["model_override"] == "org_a"


def test_route_weights_profile_applied_on_openai_backend(monkeypatch):
    """P2: 重みプロファイルはバックエンド非依存。openai でも適用される。"""
    import dataclasses

    from fastapi.testclient import TestClient

    from app.api import routes
    from app.api.main import app
    from app.evaluators.adapter_resolver import AdapterChoice
    from app.scoring.weights import ScoringWeights

    choice = AdapterChoice(
        org_id="org_b", kind="weights_profile", model="base", weights_profile_path="/p/org_b.yaml"
    )
    monkeypatch.setattr(routes, "_get_adapter_resolver", lambda config: _StubResolver(choice))
    monkeypatch.setattr(routes, "create_evaluator", lambda config, **k: _StubEvaluator())

    called: dict = {}

    def fake_load_profile(path):
        called["path"] = path
        return ScoringWeights()

    monkeypatch.setattr(routes, "_load_weights_profile", fake_load_profile)

    with TestClient(app) as c:
        # backend を openai に差し替えても重みプロファイルが適用されること
        c.app.state.config = dataclasses.replace(c.app.state.config, llm_backend="openai")
        assert c.post("/api/analyze", json=_analyze_payload("org_b")).status_code == 200
    assert called.get("path") == "/p/org_b.yaml"
