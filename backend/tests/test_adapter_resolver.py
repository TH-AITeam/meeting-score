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
