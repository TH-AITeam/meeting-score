"""meeting_builder のテスト (Issue #11)。"""

from __future__ import annotations

import sys
from typing import Any

from app.asr.base import Utterance as AsrUtterance
from app.asr.base import Word
from app.asr.meeting_builder import (
    OpenAIMetaExtractor,
    _format_timestamp,
    _parse_json_relaxed,
    build_meeting_input,
)


def _utt(uid: str, speaker: str, start: float, end: float, text: str) -> AsrUtterance:
    return AsrUtterance(
        utterance_id=uid,
        speaker=speaker,
        start_sec=start,
        end_sec=end,
        text=text,
        words=[Word(text=text, start_sec=start, end_sec=end)],
    )


class _FakeExtractor:
    def __init__(self, meta: dict[str, Any]) -> None:
        self.meta = meta
        self.calls: list[str] = []

    def extract(self, transcript: str) -> dict[str, Any]:
        self.calls.append(transcript)
        return self.meta


def test_build_meeting_input_without_extractor() -> None:
    """meta_extractor=None なら空メタで構築する。"""
    utts = [
        _utt("u001", "SPEAKER_00", 0.0, 5.0, "はじめましょう"),
        _utt("u002", "SPEAKER_01", 5.0, 10.0, "了解です"),
    ]
    mi = build_meeting_input(utts, meeting_id="m001")
    assert mi.meeting_id == "m001"
    assert mi.title == ""
    assert mi.goal == ""
    assert mi.agenda == []
    assert mi.decision_points == []
    assert len(mi.utterances) == 2
    assert mi.utterances[0].text == "はじめましょう"
    assert mi.utterances[0].timestamp == "00:00:00"
    assert mi.utterances[1].timestamp == "00:00:05"


def test_build_meeting_input_with_extractor_populates_meta() -> None:
    """meta_extractor が返す値が反映される。"""
    extractor = _FakeExtractor(
        {
            "title": "新機能企画",
            "goal": "初回リリース範囲を決める",
            "agenda": ["対象ユーザー", "機能範囲"],
            "decision_points": ["初回に含める機能"],
        }
    )
    utts = [_utt("u001", "S0", 0.0, 5.0, "では始めます")]
    mi = build_meeting_input(utts, meeting_id="m001", meta_extractor=extractor)
    assert mi.title == "新機能企画"
    assert mi.goal == "初回リリース範囲を決める"
    assert mi.agenda == ["対象ユーザー", "機能範囲"]
    assert mi.decision_points == ["初回に含める機能"]


def test_build_meeting_input_extractor_failure_uses_defaults() -> None:
    """extractor が例外を投げたら default 値で続行する。"""

    class _BrokenExtractor:
        def extract(self, transcript: str) -> dict[str, Any]:
            msg = "boom"
            raise RuntimeError(msg)

    utts = [_utt("u001", "S0", 0.0, 5.0, "test")]
    mi = build_meeting_input(
        utts,
        meeting_id="m001",
        meta_extractor=_BrokenExtractor(),
        default_title="fallback title",
        default_goal="fallback goal",
    )
    assert mi.title == "fallback title"
    assert mi.goal == "fallback goal"
    assert mi.agenda == []


def test_build_meeting_input_handles_none_in_meta() -> None:
    """LLM が None を返してきても default で補完。"""
    extractor = _FakeExtractor(
        {"title": None, "goal": None, "agenda": None, "decision_points": None}
    )
    utts = [_utt("u001", "S0", 0.0, 5.0, "test")]
    mi = build_meeting_input(
        utts,
        meeting_id="m001",
        meta_extractor=extractor,
        default_title="def",
    )
    assert mi.title == "def"
    assert mi.agenda == []
    assert mi.decision_points == []


def test_format_timestamp() -> None:
    assert _format_timestamp(0.0) == "00:00:00"
    assert _format_timestamp(65.5) == "00:01:05"
    assert _format_timestamp(3661.0) == "01:01:01"
    assert _format_timestamp(-1.0) == "00:00:00"


def test_transcript_passed_to_extractor_contains_timestamps() -> None:
    extractor = _FakeExtractor({"title": "", "goal": "", "agenda": [], "decision_points": []})
    utts = [
        _utt("u001", "田中", 0.0, 5.0, "では始めます"),
        _utt("u002", "鈴木", 65.0, 70.0, "了解です"),
    ]
    build_meeting_input(utts, meeting_id="m001", meta_extractor=extractor)
    transcript = extractor.calls[0]
    assert "[00:00:00] 田中: では始めます" in transcript
    assert "[00:01:05] 鈴木: 了解です" in transcript


def test_parse_json_relaxed_strips_code_fence() -> None:
    """LLM が ```json ... ``` で囲んでも JSON を取り出せる。"""
    text = '```json\n{"title": "x", "goal": "y", "agenda": [], "decision_points": []}\n```'
    parsed = _parse_json_relaxed(text)
    assert parsed["title"] == "x"


def test_parse_json_relaxed_plain_json() -> None:
    text = '{"title": "x", "goal": "y", "agenda": [], "decision_points": []}'
    parsed = _parse_json_relaxed(text)
    assert parsed["goal"] == "y"


def test_openai_meta_extractor_calls_client(monkeypatch) -> None:
    """OpenAIMetaExtractor が injected client を使ってリクエストを送る。"""
    captured: dict[str, Any] = {}

    class _FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class _FakeChoice:
        def __init__(self, content: str) -> None:
            self.message = _FakeMessage(content)

    class _FakeResponse:
        def __init__(self, content: str) -> None:
            self.choices = [_FakeChoice(content)]

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse(
                '{"title": "T", "goal": "G", "agenda": ["A"], "decision_points": []}'
            )

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self) -> None:
            self.chat = _FakeChat()

    extractor = OpenAIMetaExtractor(model="dummy", client=_FakeClient())
    meta = extractor.extract("これは会議書き起こしです。")
    assert meta["title"] == "T"
    assert meta["agenda"] == ["A"]
    assert captured["model"] == "dummy"
    # response_format は json_schema 強制
    assert captured["response_format"]["type"] == "json_schema"
    # transcript がプロンプトに展開されている
    assert "これは会議書き起こしです。" in captured["messages"][0]["content"]


def test_openai_meta_extractor_uses_placeholder_key_for_compatible_endpoint(
    monkeypatch,
) -> None:
    """OpenAI 互換 endpoint 指定時は api_key 未指定でも SDK に placeholder を渡す。"""
    captured: dict[str, Any] = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", type("OpenAIModule", (), {"OpenAI": _FakeOpenAI}))

    extractor = OpenAIMetaExtractor(model="dummy", endpoint="http://localhost:8001/v1")
    extractor._get_client()

    assert captured["base_url"] == "http://localhost:8001/v1"
    assert captured["api_key"] == "EMPTY"


def test_openai_meta_extractor_without_endpoint_keeps_sdk_default_auth(monkeypatch) -> None:
    """公式 OpenAI 利用時は api_key 未指定なら SDK の環境変数解決に任せる。"""
    captured: dict[str, Any] = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", type("OpenAIModule", (), {"OpenAI": _FakeOpenAI}))

    extractor = OpenAIMetaExtractor(model="dummy")
    extractor._get_client()

    assert captured == {"timeout": 60.0}
