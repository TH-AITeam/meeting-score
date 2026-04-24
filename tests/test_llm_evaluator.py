"""LLM evaluator normalization tests."""

from app.evaluators.llm_evaluator import _default_result, _parse_response, _safe_result
from app.schemas.models import SpeechType


def test_parse_response_extracts_json_block():
    text = """```json
{"speech_type":"論点整理","scores":{},"penalties":{},"reason":"test"}
```"""
    parsed = _parse_response(text)
    assert parsed["speech_type"] == "論点整理"


def test_safe_result_clamps_scores_and_penalties():
    result = _safe_result(
        {
            "speech_type": " 論点整理 ",
            "scores": {
                "issue_clarification": 9,
                "decision_progress": -1,
                "risk_detection": "2",
            },
            "penalties": {
                "duplication": -9,
                "verbosity": 2,
                "off_topic": "-2",
            },
            "reason": "境界値の確認。",
        }
    )

    assert result.speech_type == SpeechType.ISSUE_CLARIFICATION.value
    assert result.scores.issue_clarification == 3
    assert result.scores.decision_progress == 0
    assert result.scores.risk_detection == 2
    assert result.penalties.duplication == -3
    assert result.penalties.verbosity == 0
    assert result.penalties.off_topic == -2


def test_default_result_marks_failure():
    result = _default_result()
    assert result.evaluation_failed is True
    assert result.speech_type == SpeechType.INFO_SHARING.value
