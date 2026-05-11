"""文脈ウィンドウ生成のテスト"""

from app.context_builder.builder import build_contexts
from app.schemas.models import MeetingInput, Utterance


def _make_meeting(n: int) -> MeetingInput:
    return MeetingInput(
        meeting_id="m001",
        title="テスト",
        goal="テスト目的",
        utterances=[
            Utterance(
                utterance_id=f"u{i + 1:03d}",
                speaker=f"Speaker{i}",
                timestamp=f"00:{i:02d}:00",
                text=f"発言{i}",
            )
            for i in range(n)
        ],
    )


def test_context_window_sizes():
    """前後3発言の文脈ウィンドウが正しく生成される"""
    meeting = _make_meeting(10)
    contexts = build_contexts(meeting, before_count=3, after_count=3)

    assert len(contexts) == 10

    # 最初の発言: 前0件、後3件
    assert len(contexts[0].before_utterances) == 0
    assert len(contexts[0].after_utterances) == 3

    # 4番目の発言: 前3件、後3件
    assert len(contexts[3].before_utterances) == 3
    assert len(contexts[3].after_utterances) == 3

    # 最後の発言: 前3件、後0件
    assert len(contexts[9].before_utterances) == 3
    assert len(contexts[9].after_utterances) == 0


def test_context_includes_meeting_info():
    """会議情報が文脈に含まれる"""
    meeting = _make_meeting(3)
    meeting.agenda = ["議題1", "議題2"]
    meeting.decision_points = ["決定1"]
    contexts = build_contexts(meeting)

    assert contexts[0].meeting_goal == "テスト目的"
    assert contexts[0].agenda == ["議題1", "議題2"]
    assert contexts[0].decision_points == ["決定1"]


def test_current_topic_estimated():
    """アジェンダから現在の議題が推定される"""
    meeting = _make_meeting(6)
    meeting.agenda = ["議題A", "議題B", "議題C"]
    contexts = build_contexts(meeting)

    # 6発言 / 3議題 = 2発言ずつ
    assert contexts[0].current_topic == "議題A"
    assert contexts[1].current_topic == "議題A"
    assert contexts[2].current_topic == "議題B"
    assert contexts[3].current_topic == "議題B"
    assert contexts[4].current_topic == "議題C"
    assert contexts[5].current_topic == "議題C"


def test_current_topic_empty_agenda():
    """アジェンダが空なら current_topic は空文字"""
    meeting = _make_meeting(3)
    meeting.agenda = []
    contexts = build_contexts(meeting)

    assert contexts[0].current_topic == ""
    assert contexts[2].current_topic == ""


def test_current_topic_from_utterance_topic():
    """発言に topic が設定されていればそちらを優先する"""

    meeting = MeetingInput(
        meeting_id="m001",
        title="テスト",
        goal="テスト目的",
        agenda=["議題A", "議題B"],
        utterances=[
            Utterance(
                utterance_id="u001",
                speaker="A",
                timestamp="00:00:00",
                text="発言1",
                topic="明示議題X",
            ),
            Utterance(
                utterance_id="u002", speaker="B", timestamp="00:01:00", text="発言2"
            ),
            Utterance(
                utterance_id="u003", speaker="A", timestamp="00:02:00", text="発言3"
            ),
        ],
    )
    contexts = build_contexts(meeting)
    # u001 は発言の topic フィールドを優先
    assert contexts[0].current_topic == "明示議題X"
    # u002, u003 はフォールバック推定
    assert contexts[1].current_topic != ""


def test_current_topic_from_topic_transitions():
    """topic_transitions による議題マーカーが使われる"""
    from app.schemas.models import TopicTransition

    meeting = MeetingInput(
        meeting_id="m001",
        title="テスト",
        goal="テスト目的",
        agenda=["議題A", "議題B"],
        topic_transitions=[
            TopicTransition(utterance_id="u001", topic="対象ユーザーの確認"),
            TopicTransition(utterance_id="u003", topic="スケジュール確認"),
        ],
        utterances=[
            Utterance(
                utterance_id="u001", speaker="A", timestamp="00:00:00", text="発言1"
            ),
            Utterance(
                utterance_id="u002", speaker="B", timestamp="00:01:00", text="発言2"
            ),
            Utterance(
                utterance_id="u003", speaker="A", timestamp="00:02:00", text="発言3"
            ),
            Utterance(
                utterance_id="u004", speaker="B", timestamp="00:03:00", text="発言4"
            ),
        ],
    )
    contexts = build_contexts(meeting)
    assert contexts[0].current_topic == "対象ユーザーの確認"
    assert contexts[1].current_topic == "対象ユーザーの確認"
    assert contexts[2].current_topic == "スケジュール確認"
    assert contexts[3].current_topic == "スケジュール確認"
