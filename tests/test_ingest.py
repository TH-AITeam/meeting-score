"""Input normalization tests."""

from app.ingest.loader import load_meeting_from_dict


def test_load_basic():
    data = {
        "meeting_id": "m001",
        "title": "テスト会議",
        "goal": "テスト目的",
        "utterances": [
            {
                "utterance_id": "u001",
                "speaker": "A",
                "timestamp": "00:01:00",
                "text": "テスト発言",
            }
        ],
    }
    meeting = load_meeting_from_dict(data)
    assert meeting.meeting_id == "m001"
    assert len(meeting.utterances) == 1
    assert meeting.agenda == []
    assert meeting.decision_points == []


def test_auto_fill_missing_fields():
    data = {
        "meeting_id": "m002",
        "title": "テスト",
        "goal": "テスト",
        "utterances": [
            {"text": "発言1"},
            {"text": "発言2"},
        ],
    }
    meeting = load_meeting_from_dict(data)
    assert len(meeting.utterances) == 2
    assert meeting.utterances[0].utterance_id == "u001"
    assert meeting.utterances[1].utterance_id == "u002"
    assert meeting.utterances[0].speaker == "Speaker 1"


def test_preserves_input_order():
    data = {
        "meeting_id": "m003",
        "title": "テスト",
        "goal": "テスト",
        "utterances": [
            {"utterance_id": "u003", "speaker": "C", "timestamp": "00:03:00", "text": "3"},
            {"utterance_id": "u001", "speaker": "A", "timestamp": "00:01:00", "text": "1"},
            {"utterance_id": "u002", "speaker": "B", "timestamp": "00:02:00", "text": "2"},
        ],
    }
    meeting = load_meeting_from_dict(data)
    assert [u.utterance_id for u in meeting.utterances] == ["u003", "u001", "u002"]


def test_non_padded_ids_preserve_order():
    data = {
        "meeting_id": "m004",
        "title": "テスト",
        "goal": "テスト",
        "utterances": [
            {"utterance_id": "u1", "speaker": "A", "timestamp": "00:01:00", "text": "first"},
            {"utterance_id": "u10", "speaker": "B", "timestamp": "00:10:00", "text": "tenth"},
            {"utterance_id": "u2", "speaker": "C", "timestamp": "00:02:00", "text": "second"},
        ],
    }
    meeting = load_meeting_from_dict(data)
    assert [u.utterance_id for u in meeting.utterances] == ["u1", "u10", "u2"]
