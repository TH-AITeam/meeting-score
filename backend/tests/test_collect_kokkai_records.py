"""scripts/collect_kokkai_records.py の軽量テスト。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.collect_kokkai_records import KokkaiAPIError, payload_records, write_page


def test_payload_records_reads_meeting_list_records() -> None:
    records = [{"issueID": "100"}]

    assert payload_records("meeting_list", {"meetingRecord": records}) == records


def test_payload_records_raises_for_api_error_payload() -> None:
    with pytest.raises(KokkaiAPIError, match="検索条件の入力に誤りがあります"):
        payload_records(
            "meeting",
            {
                "message": "検索条件の入力に誤りがあります。",
                "details": ["from:開会日付をYYYY-MM-DD形式で入力してください。"],
            },
        )


def test_write_page_preserves_existing_snapshot(tmp_path: Path) -> None:
    page_path = tmp_path / "start_record_000000101.json"
    write_page(page_path, {"meetingRecord": [{"issueID": "old"}]})

    with pytest.raises(KokkaiAPIError, match="既に存在"):
        write_page(page_path, {"meetingRecord": [{"issueID": "new"}]})

    assert json.loads(page_path.read_text(encoding="utf-8")) == {
        "meetingRecord": [{"issueID": "old"}]
    }
