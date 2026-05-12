"""scripts/measure_asr_metrics.py の軽量テスト。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.measure_asr_metrics import _macro, _named_entity_recall


def test_named_entity_recall_returns_none_without_entities() -> None:
    """固有名詞リストが空なら JSON non-compliant な NaN ではなく None にする。"""
    assert _named_entity_recall("会議の書き起こし", []) is None
    assert _macro([None]) is None


def test_named_entity_recall_counts_present_entities() -> None:
    assert _named_entity_recall("田中さんが渋谷で確認しました", ["田中", "新宿"]) == 0.5
