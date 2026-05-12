"""文脈ウィンドウ生成モジュール

各発言の評価時に必要な文脈(前後の発言、会議目的、議題、現在の議題)を組み立てる。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.models import MeetingInput, Utterance


@dataclass
class EvaluationContext:
    """1発言の評価に使う文脈"""

    meeting_goal: str
    agenda: list[str]
    decision_points: list[str]
    current_topic: str
    before_utterances: list[Utterance]
    target_utterance: Utterance
    after_utterances: list[Utterance]
    meeting_type: str | None = None


def _estimate_current_topic(agenda: list[str], index: int, total: int) -> str:
    """発言位置からアジェンダ上の現在議題を推定する（フォールバック用）

    発言列を議題数で均等分割し、各発言がどの議題区間に属するか推定する。
    明示的な議題情報がない場合にのみ使用される。
    """
    if not agenda:
        return ""
    segment_size = max(1, total // len(agenda))
    topic_idx = min(index // segment_size, len(agenda) - 1)
    return agenda[topic_idx]


def _build_topic_map(meeting: MeetingInput) -> dict[str, str]:
    """topic_transitions から utterance_id → topic のマッピングを構築する

    各遷移マーカー以降の発言に、その議題を割り当てる。
    """
    if not meeting.topic_transitions:
        return {}

    # utterance_id の順序インデックスを作る
    id_to_idx = {u.utterance_id: i for i, u in enumerate(meeting.utterances)}
    # 遷移マーカーをソート
    sorted_transitions = sorted(
        meeting.topic_transitions,
        key=lambda t: id_to_idx.get(t.utterance_id, 0),
    )

    topic_map: dict[str, str] = {}
    trans_idx = 0
    current_topic = ""

    for i, u in enumerate(meeting.utterances):
        # 次の遷移マーカーに到達したら議題を切り替え
        while trans_idx < len(sorted_transitions):
            marker = sorted_transitions[trans_idx]
            marker_idx = id_to_idx.get(marker.utterance_id, 0)
            if i >= marker_idx:
                current_topic = marker.topic
                trans_idx += 1
            else:
                break
        if current_topic:
            topic_map[u.utterance_id] = current_topic

    return topic_map


def _resolve_topic(
    utterance: Utterance,
    topic_map: dict[str, str],
    agenda: list[str],
    index: int,
    total: int,
) -> str:
    """発言の議題を解決する。優先順位: 発言の topic > topic_transitions > 均等分割推定"""
    if utterance.topic:
        return utterance.topic
    if utterance.utterance_id in topic_map:
        return topic_map[utterance.utterance_id]
    return _estimate_current_topic(agenda, index, total)


def build_contexts(
    meeting: MeetingInput,
    before_count: int = 3,
    after_count: int = 3,
) -> list[EvaluationContext]:
    """全発言の文脈ウィンドウを生成する"""
    contexts: list[EvaluationContext] = []
    utterances = meeting.utterances
    total = len(utterances)
    topic_map = _build_topic_map(meeting)

    for i, target in enumerate(utterances):
        start = max(0, i - before_count)
        end = min(total, i + after_count + 1)

        before = utterances[start:i]
        after = utterances[i + 1 : end]

        current_topic = _resolve_topic(target, topic_map, meeting.agenda, i, total)

        contexts.append(
            EvaluationContext(
                meeting_goal=meeting.goal,
                agenda=meeting.agenda,
                decision_points=meeting.decision_points,
                current_topic=current_topic,
                before_utterances=before,
                target_utterance=target,
                after_utterances=after,
                meeting_type=meeting.meeting_type,
            )
        )

    return contexts
