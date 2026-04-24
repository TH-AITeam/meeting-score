"""Meeting analysis service.

API 層から独立した分析パイプラインを提供する。
"""

from __future__ import annotations

from app.aggregation.aggregator import build_meeting_summary
from app.context_builder.builder import build_contexts
from app.evaluators.llm_evaluator import evaluate_utterance
from app.reporting.reporter import format_meeting_summary_for_ui
from app.schemas.models import EvaluatedUtterance, MeetingInput
from app.scoring.calculator import calculate_total_score
from app.scoring.rule_corrections import apply_rule_corrections
from app.scoring.weights import AppConfig


class AnalysisEvaluationError(RuntimeError):
    """Raised when every utterance evaluation failed."""


async def run_analysis(meeting_data: MeetingInput, config: AppConfig) -> dict:
    """会議データを分析し、UI向けのサマリー辞書を返す。"""
    contexts = build_contexts(
        meeting_data,
        before_count=config.context_before,
        after_count=config.context_after,
    )

    evaluated: list[EvaluatedUtterance] = []
    failed_count = 0

    for ctx in contexts:
        target = ctx.target_utterance

        result = evaluate_utterance(
            ctx,
            model=config.llm_model,
            max_tokens=config.llm_max_tokens,
            max_retries=config.llm_max_retries,
        )

        if result.evaluation_failed:
            failed_count += 1

        total = calculate_total_score(
            result.scores,
            result.penalties,
            config.weights,
        )

        evaluated.append(
            EvaluatedUtterance(
                utterance_id=target.utterance_id,
                speaker=target.speaker,
                timestamp=target.timestamp,
                text=target.text,
                speech_type=result.speech_type,
                scores=result.scores,
                penalties=result.penalties,
                total_score=total,
                reason=result.reason,
            )
        )

    if contexts and failed_count == len(contexts):
        raise AnalysisEvaluationError(
            "LLM による評価がすべて失敗しました。APIキーの設定やAPIの状態を確認してください。"
        )

    evaluated = apply_rule_corrections(evaluated, config.weights)

    summary = build_meeting_summary(
        meeting_id=meeting_data.meeting_id,
        title=meeting_data.title,
        goal=meeting_data.goal,
        evaluated=evaluated,
        top_count=config.top_utterances_count,
        top_axis_count=config.top_per_axis_count,
    )

    return format_meeting_summary_for_ui(summary)
