# SFT データ統計サマリ (Issue #13)

自動生成 (`scripts/build_sft_dataset.py`)。手で編集しないこと。

## 件数

| split | 会議数 | 発言(行)数 |
|---|---:|---:|
| train | 28 | 485 |
| val | 6 | 159 |
| test | 6 | 133 |
| **合計** | 40 | 777 |

> 完了条件 train >= 5000 件: **❌ 未達 (あと 4515 件)**

## ソース別

| source | 件数 |
|---|---:|
| distilled | 777 |

## speech_type 分布

| speech_type | 件数 |
|---|---:|
| 情報共有 | 353 |
| 意思決定促進 | 113 |
| 質問 | 91 |
| 根拠提示 | 75 |
| 雑談/脱線 | 53 |
| 懸念提示 | 45 |
| 提案 | 21 |
| 要約 | 17 |
| 論点整理 | 9 |

## 軸スコア分布 (0-3)

| 軸 | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| issue_clarification | 194 | 335 | 233 | 15 |
| decision_progress | 212 | 393 | 169 | 3 |
| risk_detection | 478 | 186 | 99 | 14 |
| actionability | 441 | 278 | 54 | 4 |
| groundedness | 308 | 185 | 206 | 78 |
| novelty | 435 | 242 | 94 | 6 |
| summarization | 574 | 176 | 21 | 6 |

## 減点軸分布 (-3-0)

| 軸 | -3 | -2 | -1 | 0 |
|---|---:|---:|---:|---:|
| duplication | 0 | 17 | 188 | 572 |
| verbosity | 0 | 15 | 210 | 552 |
| off_topic | 0 | 17 | 35 | 725 |
| unsupported_assertion | 0 | 0 | 35 | 742 |
| override | 0 | 0 | 0 | 777 |

## reject 内訳

| 理由 | 件数 |
|---|---:|
| extreme_all_zero | 64 |

