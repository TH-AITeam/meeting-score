# #16 [chore] OpenAI 依存の完全撤去と移行ドキュメント

**Labels**: `chore`, `breaking-change`, `P1`
**Milestone**: v0.5

## 概要

#11 でローカル推論基盤が立ち、#13 で自前モデルができたら、OpenAI 経由のコードと依存を片付ける。

## やること

- [ ] `app/evaluators/llm_evaluator.py` を deprecated に(コメントで明示、削除はしない)
- [ ] `pyproject.toml` から `openai` を **optional 依存** に降格 (`[project.optional-dependencies] openai-eval`)
- [ ] `.env.example` から `OPENAI_API_KEY` を `OPENAI_API_KEY=  # optional, only for distillation` に変更
- [ ] `Readme.md` のセットアップ手順をローカル推論ベース(#11)に書き換え
- [ ] 推論サーバ起動の Quick Start を追加
- [ ] 蒸留用途(#12)でだけ OpenAI を使う、と用途を限定して明記
- [ ] サンプル `.env.example` を更新
  - `LLM_BACKEND=local`
  - `LLM_ENDPOINT=http://localhost:8001/v1`
  - `LLM_MODEL=<採用モデル(#17)>`

## 完了条件

- OpenAI API キーなしで全機能が動く
- `pyproject.toml` のデフォルトインストールに `openai` が含まれない
- README にローカル運用の動作手順がまとまっている

## 注意

完全削除はしない。ベンチマーク用途と蒸留用途で残す価値があるので「optional」扱いに留める。
