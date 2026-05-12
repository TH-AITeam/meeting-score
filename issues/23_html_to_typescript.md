# #23 [refactor] UI を単一 HTML から TypeScript (Vite + React) に移行

**Labels**: `enhancement`, `infra`, `P2`
**Milestone**: なし（v0.3 以降の手隙タイミングで対応）

## 概要

現状 `ui/index.html` は 1ファイル 824 行に HTML / CSS / JavaScript が混在している。コンポーネント分割・型安全・テスト容易性のため **Vite + React + TypeScript** ベースの SPA に移行する。

## 現状

- `ui/index.html` (823 行, 23KB)
  - `<style>` に CSS 約 200 行（CSS 変数 + コンポーネントクラス約 20 種）
  - `<script>` に JS 約 350 行、関数 49 個（DOM 操作・fetch・タブ切替・サンプル一覧・分析結果描画）
- `app/api/main.py` で `StaticFiles(directory="ui", html=True)` として配信
- API は `/api/samples` / `/api/analyze` / `/api/analyze/sample/{filename}` の3エンドポイント
- ビルドは現状なし（直接ブラウザでパース）

## 移行先の構成案

```
ui/
├── src/
│   ├── main.tsx                       # エントリ
│   ├── App.tsx
│   ├── components/
│   │   ├── InputView.tsx              # アップロード + サンプル一覧
│   │   ├── LoadingView.tsx
│   │   ├── ResultsView.tsx            # タブ枠
│   │   ├── tabs/
│   │   │   ├── SummaryTab.tsx
│   │   │   ├── SpeakersTab.tsx
│   │   │   └── ImprovementsTab.tsx
│   │   └── charts/
│   │       └── BarChart.tsx
│   ├── api/
│   │   ├── client.ts                  # fetch ラッパー
│   │   └── types.ts                   # API 応答スキーマの TS 型
│   ├── styles/
│   │   └── tokens.css                 # CSS 変数（既存の `:root` 由来）
│   └── utils/
│       └── scoring.ts                 # スコア → バッジクラス変換等
├── index.html                         # Vite のテンプレート
├── package.json
├── tsconfig.json
├── vite.config.ts
└── dist/                              # ビルド成果物（FastAPI が配信）
```

## やること

### スキャフォルディング
- [ ] `ui/package.json` 追加（`react`, `react-dom`, `typescript`, `vite`, `@vitejs/plugin-react`）
- [ ] `tsconfig.json` (`strict: true`)
- [ ] `vite.config.ts` で `build.outDir = "dist"`、`server.proxy` で `/api` を FastAPI に
- [ ] `ui/index.html` を Vite テンプレートに置換（既存ファイルはバックアップ）

### 既存ロジックの移植
- [ ] `App.tsx` でビュー切替（input → loading → results）
- [ ] `InputView`: サンプル一覧表示、JSON アップロード、`POST /api/analyze` 呼び出し
- [ ] `LoadingView`: 「分析中...」プログレスメッセージのアニメーション（既存 `animateProgress` 相当）
- [ ] `ResultsView`: タブ切替（summary / speakers / improvements）
- [ ] `SummaryTab`: 全体スコア・話者数・発言数・トピック数
- [ ] `SpeakersTab`: 話者ごとのスコアカード + バーチャート
- [ ] `ImprovementsTab`: 改善提案リスト

### 型・API
- [ ] `api/types.ts` に API 応答スキーマを定義（FastAPI 側の Pydantic と整合）
- [ ] `api/client.ts` で `fetchSamples()` / `analyzeSample(filename)` / `analyze(data)` を関数化
- [ ] 将来的に OpenAPI スキーマから自動生成（後追い検討）

### CSS
- [ ] 既存の `<style>` を `src/styles/tokens.css` + CSS Modules に展開
- [ ] CSS 変数（`--bg`, `--primary` 等）はそのまま使う
- [ ] 既存クラス名（`card`, `btn`, `score-badge` 等）の見た目を維持

### FastAPI 連携
- [ ] `ui/dist/` を `StaticFiles` 対象に変更（`app/api/main.py`）
- [ ] 開発時は Vite の dev server (`localhost:5173`) → API は `localhost:8000` にプロキシ
- [ ] 本番ビルドは `npm run build` → `dist/` を生成 → FastAPI が `dist/` を `/` に mount

### 品質
- [ ] Issue #19 / #20 の CI に `npm run typecheck` `npm run lint` を追加
- [ ] ESLint + Prettier 導入（最低限 React + TS 推奨設定）

## 完了条件

- ブラウザで `localhost:8000` を開いて、既存 UI と **見た目・操作感が等価**であること
  - サンプル一覧表示 → 分析 → 結果表示の一気通貫が動く
  - JSON アップロード → 分析が動く
  - タブ切替・バーチャート描画が動く
- `npm run build` で `ui/dist/` が生成され、FastAPI がそれを配信して旧 UI と同じ動作
- `npm run typecheck` が 0 件で通る
- README に開発手順（dev server 起動・本番ビルド）を追記

## 補足

- 既存 `ui/index.html` は `ui/index.html.bak` として残し、移行完了後に削除（PR で別途）
- フレームワーク選択肢: React / Vue / Svelte。**React 採用**（コミュニティ・型サポート・既存知識）
- スタイル選択肢: 素 CSS / CSS Modules / Tailwind / styled-components。**素 CSS + CSS Modules 採用**（既存スタイルとの差分が最小）
- 状態管理: React 標準の `useState` / `useReducer` で十分（Redux 等は不要）

## 非ゴール

- リアルタイム配信中の評価 UI
- 認証画面
- マルチページ化（Router 導入は別 Issue）
- i18n（多言語化）

## 関連

- `ui/index.html` 全体
- `app/api/main.py:48-50`（StaticFiles マウント）
- Issue #19 / #20（CI で typecheck/lint も走らせる）
