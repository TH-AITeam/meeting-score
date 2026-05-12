# ブランチ保護設定

`main` ブランチの保護ルールと PR 運用設定を記録する。

## ブランチ保護ルール（`main`）

以下の設定を GitHub API で適用済み（2026-05-12）。API での確認方法:

```bash
gh api repos/TH-AITeam/meeting-score/branches/main/protection
```

**適用確認済み設定（2026-05-12 実行結果）:**

```
required_status_checks.strict: True
contexts: ['lint', 'test', 'Frontend (typecheck + lint + build)']
enforce_admins: True
required_linear_history: True
allow_force_pushes: False
allow_deletions: False
required_conversation_resolution: True
required_approving_review_count: 0
```

| 設定 | 値 |
|---|---|
| Require a pull request before merging | ON |
| Required approving review count | 0（個人開発フェーズ。チーム化したら 1 以上に変更） |
| Require status checks to pass before merging | ON |
| Required status checks | `lint`, `test`, `Frontend (typecheck + lint + build)` |
| Require branches to be up to date before merging | ON（strict） |
| Require conversation resolution before merging | ON |
| Require linear history | ON（squash または rebase のみ） |
| Allow force pushes | OFF |
| Allow deletions | OFF |
| Do not allow bypassing the above settings (enforce_admins) | ON |

### 必須ステータスチェック

CI ワークフロー（`.github/workflows/ci.yml`）の以下 3 ジョブが全通過しないと merge 不可:

- `Frontend (typecheck + lint + build)` — TypeScript 型チェック・ESLint・Vite ビルド
- `lint` — ruff check / ruff format --check / mypy
- `test` — pytest + カバレッジ（70% 以上）

## リポジトリマージ設定

| 設定 | 値 |
|---|---|
| Allow merge commits | OFF |
| Allow squash merging | ON（デフォルト。タイトル = PR タイトル） |
| Allow rebase merging | ON |
| Automatically delete head branches | ON |

## 設定変更手順

緊急時以外は GitHub API または Web UI で変更する。

### GitHub API で変更する場合

```bash
# ブランチ保護ルールの確認
gh api repos/TH-AITeam/meeting-score/branches/main/protection

# ブランチ保護ルールの更新（例: 必須ステータスチェックの変更）
gh api -X PUT repos/TH-AITeam/meeting-score/branches/main/protection \
  --input protection.json
```

### Web UI で変更する場合

`Settings → Branches → Branch protection rules → main` を編集。

## 緊急バイパス手順

本番障害など緊急時にブランチ保護をバイパスする場合:

1. `Settings → Branches → Branch protection rules → main → Edit`
2. **Do not allow bypassing the above settings** を一時的に OFF
3. 修正を実施
4. 対応完了後、**すぐに** 設定を ON に戻す
5. 事後に PR または commit で変更内容を記録

バイパスは最小限の時間に留め、必ず元に戻すこと。

## 設定を変えるタイミング

| イベント | 変更内容 |
|---|---|
| チームに開発者が加わる | `required_approving_review_count` を 1 以上に変更 |
| CI ジョブ名が変わる | Required status checks を更新 |
| カバレッジ閾値が 80% に上がる | `test` ジョブの設定変更（CI yml 側） |
