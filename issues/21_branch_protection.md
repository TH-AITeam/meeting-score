# #21 [chore] ブランチ保護と PR 運用ルール

**Labels**: `chore`, `quality`, `P1`
**Milestone**: v0.2

## 概要

`main` への直 push を禁止し、PR + CI 通過を必須にする。**個人開発でも入れた方がいい**(過去の自分のリリース事故防止)。

## 前提

- #20 の CI が安定稼働していること
- そうでないと自分で動けなくなる

## やること(GitHub Web UI で操作)

`Settings → Branches → Branch protection rules → Add rule` で `main` に以下を設定:

- [ ] **Require a pull request before merging**: ON
  - Required approvals: 0(個人開発の場合) / 1 以上(将来チームになったら)
- [ ] **Require status checks to pass before merging**: ON
  - Status checks: `lint`, `test` (#20 の job 名)
  - **Require branches to be up to date before merging**: ON
- [ ] **Require conversation resolution before merging**: ON
- [ ] **Require linear history**: ON (rebase or squash のみ)
- [ ] **Do not allow bypassing the above settings**: ON (自分も含めて)
- [ ] **Restrict who can push to matching branches**: 個人の場合は不要

`Settings → General → Pull Requests`:

- [ ] **Allow merge commits**: OFF
- [ ] **Allow squash merging**: ON (default message: PR title)
- [ ] **Allow rebase merging**: ON
- [ ] **Automatically delete head branches**: ON

## ドキュメント化

- [ ] `CONTRIBUTING.md` 作成
  - ブランチ命名 (`feature/xxx`, `fix/xxx`, `chore/xxx`)
  - コミット規約 (Conventional Commits 推奨: `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`)
  - PR タイトル形式
  - レビュー観点(個人開発でもセルフレビュー欄を作る)

## 完了条件

- `main` への直 push が UI で拒否される
- CI red の PR が merge できない
- 設定内容が `docs/branch_protection.md` にスクショ付きで記録される

## 注意

- **emergency hotfix が必要な状況**を想定して、自分にバイパス権限を一時的に与える手順も書いておく(`bypass` を一時的に on→off)
- レビュー必須(0でなく1以上)にすると一人では merge できないので、最初は 0 で始めて、チームができたら上げる
- PR template (#22) と組み合わせるとレビュー観点が漏れにくい
