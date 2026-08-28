# AGENTS.md

常設指示はこのファイルのみ。詳細は必要時に読む。

## 固有情報

- 本体は `Ortho4XP.py` 入口のPython/Tkinter。`Utils/src/Triangle4XP.c` はC補助、`src/ASHelper.swift` はmacOS CLI。SwiftUIではない。
- 実行経路は [docs/architecture.md](docs/architecture.md)、開発手順は [docs/development.md](docs/development.md) を読む。

## 変更ルール

- 現在のbranch/worktreeを維持し、明示依頼なしにbranch、worktree、履歴、commit、pushを操作しない。
- 変更前に入口・呼び出し元・依存・状態・副作用を確認し、推測で実装しない。
- データフロー、公開関数、CLI引数、設定キー、外部バイナリ契約を壊さない。
- 無関係な変更、rename、大規模refactorを避け、既存パターンに沿う最小差分にする。
- UI文言は英語基準で日本語対応を維持し、表示・状態・エラー経路を確認する。

## 検証・報告

- 変更範囲に対応する検証を [docs/development.md](docs/development.md) から選ぶ。
- 実行済み・未実行・環境制約を分け、未確認事項を断定しない。差分と影響を日本語で報告する。

## 必要時だけ読む文書

- `docs/TASK.md` はタスク定義時、`docs/BUG.md` はバグ対応時だけ読む。
- `docs/SKILLS.md` は技能選択時だけ読む。いずれも常設ルールではない。
