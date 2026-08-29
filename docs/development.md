# Development and Verification

## セットアップと起動

リポジトリのルートで実行する。Python依存関係の正本は `requirements.txt`、macOSの初期セットアップは `install_mac.sh` である。

```sh
./install_mac.sh
source .venv/bin/activate
python3 Ortho4XP.py
```

`Ortho4XP.py` を引数なしで起動するとGUI、`python3 Ortho4XP.py <lat> <lon>` では既存タイル設定、`python3 Ortho4XP.py <lat> <lon> <imagery> <zl>` では指定値でCLI処理を行う。

`GDAL`等のネイティブ依存関係はHomebrew側のバージョンとの整合が必要である。OS別の利用者向け説明は `Install_Instructions.txt` に残すが、起動ファイルは実在する `Ortho4XP.py` を使う。

## Cユーティリティ

ネイティブmacOSビルドはリポジトリのルートから実行する。

```sh
cmake -S Utils -B Utils/build
cmake --build Utils/build
```

`Utils/run/configure.py` と `build.py` は、`Utils/` を作業ディレクトリにして使うクロスビルド補助である。`mac` はosxcross、`win` はMinGWのtoolchainを指定する。

```sh
cd Utils
python3 run/configure.py mac release
python3 run/build.py mac release
```

`mac` は `lin` または `win` に置き換えられる。

`python3 run/install.py release` は `build/release/{lin,mac,win}/` の成果物を各 `Utils/` 配下へコピーするため、必要な対象をすべてビルドしてから実行する。

## Swift補助ツール

`src/ASHelper.swift` と配置済みの `Utils/mac/ASHelper` は別の成果物である。Swiftソースを変更した場合は、リポジトリルートから専用スクリプトで実行バイナリを再生成する。CユーティリティのCMakeビルドだけでは更新されない。

```sh
./Utils/run/build_ashelper.sh
```

このスクリプトは書き込み可能な一時Module Cacheで現在のmacOS向けバイナリをビルドし、成功後に実行時配置先の `Utils/mac/ASHelper` を更新する。`src/ASHelper` と `src/ASHelper_test` は実行時の配置先ではない。

## 検証の選び方

- Python変更では、変更ファイルの構文確認と、関連する入口・モジュール import を確認する。
- `test_airport_array.py` と `scratch/test_*.py` は、pytest等の統合テストではなく、入力データやmacOS固有環境に依存する手動スクリプトである。スクリプトごとのパス、生成物、後始末を確認してから実行する。
- `scratch/test_config.py` 等は設定ファイルを作成し、RAMディスク系スクリプトはマウント・symlink・キャッシュを変更し得る。対象を限定し、実行後の状態を確認する。
- Cソース変更では対象プラットフォームのCMake configure/buildを行う。
- `ASHelper.swift` 変更ではSwiftソースのビルドに加え、`Utils/mac/ASHelper` を使う画像変換・アップスケール経路を確認する。
- GUI変更ではTkinterの表示、ボタン状態、バックグラウンド処理の完了・失敗・キャンセル経路を手動確認する。自動テストがないことを成功の根拠にしない。

構文確認、ビルド、限定的なスクリプト実行だけでは、実際のProvider応答、長時間のタイル生成、GUI操作、利用者データへの影響まで保証しない。未実行の範囲を最終報告に明記する。
