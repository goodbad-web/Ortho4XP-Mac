# Architecture

本リポジトリは、X-Plane向けシーナリーを生成するPython/Tkinterアプリである。SwiftUIアプリではない。

## 入口と実行経路

- `Ortho4XP.py` が実行入口で、`src/` と `Providers/` をモジュール検索パスに追加する。
- 起動時に必要なディレクトリを確認・作成し、RAMディスクの孤立symlinkを復旧する。
- 引数なしでは `src/O4_GUI_Utils.py` のTkinter GUIを起動する。
- 引数ありではタイル設定を読み込み、次の順で処理する。
  `O4_Vector_Map.build_poly_file` → `O4_Mesh_Utils.build_mesh` → `O4_Mask_Utils.build_masks` → `O4_Tile_Utils.build_tile`

## 主な責務

| モジュール | 責務 |
| --- | --- |
| `O4_File_Names` | 入力・キャッシュ・生成物のパス定義 |
| `O4_Config_Utils` | グローバル設定とタイル設定の読み書き |
| `O4_Imagery_Utils` | 画像取得、前処理、アップスケール、DDS変換 |
| `O4_Vector_Map` / `O4_Vector_Utils` | OSM・空港情報からのベクター形状処理 |
| `O4_DEM_Utils` / `O4_Mesh_Utils` | 標高データ取得と `Triangle4XP` によるメッシュ生成 |
| `O4_Mask_Utils` | 水面等のマスク生成。NumPy、OpenCV、scikit-fmmを使用 |
| `O4_Tile_Utils` / `O4_DSF_Utils` | タイル生成、テクスチャ・DSF出力 |
| `O4_GUI_Utils` / `O4_UI_Utils` | GUIと進捗・ログ・UI設定 |
| `O4_Parallel_Utils` | スレッド処理とmacOSで安定させるspawn型multiprocessing |
| `O4_RAMDisk_Utils` | macOS RAMディスク、symlink、キャッシュ復旧 |

`Ortho4XP.py` では、他モジュールの変数を変更し得る `O4_Config_Utils` を最後にimportする既存順序を維持する。

## 外部実行ファイルとの境界

- `Utils/{lin,mac,win}/` に `Triangle4XP`、`DSFTool`、DDS変換ツール等の実行ファイルを配置する。
- `Utils/CMakeLists.txt` は `Utils/src/Triangle4XP.c` から `Triangle4XP` をビルドする。補助スクリプトは `Utils/run/` にある。
- `src/ASHelper.swift` はMetal・Vision・CoreImageを使うmacOS用CLIで、実行時の配置先は `Utils/mac/ASHelper` である。CMakeやSwiftUIのターゲットではない。
- Provider定義は `Providers/` にあり、ネットワーク取得・画像形式・利用可能性を変更する場合は呼び出し元とフォールバックを合わせて確認する。

## 生成データ

`Elevation_data/`、`OSM_data/`、`Orthophotos/`、`Masks/`、`Tiles/`、`tmp/` 等は入力または生成キャッシュを含む。コード変更の検証でこれらを一括削除・上書きしない。対象を限定し、必要ならユーザーに確認する。
