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
| `O4_GSI_DEM_Utils` | GSI DEM ZIPの検査・管理、catalog、GeoTIFF/VRT/HGT生成 |
| `O4_Mask_Utils` | 水面等のマスク生成。NumPy、OpenCV、scikit-fmmを使用 |
| `O4_Tile_Utils` / `O4_DSF_Utils` | タイル生成、テクスチャ・DSF出力 |
| `O4_GUI_Utils` / `O4_UI_Utils` | GUIと進捗・ログ・UI設定 |
| `O4_Parallel_Utils` | スレッド処理とmacOSで安定させるspawn型multiprocessing |
| `O4_RAMDisk_Utils` | macOS RAMディスク、symlink、キャッシュ復旧 |

`Ortho4XP.py` では、他モジュールの変数を変更し得る `O4_Config_Utils` を最後にimportする既存順序を維持する。

## MUXPメッシュ更新

タイル設定で`muxp_enabled=True`を明示した場合、`MUXP/`（グローバル設定の`muxp_folder`で変更可能）を再帰検索し、MUXPヘッダーの`tile`が一致する`.muxp`だけを決定的な相対パス順で選択する。同一IDは数値的に最新versionを使い、同一versionで内容が異なる場合は失敗する。MUXP処理は`src/muxp_engine/`に固定取り込みしたヘッドレスエンジンで行う。

Build Imagery/DSFとBuild Allの両方で、生成DSFを`.dsf.tmp`として作成した後、公開直前の同じトランザクション境界でMUXPを適用する。strict検証、source_dsf互換性、コマンド処理、DSF再書き込み、manifest作成のいずれかに失敗した場合はDSFを公開しない。処理結果はタイル内の`Ortho4XP_muxp.json`とビルドログへ記録し、MUXP適用前のDSFは`muxp_backups/`に世代保存する。

## 外部実行ファイルとの境界

- `Utils/{lin,mac,win}/` に `Triangle4XP`、`DSFTool`、DDS変換ツール等の実行ファイルを配置する。
- `Utils/CMakeLists.txt` は `Utils/src/Triangle4XP.c` から `Triangle4XP` をビルドする。補助スクリプトは `Utils/run/` にある。
- `src/ASHelper.swift` はMetal・Vision・CoreImageを使うmacOS用CLIで、実行時の配置先は `Utils/mac/ASHelper` である。CMakeやSwiftUIのターゲットではない。
- Provider定義は `Providers/` にあり、ネットワーク取得・画像形式・利用可能性を変更する場合は呼び出し元とフォールバックを合わせて確認する。

## 生成データ

`Elevation_data/`、`OSM_data/`、`Orthophotos/`、`Masks/`、`Tiles/`、`tmp/` 等は入力または生成キャッシュを含む。コード変更の検証でこれらを一括削除・上書きしない。対象を限定し、必要ならユーザーに確認する。

## GSI DEMの入力と生成物

GSIの原本ZIPと生成物は、次の専用領域で分離する。`input/` はダウンロード元を変更せずに取り込んだZIPの管理場所、`output/` はOrtho4XPで利用する生成物の場所である。

```text
Elevation_data/GSI/
├── input/
│   ├── catalog.json
│   ├── DEM1A/YYYYMMDD/*.zip
│   ├── DEM5A/  DEM5B/  DEM5C/
│   ├── DEM10A/ DEM10B/
│   └── _quarantine/
└── output/
    ├── *.tif  *.vrt  *.json
    └── hgt/*.hgt
```

`src/O4_GSI_DEM_Utils.py` がCLIとSupportのGSIダイアログから共有される境界であり、`scan_gsi_input()`、`import_gsi_archives()`、`build_gsi_dem()`が公開APIである。catalogはZIPのSHA-256、検査時点のサイズ、製品種別、作成年月日、メッシュコード、状態を記録し、一時ファイルから原子的に置換する。`build`は状態が`ready`で、catalog記録から変更されていないZIPだけを読む。

既定のGeoTIFF出力は`compact_int16`で、Int16のraw値にscale=0.25、offset=0.0を適用し、NoData=-32768として保存する。ZSTD圧縮を優先し、利用できないGDALではDEFLATEへフォールバックする。VRTへ束ねるGeoTIFFは同じdtype、scale、offset、NoData契約でなければならない。既存のFloat32 GeoTIFF/VRTとHGTは`float32_legacy`または従来経路として読み込める。

入力ZIPの分類優先度はDEM1A、DEM5A、DEM5B、DEM5C、DEM10A、DEM10Bの順である。JGD2000、JGD2011、JGD2024、WGS84はWGS84地理座標へ変換するが、未知のCRSは`--source-crs`を明示しない限り拒否する。JGD2024のPROJ定義がない環境では近似処理を行わずエラーにする。既存の`make_gsi_geotiff_5m.py`、`make_gsi_hgt.py`、`Ortho4XP.cfg`はこの経路から変更しない。
