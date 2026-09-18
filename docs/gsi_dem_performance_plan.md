# GSI DEM処理の高速化計画

## 目的

実GSI入力を維持したまま、`build GSI DEM` の反復実行時間と不要なメモリ/I/Oを削減する。
対象は `src/O4_GSI_DEM_Utils.py` のcatalog選択、ZIP/XML読込、raster配置であり、出力形式・CRS・NoData・解像度選択・CLI/GUI公開契約は変更しない。

## 現状の基準

2026-09-18時点の入力は次の規模である。

- ready ZIP: 273件
- 入力容量: 約3.94 GiB
- XML: 12,925件
- 製品: DEM1A 37件、DEM5A 118件、DEM10B 118件
- 最大ZIP: 約216 MiB、最大200 XML

現行`build_gsi_dem()`はbuild開始時に全catalog scanを行う。scanは全ZIPのSHA-256とXML payload検証を行い、さらに各regionのbuildでZIP/XMLを再読込する。

実入力のcatalogを書き換えないscanでは、78秒経過時点でも完了せず、`_validate_metadata_payload()`のfloat変換中に停止した。これは完了時間ではなく下限値として扱う。

## 実装方針

### 1. build時のcandidate-only検証

通常のbuildではcatalog全体を再スキャンしない。

1. `catalog.json`を読み込む。
2. mesh/bbox/HGTごとのcandidateをcatalog metadataから選ぶ。
3. candidate ZIPだけを存在確認・サイズ確認・SHA-256確認する。
4. candidateが変更されていれば従来通りbuildを中止する。
5. 全入力の厳密検査は明示的な`scan`で行う。

catalogが存在しない、形式不正、入力ZIPの登録漏れがある場合は、既存のstrict scanへフォールバックする。catalogに登録されていない新規ZIPを通常buildが黙って無視しないよう、ZIP相対パス集合をcheap checkする。

### 2. build内のXML metadata cache

同じZIPを複数regionで読む場合、ZIP内XMLの軽量metadataだけをbuild単位でcacheする。

cacheするもの:

- XML member名
- product/date
- mesh codes
- source bounds
- width/height/startPoint
- source CRS

tupleListのnumpy配列やElementTree全体は無制限に保持しない。対象regionに交差するXMLだけを再読込し、payloadを値配列へ変換する。

### 3. 検証と値パースの二重処理削減

strict scanでは従来通りpayloadを検証する。buildではcatalogでSHA-256が一致した入力を信頼し、交差するXMLだけを値パースする。

交差XMLの処理では、metadataを再解析せず、検証と値配列生成を1回のループにまとめる。`データなし`、`<= -9990`、startPoint、grid超過の扱いは現行仕様を維持する。

### 4. `_insert_block()`の整列時高速経路

source/outputの解像度・境界が整数cellに整列する場合だけ、`np.ix_`による全面index配列を使わず、slice代入へ切り替える。

- NoData/NaNは現行と同じく上書きしない。
- 非整列、point-grid、座標変換境界では現行nearest-neighbor経路を使用する。
- まず通常GeoTIFF経路だけを対象とし、HGT point-gridは別検証後に扱う。

## 変更対象

- `docs/gsi_dem_performance_plan.md`
- `src/O4_GSI_DEM_Utils.py`
- `tests/test_gsi_dem.py`

既存の`src/O4_DEM_Utils.py`のcustom DEM window read、ZIP並列化、catalog schemaの大幅変更は今回の範囲外とする。

## 検証計画

### 自動テスト

- 既存GSIテスト全件
- catalogがあるbuildで全ZIPscanを呼ばないこと
- candidate ZIPの変更検出
- catalog欠落時のstrict scan fallback
- XML metadata cacheの再利用
- malformed XML/NoData/startPoint契約
- compact_int16、Float32、VRT、HGTの既存契約
- 整列slice経路と従来経路の同値性
- cancel時の一時出力cleanup

### 実データ比較

canonical outputは変更せず、一時outputで次を比較する。

- mesh `51320000`
- 既存21mesh相当の一括build
- cold build
- 同一入力のwarm build

比較項目:

- 出力shape
- valid/missing cell数
- elevation統計
- scale/offset適用後の値
- VRT contract
- 入力ZIP数・XML数・parse回数
- wall/user/sys時間
- peak RSS

## 実装順序

1. 本計画を保存
2. candidate-only検証とstrict fallbackを実装
3. XML metadata cacheを実装
4. payload検証/値パースの重複を削減
5. 整列slice経路を実装
6. 既存テストと実データ一時outputで検証

commit/pushは別途明示依頼があるまで行わない。

## 実装状況

2026-09-18時点で、次を実装した。

- catalogが入力ZIP集合と一致する場合、buildはstrict全体scanを使わずcatalogを読む
- build対象regionのcandidate ZIPだけをサイズ/SHA-256検証する
- catalogがない、壊れている、ZIP集合が一致しない場合はstrict scanへfallbackする
- build中の同一ZIPについてXML metadataをcacheする
- candidate XMLのpayload検証と値配列生成を1回の処理に統合する
- 通常GeoTIFFで解像度・境界が整列する場合のslice配置経路を追加する

未実装:

- catalogへの永続XML member index
- HGT point-gridのstride高速化
- custom DEMのGDAL window read
- ZIP並列化

## 実装後の検証結果

- `tests/test_gsi_dem.py`: 22 passed
- `py_compile`: 成功
- 実GSI入力の`51320000`: candidate 3 ZIPだけを検証して一時outputへ1m GeoTIFF/VRT/manifestを生成、約13.68秒
- canonical `Elevation_data/GSI/output`は変更していない

実GSI全体のstrict scanは約78秒経過時点でも完了しなかったため、13.68秒との値は同一処理同士の厳密なbefore/after比較ではなく、build経路から全体scanを除去できたことの確認値である。
