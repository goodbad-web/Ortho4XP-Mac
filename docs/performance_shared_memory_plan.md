# All-in-one 性能改善計画（RAM diskなし）

## 目的

RAM diskを前提にせず、SSD上の永続キャッシュ・transaction・rollback契約を維持したまま、all-in-oneの画像変換で発生する中間ファイル交換と直列待ちを削減する。

主対象は次の経路とする。

```text
download / image assembly
        ↓
bounded RGB8 handoff
   ├─ cache-writer thread → atomic JPEG cache on SSD
   └─ resident ASHelper → MetalFX Spatial direct DDS
                              ↓
                     in-memory DDS validation
                              ↓
                    SSD transaction temporary
                              ↓
                         atomic activation
```

初期対象はopaque provider JPEG、RGB8、`TextureConverter`、`metalfx_spatial`、BC1/BC3、GPU対応color filterとする。combined imagery、RGBA、WebP、非対応color filter、特殊providerは既存path-based/CPU経路へfallbackする。

## 非対象・互換性

- RAM disk機能は削除・変更せず、今回の性能比較では無効化する。
- `build_all`、`build_continuous`、`build_tile_list`のシグネチャは変更しない。
- 公開CLI引数は変更しない。
- 既存のJSONL path transport、CPU multiprocessing、DDS検証、transaction/rollbackをfallbackとして残す。
- vector、mesh、OSM、Triangle4XP、DSFTool、overlayのGPU化は今回のshared-memory初期対象外とする。
- `enable_parallel_overlay=False`、`max_parallel_tiles=1`を維持する。

## 実装フェーズ

### Phase 0：ベースライン

RAM diskなし・同一JPEG/DEM/OSM cache・別stagingで次を各3回実行し、中央値を比較する。

1. 旧来逐次＋CPU
2. SSD streaming＋CPU
3. SSD streaming＋MetalFX direct DDS
4. shared-memory＋MetalFX direct DDS

all-in-one全体時間を主指標とし、vector、mesh、mask、imagery/DSF、overlay、download、cache write、queue wait、GPU batch、CPU fallback、peak RSS、memory pressureを個別記録する。固定の速度目標はベースライン後に設定する。

### Phase 1：SSD streaming

- download完了後、JPEGをSSDへatomic保存する。
- 保存完了後すぐbounded conversion schedulerへ投入する。
- queue満杯時はdownload workerへbackpressureを返す。
- CPU/GPU taskを分離し、GPUはdirect DDS batchを使う。
- ASHelperはtile-local resident serverを再利用する。
- 初期値は`enable_streaming_conversion=False`とし、`+34+133`の受入後に既定値を`True`へ変更する。

### Phase 2：shared payload

`build_jpeg_ortho()`の公開契約を変更せず、内部専用payload builderを追加する。画像組み立て後にRGB8 bufferを一度だけ作り、cache-writerとGPU handoffで共有する。

cache-writerはPython内の単一threadとし、JPEGの画質・命名・cache layoutを変更しない。cache writeはtemporary＋`os.replace`で行い、tile成功前に全cache writeの完了を待つ。cancel時は完了済みcacheを残し、未完了temporaryだけを削除する。

### Phase 3：ASHelper shared-memory transport

新規`src/O4_Shared_Memory.py`でsegmentの所有、budget、lease、cleanup、stale回収を管理する。Pythonは`multiprocessing.shared_memory.SharedMemory`を使い、Swiftは`shm_open`＋`mmap`を使う。

既存`convert_batch`へ`transport`を追加する。

```json
{
  "op": "convert_batch",
  "transport": "shared_memory",
  "gpu": true,
  "tasks": [
    {
      "id": "texture-0001",
      "input_shared_memory": {
        "name": "/ortho4xp-...",
        "offset": 0,
        "capacity": 50331648,
        "used_bytes": 50331648,
        "width": 4096,
        "height": 4096,
        "stride": 12288,
        "pixel_format": "RGB8",
        "read_only": true
      },
      "output_shared_memory": {
        "name": "/ortho4xp-...",
        "offset": 0,
        "capacity": 100000000,
        "used_bytes": 0,
        "width": 4096,
        "height": 4096,
        "stride": 0,
        "pixel_format": "DDS",
        "read_only": false
      },
      "format": "BC3"
    }
  ]
}
```

入力はRGB8、row-major、sRGB、alphaなしに限定する。出力はPythonが容量を計算して確保し、ASHelperはDDS bytesと`used_bytes`を返す。PythonはSSDへ書く前にheader、format、dimensions、mipmap、payload lengthをメモリ上で検証する。

### Phase 4：障害復旧

- shared transport固有エラー：当該tileのshared transportを無効化し、path-based GPUへdowngradeする。
- ASHelper/Metal crashまたはtimeout：in-flight outputを破棄し、serverを1回だけ再起動する。失敗画像はSSD cache pathから既存CPU経路へ送る。
- 再発時：当該tileのGPU処理を停止し、CPU継続する。
- CPU fallback失敗時：tile failureとしてtransaction rollbackする。
- requestには`server_session_id`、`batch_id`、`task_id`、`generation`を含め、古い応答・書き込みを破棄する。

### Phase 5：mask/DEM

画像direct DDSが受入された後、mask blurとDEM smoothingのrawファイル交換をshared buffer化する。初期shared-memory実装とは分離し、既存CPU/OpenCV fallbackを残す。

## メモリ制御

ユーザー設定は次とする。

```ini
enable_shared_memory_handoff=False
shared_memory_budget_gb=0
```

`shared_memory_budget_gb=0`は、物理メモリの8%を2GiB〜8GiBへclampした自動値とする。内部では`max_inflight_bytes`として扱う。

task見積もりは次とする。

```text
input_rgb + 2 × output_rgba + dds_capacity + mask_bytes + 25% safety margin
```

GPU batchは初期最大8画像とし、budgetが先に到達した場合はbatchを縮小する。allocation失敗時はbatchを半分にして1回だけ再試行し、単一taskでも確保できなければ既存CPUへfallbackする。

## 計測・ログ

`Ortho4XP_performance.json`へ次を追加する。

- `handoff_transport`
- `shared_alloc_bytes`
- `shared_copy_ms`
- `shared_wait_ms`
- `cache_write_ms`
- `cache_pending_bytes`
- `gpu_batch_effective`
- `shared_fallback`
- `shared_cleanup_failures`
- Python/ASHelper/tile全体のpeak RSS
- memory pressure events

通常ログはbatch要約と異常時のtask ID、transport、fallback理由に限定する。詳細なlifetime・queue・buffer情報はJSONへ保存する。

## テスト

- shared segmentのallocation、lease、unlink、stale cleanup
- memory budget backpressureとbatch縮小
- RGB8 descriptorの範囲・stride・shape検証
- DDS output容量不足の再試行
- task単位のpartial failureとCPU fallback
- ASHelper crash、timeout、restart、generation不一致
- cancel時のqueue排出とsegment解放
- cache-writer失敗とatomic cache保存
- duplicate task抑止
- shared DDSのmemory validation
- transaction rollback

実GPUとネットワークに依存しないfake ASHelperを使用する。Swift側はdescriptor/mapping/境界条件をfixtureで検証し、実Metal dispatchは`+34+133`の手動検証と分離する。

## 受入条件

- 旧経路比でall-in-one全体時間が5%以上悪化しない。
- shared-memoryはSSD streaming比でimagery/DSF stageが中央値5%以上改善し、全体時間も悪化しない場合だけ既定化候補とする。
- GPU成功はDDS生成成功ではなく、`effective_backend=metalfx_spatial`、`dispatch=direct_dds`、fallbackなしで判定する。
- DDS構造、DSF参照、画像枚数、欠落、重複、transaction残留を検証する。
- GPU/CPU画素差は同一fixtureから決定したMAE/RMSE/max_abs閾値で判定する。
- GPU非対応環境ではCPU/fallback合格とGPU未確認を分けて報告する。
