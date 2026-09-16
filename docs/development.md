# Development and Verification

## セットアップと起動

リポジトリのルートで実行する。Python依存関係の正本は `requirements.txt`、macOSの初期セットアップは `install_mac.sh` である。

```sh
./install_mac.sh
.venv/bin/python Ortho4XP.py
```

`Ortho4XP.py` を引数なしで起動するとGUI、`.venv/bin/python Ortho4XP.py <lat> <lon>` では既存タイル設定、`.venv/bin/python Ortho4XP.py <lat> <lon> <imagery> <zl>` では指定値でCLI処理を行う。繰り返し起動する場合は `source .venv/bin/activate` の後に `python Ortho4XP.py` としてもよい。

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
../.venv/bin/python run/configure.py mac release
../.venv/bin/python run/build.py mac release
```

`mac` は `lin` または `win` に置き換えられる。

`../.venv/bin/python run/install.py release` は `build/release/{lin,mac,win}/` の成果物を各 `Utils/` 配下へコピーするため、必要な対象をすべてビルドしてから実行する。

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

Metalを使えるApple Silicon MacでGPU経路を確認する場合は、決定的な画像・低解像度マスクを一時生成して専用ランナーを実行する。

```sh
./Utils/run/verify_metal.sh --keep-artifacts
```

MetalFX SpatialとCore Image Lanczosの2倍アップスケールをM5 Max上で比較する場合は、検証専用オプションを追加する。

```sh
./Utils/run/verify_metal.sh --compare-upscale --compare-runs 5 --keep-artifacts
```

この比較は同一の決定的な入力と高解像度の正解画像を使い、RGBのMAE/RMSE/PSNRと、ASHelperプロセス・画像入出力を含むmedian/p95時間を報告する。バックエンドの正規記録名は`ci_lanczos`、`metalfx_spatial`、`tensorops`で、旧`lanczos`と`fp8_tensorops`は互換aliasとして受け付ける。新規または未設定の`upscale_backend`は`metalfx_spatial`を既定値とし、既存の明示設定は変更しない。不透明RGBは直接MetalFXへ送り、直接プロバイダのJPEGはMetalFX batch後に色補正・マスク・DDS化する。RGBAはRGBをstraight-alphaのままMetalFXで処理し、alphaを`CIBicubicScaleTransform`で2倍化して再合成する。alpha分離・再合成に失敗した場合、または非対応・GPU実行失敗時はCore Image Lanczosへフォールバックする。

TensorOpsとMetalFXを同一条件で比較する標準レーンは、1回のウォームアップ後に指定回数（既定5回）を測定する。512px、2048px、4096pxの決定的なprovider形式入力について、PNGの品質ゲートとdirect DDSのサイズ、BC3、mipmap、effective backendを検証し、wall/user/sys時間、peak RSS、TensorOps/MetalFX・readback・DDSの内訳をJSONLへ記録する。

```sh
./Utils/run/verify_metal.sh --compare-tensorops --compare-runs 5 \
  --keep-artifacts --record-jsonl /private/tmp/ortho4xp-backend-comparison.jsonl
```

MetalFX出力の品質ゲートは同じ入力に対するFP8 TensorOps PNG出力を基準にし、PSNR低下0.25dB以内、MAE/RMSE増加5%以内とする。`--compare-tensorops`は実機向けの明示的な比較レーンなので、4096px direct DDSを含む実行時間とディスク使用量を見込んでから実行する。

MetalFXの単画像RGBA経路と順次batch経路は、次で確認できる。

```sh
./Utils/mac/ASHelper --metalfx-spatial-upscale input-rgba.png output.png
./Utils/mac/ASHelper --metalfx-spatial-upscale-batch \
  input-1.jpg output-1.png input-2.png output-2.png
```

ログの`backend`、`effective_backend`、`alpha_mode`、`dispatch`、`duration_ms`と、batchの`batch_tasks`、`batch_success`、`batch_fallback`を記録する。`verify_metal.py --compare-upscale`はopaque、RGBA、batchを測定する。MetalFXを利用できない環境でRGBA単画像を実行した場合も、ASHelper内のLanczos fallbackが成功すれば出力を残し、実際にMetalFX dispatchが発生したかはログの`effective_backend`で区別する。

直接プロバイダのopaque JPEGを実タイルで処理するときは、PNG中間ファイルを作らないdirect DDS経路を使用する。MetalFXは`--metalfx-spatial-dds-batch <request.json>`、TensorOpsは`--tensorops-dds-batch <request.json>`（`--fp8-tensorops-dds-batch`はalias）へ入力、マスク、色補正、BC1/BC3形式、`.gpu.tmp.dds`出力先を渡す。ASHelperがreadbackから既存DDS圧縮までを同一プロセス内で実行し、検証済みの一時DDSだけを`os.replace`で公開するため、途中失敗時に部分出力を公開しない。RGBA、ローカル合成、WebP、非対応providerは従来の経路を維持する。

direct DDS batchは既定2 worker、最大4 worker、1プロセス8画像chunkで実行する。verbosity 1では`MetalFX Spatial DDS batch: completed/total`、続けて`png_intermediate=false`、`metalfx_ms`、`readback_ms`、`dds_ms`、`temporary_bytes`、fallback理由を表示する。単独経路と性能を比較する場合は、同じ入力を次のように実行する。

TensorOps direct DDSは1プロセス8画像のchunkを単位にし、画像別activation/output、DDS一時payloadとmipmap、親プロセスRSS、物理メモリの70%を含む推定working setからworker数を動的に決める（最大4）。512px級は安全な範囲で並列化し、2048px以上は通常1 workerに制限する。物理メモリを取得できない場合も安全のため1 workerとする。chunk終了時にASHelperを終了するため、Metal/PNG/CGImageの一時リソースをプロセス境界で回収できる。ログには`batch_workers`、`batch_chunks`、`chunk_size=8`、`memory_budget_mb`、`parent_rss_mb`、`estimated_worker_mb`、`estimated_total_mb`、`peak_rss_mb`、`rss_after_item_mb`、`signal=9`（SIGKILL時）、`fallback_reasons`を記録する。TensorOps batchが失敗した場合は元JPEGを使って失敗画像だけをMetalFXへ再処理し、MetalFXも失敗した画像だけを`ci_lanczos`へ送る。成功済みDDSは再処理しない。`--tensorops-upscale`とそのbatchはPNG互換CLIとして残るが、実タイルのdirect DDSでは`png_intermediate=false`となり、`_tensorops_upscaled.png`や`tile_input.png`を生成しない。

```sh
cat >/private/tmp/metalfx-direct-dds.json <<'JSON'
{"version":1,"items":[{"input":"/path/input.jpg","mask":"none","output":"/private/tmp/output.gpu.tmp.dds","format":"BC1","color":{"r":1.0,"g":1.0,"b":1.0,"contrast":1.0,"brightness":0.0,"saturation":1.0}}]}
JSON
Utils/mac/ASHelper --metalfx-spatial-dds-batch /private/tmp/metalfx-direct-dds.json
```

受入時は、旧MetalFX PNG経路、新direct DDS経路、CI Lanczosをwall/user/sys時間、CPU使用率、peak RSS、PNG中間ファイルの有無、DDS検証結果で比較する。GPU処理の実証には`gpucapture`の`.gputrace`、`gpudebug --oneshot --json`、`metalperftrace`の成果物を別々に保存し、DDSの成功だけでMetalFX dispatchやNeural Accelerator使用を推測しない。

精度ラダーはFP16基準、FP8、FP4、INT2の順に実行する。FP8が画質ゲートを通過しない限りFP4以降は実行せず、FP4が通過しない限りINT2も実行しない。既定ゲートは、共通参照画像に対するFP16比でPSNR低下0.25dB以内、MAE/RMSE増加5%以内、2倍サイズ、有限値、継ぎ目なしである。

```sh
./Utils/run/verify_metal.sh --precision-ladder \
  --fp16-pack /path/to/fp16.fp8sr \
  --fp8-pack /path/to/fp8.fp8sr \
  --fp4-pack /path/to/fp4.fp8sr \
  --int2-pack /path/to/int2.fp8sr \
  --record-jsonl /private/tmp/ortho4xp-execution.jsonl \
  --gpu-tools --keep-artifacts
```

FP16/FP4/INT2の決定的パックは次のように生成できる。FP8SR v1は従来どおり受け付け、非FP8 dtypeはv2パックとして扱う。

```sh
.venv/bin/python tools/fp8sr_pack.py --create-fixture /private/tmp/fp4sr --dtype MetalFloat4E2M1
.venv/bin/python tools/fp8sr_pack.py --create-fixture /private/tmp/int2sr --dtype Int2
```

`--gpu-tools`を指定した場合だけ、`gpucapture`で`.gputrace`を作成し、`gpudebug --oneshot --json`でcompute dispatchを確認する。さらにM5対応GPUでは`gpudebug profile run --gpu-state high --exec serial`で性能カウンタを収集し、Neural Accelerator utilizationが0より大きい場合だけ`neural_accelerator_confirmed=true`としてprofile証跡を保存する。`metalperftrace collect/overview`の成果物も保存する。ツール不在、capturable process不在、Metal layerの記録なし、profile非対応は理由付き`SKIP`となる。TensorOps dispatchの存在だけではNeural Accelerator使用の証明にならない。

この検証はMetalデバイス、Core ImageのMetalコンテキスト、ASHelperの直接変換、`--convert-batch-v3` の64件並列変換、DDSのヘッダ・Mip数・マスク透明度、色補正の作用、DDS書き込み失敗時の終了コードを確認する。`--keep-artifacts` を省略すると、成功時の生成物は終了時に削除される。失敗時は調査用に生成物を残し、出力された `kept_artifacts` を確認できる。Metal対応ホストでも実行プロセスのサンドボックスからデバイスが見えない場合があり、その場合は `metal_host_supported=true` と表示されるため、ホストのターミナルなど隔離されていないCLIから再実行する。MetalデバイスがないMacではCPU/fallbackの確認だけを行い、GPU固有の判定はスキップする。実データのタイル生成・GUI操作は既存の手動確認範囲であり、このランナーには含めない。

FP8 TensorOpsの固定契約と外部モデルパックは、[FP8SRパック仕様](fp8sr-pack.md)に従う。パックの形式検証と決定的フィクスチャ生成は次で行う。

```sh
.venv/bin/python tools/fp8sr_pack.py --validate /path/to/model.fp8sr
.venv/bin/python tools/fp8sr_pack.py --create-fixture /private/tmp/ortho4xp-fp8-fixture
```

実用モデルの学習は任意の`requirements-train.txt`を使う。LR/HRの相対パス対応ペアからFP16モデルを学習し、validationでLanczos以上になった場合だけFP8 E4M3へ量子化する。学習、FP16SR v2パック生成、FP8SR v1パック生成、品質検証を一括で行う場合は次を使う。

```sh
.venv/bin/python -m pip install -r requirements-train.txt
.venv/bin/python tools/train_fp8sr.py run \
  --dataset /path/to/aerial-pairs \
  --output-dir /private/tmp/ortho4xp-fp8sr-training \
  --device auto \
  --gpu-tools \
  --record-jsonl /private/tmp/ortho4xp-fp8sr-training/execution.jsonl
```

学習成果物は外部ディレクトリへ出力し、PyTorchは通常のOrtho4XP起動経路へ追加しない。小規模な実データ学習とM5 Max実機TensorOps dispatchを最終受入条件とする。

macOS 27、FP8 TensorOps対応Apple Silicon Macでは、ASHelperを再ビルドした後に単画像・batchの実行を確認する。次のコマンドは入力8x8のフィクスチャから16x16 PNGを生成する。

```sh
./Utils/run/build_ashelper.sh
Utils/mac/ASHelper --tensorops-upscale \
  /private/tmp/ortho4xp-fp8-fixture \
  /private/tmp/ortho4xp-fp8-fixture/input.png \
  /private/tmp/ortho4xp-fp8-fixture/output.png
Utils/mac/ASHelper --tensorops-upscale-batch \
  /private/tmp/ortho4xp-fp8-fixture \
  /private/tmp/ortho4xp-fp8-fixture/input.png \
  /private/tmp/ortho4xp-fp8-fixture/batch-output.png
```

`tensorops_dispatch=ready`はTensorOpsパイプライン初期化、`tensorops_dispatch=completed`は画像出力までの完了を示す。これはGPU dispatchの実行証拠であり、Neural Acceleratorの使用証明ではない。`--gpu-tools`のprofile結果、またはXcode GPU traceのNeural Acceleratorカウンタで別途確認する。現行のXcode環境で`xcrun metal`がMetal Toolchain不足を報告する場合、Swift側のランタイムコンパイル確認とGPU traceの確認は未実行として分けて報告する。

TensorOpsの入力幅または高さが2048pxを超える場合、ASHelperは2048px以下の中心領域と1px haloへ自動分割する。4096x4096は4タイルを処理し、各タイルの中心2倍領域だけを最終画像へコピーする。`tensorops_dispatch=tiled`、`tile_count`、`tile_core_size`、`tile_input_sizes`、`tile_halo`は実行記録へ保存される。タイルのGPU失敗・非有限値・出力サイズ不正は部分出力を採用せず、上位の既存Lanczosフォールバックへ渡す。4GiB相当のアドレス境界は公式Metal上限ではなく、M5 Max/macOS 27で観測した単一ディスパッチの安全運用上の閾値として扱う。

構文確認、ビルド、限定的なスクリプト実行だけでは、実際のProvider応答、長時間のタイル生成、GUI操作、利用者データへの影響まで保証しない。未実行の範囲を最終報告に明記する。
