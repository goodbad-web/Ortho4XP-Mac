# FP8SR TensorOpsモデルパック

Ortho4XPの`TensorOps`は、外部の`FP8SR`パックを使う2倍RGB超解像バックエンドです。重みはリポジトリへ同梱せず、`fp8_model_path`へパックのディレクトリを指定します。通常の起動経路はCore MLへ依存しません。既存の`FP8SR` v1はFP8 E4M3固定で、FP16/FP4/INT2の実験用バリアントはv2として扱います。

## 固定契約

パック直下に`manifest.json`と、manifestから参照される以下のファイルを置きます。

- `format`: `FP8SR`
- `version`: `1`（FP8 E4M3互換）または`2`（精度バリアント）
- `upscale_factor`: `2`
- `layout`: `NHWC`
- 入力/出力: RGB 3チャンネル
- `weight_dtype`: `Float16`、`MetalFloat8E4M3`、`MetalFloat4E2M1`、`Int2`
- `activation_dtype`: `Float16`
- `accumulation_dtype`: `Float16`
- `weight_row_stride_bytes`: `128`
- グラフ: `3x3: 3→32→32→12 + PixelShuffle2`

各層は`name`、`kernel`、`in_channels`、`out_channels`、`weights`、`bias`、`scale`を持ちます。重みはdtypeごとの生バイト列で、論理的なK方向を32要素境界へパディングし、各K行のストライドを128バイトにします。INT2はsigned 2-bit、FP4はMetal E2M1のpacked表現です。biasは出力チャンネルごとのリトルエンディアンFP16で、scaleは有限な正数でなければなりません。現在の実行時は`conv0`（3→32）、`conv1`（32→32）、`conv2`（32→12）以外を受け付けません。

検証と決定的な小型フィクスチャの生成には次を使えます。

```sh
.venv/bin/python tools/fp8sr_pack.py --validate /path/to/model.fp8sr
.venv/bin/python tools/fp8sr_pack.py --create-fixture /private/tmp/ortho4xp-fp8-fixture
```

## CLIとフォールバック

単画像は次の形式です。

```sh
Utils/mac/ASHelper --tensorops-upscale \
  /path/to/model.fp8sr input.png output.png
```

複数画像は同じプロセスへ入力/出力ペアを渡します。モデルパックの検証・Metalライブラリ・TensorOpsパイプラインはプロセス内で一度だけ初期化されます。

```sh
Utils/mac/ASHelper --tensorops-upscale-batch \
  /path/to/model.fp8sr input-1.png output-1.png input-2.png output-2.png
```

Ortho4XPで`upscale_backend=tensorops`を選んだ場合、macOS 27未満、TensorOps非対応、パック不在/不正、透明入力、GPU実行失敗、非有限値、出力サイズ不正ではCore Image Lanczosへフォールバックします。通常のタイル処理では、条件を満たす直接JPEGだけをbatch経路へ集約します。マスク、色補正、結合プロバイダ、高ズームの前処理が必要な画像は個別経路を使います。旧`upscale_backend=fp8_tensorops`と旧CLIは互換aliasです。

ログには少なくとも次の実行証拠を出します。

```text
tensorops_dispatch=ready dtype=MetalFloat8E4M3 activation=Float16 accumulation=Float16 ...
tensorops_dispatch=completed dtype=MetalFloat8E4M3 activation=Float16 accumulation=Float16 output=...
```

これはTensorOps dispatchの証拠であり、Neural Acceleratorの実使用を意味しません。Neural Acceleratorの最終確認はXcode GPU traceで別途行います。

速度・画質の検証は次で実行できます。`--fp8-pack`を省略すると一時ディレクトリへ決定的な小型パックを生成します。

```sh
./Utils/run/verify_metal.sh --compare-fp8 --compare-runs 5
```

Core MLのコンパイル済み参照モデル（`.mlmodelc`）を明示した場合だけ、検証ランナーが一時的なCore MLヘルパーをビルドし、その出力をFP8SR出力の画質基準にします。通常のASHelperとOrtho4XPプロセスはCore MLへ依存しません。

```sh
./Utils/run/verify_metal.sh --compare-fp8 \
  --fp8-pack /path/to/model.fp8sr \
  --coreml-reference-model /path/to/reference.mlmodelc
```

## Core ML参照モデル

Core MLは参照品質の検証ランナーから明示的に指定する場合だけ使用する想定です。Core MLモデルがFP8であってもNeural Engine実行は保証されないため、Core ML実行をFP8高速化の証拠にはしません。通常のOrtho4XP実行へCore ML依存を追加しない方針です。

## FP16学習からFP8SR生成

実用モデルを作る場合は、通常のOrtho4XP依存へPyTorchを追加せず、学習用依存だけを導入します。

```sh
.venv/bin/python -m pip install -r requirements-train.txt
```

学習データはLR/HRの相対パス対応ペアで、HRがLRの正確な2倍である必要があります。

```text
dataset/
  train/lr/
  train/hr/
  val/lr/
  val/hr/
```

学習・FP16パック生成・FP8量子化・validation検証を個別に実行できます。

```sh
.venv/bin/python tools/train_fp8sr.py train \
  --dataset /path/to/aerial-pairs \
  --output-dir /private/tmp/ortho4xp-fp8sr-training \
  --device auto

.venv/bin/python tools/train_fp8sr.py export-fp16 \
  --checkpoint /private/tmp/ortho4xp-fp8sr-training/fp16_best.safetensors \
  --output-pack /private/tmp/ortho4xp-fp8sr-training/fp16-pack

.venv/bin/python tools/train_fp8sr.py quantize-fp8 \
  --checkpoint /private/tmp/ortho4xp-fp8sr-training/fp16_best.safetensors \
  --output-pack /private/tmp/ortho4xp-fp8sr-training/fp8-pack

.venv/bin/python tools/train_fp8sr.py verify \
  --dataset /path/to/aerial-pairs \
  --fp16-pack /private/tmp/ortho4xp-fp8sr-training/fp16-pack \
  --fp8-pack /private/tmp/ortho4xp-fp8sr-training/fp8-pack \
  --output-dir /private/tmp/ortho4xp-fp8sr-training/verification \
  --record-jsonl /private/tmp/ortho4xp-fp8sr-training/execution.jsonl \
  --gpu-tools
```

`run`サブコマンドでは同じ処理を一括実行できます。FP16のvalidation品質がLanczos以上でない場合はFP8量子化を停止します。FP8はFP16比でPSNR低下0.25 dB以内、MAE/RMSE増加5%以内、2倍サイズ、有限値を満たした場合だけ採用候補になります。FP8不合格でも候補パックと検証結果は保存されますが、本番設定へ自動採用しません。

FP8量子化は層単位の対称PTQです。重みはE4M3、活性値・累積・biasはFP16のままで、各層のscaleをmanifestへ保存します。学習済み重みとパックはリポジトリへ同梱せず、出力ディレクトリとSHA-256を実行記録で管理します。
