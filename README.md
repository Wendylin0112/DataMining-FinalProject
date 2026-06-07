# Power-Line Asset Defect Classification

本專案使用高解析度無人機彩色影像進行電力線路資產巡檢分類。最終流程會先判斷影像中的設備類別，再判斷該設備是否為正常或缺陷。

標籤定義：

```text
normal / good     -> 0
defective / bad   -> 1
```

設備類別：

```text
vari-grip
lightning-rod-suspension
polymer-insulator-upper-shackle
glass-insulator
yoke-suspension
```

## Final Model

最終提交版本為：

```text
test_submission_hybrid_deep_p097_comp_balanced.csv
```

這份提交是目前最佳版本，對應 `Final Testing Results/第二次.json`。

Final testing score：

```text
overall accuracy:  0.9455
precision:         0.9206
recall:            0.8756
f1-score:          0.8975
```

Confusion matrix：

```text
tn: 583
fp: 17
fn: 28
tp: 197
```

相較第一次提交，最終版本主要提升 recall 與 f1：

```text
第一次 p097:
  accuracy  0.9188
  precision 0.9341
  recall    0.7556
  f1        0.8354

最終 comp_balanced:
  accuracy  0.9455
  precision 0.9206
  recall    0.8756
  f1        0.8975
```

## Dataset

程式預期資料夾結構如下：

```text
train_dataset/
  vari-grip/
    good/
    bad/
  lightning-rod-suspension/
    good/
    bad/
  polymer-insulator-upper-shackle/
    good/
    bad/
  glass-insulator/
    good/
    bad/
  yoke-suspension/
    good/
    bad/

test_dataset/
  images/

test_submission_template.csv
```

訓練資料分布：

```text
component                         good  bad
vari-grip                          358  638
lightning-rod-suspension           459   20
polymer-insulator-upper-shackle    653   60
glass-insulator                    600  675
yoke-suspension                    299  120
```

## Method

最終方法分成兩個階段。

第一階段是 component classifier：

```text
input image -> EfficientNet-B0 -> 5-class component prediction
```

第二階段是 hybrid defect classifier：

```text
predicted component -> select defect route -> output normal / defective
```

Hybrid defect route：

```text
vari-grip:
  使用 vari-grip defect specialist

glass-insulator:
  使用 glass-insulator defect specialist

yoke-suspension:
  使用 yoke-suspension defect specialist

polymer-insulator-upper-shackle:
  bad 樣本較少，使用 global defect classifier

lightning-rod-suspension:
  bad 樣本非常少，使用 global defect classifier
```

這樣設計的原因是 `polymer-insulator-upper-shackle` 與 `lightning-rod-suspension` 的 bad 樣本數較少，若單獨訓練 specialist 容易 overfit；資料量較充足的三個類別則使用各自的 specialist model。

## Model Details

主要模型設定：

```text
Backbone:
  torchvision EfficientNet-B0

Weights:
  ImageNet pretrained weights

Component task:
  5-class classification

Defect task:
  binary classification

Loss:
  component classifier -> CrossEntropyLoss + class weights
  defect classifier    -> BCEWithLogitsLoss + pos_weight

Sampling:
  WeightedRandomSampler

Threshold:
  validation-based route threshold tuning
  final submission additionally uses component-level threshold post-processing
```

資料增強：

```text
Resize
RandomResizedCrop
RandomHorizontalFlip
RandomRotation
ColorJitter
ImageNet Normalize
```

## Environment

已確認可用的 conda environment：

```text
aicup_cuda
```

GPU / package：

```text
torch 2.11.0+cu128
torchvision 0.26.0+cu128
GPU: NVIDIA GeForce RTX 3050 4GB Laptop GPU
```

## Reproduce Final Submission

最終提交檔是由兩步產生。

Step 1. 訓練 p097 base model：

```powershell
conda run --no-capture-output -n aicup_cuda python -u train_hybrid_deep_classifier.py --batch-size 12 --num-workers 0 --specialist-min-bad 80 --specialist-min-good 100 --defect-epochs 6 --lr-finetune 3e-5 --target-defect-precision 0.97 --binary-sampler label --defect-pos-weight-multiplier 1.0 --log-every 25 --output test_submission_hybrid_deep_p097.csv --detailed-output test_predictions_hybrid_deep_p097_detailed.csv --model-dir hybrid_deep_models_p097
```

Step 2. 對低 recall 類別做 component-level threshold post-processing：

```powershell
python adjust_submission_thresholds.py --detailed test_predictions_hybrid_deep_p097_detailed.csv --output test_submission_hybrid_deep_p097_comp_balanced.csv --detailed-output test_predictions_hybrid_deep_p097_comp_balanced_detailed.csv --component-override lightning-rod-suspension=0.40 --component-override polymer-insulator-upper-shackle=0.50 --component-override yoke-suspension=0.55
```

這個後處理主要針對第一次提交中 recall 偏低但 precision 仍高的類別：

```text
lightning-rod-suspension
polymer-insulator-upper-shackle
yoke-suspension
```

## Training Epochs

最終 base model 訓練回合：

```text
component classifier:
  6 epochs

global defect classifier:
  6 epochs

each defect specialist:
  6 epochs
```

本次符合 specialist 條件的類別：

```text
vari-grip
glass-insulator
yoke-suspension
```

因此共訓練五個模型：

```text
1. component classifier              6 epochs
2. global defect classifier          6 epochs
3. vari-grip defect specialist       6 epochs
4. glass-insulator defect specialist 6 epochs
5. yoke-suspension defect specialist 6 epochs
```

合計：

```text
30 model-epochs
```

在 RTX 3050 4GB Laptop GPU 上完整訓練約 40 分鐘左右，實際時間會受 GPU、CPU、磁碟、batch size、num_workers 與 pretrained weights 是否已快取影響。

## Output Files

核心程式：

```text
app.py
train_hybrid_deep_classifier.py
adjust_submission_thresholds.py
```

最終模型與輸出：

```text
hybrid_deep_models_p097/
  component_model.pt
  defect_global.pt
  defect_vari-grip.pt
  defect_glass-insulator.pt
  defect_yoke-suspension.pt
  metadata.json
  metadata.joblib

test_submission_hybrid_deep_p097_comp_balanced.csv
test_predictions_hybrid_deep_p097_comp_balanced_detailed.csv
```

其他非最終版本的 submission 與 detailed prediction CSV 已整理到：

```text
csv_history/
```

最終測試結果整理於：

```text
Final Testing Results/
```

其中 `第二次.json` 是最終版本的 scoring result。

## Local Web App

`app.py` 是 Streamlit local web app，已改成載入最終 hybrid deep model。

使用的模型資料夾：

```text
hybrid_deep_models_p097/
```

此資料夾包含 `.pt` model weights，因檔案較大已被 `.gitignore` 排除；若從 GitHub clone 專案，需要自行重新訓練或複製此資料夾到專案根目錄。

若 `aicup_cuda` 尚未安裝 Streamlit：

```powershell
conda run --no-capture-output -n aicup_cuda python -m pip install streamlit
```

啟動 app：

```powershell
conda run --no-capture-output -n aicup_cuda python -m streamlit run app.py
```

App 會先使用 component classifier 判斷設備類別，再依據 component 選擇 global defect model 或 specialist defect model。預設 threshold mode 使用最終提交 `test_submission_hybrid_deep_p097_comp_balanced.csv` 的 component-level thresholds。
