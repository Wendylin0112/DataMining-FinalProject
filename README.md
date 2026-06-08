# Power-Line Asset Defect Classification

這是一個電力線路資產巡檢影像分類專案。模型會先辨識照片中的設備類別，再判斷該設備是否為正常或有缺失。

線上 Demo：

https://datamining-finalproject-jwxzxr4njtsox4z9quebt7.streamlit.app/

```text
normal / good   -> 0
defective / bad -> 1
```

支援的設備類別：

```text
vari-grip
lightning-rod-suspension
polymer-insulator-upper-shackle
glass-insulator
yoke-suspension
```

## 快速使用

如果只是想使用模型，不需要下載專案，也不需要安裝 Python。

1. 開啟線上 Demo：

```text
https://datamining-finalproject-jwxzxr4njtsox4z9quebt7.streamlit.app/
```

2. 上傳一張或多張巡檢圖片，支援 `jpg`、`jpeg`、`png`。

3. 網頁會輸出：

```text
設備類別
defect score
threshold
normal / defective 預測結果
component probabilities
```

4. 若一次上傳多張圖片，可以用左側的圖片下拉選單或左右箭頭按鈕切換目前預覽的照片；右側結果表格可以勾選並移除照片，也可以下載 CSV 保存批次推論結果。

預測結果中：

```text
NORMAL / good (0)      代表模型判斷為正常
DEFECTIVE / bad (1)    代表模型判斷為有缺失
```

## 本機執行

如果想下載到本地端執行，可以照以下步驟。

### 1. 下載專案

```powershell
git clone <your-repository-url>
cd DataMining
```

請確認專案根目錄有以下檔案與資料夾：

```text
app.py
requirements.txt
hybrid_deep_models_p097/
```

其中 `hybrid_deep_models_p097/` 是 App 會使用的最終模型權重資料夾。

必要內容如下：

```text
hybrid_deep_models_p097/
  metadata.json
  component_model.pt
  defect_global.pt
  defect_vari-grip.pt
  defect_glass-insulator.pt
  defect_yoke-suspension.pt
```

### 2. 建立 Python 環境

本專案示範使用的 conda 環境名稱為 `powerline_defect`。環境名稱可以自行調整；如果你已經有可用的 Python / CUDA / PyTorch 環境，也可以直接使用自己的環境執行。

```powershell
conda create -n powerline_defect python=3.11 -y
conda activate powerline_defect
```

安裝套件：

```powershell
pip install -r requirements.txt
```

CUDA / GPU 說明：

- 使用 NVIDIA GPU 時，請依照自己的 GPU、驅動版本與作業系統，在目前啟用的虛擬環境中安裝相容的 CUDA 版 `torch` / `torchvision`。
- 如果本機環境已經安裝可用的 CUDA 與 PyTorch，可以使用原本的環境，不一定要建立 `powerline_defect`。
- 如果沒有 GPU 或不需要 GPU，也可以使用 CPU 版 PyTorch，但訓練速度會較慢。
- PyTorch / CUDA 安裝指令可能隨版本調整，建議以 PyTorch 官方網站產生的安裝指令為準：https://pytorch.org/get-started/locally/

`requirements.txt` 主要包含：

```text
streamlit
torch
torchvision
pandas
pillow
scikit-learn
joblib
```

### 3. 啟動 App

```powershell
streamlit run app.py
```

或使用 conda run：

```powershell
conda run --no-capture-output -n powerline_defect python -m streamlit run app.py
```

啟動後會在瀏覽器開啟本機網址，通常是：

```text
http://localhost:8501
```

## 專案主入口

```text
app.py
  Streamlit 本機與雲端 App 入口。
  使用 final hybrid deep model，讓使用者上傳圖片並取得預測結果。

train_hybrid_deep_classifier.py
  訓練 hybrid deep model 的主程式。
  會訓練設備類別分類器、global defect classifier，以及部分設備類別的 specialist defect classifiers。

adjust_submission_thresholds.py
  根據 detailed prediction CSV 微調各設備類別的 defect threshold。
  最終 submission 使用這個腳本進行 component-level threshold post-processing。

Final Testing Results/
  保存正式提交結果、評分 JSON、最終 submission 備份與成績摘要。
```

## 模型方法

本專案最終採用 hybrid deep learning 架構，而不是單一二分類模型。

第一階段：設備類別分類。

```text
input image -> EfficientNet-B0 -> 5-class component classifier
```

第二階段：根據預測出的設備類別，選擇對應的 defect route。

```text
predicted component -> select defect route -> normal / defective
```

Defect route 設計如下：

```text
vari-grip
  使用 vari-grip defect specialist

glass-insulator
  使用 glass-insulator defect specialist

yoke-suspension
  使用 yoke-suspension defect specialist

polymer-insulator-upper-shackle
  bad 資料量較少，使用 global defect classifier

lightning-rod-suspension
  bad 資料量太少，使用 global defect classifier
```

這樣設計的原因是不同設備類別的缺失型態不同。資料量足夠的類別適合訓練各自的 specialist defect model；資料量不足的類別如果單獨訓練容易 overfit，因此使用 global defect classifier 會比較穩定。

## 訓練設定

最終模型使用 torchvision 的 ImageNet pretrained EfficientNet-B0 作為 backbone。

```text
Backbone:
  EfficientNet-B0

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
  final submission 使用 component-level threshold post-processing
```

資料增強包含：

```text
Resize
RandomResizedCrop
RandomHorizontalFlip
RandomRotation
ColorJitter
ImageNet Normalize
```

最終 p097 base model 的訓練回合：

```text
component classifier              6 epochs
global defect classifier          6 epochs
vari-grip defect specialist       6 epochs
glass-insulator defect specialist 6 epochs
yoke-suspension defect specialist 6 epochs
```

總計約 30 model-epochs。在 NVIDIA GeForce RTX 3050 4GB Laptop GPU 上，完整訓練大約需要 40 分鐘左右。實際時間會受 GPU、CPU、batch size、num_workers、硬碟速度與是否已下載 pretrained weights 影響。

## 專案結構

建議的完整結構如下：

```text
DataMining/
  README.md
  requirements.txt
  .gitignore

  app.py
  train_hybrid_deep_classifier.py
  adjust_submission_thresholds.py

  test_submission_template.csv
  test_submission_hybrid_deep_p097_comp_balanced.csv
  test_predictions_hybrid_deep_p097_comp_balanced_detailed.csv

  hybrid_deep_models_p097/
    metadata.json
    metadata.joblib
    component_model.pt
    defect_global.pt
    defect_vari-grip.pt
    defect_glass-insulator.pt
    defect_yoke-suspension.pt

  app_examples/
    glass-insulator-good.jpg
    glass-insulator-bad.jpg
    lightning-rod-suspension-good.jpg
    lightning-rod-suspension-bad.jpg
    polymer-insulator-upper-shackle-good.jpg
    polymer-insulator-upper-shackle-bad.jpg
    vari-grip-good.jpg
    vari-grip-bad.jpg
    yoke-suspension-good.jpg
    yoke-suspension-bad.jpg

  Final Testing Results/
    README.md
    results_summary.csv
    final_submission.csv
    第一次.json
    第二次.json
    第三次.json
    第一組_test_version1 (hybrid_deep_p097).csv
    第一組_test_version2 (hybrid_deep_p097_comp_balanced).csv
```

如果需要重新訓練，還需要另外準備資料集：

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
```

`train_dataset/` 和 `test_dataset/` 不會放進 GitHub，避免 repository 過大。`hybrid_deep_models_p097/` 是 final App 需要的模型權重，因此本專案會保留在 GitHub 中，讓使用者 clone 後可以直接執行 App。

## 重新訓練

重現最終 p097 base model 的訓練指令：

```powershell
conda run --no-capture-output -n powerline_defect python -u train_hybrid_deep_classifier.py --batch-size 12 --num-workers 0 --specialist-min-bad 80 --specialist-min-good 100 --defect-epochs 6 --lr-finetune 3e-5 --target-defect-precision 0.97 --binary-sampler label --defect-pos-weight-multiplier 1.0 --log-every 25 --output test_submission_hybrid_deep_p097.csv --detailed-output test_predictions_hybrid_deep_p097_detailed.csv --model-dir hybrid_deep_models_p097
```

`python -u` 與 `--log-every 25` 會讓訓練過程即時輸出目前訓練到哪一個模型、哪一個 epoch，以及 batch 進度。

訓練完成後會產生：

```text
hybrid_deep_models_p097/
test_submission_hybrid_deep_p097.csv
test_predictions_hybrid_deep_p097_detailed.csv
```

## 重現最終 Submission

最終提交不是直接使用 p097 base output，而是在 detailed prediction 上做 component-level threshold post-processing。

```powershell
python adjust_submission_thresholds.py --detailed test_predictions_hybrid_deep_p097_detailed.csv --output test_submission_hybrid_deep_p097_comp_balanced.csv --detailed-output test_predictions_hybrid_deep_p097_comp_balanced_detailed.csv --component-override lightning-rod-suspension=0.40 --component-override polymer-insulator-upper-shackle=0.50 --component-override yoke-suspension=0.55
```

最終採用的 submission：

```text
test_submission_hybrid_deep_p097_comp_balanced.csv
```

## 最終成績

正式提交結果保存在：

```text
Final Testing Results/第二次.json
```

最終成績：

```text
overall accuracy: 0.9455
precision:        0.9206
recall:           0.8756
f1-score:         0.8975
```

Confusion matrix：

```text
tn: 583
fp: 17
fn: 28
tp: 197
```

三次提交比較：

```text
第一次 p097:
  accuracy  0.9188
  precision 0.9341
  recall    0.7556
  f1        0.8354

第二次 p097_comp_balanced:
  accuracy  0.9455
  precision 0.9206
  recall    0.8756
  f1        0.8975

第三次:
  accuracy  0.9455
  precision 0.9369
  recall    0.8578
  f1        0.8956
```

第二次的 `p097_comp_balanced` 在 precision 維持 0.90 以上的情況下有最高的 recall 與 f1，因此選為最終模型與最終 submission。

## Streamlit Cloud 部署

目前 App 已部署在 Streamlit Community Cloud：

```text
https://datamining-finalproject-jwxzxr4njtsox4z9quebt7.streamlit.app/
```

部署需要 GitHub repository 中包含：

```text
app.py
requirements.txt
hybrid_deep_models_p097/
app_examples/
```

Streamlit Cloud 會自動讀取 `requirements.txt` 安裝套件，並以 `app.py` 作為 main file 執行。

## GitHub 注意事項

`.gitignore` 已排除大型或非必要檔案，但特別保留 final App 需要的 `hybrid_deep_models_p097/`：

```text
train_dataset/
test_dataset/
*.pt
*.pth
*.joblib
hybrid_deep_models*/
!hybrid_deep_models_p097/
!hybrid_deep_models_p097/**
csv_history/
__pycache__/
```

GitHub repo 建議保留：

```text
README.md
requirements.txt
app.py
train_hybrid_deep_classifier.py
adjust_submission_thresholds.py
test_submission_template.csv
test_submission_hybrid_deep_p097_comp_balanced.csv
test_predictions_hybrid_deep_p097_comp_balanced_detailed.csv
hybrid_deep_models_p097/
Final Testing Results/
```
