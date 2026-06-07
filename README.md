# Power-Line Asset Defect Classification

這個專案用於高解析度無人機電力線路巡檢影像的資產類別分類與缺失判斷。模型會先判斷照片中的設備類別，再判斷是否為 `normal` 或 `defective`。

```text
normal / good   -> 0
defective / bad -> 1
```

支援的 5 個設備類別：

```text
vari-grip
lightning-rod-suspension
polymer-insulator-upper-shackle
glass-insulator
yoke-suspension
```

## 主入口

這份專案最重要的入口如下：

```text
app.py
  Streamlit 本機網頁 App。給使用者上傳圖片，直接輸出設備類別、defect 分數與 normal/defective 結果。

train_hybrid_deep_classifier.py
  最終 hybrid deep model 的訓練與 test set 推論主程式。

adjust_submission_thresholds.py
  根據 detailed prediction CSV 微調各設備類別的 defect threshold，輸出最終 submission CSV。

Final Testing Results/
  保存正式提交結果、評分 JSON、最終 submission 備份與成績摘要。
```

目前最終採用的 submission：

```text
test_submission_hybrid_deep_p097_comp_balanced.csv
```

對應的 detailed prediction：

```text
test_predictions_hybrid_deep_p097_comp_balanced_detailed.csv
```

## 專案結構

完整可使用的資料夾建議如下。

```text
DataMining/
  README.md
  .gitignore

  app.py
  train_hybrid_deep_classifier.py
  adjust_submission_thresholds.py

  test_submission_template.csv
  test_submission_hybrid_deep_p097_comp_balanced.csv
  test_predictions_hybrid_deep_p097_comp_balanced_detailed.csv

  Final Testing Results/
    README.md
    results_summary.csv
    final_submission.csv
    第一次.json
    第二次.json
    第三次.json
    第一組_test_version1 (hybrid_deep_p097).csv
    第一組_test_version2 (hybrid_deep_p097_comp_balanced).csv

  hybrid_deep_models_p097/
    metadata.json
    component_model.pt
    defect_global.pt
    defect_vari-grip.pt
    defect_glass-insulator.pt
    defect_yoke-suspension.pt

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

`hybrid_deep_models_p097/`、`train_dataset/`、`test_dataset/` 通常不會放進 GitHub，因為模型權重與資料集檔案較大。這些資料夾已經由 `.gitignore` 排除；如果其他人 clone 專案後要執行 App，需要另外取得 `hybrid_deep_models_p097/`，或是自行重新訓練產生。

## 安裝環境

建議使用 conda 建立獨立環境。以下環境名稱沿用本專案開發時使用的 `aicup_cuda`。

```powershell
conda create -n aicup_cuda python=3.11 -y
conda activate aicup_cuda
```

安裝 PyTorch 時請依照自己的 CUDA 版本選擇官方指令。本專案開發與訓練時使用 GPU：

```text
NVIDIA GeForce RTX 3050 4GB Laptop GPU
```

本機曾使用的主要套件版本：

```text
torch 2.11.0+cu128
torchvision 0.26.0+cu128
```

若環境中已經有可用的 PyTorch / torchvision，只需要補上 App 與資料處理套件：

```powershell
pip install streamlit pandas pillow scikit-learn joblib
```

若需要重新安裝 PyTorch，請到 PyTorch 官方網站選擇符合 CUDA 版本的安裝指令：

```text
https://pytorch.org/get-started/locally/
```

## 執行本機網頁 App

App 入口是：

```text
app.py
```

執行前請確認根目錄下有最終模型資料夾：

```text
hybrid_deep_models_p097/
  metadata.json
  component_model.pt
  defect_global.pt
  defect_vari-grip.pt
  defect_glass-insulator.pt
  defect_yoke-suspension.pt
```

啟動 App：

```powershell
conda run --no-capture-output -n aicup_cuda python -m streamlit run app.py
```

啟動後瀏覽器會開啟 Streamlit 頁面。使用者可以上傳 JPG / PNG 圖片，App 會輸出：

```text
設備類別
defect score
threshold
normal / defective 預測
component probabilities
```

App 預設使用與最終 submission 相同的 component-level thresholds：

```text
lightning-rod-suspension           -> 0.40
polymer-insulator-upper-shackle    -> 0.50
yoke-suspension                    -> 0.55
其他 route                         -> 使用 metadata.json 中保存的 validation route threshold
```

## 模型方法

最終模型採用 hybrid deep learning 流程，不是單一二分類模型。

第一階段是設備類別分類：

```text
input image -> EfficientNet-B0 -> 5-class component classifier
```

第二階段是 defect 判斷：

```text
predicted component -> 選擇對應 defect route -> normal / defective
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

這樣設計的原因是每個設備類別的缺失型態不同。資料量足夠的類別適合訓練自己的 specialist defect model；資料量不足的類別如果單獨訓練容易 overfit，因此使用 global defect classifier 會比較穩定。

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
  final submission 再做 component-level threshold post-processing
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

總計約 30 model-epochs。在 RTX 3050 4GB Laptop GPU 上，完整訓練大約需要 40 分鐘左右，實際時間會受 batch size、num_workers、GPU、CPU、硬碟速度與是否已下載 pretrained weights 影響。

## 重新訓練

訓練前請確認資料集結構如下：

```text
train_dataset/
  vari-grip/good/
  vari-grip/bad/
  lightning-rod-suspension/good/
  lightning-rod-suspension/bad/
  polymer-insulator-upper-shackle/good/
  polymer-insulator-upper-shackle/bad/
  glass-insulator/good/
  glass-insulator/bad/
  yoke-suspension/good/
  yoke-suspension/bad/

test_dataset/
  images/

test_submission_template.csv
```

重現最終 p097 base model 的訓練指令：

```powershell
conda run --no-capture-output -n aicup_cuda python -u train_hybrid_deep_classifier.py --batch-size 12 --num-workers 0 --specialist-min-bad 80 --specialist-min-good 100 --defect-epochs 6 --lr-finetune 3e-5 --target-defect-precision 0.97 --binary-sampler label --defect-pos-weight-multiplier 1.0 --log-every 25 --output test_submission_hybrid_deep_p097.csv --detailed-output test_predictions_hybrid_deep_p097_detailed.csv --model-dir hybrid_deep_models_p097
```

`python -u` 與 `--log-every 25` 會讓訓練過程即時輸出目前訓練到哪一個模型、哪一個 epoch，以及 batch 進度。

訓練完成後，會產生：

```text
hybrid_deep_models_p097/
  metadata.json
  component_model.pt
  defect_global.pt
  defect_vari-grip.pt
  defect_glass-insulator.pt
  defect_yoke-suspension.pt

test_submission_hybrid_deep_p097.csv
test_predictions_hybrid_deep_p097_detailed.csv
```

## 重現最終 Submission

最終提交不是直接使用 p097 base output，而是在 detailed prediction 上做 component-level threshold post-processing。

```powershell
python adjust_submission_thresholds.py --detailed test_predictions_hybrid_deep_p097_detailed.csv --output test_submission_hybrid_deep_p097_comp_balanced.csv --detailed-output test_predictions_hybrid_deep_p097_comp_balanced_detailed.csv --component-override lightning-rod-suspension=0.40 --component-override polymer-insulator-upper-shackle=0.50 --component-override yoke-suspension=0.55
```

這個版本是本專案最終提交版本：

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

## GitHub 上傳注意事項

`.gitignore` 已排除大型或非必要檔案：

```text
train_dataset/
test_dataset/
hybrid_deep_models*/
csv_history/
*.pt
*.pth
*.joblib
__pycache__/
```

GitHub repo 內建議保留：

```text
README.md
app.py
train_hybrid_deep_classifier.py
adjust_submission_thresholds.py
test_submission_template.csv
test_submission_hybrid_deep_p097_comp_balanced.csv
test_predictions_hybrid_deep_p097_comp_balanced_detailed.csv
Final Testing Results/
```

如果希望其他人 clone 後可以直接執行 App，需要另外提供 `hybrid_deep_models_p097/` 模型權重，例如放在雲端硬碟、GitHub Release，或使用 Git LFS 管理大型模型檔案。
