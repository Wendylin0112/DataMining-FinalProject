# Final Testing Results

本資料夾整理最終測試提交結果。

## Best Submission

最終採用版本：

```text
test_submission_hybrid_deep_p097_comp_balanced.csv
```

本資料夾中的對應檔案：

```text
第一組_test_version2 (hybrid_deep_p097_comp_balanced).csv
第二次.json
```

`第二次.json` 是目前最佳結果：

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

## Submission History

```text
第一次.json
  file: 第一組_test_version1 (hybrid_deep_p097).csv
  summary: high precision, low recall

第二次.json
  file: 第一組_test_version2 (hybrid_deep_p097_comp_balanced).csv
  summary: best overall f1 and recall while keeping precision above 0.90

第三次.json
  summary: higher precision than second run, but lower recall and f1
```

完整比較請見：

```text
results_summary.csv
```

