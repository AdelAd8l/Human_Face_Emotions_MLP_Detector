# Human Face Emotion Classification with MLPs

**Author:** Adel Ahmed Fathi Amer (ID 224100177)

Five-class facial emotion classifier (Angry, Fear, Happy, Sad, Surprise) built with **multi-layer perceptrons only**
(no CNNs / transformers / pre-trained networks), using the Kaggle dataset
[Human Face Emotions](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions).

| File | What it is |
|---|---|
| `emotion_mlp.ipynb` | The full study, all cells executed (data cleaning, 13 preprocessing pipelines, architecture / activation / regularisation / optimiser / imbalance / augmentation experiments, random search, final test evaluation) |
| `emotion_mlp_report.pdf` | 10-page report with the results and their interpretation |
| `results/all_runs.csv` | Every one of the 132 evaluated configurations (validation accuracy, macro-F1, ...) |
| `results/final_results.json` | Final test metrics, per-class report, confusion matrix |
| `figs/` | All figures produced by the notebook |

## Results (held-out test set, 4,323 images)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Single MLP (mean of 5 seeds) | 60.1 % +- 0.3 | 0.573 |
| Ensemble of 5 MLPs | **62.2 %** | **0.593** |
| Majority-class baseline | 32.1 % | 0.097 |
| Plain MLP on raw pixels (validation) | 47.0 % | 0.433 |

Final model: standardised **HOG + LBP** features -> MLP 1024-512 (GELU, batch-norm, dropout 0.1, label smoothing 0.1,
AdamW) trained with horizontally flipped copies of the training images.

## Key findings

* 30,232 of the 59,099 images (51 %) are exact duplicates; they were removed to avoid train/test leakage.
* The input representation (HOG/LBP instead of raw pixels) gave about +10 points macro-F1; architecture, activation and
  regularisation changes were mostly within +-1 point (seed noise).
* All MLPs memorise the training set (~100 % train accuracy), so generalisation, not capacity, is the bottleneck.

## Reproduce

```bash
pip install -r requirements.txt
jupyter notebook emotion_mlp.ipynb      # run all cells (~40 min on a GPU); the dataset is fetched with kagglehub
```

The dataset is **not** included in this repository. `kagglehub` downloads it (about 700 MB) on first run.
A CUDA GPU is recommended; the code falls back to CPU but will be much slower.
