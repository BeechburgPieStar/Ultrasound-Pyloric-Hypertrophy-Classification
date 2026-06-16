# Ultrasound Pyloric Hypertrophy Classification System

A deep-learning pipeline for binary classification of ultrasound images
(normal vs. pyloric hypertrophy), with five-fold cross-validation fine-tuning,
hyperparameter grid search, and an uncertainty-aware ensemble inference stage.

---

## 1. Data Preparation

Before launching fine-tuning, prepare your raw ultrasound images together with a
standard Excel index file (`.xlsx`). This Excel file is the core of data loading
and **must** contain the following four columns, named exactly as shown:

| Column      | Type    | Description                                                                 |
|-------------|---------|-----------------------------------------------------------------------------|
| `file_path` | string  | Path to each image on disk.                                                 |
| `label`     | integer | Binary label: `0` = normal, `1` = pyloric hypertrophy.                      |
| `fold`      | integer | A value from `1` to `5`, used as the grouping key for five-fold cross-validation. |
| `text`      | string  | Clinical text description of the ultrasound findings for that sample.       |

**Supported image formats:** standard formats such as `.jpg` and `.png`.

---

## 2. Training

1. Open `train.py` and change the Excel path to your own (relative) path.
2. Confirm the `ParaMeters` block at the end of the file (the hyperparameter
   grid-search ranges).
3. Run the full automated five-fold fine-tuning and grid search:

   ```bash
   python train.py
   ```

After training finishes, go to the output root directory created automatically by
the script at `./outputs/`. Open `Hyperparameters_Summary.xlsx`, which is sorted
by **AUC in descending order**, to directly identify the best-performing model
weights.

---

## 3. Testing

1. Open the test script and locate the `if __name__ == "__main__":` block at the
   bottom.
2. Set the configuration values:
   - `TEST_EXCEL` — path to your test-set Excel file.
   - `WEIGHTS_DIR` — folder containing the trained weights.
   - `target_pass_rate` — the desired uncertainty pass-through ratio (optional).
3. Run the test:

   ```bash
   python test.py
   ```

The system automatically loads all five fold models in parallel and performs
ensemble majority-vote inference. It then computes a dynamic threshold based on a
normalized-entropy algorithm to flag and reject high-risk (high-uncertainty)
samples.

---
## 6. Demo

[pyloric-ultrasound-demo](https://huggingface.co/spaces/noahwang1996/pyloric-ultrasound-demo)

## 5. Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

> **Note on `requirements.txt`:** the provided file lists conflicting versions for
> a few packages — `torch` (`2.8.0+cu126` vs `2.4.1`), `torchvision`
> (`0.23.0+cu126` vs `0.19.1`), and `tqdm` (`4.67.3` vs `4.64.1`). Keep only one
> version of each (matching your CUDA setup) before installing to avoid resolver
> errors. The `+cu126` builds require a matching CUDA 12.6 environment.
