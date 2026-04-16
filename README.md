# Facial Keypoints Detection  

A **hands-on, end-to-end starter project** for locating key facial landmarks (eyes, nose, mouth, eyebrows) on 96×96‑pixel grayscale images.  
The repo packages two complementary baselines:

1. **An R quick‑start** that mirrors the original Kaggle tutorial (mean coordinates + simple patch‑matching).  
2. **A Python/Lasagne convolutional neural‑network** baseline that you can train on GPU or CPU.

The code is intentionally minimal—no framework lock‑in—so you can rip things out, swap pieces, or rebuild from scratch.

---

## Table of Contents
1. [Project structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Dataset](#dataset)
4. [R baseline](#r-baseline)
5. [Python + CNN baseline](#python--cnn-baseline)
6. [Training tips](#training-tips)
7. [Evaluation & submission](#evaluation--submission)
8. [AWS one-liner](#aws-one-liner) (cheap GPU)
9. [Troubleshooting FAQ](#troubleshooting-faq)
10. [Citation & license](#citation--license)

---

## Project structure
```text
.
├── data/                # put training.csv, test.csv, submissionFileFormat.csv here
├── r/                   # R scripts & notebooks
│   ├── 01_mean_benchmark.R
│   └── 02_patch_search.R
├── python/
│   ├── train_cnn.py     # Lasagne/Theano implementation
│   ├── model_zoo.py     # a few tiny model variants
│   └── kaggle_submit.py # generates submission
└── README.md            # you are here
```
Feel free to reorganise—paths are loaded from a single config file (`config.yml`).

---

## Prerequisites
### R route  
* R ≥ 4.1 (tested)  
* Packages: `doMC`, `foreach`, `reshape2`, `tidyverse`  
  ```r
  install.packages(c("doMC", "foreach", "reshape2", "tidyverse"))
  ```

### Python route  
* Python ≥ 3.9 (conda or venv)  
* `pip install -r requirements.txt` *(Lasagne, Theano‑PyMC, numpy, pandas, matplotlib, kaggle)*  
* **GPU optional** but recommended.  CUDA 11.x + cuDNN 8.x works.

---

## Dataset
Grab the files from Kaggle:
```bash
kaggle competitions download -c facial-keypoints-detection
unzip facial-keypoints-detection.zip -d data/
```
If the Kaggle CLI is blocked, manually download via browser and drop the three CSVs into `data/`.

---

## R baseline
The original tutorial distilled to two scripts.

### 1. Mean‑coordinate benchmark
```r
source("r/01_mean_benchmark.R")
```
* Reads `training.csv` → computes column means → writes `submission_means.csv`.
* LB ≈ **3.96 RMSE** (don’t get excited—this is the throwaway baseline).

### 2. Patch‑search benchmark
```r
source("r/02_patch_search.R")
```
* Crafts a 21×21 average patch per keypoint and slides it ±2 pixels.
* LB ≈ **3.81 RMSE**.  Slightly better, still miles from SOTA.

Both scripts parallelise with **doMC**; tweak `options(mc.cores = …)` to taste.

---

## Python + CNN baseline
A 3‑layer convolutional net (~90 k params) that trains in **≤15 min** on an RTX 3060, or ≈2 h on CPU.

```bash
# create env (conda example)
conda create -n keypoints python=3.10
conda activate keypoints
pip install -r requirements.txt

# quick run (defaults to ~35 epochs)
python python/train_cnn.py \
    --train data/training.csv \
    --test  data/test.csv \
    --out   output/model.npz

# generate submission
python python/kaggle_submit.py --model output/model.npz --outfile submission_cnn.csv
```
Default model nets **≈2.85 RMSE**.  Not leaderboard‑winning, but a solid springboard:
* Swap in `model_zoo.py::TinyResNet` or `FPN` for instant gains (2.5‑ish).
* Flip `--flip`, `--rotate`, `--jitter` flags for more aggressive augmentation.
* Replace MSE with **Wing loss** or **Huber** for robustness to outliers.

---

## Training tips
* **Normalise** coordinates to [-1, 1] before feeding networks; rescale at inference.
* Treat missing keypoints as **masked loss** (see code).  Zero‑imputation will tank accuracy.
* Wider conic kernels (e.g. 7×7) pick up coarse hairline cues; stack with 3×3 for fine‑grained detail.
* Add **spatial‑dropout** on conv blocks—dramatically lowers overfitting on 7 k images.
* Early‑stopping on a 10 % hold‑out lifts reliability; Kaggle’s test set is only 1 800 images.

---

## Evaluation & submission
```bash
kaggle competitions submit facial-keypoints-detection \
    --file submission_cnn.csv \
    --message "2.85 RMSE – 3-layer CNN"
```
The competition scores **RMSE across *available* keypoints** (some images annotate fewer than 30).  Missing predictions count as huge error—always emit all rows present in `submissionFileFormat.csv`.

---

## AWS one-liner
Cheap (<€0.05/hr) GPU via spot instance:
```bash
aws ec2 run-instances \
    --image-id ami-1f3e225a \
    --instance-type g4dn.xlarge \
    --key-name my-key \
    --instance-market-options 'MarketType="spot"' \
    --user-data file://cloud/bootstrap.sh
```
`bootstrap.sh` installs CUDA 11.8, cuDNN, Miniconda, pulls this repo, and fires `train_cnn.py` with sane defaults.

Remember: **terminate the instance** after training.

---

## Troubleshooting FAQ
> **Q**: R complains *“X11 not available”* when plotting images.
>
> **A**: Use `options(bitmapType = "cairo")` or run the scripts headless (`Rscript`).

> **Q**: Theano warns about *"No GPU detected"*.
>
> **A**: Check `nvidia-smi`.  If it shows your card, ensure `THEANO_FLAGS="device=cuda,floatX=float32"` is exported **before** Python starts.

> **Q**: My CNN overfits after a few epochs.
>
> **A**: Crank up data augmentation (`--jitter 4 --rotate 15`) or add another dropout layer.  The dataset is tiny.

> **Q**: Submission has NaNs.
>
> **A**: Likely missing keypoints in some test images.  The helper script applies mean‑imputation—double‑check you’re using the wrapper.

---

## Citation & license
```
@misc{petterson2013facial,
  title        = {{Facial Keypoints Detection}},
  author       = {James Petterson and Will Cukierski},
  year         = 2013,
  howpublished = {\url{https://kaggle.com/competitions/facial-keypoints-detection}}
}
```

* Dataset © Kaggle / Université de Montréal – Yoshua Bengio lab.  
* This code released under the **MIT License**—do whatever you like, just don’t sue.

---

**Enjoy hacking—rip it apart, question everything, and push the RMSE into the basement.**
