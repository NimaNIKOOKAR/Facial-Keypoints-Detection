import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed

IMG_SIZE = 96
N_KEYPOINTS = 15
IMAGENET_MEAN = 0.449
IMAGENET_STD = 0.226

LOW_MISSINGNESS_COLS = [
    "left_eye_center_x", "left_eye_center_y",
    "right_eye_center_x", "right_eye_center_y",
    "nose_tip_x", "nose_tip_y",
    "mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y",
]

SYMMETRIC_X_PAIRS = [
    ("left_eye_center_x", "right_eye_center_x"),
]

SYMMETRIC_Y_PAIRS = [
    ("left_eye_center_y", "right_eye_center_y"),
]


def fill_missing_keypoints(df, low_missingness_cols=None):
    """
    Imputes ONLY the near-complete columns (default: LOW_MISSINGNESS_COLS).
    Everything else is left as NaN on purpose - those columns are missing on
    the majority of rows in this dataset, and imputing them would mean the
    model gets trained against a repeated constant for most samples. The
    masked loss in run_epoch() is what handles those, not this function.

    Symmetry fill for x-coordinates uses the mirror relationship
    (left_x ~ IMG_SIZE - right_x), not equality - the two sides are
    reflections of each other, not duplicates.
    """
    df_filled = df.copy()
    cols = low_missingness_cols if low_missingness_cols is not None else LOW_MISSINGNESS_COLS

    for left_col, right_col in SYMMETRIC_X_PAIRS:
        if left_col in cols and right_col in cols:
            df_filled[left_col] = df_filled[left_col].fillna(IMG_SIZE - df_filled[right_col])
            df_filled[right_col] = df_filled[right_col].fillna(IMG_SIZE - df_filled[left_col])

    for left_col, right_col in SYMMETRIC_Y_PAIRS:
        if left_col in cols and right_col in cols:
            df_filled[left_col] = df_filled[left_col].fillna(df_filled[right_col])
            df_filled[right_col] = df_filled[right_col].fillna(df_filled[left_col])

    for col in cols:
        if col in df_filled.columns:
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())

    return df_filled


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sharpen_image(img, amount=1.0, radius=1.0):
    """
    Unsharp masking: sharpened = img + amount * (img - blurred(img)).
    This is NOT deblurring (no attempt to invert an unknown blur kernel) —
    it boosts local contrast at edges, which is what actually reads as
    "sharper" at 96x96 resolution and is stable/cheap unlike blind
    deconvolution. Operates on a single (96, 96) float array in [0, 1].
    """
    blurred = gaussian_filter(img, sigma=radius)
    sharpened = img + amount * (img - blurred)
    return np.clip(sharpened, 0.0, 1.0)


def preprocess_images_parallel(images, sharpen_amount=1.0, sharpen_radius=1.0, n_jobs=-1, verbose=True):
    """
    Applies sharpen_image across an (N, 96, 96) array using joblib, run ONCE
    up front rather than per-epoch. n_jobs=-1 uses all available cores.
    verbose=True prints joblib's own progress output so you can confirm
    it's actually dispatching to multiple workers, not silently running
    single-threaded.
    """
    import os

    resolved_jobs = os.cpu_count() if n_jobs == -1 else n_jobs
    if verbose:
        print(f"preprocess_images_parallel: dispatching {len(images)} images across n_jobs={n_jobs} "
              f"(resolves to {resolved_jobs} workers on this machine)")

    sharpened = Parallel(n_jobs=n_jobs, backend="threading", verbose=10 if verbose else 0)(
        delayed(sharpen_image)(img, sharpen_amount, sharpen_radius) for img in images
    )
    return np.stack(sharpened)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class KeypointsDataset(Dataset):
    """
    Parses the Kaggle training.csv format: last column 'Image' is a
    space-separated pixel string, preceding columns are x/y pairs per
    keypoint with NaN where a keypoint wasn't annotated.
    """

    def __init__(self, csv_path, train=True, sharpen=False, sharpen_amount=1.0, sharpen_radius=1.0, n_jobs=-1):
        df = pd.read_csv(csv_path)
        self.train = train

        images = df["Image"].apply(lambda s: np.array(s.split(), dtype=np.float32))
        self.images = np.stack(images.values).reshape(-1, IMG_SIZE, IMG_SIZE)
        self.images /= 255.0

        if sharpen:
            # Done once here, not in __getitem__ — otherwise every epoch
            # re-runs the same sharpening on the same pixels for no benefit.
            self.images = preprocess_images_parallel(
                self.images, sharpen_amount=sharpen_amount, sharpen_radius=sharpen_radius, n_jobs=n_jobs
            )

        if train:
            coord_cols = [c for c in df.columns if c != "Image"]
            assert len(coord_cols) == N_KEYPOINTS * 2, (
                f"expected {N_KEYPOINTS * 2} coordinate columns, got {len(coord_cols)}"
            )
            self.coord_cols = coord_cols  # exact order, reused by make_submission()
            # Impute only the near-complete columns; leave everything else NaN
            # so the mask below still excludes it from the loss. See
            # fill_missing_keypoints() docstring for why the two are split.
            df = fill_missing_keypoints(df)
            coords = df[coord_cols].values.astype(np.float32)
            self.mask = ~np.isnan(coords)
            coords = np.nan_to_num(coords, nan=0.0)
            self.coords = (coords / (IMG_SIZE / 2.0)) - 1.0
        else:
            self.coord_cols = None
            self.coords = None
            self.mask = None

        # ImageId if the CSV has one (Kaggle's test.csv does); otherwise
        # fall back to 1-indexed row position. Needed to join predictions
        # against submissionFileFormat.csv later.
        if "ImageId" in df.columns:
            self.image_ids = df["ImageId"].values
        else:
            self.image_ids = np.arange(1, len(df) + 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.repeat(img[None, :, :], 3, axis=0)  # 1ch -> 3ch, no conv1 surgery
        img = torch.from_numpy(img.astype(np.float32))

        if self.train:
            target = torch.from_numpy(self.coords[idx])
            mask = torch.from_numpy(self.mask[idx].astype(np.float32))
            return img, target, mask
        return img


def make_loaders(dataset, batch_size=64, val_frac=0.1, num_workers=2, seed=None):
    """Split a training-mode KeypointsDataset into train/val DataLoaders."""
    n_val = int(len(dataset) * val_frac)
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model():
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, N_KEYPOINTS * 2)
    return model


def masked_mse(preds, targets, mask):
    diff2 = (preds - targets) ** 2 * mask
    return diff2.sum() / mask.sum().clamp(min=1.0)


def set_backbone_trainable(model, trainable: bool):
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, device, train=True, scaler=None):
    """
    scaler: pass a torch.amp.GradScaler('cuda') to enable mixed precision.
    Only does anything useful on CUDA - AMP has no benefit on CPU/MPS, and
    passing a scaler there is harmless but pointless.
    """
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    use_amp = scaler is not None and device.type == "cuda"

    with torch.set_grad_enabled(train):
        for imgs, targets, mask in loader:
            imgs, targets, mask = imgs.to(device), targets.to(device), mask.to(device)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    preds = model(imgs)
                    loss = masked_mse(preds, targets, mask)
            else:
                preds = model(imgs)
                loss = masked_mse(preds, targets, mask)

            if train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches


def train_head(model, train_loader, val_loader, device, epochs=8, lr=1e-3, verbose=True, amp=True):
    """Phase 1: freeze backbone, train the regression head only.
    amp=True enables mixed precision on CUDA (no-op elsewhere)."""
    set_backbone_trainable(model, trainable=False)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda') if (amp and device.type == "cuda") else None

    history = {"train": [], "val": []}
    for epoch in range(epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True, scaler=scaler)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False, scaler=scaler)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if verbose:
            print(f"[head] epoch {epoch+1}/{epochs}  train {train_loss:.4f}  val {val_loss:.4f}")
    return history


def train_finetune(
    model,
    train_loader,
    val_loader,
    device,
    epochs=25,
    lr_head=1e-3,
    lr_backbone=1e-4,
    checkpoint_path=None,
    verbose=True,
    amp=True,
):
    """Phase 2: unfreeze everything, discriminative LR, cosine schedule.
    amp=True enables mixed precision on CUDA (no-op elsewhere)."""
    set_backbone_trainable(model, trainable=True)
    optimizer = torch.optim.Adam(
        [
            {"params": model.fc.parameters(), "lr": lr_head},
            {
                "params": [p for n, p in model.named_parameters() if not n.startswith("fc.")],
                "lr": lr_backbone,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda') if (amp and device.type == "cuda") else None

    history = {"train": [], "val": []}
    best_val = float("inf")
    for epoch in range(epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True, scaler=scaler)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False, scaler=scaler)
        scheduler.step()
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if verbose:
            print(f"[ft] epoch {epoch+1}/{epochs}  train {train_loss:.4f}  val {val_loss:.4f}")

        if checkpoint_path is not None and val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    history["best_val"] = best_val
    return history


# --------------------------------------------------------------------------
# Evaluation / inspection
# --------------------------------------------------------------------------
def pixel_rmse(model, loader, device):
    """RMSE in pixel space (comparable to the Kaggle leaderboard scale),
    converting back from the [-1, 1] normalized targets used in training."""
    model.eval()
    sq_errs, n = 0.0, 0
    with torch.no_grad():
        for imgs, targets, mask in loader:
            imgs, targets, mask = imgs.to(device), targets.to(device), mask.to(device)
            preds = model(imgs)

            preds_px = (preds + 1) * (IMG_SIZE / 2)
            targets_px = (targets + 1) * (IMG_SIZE / 2)

            diff2 = (preds_px - targets_px) ** 2 * mask
            sq_errs += diff2.sum().item()
            n += mask.sum().item()
    return (sq_errs / n) ** 0.5


def show_sample(dataset, idx):
    """Requires matplotlib; imported lazily so the module doesn't force
    a plotting backend on non-notebook callers."""
    import matplotlib.pyplot as plt

    img, target, mask = dataset[idx]
    raw = img[0].numpy() * IMAGENET_STD + IMAGENET_MEAN
    xs = (target[0::2].numpy() + 1) * (IMG_SIZE / 2)
    ys = (target[1::2].numpy() + 1) * (IMG_SIZE / 2)
    present = mask[0::2].numpy().astype(bool)

    plt.imshow(raw, cmap="gray")
    plt.scatter(xs[present], ys[present], c="red", s=15)
    plt.title(f"sample {idx} — {present.sum()} keypoints")
    plt.show()


# --------------------------------------------------------------------------
# Test-set inference and Kaggle submission
# --------------------------------------------------------------------------
def predict_all(model, test_dataset, device, batch_size=64):
    """
    Runs the model over every image in a test-mode KeypointsDataset
    (train=False, so __getitem__ returns just the image tensor) and returns
    predictions in PIXEL space as an (N, 30) numpy array, ordered by the
    dataset's iteration order (i.e. matches test_dataset.image_ids).
    """
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            preds_px = (preds + 1) * (IMG_SIZE / 2.0)
            all_preds.append(preds_px.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def make_submission(
    model,
    test_csv_path,
    submission_format_path,
    coord_cols,
    out_path,
    device,
    batch_size=64,
    clip_to_image=True,
):
    """
    Produces a Kaggle-format submission CSV.

    coord_cols: pass train_dataset.coord_cols from the KeypointsDataset you
    trained on, NOT a separately hardcoded list — this guarantees prediction
    column N corresponds to the same feature name the model was trained
    against, even if some future training.csv has columns in a different
    order.

    submission_format_path: submissionFileFormat.csv from Kaggle. The
    competition does NOT require all 30 keypoints for every test image, so
    this file tells us exactly which (ImageId, FeatureName) rows to emit —
    predicting all 30 for every image and shipping that as-is would produce
    a malformed submission.
    """
    test_ds = KeypointsDataset(test_csv_path, train=False)
    preds_px = predict_all(model, test_ds, device, batch_size=batch_size)

    if clip_to_image:
        # model can extrapolate slightly outside [0, 96]; the ground truth
        # never does, so clipping is a safe, free RMSE improvement
        preds_px = np.clip(preds_px, 0, IMG_SIZE)

    wide = pd.DataFrame(preds_px, columns=coord_cols)
    wide.insert(0, "ImageId", test_ds.image_ids)

    long = wide.melt(id_vars="ImageId", var_name="FeatureName", value_name="Location")

    fmt = pd.read_csv(submission_format_path)
    fmt = fmt.drop(columns=["Location"])  # placeholder '?' column, replaced below

    submission = fmt.merge(long, on=["ImageId", "FeatureName"], how="left")
    missing = submission["Location"].isna().sum()
    if missing:
        print(f"warning: {missing} required rows had no matching prediction — check coord_cols/ImageId alignment")

    submission = submission[["RowId", "ImageId", "FeatureName", "Location"]]
    submission.to_csv(out_path, index=False)
    print(f"wrote {len(submission)} rows to {out_path}")
    return submission


def plot_test_predictions(model, test_csv_path, coord_cols, device, n=9, indices=None, cols=3):
    """
    Visual sanity check on test.csv: runs the model and overlays predicted
    keypoints on a grid of test images. There's no ground truth for test.csv
    (that's the whole point of a test set), so this is purely an eyeball
    check — if points land off-face or clustered in a corner, something's
    wrong upstream (column order, normalization, a stale checkpoint), not
    a "the model needs more epochs" situation.
    """
    import matplotlib.pyplot as plt

    test_ds = KeypointsDataset(test_csv_path, train=False)
    if indices is None:
        indices = np.random.choice(len(test_ds), size=min(n, len(test_ds)), replace=False)
    n = len(indices)
    rows = int(np.ceil(n / cols))

    model.eval()
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            img = test_ds[idx].unsqueeze(0).to(device)
            pred = model(img)[0].cpu().numpy()
            pred_px = (pred + 1) * (IMG_SIZE / 2.0)
            xs, ys = pred_px[0::2], pred_px[1::2]

            raw = test_ds[idx][0].numpy() * IMAGENET_STD + IMAGENET_MEAN
            ax.imshow(raw, cmap="gray")
            ax.scatter(xs, ys, c="red", s=12)
            ax.set_title(f"img {test_ds.image_ids[idx]}", fontsize=9)
            ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()