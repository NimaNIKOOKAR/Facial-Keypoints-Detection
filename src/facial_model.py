import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader ,Dataset
from tqdm import tqdm


import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.optim as optim

import numpy as np
from matplotlib import pyplot as plt

class FacialKeypointsDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

        # Select only facial keypoint coordinates
        self.coord_cols = [
            col for col in self.df.columns
            if col.endswith(("_x", "_y"))
        ]

        if len(self.coord_cols) != 30:
            raise ValueError(
                f"Expected 30 coordinates, got {len(self.coord_cols)}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image preprocessing
        image = np.fromstring(
            row["Image"],
            sep=" ",
            dtype=np.float32
        ).reshape(96, 96) / 255.0

        image = torch.from_numpy(image).unsqueeze(0)

        # Keypoint preprocessing
        keypoints = row[self.coord_cols].to_numpy(
            dtype=np.float32
        )

        # Normalize coordinates to [0, 1]
        keypoints = keypoints / 95.0

        keypoints = torch.from_numpy(keypoints.copy())

        return image, keypoints

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.sigmoid(self.regressor(x))


def evaluate_model(model, data_loader, device):

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for images, keypoints in data_loader:

            images = images.to(device)

            outputs = model(images)

            predictions.append(outputs.cpu())
            targets.append(keypoints.cpu())

    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)

    # Convert to pixel coordinates
    predictions_px = predictions * 95.0
    targets_px = targets * 95.0

    # Ignore missing coordinates
    valid = torch.isfinite(targets_px)

    if not valid.any():
        raise ValueError("No valid target coordinates found.")

    errors = predictions_px[valid] - targets_px[valid]

    mae = errors.abs().mean().item()

    rmse = torch.sqrt(
        (errors ** 2).mean()
    ).item()

    print(f"MAE:  {mae:.4f} pixels")
    print(f"RMSE: {rmse:.4f} pixels")

    return predictions, targets

def visualize_predictions(
    model,
    data_loader,
    device,
    n=5
):
    model.eval()

    images, targets = next(iter(data_loader))

    with torch.no_grad():
        outputs = model(images.to(device)).cpu()

    n = min(n, len(images))

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(4 * n, 4),
        squeeze=False
    )

    for i in range(n):

        ax = axes[0, i]

        # Display image
        image = images[i].squeeze().cpu().numpy()

        ax.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        # Convert normalized coordinates to pixels
        pred = outputs[i].numpy().reshape(-1, 2) * 95.0
        actual = targets[i].numpy().reshape(-1, 2) * 95.0

        # Valid keypoints
        actual_valid = np.isfinite(actual).all(axis=1)
        pred_valid = np.isfinite(pred).all(axis=1)

        # Predicted points
        ax.scatter(
            pred[pred_valid, 0],
            pred[pred_valid, 1],
            c="red",
            marker="x",
            s=40,
            label="Predicted",
            zorder=2
        )

        # Actual points
        ax.scatter(
            actual[actual_valid, 0],
            actual[actual_valid, 1],
            c="lime",
            edgecolors="black",
            marker="o",
            s=45,
            label="Actual",
            zorder=3
        )

        # MAE on fully observed landmarks
        comparable = actual_valid & pred_valid

        if comparable.any():

            errors = np.abs(
                pred[comparable] - actual[comparable]
            )

            mae = errors.mean()

            mae_text = f"{mae:.2f} px"

        else:
            mae_text = "N/A"

        ax.set_title(
            f"Image {i + 1}\n"
            f"Actual: {actual_valid.sum()}/15\n"
            f"MAE: {mae_text}"
        )

        ax.set_xlim(-0.5, 95.5)
        ax.set_ylim(95.5, -0.5)

        ax.axis("off")

        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.show()

    
def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=10,
    learning_rate=0.001
):
    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(num_epochs):

        # Training
        model.train()

        train_error_sum = 0.0
        train_valid_count = 0

        for images, targets in tqdm(train_loader):

            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            predictions = model(images)

            # Ignore missing target values
            valid = torch.isfinite(targets)

            if not valid.any():
                continue

            loss = criterion(
                predictions[valid],
                targets[valid]
            )

            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            optimizer.step()

            count = valid.sum().item()

            train_error_sum += loss.item() * count
            train_valid_count += count

        # Validation
        model.eval()

        val_error_sum = 0.0
        val_valid_count = 0

        with torch.no_grad():

            for images, targets in val_loader:

                images = images.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)

                predictions = model(images)

                valid = torch.isfinite(targets)

                if not valid.any():
                    continue

                loss = criterion(
                    predictions[valid],
                    targets[valid]
                )

                count = valid.sum().item()

                val_error_sum += loss.item() * count
                val_valid_count += count

        train_loss = (
            train_error_sum / train_valid_count
            if train_valid_count else float("nan")
        )

        val_loss = (
            val_error_sum / val_valid_count
            if val_valid_count else float("nan")
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

    return model, history