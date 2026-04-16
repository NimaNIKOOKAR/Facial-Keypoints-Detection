import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Dataset
# =========================
class FacialKeypointsDataset(Dataset):
    def __init__(self, df):
        self.df = df.copy()

        self.df["Image"] = self.df["Image"].apply(
            lambda x: np.fromstring(x, sep=" ", dtype=np.float32).reshape(96, 96)
        )

        self.df = self.df.dropna().reset_index(drop=True)

        self.images = np.stack(self.df["Image"].values) / 255.0
        self.keypoints = self.df.drop(columns=["Image"]).values.astype(np.float32)

        # Normalize keypoints from [0, 96] to [-1, 1]
        self.keypoints = (self.keypoints - 48.0) / 48.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = np.expand_dims(image, axis=0)  # (1, 96, 96)
        keypoints = self.keypoints[idx]

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(keypoints, dtype=torch.float32),
        )


# =========================
# Model
# =========================
class KeypointCNN(nn.Module):
    def __init__(self, num_outputs=30):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, num_outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


# =========================
# Utils
# =========================
def denormalize_keypoints(x):
    return x * 48.0 + 48.0


def plot_predictions(images, preds, n=6):
    plt.figure(figsize=(12, 6))

    for i in range(n):
        plt.subplot(2, 3, i + 1)
        img = images[i].squeeze()
        keypoints = preds[i]

        plt.imshow(img, cmap="gray")
        plt.scatter(keypoints[0::2], keypoints[1::2], s=10)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# =========================
# Training
# =========================
def train_model(csv_path="training.csv", epochs=15, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(csv_path)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    train_dataset = FacialKeypointsDataset(train_df)
    val_dataset = FacialKeypointsDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = KeypointCNN(num_outputs=30).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, keypoints in train_loader:
            images = images.to(device)
            keypoints = keypoints.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, keypoints in val_loader:
                images = images.to(device)
                keypoints = keypoints.to(device)

                outputs = model(images)
                loss = criterion(outputs, keypoints)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_keypoint_model.pth")
            print("Best model saved.")

    return model, val_loader, device

