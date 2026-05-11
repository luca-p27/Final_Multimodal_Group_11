"""
geo_baseline.py

Geographic-only baseline: trains a small two-layer MLP on raw lat/lon to predict
species, with no image input. Useful for comparing against multimodal models.

Usage:
    python geo_baseline.py
    python geo_baseline.py --data_path /path/to/CrypticBio-Common.csv
"""

import argparse
import copy
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DEFAULT = os.path.abspath(os.path.join(_HERE, '..', '..', 'input', 'CrypticBio-Common.csv'))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class GeographicDataset(Dataset):
    def __init__(self, X, y, label_encoder):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(label_encoder.transform(y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def clean_data(df):
    df = df.dropna(subset=['decimalLatitude', 'decimalLongitude', 'scientificName'])

    df = df[
        (df['decimalLatitude'].between(-90, 90)) &
        (df['decimalLongitude'].between(-180, 180))
    ]

    df['species'] = df['scientificName']
    return df


def main():
    p = argparse.ArgumentParser(
        description='Geographic-only MLP baseline.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data_path', default=DATA_DEFAULT,
                   help='path to CrypticBio CSV')
    args = p.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 128 if torch.cuda.is_available() else 64
    epochs = 30
    patience = 5

    # Load data
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} samples")

    # Clean
    df = clean_data(df)

    # Filter rare species
    counts = df['species'].value_counts()
    df = df[df['species'].isin(counts[counts >= 5].index)]

    print(f"Samples after cleaning: {len(df)}")
    print(f"Number of species: {df['species'].nunique()}")

    X = df[['decimalLatitude', 'decimalLongitude']].to_numpy(dtype=np.float32)
    y = df['species'].to_numpy()

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Encode labels
    le = LabelEncoder()
    le.fit(y_train)
    num_classes = len(le.classes_)

    # Datasets
    train_ds = GeographicDataset(X_train, y_train, le)
    val_ds = GeographicDataset(X_val, y_val, le)
    test_ds = GeographicDataset(X_test, y_test, le)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Model
    model = SimpleMLP(input_dim=2, num_classes=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):

        # train
        model.train()
        train_correct = 0
        total = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(Xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            _, pred = torch.max(outputs, 1)
            train_correct += (pred == yb).sum().item()
            total += yb.size(0)

        train_acc = train_correct / total

        # validation
        model.eval()
        val_correct = 0
        total = 0

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                outputs = model(Xb)

                _, pred = torch.max(outputs, 1)
                val_correct += (pred == yb).sum().item()
                total += yb.size(0)

        val_acc = val_correct / total

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")

        # early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_model)

    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            outputs = model(Xb)

            _, pred = torch.max(outputs, 1)

            preds.extend(pred.cpu().numpy())
            labels.extend(yb.numpy())

    acc = accuracy_score(labels, preds)
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)")


if __name__ == "__main__":
    main()
