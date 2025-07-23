import soundata
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import os

# --------------------------
# STEP 1: Load UrbanSound8K
# --------------------------
print("Loading UrbanSound8K...")
dataset = soundata.initialize("urbansound8k")
dataset.download()

clip_ids = dataset.clip_ids[:500]  # use small subset for speed
X = []
y = []

print("Extracting MFCC features...")
for clip_id in clip_ids:
    clip = dataset.clip(clip_id)
    labels = clip.tags.labels
    if not labels:
        continue
    try:
        audio, sr = clip.audio
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        # Pad or crop MFCC to (40, 44) for uniform input
        if mfcc.shape[1] < 44:
            mfcc = np.pad(mfcc, ((0, 0), (0, 44 - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :44]

        X.append(mfcc)
        y.append(labels[0])  # e.g., 'dog_bark'
    except Exception as e:
        print(f"Failed on {clip_id}: {e}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X = np.array(X)
y = np.array(y_encoded)

# --------------------------
# STEP 2: PyTorch Dataset
# --------------------------
class MFCCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [B, 1, 40, 44]
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = MFCCDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# --------------------------
# STEP 3: CNN Model
# --------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),    # [B, 16, 40, 44]
            nn.ReLU(),
            nn.MaxPool2d(2),                               # [B, 16, 20, 22]
            nn.Conv2d(16, 32, kernel_size=3),              # [B, 32, 18, 20]
            nn.ReLU(),
            nn.MaxPool2d(2),                               # [B, 32, 9, 10]
            nn.Flatten(),
            nn.Linear(32 * 9 * 10, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------
# STEP 4: Train the Model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(le.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

print("Training CNN...")
for epoch in range(5):
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}")

# --------------------------
# STEP 5: Save Model + Labels
# --------------------------
print("Saving model...")
os.makedirs("model", exist_ok=True)

model_path = os.path.abspath("model/cnn_model.pth")
label_map_path = os.path.abspath("model/label_mapping.npy")

torch.save(model.state_dict(), model_path)
np.save(label_map_path, le.classes_)

print(f"Model saved to {model_path}")
print(f"Labels saved to {label_map_path}")
