import torch
import torch.nn as nn
import numpy as np

# Define same CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 9 * 10, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Load label map
label_map = np.load("model/label_mapping.npy", allow_pickle=True)

# Load model
model = SimpleCNN(num_classes=len(label_map))
model.load_state_dict(torch.load("model/cnn_model.pth", map_location="cpu"))
model.eval()

def predict_mfcc_tensor(mfcc_tensor):
    with torch.no_grad():
        output = model(mfcc_tensor)
        predicted_idx = output.argmax(dim=1).item()
        return label_map[predicted_idx]
