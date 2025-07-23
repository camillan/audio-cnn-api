import librosa
import numpy as np
import torch
from tempfile import NamedTemporaryFile

def audio_to_mfcc(file_bytes):
    # Save file temporarily
    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()

        y, sr = librosa.load(tmp.name, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Pad/crop to match training input (40 x 44)
        if mfcc.shape[1] < 44:
            mfcc = np.pad(mfcc, ((0, 0), (0, 44 - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :44]

        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 40, 44]
        return mfcc_tensor
