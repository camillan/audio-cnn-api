# 🎷 Audio Classification API with CNN + FastAPI

A lightweight, end-to-end audio classification system trained on the UrbanSound8K dataset. It uses MFCC spectrograms, a custom PyTorch CNN, and is deployed via a FastAPI web server with support for .wav file uploads.

## 🔍 What It Does

* Accepts .wav audio files via a POST endpoint
* Converts audio into MFCC spectrograms
* Predicts one of 10 common urban sound classes using a trained CNN
* Returns the predicted label as a JSON response

## 🔧 Features

* Trains a simple CNN model using PyTorch
* Uses MFCC audio features (via librosa)
* REST API powered by FastAPI
* Accepts `.wav` files as input
* Returns predicted audio class (e.g., "dog\_bark", "siren")
* Includes a `samples/` folder with demo audio files
* Can be queried via FastAPI UI or `curl`

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/camillan/audio-cnn-api.git
cd audio-cnn-api
```

### 2. Set Up Environment

Use Conda:

```bash
conda create -n audio-env python=3.9 -y
conda activate audio-env
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Download & Prepare Dataset

We use the `soundata` library to load UrbanSound8K.
Run:

```bash
python train/train_cnn.py
```

This script will:

* Download UrbanSound8K
* Extract MFCC features
* Train the CNN
* Save model and label mappings to the `model/` directory

### 4. Run the API Server

```bash
uvicorn app.main:app --reload
```

API will be available at: `http://localhost:8000`

### 5. Try a Prediction

Make sure the server is running, then in another terminal:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@samples/sample_000.wav"
```

Or go to `http://localhost:8000/docs` in your browser to use the FastAPI Swagger UI.

### 6. Sample Files

We include a `samples/` folder with 100 randomly selected `.wav` files from UrbanSound8K for testing. These were copied from the dataset using:

```python
import pathlib, random, shutil
files = list(pathlib.Path('~/sound_datasets/urbansound8k/audio').expanduser().rglob('*.wav')) 
sampled = random.sample(files, 100)
pathlib.Path('samples').mkdir(exist_ok=True)
for i, f in enumerate(sampled):
    shutil.copy(f, f'samples/sample_{i:03}.wav')
```

## 📁 Project Structure

```
audio-cnn-api/
├── app/
│   ├── main.py          # FastAPI app
│   └── model.py         # Model loading & prediction
├── model/               # Saved model + label map
├── samples/             # Sample audio files
├── train/
│   └── train_cnn.py     # Model training script
├── requirements.txt
└── README.md
```

## 📣 Summary

* Built a complete ML inference API using FastAPI
* End-to-end flow: data loading, feature extraction, training, serialization, deployment
* Lightweight approach: minimal dependencies, no cloud infra
* Practiced model serving, curl-based testing, and audio-specific preprocessing
* Included sample data for easy testing and reproducibility via both browser and CLI

## 🧠 About the Author

This project was built as part of my transition into Machine Learning Engineering. I wanted to practice real-world ML deployment skills, including data ingestion, model serving, and API design.

Feel free to reach out!
