# 🎷 Audio Classification API with CNN + FastAPI

A lightweight, end-to-end audio classification system trained on the UrbanSound8K dataset. It uses MFCC spectrograms, a custom PyTorch CNN, and is deployed via a FastAPI web server with support for `.wav` file uploads.

## 🔍 What It Does

* 🎹 Accepts `.wav` audio files via a POST endpoint
* 📊 Converts audio into MFCC spectrograms
* 🧠 Predicts one of 10 common urban sound classes using a trained CNN
* ↺ Returns the predicted label as a JSON response

## 🚀 How to Run It

### 1. Clone the repo and install dependencies

```bash
git clone https://github.com/camillan/audio-cnn-api.git
cd audio-cnn-api
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train/train_cnn.py
```

This will:

* Download UrbanSound8K using `soundata`
* Extract MFCCs
* Train a CNN
* Save the model to `model/cnn_model.pth` and `label_mapping.npy`

### 3. Start the API server

```bash
uvicorn app.main:app --reload
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) to test it via Swagger UI.

---

## 🧪 Test with a Sample `.wav`

Use any `.wav` from:

```bash
~/sound_datasets/urbansound8k/audio/fold1/
```

Or upload your own!

---

## 🛠️ Tech Stack

* `FastAPI` – API for real-time prediction
* `PyTorch` – Lightweight CNN for MFCC classification
* `librosa` – Audio preprocessing
* `soundata` – UrbanSound8K dataset download & management
* `numpy`, `scikit-learn` – Utilities

---

## 💡 Ideas for Improvements

* Add confidence scores or top-3 predictions
* Add Streamlit/Gradio UI
* Containerize with Docker
* Deploy on Hugging Face Spaces or Render
* Support other datasets or languages

---

## 🧠 About the Author

This project was built as part of my transition into Machine Learning Engineering. It demonstrates real-world ML deployment skills, including data ingestion, model serving, and API design.

Feel free to fork or reach out!
