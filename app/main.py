from fastapi import FastAPI, UploadFile, File
from app.utils import audio_to_mfcc
from app.model import predict_mfcc_tensor

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Audio classifier is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    mfcc_tensor = audio_to_mfcc(audio_bytes)
    prediction = predict_mfcc_tensor(mfcc_tensor)
    return {"prediction": prediction}