import requests

with open("samples/sample_000.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("sample_000.wav", f, "audio/wav")},
    )

print(response.json())
