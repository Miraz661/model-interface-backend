from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
import gdown
import os

app = FastAPI()

GOOGLE_DRIVE_FILE_ID = "1C_duCoG3tF1yJEX9ZUXrwJ2AkPlLXfaa"
MODEL_PATH = "deit_model.pth"
class_names = ["Normal", "Pneumonia"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    global model
    model = timm.create_model('deit_base_patch16_224', pretrained=False, num_classes=2)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

        return {
            "class": class_names[predicted_class.item()],
            "confidence": round(confidence.item() * 100, 2),
            "probabilities": {
                class_names[0]: round(probs[0][0].item() * 100, 2),
                class_names[1]: round(probs[0][1].item() * 100, 2),
            }
        }
