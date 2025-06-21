from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
class_names = ["Normal", "Pneumonia"]  # ✅ Change as per your classes

model = timm.create_model('deit_base_patch16_224', pretrained=False, num_classes=2)
checkpoint = torch.load("D:\Research\deit_model_b16_01.pth", map_location="cpu")
model.load_state_dict(checkpoint['model_state'])  # if you saved full checkpoint
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # optional, based on your training
])


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

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
