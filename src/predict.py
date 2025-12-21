import sys
import torch
from torchvision import transforms
from PIL import Image

from src.model import AIDetectorCNN


def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AIDetectorCNN().to(device)
    model.load_state_dict(
        torch.load("models/ai_detector.pth", map_location=device)
    )
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = "REAL" if predicted.item() == 0 else "FAKE"
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence.item():.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict <image_path>")
        sys.exit(1)

    predict(sys.argv[1])
