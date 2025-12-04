import torch
from PIL import Image
from torchvision import transforms
from model import RealFakeCNN


def predict(image_path, model_path="models/model.pth"):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    model = RealFakeCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        output = model(x).item()

    print(f"Prediction Score (0 = Real, 1 = AI-generated): {output:.4f}")
    return output


if __name__ == "__main__":
    import sys
    predict(sys.argv[1])
