import os
import torch
from PIL import Image

from src.model import AIDetectorCNN
from src.utils import get_transforms


def test_model_prediction():
    """
    Basic test to check if the prediction pipeline runs without error.
    """

    model_path = "models/ai_detector.pth"
    test_image_path = "data/real/00001.jpg"

    assert os.path.exists(model_path), "Model file not found"
    assert os.path.exists(test_image_path), "Test image not found"

    device = torch.device("cpu")

    model = AIDetectorCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = get_transforms(train=False)

    image = Image.open(test_image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    assert output.shape[-1] == 2, "Model output should have 2 classes"
