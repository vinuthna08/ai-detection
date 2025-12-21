import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import AIDetectorCNN


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Image transforms (IMPORTANT: normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Dataset (expects data/fake and data/real)
    dataset = datasets.ImageFolder("data", transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    print(f"Total images: {len(dataset)}")
    print(f"Classes: {dataset.classes}")

    # Model
    model = AIDetectorCNN().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5
    print("Training started...")

    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "models/ai_detector.pth")
    print("Model saved at models/ai_detector.pth")


if __name__ == "__main__":
    main()
