import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import RealFakeCNN


def main():

    # Transform images to tensors
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load dataset from data/ folder
    dataset = datasets.ImageFolder('data/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Model, loss, optimizer
    model = RealFakeCNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training started...")

    model.train()
    for epoch in range(1, 3):  
        running_loss = 0.0
        
        for images, labels in dataloader:
            labels = labels.float().unsqueeze(1)
            
            preds = model(images)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch {epoch} - Loss: {running_loss / len(dataloader):.4f}")

    # Save model
    torch.save(model.state_dict(), "models/model.pth")
    print("Model saved to models/model.pth")


if __name__ == "__main__":
    main()
