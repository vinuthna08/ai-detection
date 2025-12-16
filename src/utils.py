import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RealFakeImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir:
            data/
              ├── real/
              └── fake/
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for label, folder in enumerate(["real", "fake"]):
            folder_path = os.path.join(root_dir, folder)
            for img_name in os.listdir(folder_path):
                self.samples.append(
                    (os.path.join(folder_path, img_name), label)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
