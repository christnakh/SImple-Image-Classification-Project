import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class_name_to_id = {
    'book': 0, 'bottle': 1, 'car': 2, 'cat': 3, 'chair': 4, 'computermouse': 5, 
    'cup': 6, 'dog': 7, 'flower': 8, 'fork': 9, 'glass': 10, 'glasses': 11, 
    'headphones': 12, 'knife': 13, 'laptop': 14, 'pen': 15, 'plate': 16, 
    'shoes': 17, 'spoon': 18, 'tree': 19
}

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file, delimiter=';')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name).convert('L')
        class_name = self.labels.iloc[idx, 1]
        label = class_name_to_id[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label

def create_data_loaders(csv_file, root_dir, batch_size):
    train_dataset = ImageDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
