from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from vit_pytorch import ViT
import os
import json
import cv2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

class RGBDGraspDataset(Dataset):
    def __init__(self, datasets):
        self.data = []
        self.grasp_mapping = {
            "3 jaw chuck": 0,
            "key": 1,
            "pinch": 2,
            "power": 3,
            "tool": 4
        }

        for dataset in datasets:
            image_path = dataset['image_path']
            depth_path = dataset.get('depth_path', None)
            anno_path = dataset['anno_path']
            subdirs = dataset.get('image_subdirs', None)

            if os.path.isdir(anno_path):
                anno_files = [os.path.join(anno_path, f) for f in os.listdir(anno_path) if f.endswith('.json')]
                annotations = {}
                for file in anno_files:
                    with open(file, 'r') as f:
                        annotations.update(json.load(f))
            else:
                with open(anno_path, 'r') as f:
                    annotations = json.load(f)

            if subdirs:
                for subdir in subdirs:
                    full_image_dir = os.path.join(image_path, subdir)
                    full_depth_dir = os.path.join(depth_path, subdir) if depth_path else None
                    for img_file in os.listdir(full_image_dir):
                        if img_file in annotations:
                            rgb_path = os.path.join(full_image_dir, img_file)
                            depth_file = os.path.splitext(img_file)[0] + "_depth.png"
                            depth_filepath = os.path.join(full_depth_dir, depth_file) if full_depth_dir else None

                            label = annotations[img_file]["grip"]
                            if label != 'None' and os.path.exists(rgb_path) and (depth_filepath is None or os.path.exists(depth_filepath)):
                                self.data.append((rgb_path, depth_filepath, label))
            else:
                for img_file in os.listdir(image_path):
                    if img_file in annotations:
                        rgb_path = os.path.join(image_path, img_file)
                        depth_file = os.path.splitext(img_file)[0] + "_depth.png"
                        depth_filepath = os.path.join(depth_path, depth_file) if depth_path else None

                        label = annotations[img_file]["grip"]
                        if label != 'None' and os.path.exists(rgb_path) and (depth_filepath is None or os.path.exists(depth_filepath)):
                            self.data.append((rgb_path, depth_filepath, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_path, depth_path, label = self.data[idx]

        rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_tensor = transform(rgb_image)

        label = self.grasp_mapping[label]

        return rgb_tensor, label

# Define ViT model for RGB
class ViTRGB(nn.Module):
    def __init__(self, num_classes=5):
        super(ViTRGB, self).__init__()
        self.vit = ViT(
            image_size=224,
            patch_size=16,
            num_classes=num_classes,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=2048,
            channels=3  # RGB only
        )

    def forward(self, x):
        return self.vit(x)

if __name__ == '__main__':
    datasets = [
        {
            'image_path': '../data/DeepGrasping_JustImages',
            'depth_path': None,  # No depth data
            'anno_path': '../data/DeepGrasping_Anno',
            'image_subdirs': [f'{i:02}' for i in range(1, 11)],
        },
        {
            'image_path': '../data/Imagenet',
            'depth_path': None,  # No depth data
            'anno_path': '../data/Anno_ImageNet.json',
        },
        {
            'image_path': '../data/HandCam',
            'depth_path': None,  # No depth data
            'anno_path': '../data/Anno_HandCam4.json',
        }
    ]

    train_dataset = RGBDGraspDataset(datasets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTRGB().to(device)

    lr = 1e-3  # Slightly lower learning rate for transformer
    num_epochs = 10
    batch_size = 8
    
		# -- Accuracy: _____ % --
    # lr = 1e-3 
    # num_epochs = 10  
    # batch_size = 8
    
	# -- Accuracy: ____ % --
	# lr = 1e-3  
    # num_epochs = 10  
    # batch_size = 16  
    
	# -- Accuracy: 54.87 %
	# lr = 1e-4  
    # num_epochs = 10  
    # batch_size = 8 

	# -- Accuracy: _____ %
    # lr = 1e-4  
    # num_epochs = 10  
    # batch_size = 64

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0

        for rgb_maps, labels in tqdm(train_loader, desc="Training"):
            rgb_maps = rgb_maps.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(rgb_maps)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f"Training Loss: {running_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

    # Save the model parameters
    torch.save(model.state_dict(), "vit_rgb_model.pth")
    print("Model parameters saved to vit_rgb_model.pth")