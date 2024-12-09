import torch
from vit_pytorch import ViT

import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from dataset import RGBDGraspDataset
from utils import get_device

"""
-- To run on GPU --
1) ``cd .\final_project\vish\code\``
2) ``conda activate torch-gpu``
3) ``python .\train-RGB-ViT.py``
"""

# -- define model -- 
class ViTRGBD(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=5, dim=1024, depth=6, heads=8, mlp_dim=2048):
        super(ViTRGBD, self).__init__()
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=4  # RGB + Depth
        )

    def forward(self, x):
        return self.vit(x)

if __name__ == '__main__':
    # -- example usage -- 
    model = ViTRGBD()

    # define data
    datasets = [
        {
            'image_path': '../data/DeepGrasping_JustImages',
            'depth_path': '../data/DEPTH_DeepGrasping_JustImages',
            'anno_path': '../data/DeepGrasping_Anno',
            'image_subdirs': [f'{i:02}' for i in range(1, 11)],
        },
        {
            'image_path': '../data/Imagenet',
            'depth_path': '../data/DEPTH_Imagenet',
            'anno_path': '../data/Anno_ImageNet.json',
        },
        {
            'image_path': '../data/HandCam',
            'depth_path': '../data/DEPTH_HandCam',
            'anno_path': '../data/Anno_HandCam4.json',
        }
    ]

    # -- deep learning -- 
    lr = 1e-3
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
    
	# -- Accuracy: _____ %
	# lr = 1e-4  
    # num_epochs = 10  
    # batch_size = 8 

	# -- Accuracy: _____ %
    # lr = 1e-4  
    # num_epochs = 10  
    # batch_size = 64
    


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr,
        weight_decay=1e-4  # Added weight decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_dataset = RGBDGraspDataset(datasets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # grabbing weights from ViT
    model = ViTRGBD().to(device) 
    for name, param in model.named_parameters():
        param.requires_grad = True  # Ensure all parameters require gradients

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0

        for rgb_maps, depth_maps, labels in tqdm(train_loader, desc="Training"):
            rgb_maps, depth_maps = rgb_maps.to(device), depth_maps.to(device)
            labels = labels.to(device)
            inputs = torch.cat((rgb_maps, depth_maps), dim=1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f"Training Loss: {running_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")
        
        # Step the scheduler
        scheduler.step()