from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import json
import cv2
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get the preprocessing transforms from the model weights
weights = ViT_B_16_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

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
        self.idx_to_grasp = {v: k for k, v in self.grasp_mapping.items()}

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

        # Read image and convert to RGB
        rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to PIL Image
        rgb_image = Image.fromarray(rgb_image)
        
        # Apply transforms
        rgb_tensor = preprocess(rgb_image)

        label = self.grasp_mapping[label]

        return rgb_tensor, label

class PretrainedViTClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(PretrainedViTClassifier, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Freeze the backbone parameters
        for param in self.vit.parameters():
            param.requires_grad = False
            
        self.vit.heads = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.vit(x)

def evaluate_model(model, data_loader, criterion, device, dataset):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for rgb_maps, labels in tqdm(data_loader, desc="Evaluating"):
            rgb_maps = rgb_maps.to(device)
            labels = labels.to(device)
            
            outputs = model(rgb_maps)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[dataset.idx_to_grasp[i] for i in range(len(dataset.grasp_mapping))],
                yticklabels=[dataset.idx_to_grasp[i] for i in range(len(dataset.grasp_mapping))])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return accuracy, avg_loss

if __name__ == '__main__':
    datasets = [
        {
            'image_path': '../data/DeepGrasping_JustImages',
            'depth_path': None,
            'anno_path': '../data/DeepGrasping_Anno',
            'image_subdirs': [f'{i:02}' for i in range(1, 11)],
        },
        {
            'image_path': '../data/Imagenet',
            'depth_path': None,
            'anno_path': '../data/Anno_ImageNet.json',
        },
        {
            'image_path': '../data/HandCam',
            'depth_path': None,
            'anno_path': '../data/Anno_HandCam4.json',
        }
    ]

    # Create full dataset
    full_dataset = RGBDGraspDataset(datasets)
    
    # Split dataset into train and test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PretrainedViTClassifier().to(device)

    # Hyperparameters
    lr = 1e-3
    num_epochs = 10
    batch_size = 32

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.vit.heads.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Training loop
    best_accuracy = 0
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
        train_loss = running_loss / len(train_loader)
        print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        
        # Evaluate on test set
        test_accuracy, test_loss = evaluate_model(model, test_loader, criterion, device, full_dataset)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        # Update learning rate
        scheduler.step()
        
        # Save best model based on test accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), "vit_grasp_best.pth")

    print(f"Best test accuracy: {best_accuracy:.2f}%")
    
    # Final evaluation
    model.load_state_dict(torch.load("vit_grasp_best.pth"))
    final_test_accuracy, final_test_loss = evaluate_model(model, test_loader, criterion, device, full_dataset)
    print(f"\nFinal Test Results:")
    print(f"Loss: {final_test_loss:.4f}")
    print(f"Accuracy: {final_test_accuracy:.2f}%")
