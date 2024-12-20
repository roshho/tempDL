import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms, models
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class RGBGraspDataset(Dataset):
    def __init__(self, dataset_configs):
        self.configs = dataset_configs
        self.samples = []
        self.grasp_mapping = {}  
        self.idx_to_grasp = {}   
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self._load_all_datasets()
        
    def _load_all_datasets(self):
        for config in self.configs:
            if 'image_subdirs' in config:
                for subdir in config['image_subdirs']:
                    img_dir = os.path.join(config['image_path'], subdir)
                    anno_dir = config['anno_path']
                    
                    for img_name in os.listdir(img_dir):
                        if img_name.endswith(('.jpg', '.png')):
                            img_path = os.path.join(img_dir, img_name)
                            anno_path = os.path.join(anno_dir, f"anno_{subdir}.json")
                            
                            if os.path.exists(anno_path):
                                with open(anno_path, 'r') as f:
                                    annotations = json.load(f)
                                
                                if img_name in annotations:
                                    anno = annotations[img_name]
                                    grasp_type = anno['grip']
                                    if grasp_type not in self.grasp_mapping:
                                        idx = len(self.grasp_mapping)
                                        self.grasp_mapping[grasp_type] = idx
                                        self.idx_to_grasp[idx] = grasp_type
                                    
                                    self.samples.append({
                                        'rgb_path': img_path,
                                        'label': self.grasp_mapping[grasp_type]
                                    })
            else:
                with open(config['anno_path'], 'r') as f:
                    annotations = json.load(f)
                
                for img_name, anno in annotations.items():
                    img_path = os.path.join(config['image_path'], img_name)
                    
                    grasp_type = anno['grip']
                    if grasp_type not in self.grasp_mapping:
                        idx = len(self.grasp_mapping)
                        self.grasp_mapping[grasp_type] = idx
                        self.idx_to_grasp[idx] = grasp_type
                    
                    self.samples.append({
                        'rgb_path': img_path,
                        'label': self.grasp_mapping[grasp_type]
                    })
        
        print(f"Dataset size: {len(self.samples)}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and transform RGB image
        rgb_image = Image.open(sample['rgb_path']).convert('RGB')
        rgb_tensor = self.transform(rgb_image)
        
        return rgb_tensor, sample['label']

class PretrainedViTClassifier(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        # Load pretrained ViT
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)
        
    def forward(self, x):
        return self.vit(x)

def evaluate_model(model, data_loader, criterion, device, dataset):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for rgb_maps, labels in data_loader:
            rgb_maps, labels = rgb_maps.to(device), labels.to(device)
            outputs = model(rgb_maps)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(data_loader)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[dataset.idx_to_grasp[i] for i in range(len(dataset.grasp_mapping))],
                yticklabels=[dataset.idx_to_grasp[i] for i in range(len(dataset.grasp_mapping))])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    return accuracy, avg_loss

def train_and_evaluate(train_dataset, test_dataset, num_epochs=10, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PretrainedViTClassifier(num_classes=len(train_dataset.grasp_mapping)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_accuracy = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0

        for rgb_maps, labels in tqdm(train_loader, desc="Training"):
            rgb_maps, labels = rgb_maps.to(device), labels.to(device)
            
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

        # Evaluate on test set
        test_accuracy, test_loss = evaluate_model(model, test_loader, criterion, device, test_dataset)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        # Save the best model - TBC
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved with accuracy: {best_accuracy:.2f}%")

        scheduler.step()

    print("\nTraining Complete!")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")

if __name__ == '__main__':
    datasets = [
        {
            'image_path': '../data/DeepGrasping_JustImages',
            'anno_path': '../data/DeepGrasping_Anno',
            'image_subdirs': [f'{i:02}' for i in range(1, 11)],
        },
        {
            'image_path': '../data/Imagenet',
            'anno_path': '../data/Anno_ImageNet.json',
        },
        {
            'image_path': '../data/HandCam',
            'anno_path': '../data/Anno_HandCam4.json',
        }
    ]

    train_dataset = RGBGraspDataset(datasets[:2])
    test_dataset = RGBGraspDataset([datasets[2]])
    train_and_evaluate(train_dataset, test_dataset)
