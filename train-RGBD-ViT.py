import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import RGBDGraspDataset
from utils import get_device

class ViTRGBD(nn.Module):
    def __init__(self, num_classes=5):
        super(ViTRGBD, self).__init__()
        # Load pretrained ViT-B/16
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Modify first conv layer to accept 4 channels (RGB + Depth)
        original_conv = self.vit.conv_proj
        self.vit.conv_proj = nn.Conv2d(4, 768, kernel_size=16, stride=16)
        
        # Initialize the new conv layer with pretrained weights for RGB channels
        with torch.no_grad():
            self.vit.conv_proj.weight[:, :3] = original_conv.weight
            self.vit.conv_proj.weight[:, 3] = original_conv.weight.mean(dim=1)
        
        # Modify the classification head
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

def plot_confusion_matrix(cm, classes, save_path='confusion_matrix.png'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for rgb_maps, depth_maps, labels in tqdm(data_loader, desc="Evaluating"):
            rgb_maps, depth_maps = rgb_maps.to(device), depth_maps.to(device)
            labels = labels.to(device)
            inputs = torch.cat((rgb_maps, depth_maps), dim=1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(data_loader)
    return accuracy, avg_loss, all_predictions, all_labels

if __name__ == '__main__':
    # Training settings
    lr = 1e-3
    num_epochs = 10
    batch_size = 32
    num_classes = 5
    train_split = 0.8  # 80% training, 20% testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset configuration
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

    # Initialize model and training components
    model = ViTRGBD(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Prepare data with train/test split
    full_dataset = RGBDGraspDataset(datasets)
    train_size = int(train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Training metrics tracking
    best_train_accuracy = 0.0
    best_test_accuracy = 0.0
    training_losses = []
    training_accuracies = []
    test_losses = []
    test_accuracies = []
    best_test_predictions = None
    best_test_labels = None

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training phase
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

        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        training_losses.append(train_loss)
        training_accuracies.append(train_accuracy)
        
        # Testing phase
        test_accuracy, test_loss, test_predictions, test_labels = evaluate_model(
            model, test_loader, criterion, device
        )
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        # Update best metrics
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_test_predictions = test_predictions
            best_test_labels = test_labels
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step()

    # Final metrics and visualization
    print("\nTraining Complete!")
    print(f"Best Test Accuracy: {best_test_accuracy:.2f}%")
    print(f"Final Training Loss: {training_losses[-1]:.4f}")
    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    
    # Generate and plot confusion matrix for best test results
    cm = confusion_matrix(best_test_labels, best_test_predictions)
    class_names = [f"Class {i}" for i in range(num_classes)]  # Replace with actual class names
    plot_confusion_matrix(cm, class_names, 'test_confusion_matrix.png')
    
    # Plot training and testing metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(training_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    
    print("\nConfusion Matrix saved as 'test_confusion_matrix.png'")
    print("Training metrics plot saved as 'training_metrics.png'")
