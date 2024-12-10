import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Assuming the necessary imports and class definitions are already present

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

    # Generate confusion matrix
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

if __name__ == '__main__':
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

    train_dataset = RGBDGraspDataset(datasets[:2])
    test_dataset = RGBDGraspDataset([datasets[2]])

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
        print(f"Training Loss: {running_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

        # Evaluate on test set
        test_accuracy, test_loss = evaluate_model(model, test_loader, criterion, device, test_dataset)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved with accuracy: {best_accuracy:.2f}%")

        scheduler.step()

    print("\nTraining Complete!")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
