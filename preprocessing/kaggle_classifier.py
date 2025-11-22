import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(train_dir="dataset-split/dataset-split/train", val_dir="dataset-split/dataset-split/val", batch_size=32):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset   = datasets.ImageFolder(val_dir, transform=transform_val)

    labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(labels)
    print(f"Class distribution (train): {dict(enumerate(class_counts))}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, class_counts


def get_model(num_classes=2):
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model.to(device)


def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    print("Classification Report:\n", classification_report(all_labels, all_preds))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    return f1



def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, save_path="frame_classifier.pth", early_stopping_patience=3):
    best_f1 = 0.0
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    val_f1s = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"\nEpoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        f1 = f1_score(all_labels, all_preds, average='weighted')
        val_f1s.append(f1)

        print(f"Val Loss: {avg_val_loss:.4f}, F1-score: {f1:.4f}")
        print("Classification Report:\n", classification_report(all_labels, all_preds))
        print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))

        if f1 > best_f1:
            print(f"New best F1: {f1:.4f} (saving model)")
            best_f1 = f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            print(f"No improvement. Patience: {epochs_no_improve}/{early_stopping_patience}")
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    # Plotting
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(val_losses, label="Val Loss", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(val_f1s, label="Val F1-score", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.title("Validation F1-score")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    dataset_root = "/kaggle/input/dataset-split/dataset_split/"
    train_loader, val_loader, class_counts = get_dataloaders(
        train_dir=os.path.join(dataset_root, "train"),
        val_dir=os.path.join(dataset_root, "val"),
        batch_size=32
    )

    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = get_model(num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, save_path="frame_classifier_v5.pth")


if __name__ == "__main__":
    main()
