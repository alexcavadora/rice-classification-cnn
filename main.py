import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.nn import CrossEntropyLoss, NLLLoss
import torch.nn.functional as F
from collections import Counter
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import torchsummary
import argparse
import os
from pytorch_utils import get_device

from models.alexnet import AlexNet, AlexNetModified
from models.lenet import LeNet, LeNetModified
from models.vggnet import VGGNet, VGGNetModified

import time
from datetime import timedelta

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate CNN models on rice dataset")
    parser.add_argument('--model', type=str, default='VGG16Net', choices=['AlexNet', 'LeNet', 'VGG16Net'],
                        help='Model to train (AlexNet, LeNet, VGG16Net)')
    parser.add_argument('--modified', action='store_true',
                        help='Use modified version of the model (False for standard, True for modified)')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'RMSprop', 'AdamW'],
                        help='Optimizer to use (default: Adam)')
    parser.add_argument('--criterion', type=str, default='CrossEntropy', choices=['CrossEntropy', 'NLLLoss'],
                        help='Loss function to use (default: CrossEntropy)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--train-split', type=float, default=0.8, help='Training split ratio')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--data-path', type=str, default='rice', help='Path to dataset')
    return parser.parse_args()

def create_directories(base_path):
    """Create necessary directories if they don't exist"""
    os.makedirs(base_path, exist_ok=True)
    os.makedirs("trained_models", exist_ok=True)
    print(f"[INFO] Created directory: {base_path}")

def media_std(path_db, batch_size=64, workers=4):
    transformaciones = transforms.ToTensor()
    DB = datasets.ImageFolder(path_db, transform=transformaciones)
    DB_loader = DataLoader(DB, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)

    media = 0.0
    std = 0.0
    imgs_totales = 0

    for imagenes, _ in DB_loader:
        batch_samples = imagenes.size(0)
        imagenes = imagenes.view(batch_samples, imagenes.size(1), -1)
        media += imagenes.mean(2).sum(0)
        std += imagenes.std(2).sum(0)
        imgs_totales += batch_samples

    media /= imgs_totales
    std /= imgs_totales
    print(f'[INFO] Loaded images: {imgs_totales}')
    return media, std

def get_optimizer(optimizer_name, model_parameters, lr, weight_decay, momentum=0.9):
    """Return the specified optimizer"""
    if optimizer_name == 'Adam':
        return Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return SGD(model_parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == 'RMSprop':
        return RMSprop(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        return AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_criterion(criterion_name):
    """Return the specified loss function"""
    if criterion_name == 'CrossEntropy':
        return CrossEntropyLoss()
    elif criterion_name == 'NLLLoss':
        return NLLLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")

def plot_confusion_matrix(model, dataloader, device, class_names=None, title="Confusion Matrix", save_path=''):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Saved confusion matrix to {save_path}")
    plt.close()

    return cm

def main():
    args = parse_args()
    device = get_device()

    model_suffix = "_modified" if args.modified else ""
    model_variant = "Modified" if args.modified else "Standard"

    results_folder = f"results/{args.model.lower()}{model_suffix}"
    create_directories(results_folder)

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Model: {model_variant} {args.model}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Criterion: {args.criterion}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Results Path: {results_folder}")
    print(f"{'='*60}\n")

    mu, std = media_std(args.data_path, batch_size=args.batch_size)
    print(f"[INFO] Mean: {mu}, Standard Deviation: {std}")

    transformaciones = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1339, 0.1344, 0.1389], std=[0.3030, 0.3041, 0.3172])
    ])

    DB = datasets.ImageFolder(args.data_path, transform=transformaciones)
    total_size = len(DB)
    labels = [label for _, label in DB]
    class_counts = Counter(labels)

    print(f"\n[INFO] Dataset Statistics:")
    for id, cls_name in enumerate(DB.classes):
        print(f"  Class '{cls_name}': {class_counts[id]} images")

    train_size = int(args.train_split * total_size)
    val_size = int(args.val_split * total_size)
    test_size = total_size - train_size - val_size

    train_data, val_data, test_data = random_split(
        DB, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    trainDataLoader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valDataLoader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    testDataLoader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    print(f"\n[INFO] Dataset Splits:")
    print(f"  Training set: {len(train_data)} ({len(train_data)/total_size*100:.1f}%)")
    print(f"  Validation set: {len(val_data)} ({len(val_data)/total_size*100:.1f}%)")
    print(f"  Test set: {len(test_data)} ({len(test_data)/total_size*100:.1f}%)")
    print(f"  Number of classes: {len(DB.classes)}")

    print(f"\n[INFO] Initializing {model_variant} {args.model}...")

    if args.model == 'AlexNet':
        model = AlexNetModified(nChannels=3, nClases=len(DB.classes)) if args.modified else AlexNet(nChannels=3, nClases=len(DB.classes))
    elif args.model == 'LeNet':
        model = LeNetModified(nChannels=3, nClases=len(DB.classes)) if args.modified else LeNet(nChannels=3, nClases=len(DB.classes))
    else:  # VGG16Net
        model = VGGNetModified(nChannels=3, nClasses=len(DB.classes)) if args.modified else VGGNet(nChannels=3, nClasses=len(DB.classes))

    model = model.to(device)

    print(f"[INFO] Input shape: {trainDataLoader.dataset[0][0].shape}")
    print(f"\n[INFO] Model Architecture:")
    print(model)
    print(f"\n[INFO] Model Summary:")
    torchsummary.summary(model, input_size=trainDataLoader.dataset[0][0].shape)


    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay, args.momentum)
    criterion = get_criterion(args.criterion)

    print(f"\n[INFO] Training with {args.optimizer} optimizer and {args.criterion} loss")

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'epoch_times': []}

    patience = args.patience
    best_val_loss = np.inf
    counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    model_path = f"trained_models/{args.model.lower()}{model_suffix}_best.pth"

    print(f"\n{'='*60}")
    print("Starting Training...")
    print(f"{'='*60}\n")

    total_training_start = time.perf_counter()

    for e in range(args.epochs):
        epoch_start_time = time.perf_counter()
        model.train()
        total_train_loss = 0.0
        total_val_loss = 0.0
        train_correct = 0.0
        val_correct = 0.0

        # Training loop
        for imgs, labels in trainDataLoader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_correct += (preds.argmax(dim=1) == labels).type(torch.float).sum().item()

        # Validation loop
        model.eval()
        with torch.no_grad():
            for val_imgs, val_labels in valDataLoader:
                val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
                preds_vals = model(val_imgs)
                loss_val = criterion(preds_vals, val_labels)
                total_val_loss += loss_val.item()
                val_correct += (preds_vals.argmax(dim=1) == val_labels).type(torch.float).sum().item()

        # Calculate metrics
        epoch_duration = time.perf_counter() - epoch_start_time
        avg_train_loss = total_train_loss / len(trainDataLoader)
        avg_val_loss = total_val_loss / len(valDataLoader)
        avg_train_acc = train_correct / len(trainDataLoader.dataset)
        avg_val_acc = val_correct / len(valDataLoader.dataset)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        history['epoch_times'].append(epoch_duration)

        print(f'[INFO] Epoch {e+1:2d}/{args.epochs} | '
              f'Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f} | '
              f'Time: {timedelta(seconds=int(epoch_duration))}')

        # Early stopping + save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved (Val Loss: {best_val_loss:.4f}) -> {model_path}")
            counter = 0
        else:
            counter += 1
            print(f"No improvement ({counter}/{patience} epochs)")

        if counter >= patience:
            print(f"\n⏹️  Early stopping triggered after {e+1} epochs")
            break

        torch.cuda.empty_cache()

    total_training_time = time.perf_counter() - total_training_start
    avg_epoch_time = np.mean(history['epoch_times'])

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Total Training Time: {timedelta(seconds=int(total_training_time))}")
    print(f"  Average Epoch Time: {timedelta(seconds=int(avg_epoch_time))}")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")

    # Load best model
    model.load_state_dict(best_model_wts)
    model.eval()

    # Evaluate on test set
    print("[INFO] Evaluating on test set...")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for test_imgs, test_labels in testDataLoader:
            test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)
            preds_vals = model(test_imgs)
            preds_classes = preds_vals.argmax(dim=1)
            all_preds.extend(preds_classes.cpu().numpy())
            all_labels.extend(test_labels.cpu().numpy())

    test_macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"\n{'='*60}")
    print(f"Test Set Results:")
    print(f"  Macro F1-Score: {test_macro_f1:.4f}")
    print(f"{'='*60}\n")

    # Print detailed classification report
    print("[INFO] Detailed Classification Report (Test Set):")
    print(classification_report(all_labels, all_preds, target_names=DB.classes))

    # Save classification report
    report_path = f"{results_folder}/classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Model: {model_variant} {args.model}\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"Criterion: {args.criterion}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
        f.write(f"Test Macro F1-Score: {test_macro_f1:.4f}\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=DB.classes))
    print(f"[INFO] Classification report saved to {report_path}")

    # Plot training metrics
    plt.style.use('ggplot')

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', color='red', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', color='blue', linewidth=2)
    plt.title(f'Training and Validation Loss\n{model_variant} {args.model} - {args.optimizer}', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_plot_path = f"{results_folder}/loss_plot.png"
    plt.savefig(loss_plot_path, dpi=150)
    print(f"[INFO] Loss plot saved to {loss_plot_path}")
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label='Train Accuracy', color='red', linewidth=2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='blue', linewidth=2)
    plt.title(f'Training and Validation Accuracy\n{model_variant} {args.model} - {args.optimizer}', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_plot_path = f"{results_folder}/acc_plot.png"
    plt.savefig(acc_plot_path, dpi=150)
    print(f"[INFO] Accuracy plot saved to {acc_plot_path}")
    plt.close()

    # Plot confusion matrices
    print("\n[INFO] Generating confusion matrices...")
    plot_confusion_matrix(
        model, trainDataLoader, device,
        class_names=DB.classes,
        title=f"Train Confusion Matrix\n{model_variant} {args.model}",
        save_path=f"{results_folder}/train_confusion_matrix.png"
    )
    plot_confusion_matrix(
        model, valDataLoader, device,
        class_names=DB.classes,
        title=f"Validation Confusion Matrix\n{model_variant} {args.model}",
        save_path=f"{results_folder}/val_confusion_matrix.png"
    )
    plot_confusion_matrix(
        model, testDataLoader, device,
        class_names=DB.classes,
        title=f"Test Confusion Matrix\n{model_variant} {args.model}",
        save_path=f"{results_folder}/test_confusion_matrix.png"
    )

    print(f"\n{'='*60}")
    print(f"All results saved to: {results_folder}")
    print(f"Model weights saved to: {model_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
