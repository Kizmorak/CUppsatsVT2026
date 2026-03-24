import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn.functional as F
import argparse
from enum import Enum

class ModelOutputClasses(Enum):
    A_B = "A/B"
    A_B_NONE = "A/B/NONE"

class DatasetPaths(Enum):
    DATASET_ALL_TIs = "dataset_all_TIs"
    DATASET_NO_TIs = "dataset_no_TIs"
    DATASET_2_TIs = "dataset_2_TIs"
    def __str__(self):        
        return str(self.value)

# -------------------------
# Variables and hyperparameters
# -------------------------
# Model and training parameters
model_name = "convnextv2_atto"
default_model_output_classes = ModelOutputClasses.A_B_NONE
default_dataset_path = DatasetPaths.DATASET_ALL_TIs  # Path to your dataset folder containing 'train', 'val' and 'backtest' subfolders

# Training parameters
default_num_epochs = 10
expected_none_ratio = 15/18  # Expected ratio of NONE samples in the backtesting set (used for auto-tuning thresholds)
#Maybe do: number of layers to unfreeze for fine-tuning (0 = only head, 1 = last stage + head, 2 = last 2 stages + head, etc.)

# Augmentation options
use_random_affine = True

# Hyperparameters
workers_cpu = 0
workers_gpu = 4
batch_size_cpu = 8
batch_size_gpu = 16

# Auto-tuning options for open-set NONE thresholding
auto_tune_none_thresholds = True



# -------------------------
# Device
# -------------------------
def device_spec_setup():
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #optimize based on hardware
    batch_size = batch_size_cpu if device.type == "cpu" else batch_size_gpu
    num_workers = workers_cpu if device.type == "cpu" else workers_gpu
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    return device, batch_size, num_workers

# -------------------------
# Transforms
# -------------------------
def transforms_setup():
    # Define the mean and standard deviation for normalization (these are commonly used values for pre-trained models)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # Limited data augmentation for fine-tuning
    if use_random_affine:
        train_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                scale=(0.97, 1.03)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # Validation/backtesting should be deterministic (no random augmentation).
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_transform, eval_transform

# -------------------------
# Dataset
# -------------------------
def dataset_setup(batch_size, num_workers, dataset_path):
    train_transform, eval_transform = transforms_setup()
    
    train_dataset = ImageFolder(root=f"{dataset_path}/train", transform=train_transform)
    val_dataset = ImageFolder(root=f"{dataset_path}/val", transform=eval_transform)

    print(f"Dataset path '{dataset_path}':")

    # Balanced sampler
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets)
    class_weights = np.divide(1.0, class_counts, out=np.zeros_like(class_counts, dtype=float), where=class_counts > 0)
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    print("Train class counts:", {train_dataset.classes[i]: int(c) for i, c in enumerate(class_counts)})
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataset, train_loader, val_loader

# -------------------------
# Model
# -------------------------
def model_setup(device, model_output_classes):
    model = timm.create_model(
        model_name,
        pretrained=True
    )

    # Compatibility fallback for timm variants where ConvNeXt may miss norm_pre.
    # (convnextv2_atto seems to have it, but this ensures the code works across more timm versions without modification)
    if not hasattr(model, "norm_pre"):
        model.norm_pre = nn.Identity()
    # says copilot to fix a bug...


    model.reset_classifier(2 if model_output_classes == ModelOutputClasses.A_B else 1)
    model.to(device)

    print(f"Model: {model_name}, Output class: {model_output_classes}")

    # Freeze all layers except the head
    for p in model.parameters():
        p.requires_grad = False

    for p in model.head.parameters():
        p.requires_grad = True

    # Unfreeze the last stage of the model 
    for name, p in model.named_parameters():
        if "stages.3" in name:
            p.requires_grad = True
    return model

# -------------------------
# Loss + optimizer
# -------------------------
# Define the optimizer (AdamW is commonly used for training vision models)
def optimizer_setup(model):
    lr = 1e-4
    weight_decay = 0.05
    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": lr},
        {"params": model.stages[3].parameters(), "lr": lr * 0.1},
    ], weight_decay=weight_decay)
    return optimizer

# -------------------------
# Scheduler
# -------------------------
def scheduler_setup(optimizer, num_epochs):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    return scheduler

# -------------------------
# Training
# -------------------------
def train(model, train_loader, val_loader, optimizer, scheduler, device, model_output_classes, num_epochs):

    criterion = nn.BCEWithLogitsLoss() if model_output_classes == ModelOutputClasses.A_B_NONE else nn.CrossEntropyLoss()
    if model_output_classes == ModelOutputClasses.A_B:
        # Training loop for 2 classes with standard CrossEntropyLoss
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)

            # validation
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:

                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    preds = outputs.argmax(1)

                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = correct / total

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

            scheduler.step()
    else:
        # Training loop for binary classification with probability output
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)

            # validation
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    probs = torch.sigmoid(outputs).squeeze(1)
                    preds = (probs > 0.5).long()

                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = correct / total

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

            scheduler.step()

# -------------------------
# Evaluation on val (A/B)
# -------------------------
def evaluate_A_B(model, val_loader, device, model_output_classes, class_names):
    val_preds = []
    val_labels = []
    val_confidences = []

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            if model_output_classes == ModelOutputClasses.A_B:
                preds = outputs.argmax(1)
                confidences = F.softmax(outputs, dim=1).max(1)[0]
            else:
                probs = torch.sigmoid(outputs).squeeze(1)
                preds = (probs > 0.5).long()
                confidences = torch.where(preds == 1, probs, 1 - probs)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            val_confidences.extend(confidences.cpu().numpy())

    cm_val = confusion_matrix(val_labels, val_preds)
    print("\nValidation (A/B) confusion matrix:")
    print(cm_val)

    with open("predictions_val.txt", "w", encoding="utf-8") as f:
        f.write("Sample | Predicted | Confidence | Actual | Correct\n")
        f.write("-" * 70 + "\n")
        for i, (pred, conf, label) in enumerate(zip(val_preds, val_confidences, val_labels)):
            is_correct = "✓" if pred == label else "✗"
            f.write(f"{i:4d}   | {class_names[pred]:15s} | {conf:.4f}     | {class_names[label]:15s} | {is_correct}\n")

    print("Validation records saved to predictions_val.txt")
    print("Validation summary:")
    print(f"Total samples: {len(val_preds)}")
    print(f"Correct predictions: {sum(np.array(val_preds) == np.array(val_labels))}")
    print(f"Accuracy: {sum(np.array(val_preds) == np.array(val_labels)) / len(val_preds):.4f}")
    print(f"Average confidence: {np.mean(val_confidences):.4f}")

# -------------------------
# Evaluation on none_val (A/B/NONE)
# -------------------------
def evaluate_A_B_NONE(model, device, dataset_path, batch_size, num_workers, eval_transform, class_names):
    none_val_dataset = ImageFolder(root=f"{dataset_path}/backtesting", transform=eval_transform)
    none_val_loader = DataLoader(none_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    none_label_name = "NONE"
    none_probs = []
    none_labels = []

    def label_to_name_none(label_idx):
        if label_idx == 2:
            return none_label_name
        if 0 <= label_idx < len(class_names):
            return class_names[label_idx]
        return f"UNKNOWN_{label_idx}"

    def class_name_to_open_set_label(name):
        if name == "noMovement":
            return 2
        if name == "upMovement":
            return 1
        if name == "downMovement":
            return 0
        return None


    # Map none_val folder class indices to open-set label ids [0:down/A, 1:up/B, 2:NONE].
    none_val_to_open_label = {}
    for idx, cls_name in enumerate(none_val_dataset.classes):
        mapped = class_name_to_open_set_label(cls_name)
        if mapped is None:
            raise ValueError(f"Could not map none_val class '{cls_name}' to A/B/NONE")
        none_val_to_open_label[idx] = mapped


    with torch.no_grad():
        for images, labels in none_val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1).flatten()

            none_probs.extend(probs.cpu().numpy().tolist())
            mapped_labels = [none_val_to_open_label[int(lbl)] for lbl in labels.cpu().numpy()]
            none_labels.extend(mapped_labels)

    high_confidence_threshold = 0.7
    low_confidence_threshold = 0.3

    if auto_tune_none_thresholds and len(none_probs) > 0:
        tail_ratio = max(0.0, min(0.49, (1.0 - expected_none_ratio) / 2.0))
        low_confidence_threshold = float(np.quantile(none_probs, tail_ratio))
        high_confidence_threshold = float(np.quantile(none_probs, 1.0 - tail_ratio))


    def predict_open_set(prob, low, high):
        if prob <= low:
            return 0
        if prob >= high:
            return 1
        return 2


    none_preds = [predict_open_set(prob, low_confidence_threshold, high_confidence_threshold) for prob in none_probs]
    cm_none = confusion_matrix(none_labels, none_preds, labels=[0, 1, 2])
    none_labels_np = np.array(none_labels)
    none_preds_np = np.array(none_preds)
    none_probs_np = np.array(none_probs)

    print("\nOpen-set (A/B/NONE) confusion matrix:")
    print(cm_none)
    print("Matrix labels order:", [label_to_name_none(i) for i in [0, 1, 2]])
    print("none_val class mapping:", {k: f"{none_val_dataset.classes[k]} -> {label_to_name_none(v)}" for k, v in none_val_to_open_label.items()})
    print(f"Thresholds used: low={low_confidence_threshold:.4f}, high={high_confidence_threshold:.4f}")

    none_accuracy = sum(none_preds_np == none_labels_np) / len(none_preds)
    print("Open-set summary:")
    print(f"Total samples: {len(none_preds)}")
    print(f"Correct predictions: {sum(none_preds_np == none_labels_np)}")
    print(f"Accuracy: {none_accuracy:.4f}")
    print(f"Average confidence: {np.mean(none_probs):.4f}")

    print("\nOpen-set confidence stats by predicted class:")
    for class_idx in [0, 1, 2]:
        class_name = label_to_name_none(class_idx)
        class_conf = none_probs_np[none_preds_np == class_idx]
        if class_conf.size == 0:
            print(f"{class_name:15s} | n=0")
        else:
            print(
                f"{class_name:15s} | n={class_conf.size:4d} | "
                f"min={class_conf.min():.4f} | mean={class_conf.mean():.4f} | max={class_conf.max():.4f}"
            )

    print("\nOpen-set confidence stats by true class:")
    for class_idx in [0, 1, 2]:
        class_name = label_to_name_none(class_idx)
        mask = (none_labels_np == class_idx)
        class_conf = none_probs_np[mask]
        if class_conf.size == 0:
            print(f"{class_name:15s} | n=0")
        else:
            class_acc = np.mean(none_preds_np[mask] == class_idx)
            print(
                f"{class_name:15s} | n={class_conf.size:4d} | "
                f"mean_prob={class_conf.mean():.4f} | std_prob={class_conf.std():.4f} | recall={class_acc:.4f}"
            )

    # Per-class metrics that are robust to class imbalance
    tp = np.diag(cm_none).astype(float)
    support = cm_none.sum(axis=1).astype(float)
    pred_count = cm_none.sum(axis=0).astype(float)

    recall_per_class = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision_per_class = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(tp),
        where=(precision_per_class + recall_per_class) > 0,
    )

    balanced_accuracy = float(np.mean(recall_per_class))
    macro_f1 = float(np.mean(f1_per_class))

    # Prior-adjusted accuracy using class recalls (more informative when classes are imbalanced)
    equal_prior = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    deploy_prior = np.array([
        (1.0 - expected_none_ratio) / 2.0,
        (1.0 - expected_none_ratio) / 2.0,
        expected_none_ratio,
    ], dtype=float)
    deploy_prior = np.clip(deploy_prior, 0.0, 1.0)
    deploy_prior = deploy_prior / deploy_prior.sum()

    adjusted_acc_equal = float(np.sum(recall_per_class * equal_prior))
    adjusted_acc_deploy = float(np.sum(recall_per_class * deploy_prior))

    print("\nOpen-set adjusted metrics:")
    for class_idx in [0, 1, 2]:
        class_name = label_to_name_none(class_idx)
        print(
            f"{class_name:15s} | support={int(support[class_idx]):4d} | "
            f"precision={precision_per_class[class_idx]:.4f} | "
            f"recall={recall_per_class[class_idx]:.4f} | f1={f1_per_class[class_idx]:.4f}"
        )
    print(f"Balanced accuracy (macro recall): {balanced_accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Adjusted accuracy (equal prior A/B/NONE): {adjusted_acc_equal:.4f}")
    print(f"Adjusted accuracy (deployment prior, NONE={expected_none_ratio:.2f}): {adjusted_acc_deploy:.4f}")

    with open("predictions_none_val.txt", "w", encoding="utf-8") as f:
        f.write("Sample | Predicted | Confidence | Actual | Correct\n")
        f.write("-" * 70 + "\n")
        for i, (pred, conf, label) in enumerate(zip(none_preds, none_probs, none_labels)):
            is_correct = "✓" if pred == label else "✗"
            f.write(f"{i:4d}   | {label_to_name_none(pred):15s} | {conf:.4f}     | {label_to_name_none(label):15s} | {is_correct}\n")

    with open("open_set_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Open-set metrics (A/B/NONE)\n")
        f.write(f"Thresholds: low={low_confidence_threshold:.4f}, high={high_confidence_threshold:.4f}\n")
        f.write(f"Accuracy: {none_accuracy:.4f}\n")
        f.write(f"Balanced accuracy (macro recall): {balanced_accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Adjusted accuracy (equal prior): {adjusted_acc_equal:.4f}\n")
        f.write(f"Adjusted accuracy (deployment prior, NONE={expected_none_ratio:.2f}): {adjusted_acc_deploy:.4f}\n")
        f.write("\nPer-class metrics:\n")
        for class_idx in [0, 1, 2]:
            class_name = label_to_name_none(class_idx)
            f.write(
                f"{class_name}: support={int(support[class_idx])}, "
                f"precision={precision_per_class[class_idx]:.4f}, "
                f"recall={recall_per_class[class_idx]:.4f}, "
                f"f1={f1_per_class[class_idx]:.4f}\n"
            )

    print("Open-set records saved to predictions_none_val.txt")
    print("Open-set metrics saved to open_set_metrics.txt")

# -------------------------
# Save model
# -------------------------
def save_model(model):
    torch.save(model.state_dict(), "convnext_atto_finetuned.pth")



def setup_train_and_evaluate(model_output_classes, dataset_path, num_epochs):
    # -------------------------
    # Setup
    # -------------------------
    device, batch_size, num_workers = device_spec_setup()
    _, eval_transform = transforms_setup()
    train_dataset, train_loader, val_loader = dataset_setup(batch_size=batch_size, num_workers=num_workers, dataset_path=dataset_path)
    model = model_setup(device=device, model_output_classes=model_output_classes)
    optimizer = optimizer_setup(model=model)
    scheduler = scheduler_setup(optimizer=optimizer, num_epochs=num_epochs)

    # -------------------------
    # Training
    # -------------------------
    train(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        device=device, 
        model_output_classes=model_output_classes, 
        num_epochs=num_epochs
    )
    
    
    # -------------------------
    # Evaluation
    # -------------------------
    evaluate_A_B(model=model, val_loader=val_loader, device=device, model_output_classes=model_output_classes, class_names=train_dataset.classes)
    evaluate_A_B_NONE(
        model=model,
        device=device,
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        eval_transform=eval_transform,
        class_names=train_dataset.classes,
    )

    save_model(model=model)

if __name__ == '__main__':
        # -------------------------
    # Argument parsing to allow easy configuration from command line
    # -------------------------
    argparser = argparse.ArgumentParser(prog = "CNN Fine-tuning", description="Fine-tune ConvNeXt on A/B classification with open-set NONE evaluation")
    argparser.add_argument("-c", "--model_output_classes", type=int, default=3, help="Model output classes (3 for binary classification with probability output, 2 for standard binary classification with CrossEntropyLoss)")
    argparser.add_argument("-d", "--dataset_path", type=str, default="dataset_all_TIs", help="dataset_all_TIs, dataset_no_TIs or dataset_2_TIs")
    argparser.add_argument("-e", "--num_epochs", type=int, default=default_num_epochs, help="Number of training epochs")
    args = argparser.parse_args()
    
    if(args.model_output_classes == 2):
        model_output_classes = ModelOutputClasses.A_B
    elif(args.model_output_classes == 3):
        model_output_classes = ModelOutputClasses.A_B_NONE
    else:
        raise ValueError(f"Invalid model_output_classes argument: {args.model_output_classes}")

    if(args.dataset_path == "dataset_all_TIs"):
        dataset_path = DatasetPaths.DATASET_ALL_TIs
    elif(args.dataset_path == "dataset_no_TIs"):
        dataset_path = DatasetPaths.DATASET_NO_TIs
    elif(args.dataset_path == "dataset_2_TIs"):
        dataset_path = DatasetPaths.DATASET_2_TIs
    else:
        raise ValueError(f"Invalid dataset_path argument: {args.dataset_path}")
    
    setup_train_and_evaluate(model_output_classes, dataset_path, args.num_epochs)

""" #Grad-CAM visualization to check which parts of the image the model is focusing on for its predictions
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

target_layer = model.stages[-1].blocks[-1]

cam = GradCAM(model=model, target_layers=[target_layer])
grayscale_cam = cam(input_tensor=input_tensor)
visualization = show_cam_on_image(image, grayscale_cam[0]) """