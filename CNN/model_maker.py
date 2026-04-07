import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
import numpy as np
import argparse
from enum import Enum
from datetime import datetime, timedelta
import csv
from pathlib import Path
import threshold_estimator


class ModelNames(Enum):
    ALL_TIs = "all_TIs"
    NO_TIs = "No_TIs"
    NO_BB = "No_BB"
    NO_BB_NO_RSI = "No_BB_No_RSI"
    NO_BB_NO_OBV = "No_BB_No_OBV"
    NO_RSI = "No_RSI"
    NO_RSI_NO_OBV = "No_RSI_No_OBV"
    NO_OBV = "No_OBV"

    def __str__(self):
        return str(self.value)


# -------------------------
# Variables and hyperparameters
# -------------------------
# Model and training parameters
default_model_name = ModelNames.ALL_TIs  # Contains "train", "val", "threshold_estimation", and "backtesting".

# Training parameters
default_num_epochs = 10
default_noMov_ratio = 0.0  # Expected ratio of NOMOV samples in the backtesting set (used for auto-tuning)
# Maybe do: number of layers to unfreeze for fine-tuning (0 = only head, 1 = last stage + head, etc.)
default_num_stages_to_unfreeze = 1
# Augmentation options
use_random_affine = True

# Hyperparameters
workers_cpu = 0
workers_gpu = 4
batch_size_cpu = 8
batch_size_gpu = 16
base_lr = 1e-4
backbone_lr_scale = 0.1

# Auto-tuning options for thresholding
auto_tune_thresholds = True
# Manual fallback thresholds (used when auto_tune_thresholds == False)
low_threshold = 0.0
high_threshold = 0.0


class ModelMaker:
    def __init__(self, model_name=default_model_name, num_epochs=default_num_epochs, noMov_ratio=(0.0), num_stages_to_unfreeze=(1), thresholds=(0, 0)):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.noMov_ratio = noMov_ratio
        self.num_stages_to_unfreeze = num_stages_to_unfreeze
        self.thresholds = thresholds
        self.auto_tune_thresholds = auto_tune_thresholds
        self.lr = base_lr
        self.backbone_lr_scale = backbone_lr_scale
        self.device, self.batch_size, self.num_workers = device_spec_setup()
        self.train_transform, self.eval_transform = transforms_setup()
        (
            self.train_dataset,
            self.train_loader,
            self.val_loader,
            self.backtest_2_class_dataset,
            self.backtest_2_class_loader,
            self.fix_thresholds_dataset,
            self.fix_thresholds_loader,
        ) = dataset_setup(self)
        self.model = model_setup(self)
        self.optimizer = optimizer_setup(self)
        self.scheduler = scheduler_setup(self)
        print_summary(self)

        train_model(self)
        evaluate_model(self)


# -------------------------
# Device
# -------------------------
def device_spec_setup():
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Optimize based on hardware
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
    std = [0.229, 0.224, 0.225]

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
def dataset_setup(self):

    train_dataset = ImageFolder(root=f"datasets/{self.model_name}/train", transform=self.train_transform)
    val_dataset = ImageFolder(root=f"datasets/{self.model_name}/val", transform=self.eval_transform)
    backtest_2_class_dataset = ImageFolder(root=f"datasets/{self.model_name}/threshold_estimationAB", transform=self.eval_transform)
    fix_thresholds_dataset = ImageFolder(root=f"datasets/{self.model_name}/threshold_estimation", transform=self.eval_transform)
    
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
    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    backtest_2_class_loader = DataLoader(
        backtest_2_class_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    fix_thresholds_loader = DataLoader(
        fix_thresholds_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    return train_dataset, train_loader, val_loader, backtest_2_class_dataset, backtest_2_class_loader, fix_thresholds_dataset, fix_thresholds_loader


# -------------------------
# Model
# -------------------------
def model_setup(self):
    model = timm.create_model(
        "convnextv2_atto",
        pretrained=True
    )

    # Compatibility fallback for timm variants where ConvNeXt may miss norm_pre.
    # (convnextv2_atto seems to have it, but this ensures the code works across more timm versions without modification)
    if not hasattr(model, "norm_pre"):
        model.norm_pre = nn.Identity()
    # says copilot to fix a bug...

    model.reset_classifier(1)
    model.to(self.device)

    # Freeze all layers except the head
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze head
    for p in model.head.parameters():
        p.requires_grad = True

    # Unfreeze the last backbone stages depending on model family.
    stage_prefixes = []
    max_stages = len(model.stages)
    clamped = max(0, min(self.num_stages_to_unfreeze, max_stages))
    stage_prefixes = [f"stages.{idx}" for idx in range(
        max_stages - 1, max_stages - 1 - clamped, -1)]

    if stage_prefixes:
        for name, p in model.named_parameters():
            if any(
                name.startswith(prefix)
                    for prefix in stage_prefixes):
                p.requires_grad = True
    return model


# -------------------------
# Loss + optimizer
# -------------------------
# Define the optimizer (AdamW is commonly used for training vision models)
def optimizer_setup(self):
    weight_decay = 0.05
    trainable_named_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
    if not trainable_named_params:
        raise ValueError("No trainable parameters found. Check fine-tuning stage configuration.")

    # Use differential learning rates: lower LR for unfrozen backbone stages, base LR for head.
    head_prefixes = ("head", "fc", "classifier")
    head_params = [p for n, p in trainable_named_params if n.startswith(head_prefixes)]
    backbone_params = [p for n, p in trainable_named_params if not n.startswith(head_prefixes)]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": self.lr * self.backbone_lr_scale})
    if head_params:
        param_groups.append({"params": head_params, "lr": self.lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    print(
        f"Optimizer setup: base_lr={self.lr:.2e}, backbone_lr={self.lr * self.backbone_lr_scale:.2e}, "
        f"head_params={len(head_params)}, backbone_params={len(backbone_params)}"
    )
    return optimizer


# -------------------------
# Scheduler
# -------------------------
def scheduler_setup(self):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        self.optimizer,
        T_max=self.num_epochs,
        eta_min=1e-6
    )
    return scheduler


# --------------------------
# Print configuration summary
# --------------------------
def print_summary(self):
    print("-" * 50)
    print("Configuration summary:")
    print(f"Model name: {self.model_name}")
    print(f"Number of epochs: {self.num_epochs}")
    print(f"Base learning rate: {base_lr}")
    print(f"Backbone LR scale: {backbone_lr_scale}")
    if self.auto_tune_thresholds:
        print("Auto-tuning NOMOV thresholds")
        if self.noMov_ratio > 0.0 and self.noMov_ratio < 1.0:
            print(f"Expected NOMOV ratio for auto-tuning: {self.noMov_ratio:.4f}")
    else:
        print(f"Using manual NOMOV thresholds: {self.manual_thresholds}")
    print(f"Number of stages to unfreeze for fine-tuning: {self.num_stages_to_unfreeze}")
    print("-" * 50)


# -------------------------
# Training
# -------------------------
def train_model(self):

    criterion = nn.BCEWithLogitsLoss() 

    # Training loop for binary classification with probability output
    for epoch in range(self.num_epochs):
        self.model.train()
        running_loss = 0.0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)

        # validation
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                probs = torch.sigmoid(outputs).squeeze(1)
                preds = (probs > 0.5).long()

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total

        print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

        self.scheduler.step()


# -------------------------
# Save and load model functions
# -------------------------
def evaluate_model(self):
    backtest_2_class(self)
    low, high = fix_thresholds(self)


# -------------------------
# Evaluation on val (A/B)
# -------------------------
def backtest_2_class(self):
    backtest_2_class_preds = []
    backtest_2_class_labels = []
    backtest_2_class_confidences = []

    self.model.eval()
    with torch.no_grad():
        for images, labels in self.backtest_2_class_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)

            probs = torch.sigmoid(outputs).squeeze(1)
            preds = (probs > 0.5).long()
            confidences = torch.where(preds == 1, probs, 1 - probs)

            backtest_2_class_preds.extend(preds.cpu().numpy())
            backtest_2_class_labels.extend(labels.cpu().numpy())
            backtest_2_class_confidences.extend(confidences.cpu().numpy())

    print("\nValidation (A/B) summary:")
    print(f"Total samples: {len(backtest_2_class_preds)}")
    print(f"Correct predictions: {sum(np.array(backtest_2_class_preds) == np.array(backtest_2_class_labels))}")
    print(f"Accuracy: {sum(np.array(backtest_2_class_preds) == np.array(backtest_2_class_labels)) / len(backtest_2_class_preds):.4f}")
    print(f"Average confidence: {np.mean(backtest_2_class_confidences):.4f}")
    print(f"Macro F1: {f1_score(backtest_2_class_labels, backtest_2_class_preds, average='macro'):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(backtest_2_class_labels, backtest_2_class_preds):.4f}")

    cm_val = confusion_matrix(backtest_2_class_labels, backtest_2_class_preds)
    print("\nValidation (A/B) confusion matrix:")
    print(cm_val)


# -------------------------
# Threshold auto-tuning for 3 class classification
# -------------------------
def fix_thresholds(self):
    fix_thresholds_dataset = ImageFolder(root=f"datasets/{self.model_name}/threshold_estimation", transform=self.eval_transform)

    nomov_probs = []
    nomov_labels = []

    name_to_open_label = {
        "downMovement": 0,
        "upMovement": 1,
        "noMovement": 2,
    }

    # Map dataset label index -> open-set label, and reverse
    val_to_open = {
        idx: name_to_open_label[name]
        for idx, name in enumerate(fix_thresholds_dataset.classes)
    }

    def open_label_to_name(label: int) -> str:
        name = {v: k for k, v in name_to_open_label.items()}
        return name.get(label, "Unknown")

    # Get model probabilities on the nomov_val set and map to 3 class labels
    with torch.no_grad():
        for images, labels in self.fix_thresholds_loader:
            images = images.to(self.device)
            outputs = self.model(images)
            probs = torch.sigmoid(outputs).squeeze(1).flatten()

            nomov_probs.extend(probs.cpu().tolist())
            nomov_labels.extend(val_to_open[int(lbl)] for lbl in labels)

    manual_low_threshold, manual_high_threshold = self.thresholds

    def predict_open_set(prob, low, high):
        if prob <= low:
            return 0
        if prob >= high:
            return 1
        return 2

    print("\n3 class summary:")
    print(f"Total samples: {len(nomov_probs)}")

    # Print a summary of the 3 class evaluation before thresholding to understand the raw model outputs
    print("\n3 class evaluation (before thresholding):")

    # Adjust thresholds
    if manual_high_threshold > 0 or manual_low_threshold > 0:
        self.thresholds = manual_high_threshold, manual_low_threshold
        print(f"Manual thresholds applied: low={manual_low_threshold:.4f}, high={manual_high_threshold:.4f}")
    else:
        print(
            "No manual thresholds applied. Thresholds will be auto-tuned based on "
            "noMov_ratio and/or automatic threshold estimation."
        )
        new_thresholds = threshold_estimator.ThresholdEstimator(
            self.model, nomov_probs, nomov_labels, self.noMov_ratio, threshold_sweep_steps=20, threshold_sweep_objective="macro_f1"
        )
        self.thresholds = new_thresholds.estimate_thresholds()

    low_threshold, high_threshold = self.thresholds
    fix_thresholds_preds = [predict_open_set(prob, low_threshold, high_threshold) for prob in nomov_probs]

    # -------------------------
    # Final evaluation with selected thresholds
    # -------------------------
    # Compute predictions after threshold selection so auto-tuned thresholds are actually applied.
    nomov_labels_np = np.array(nomov_labels)
    nomov_preds_np = np.array(fix_thresholds_preds)
    nomov_probs_np = np.array(nomov_probs)
    nomov_accuracy = sum(nomov_preds_np == nomov_labels_np) / len(fix_thresholds_preds)
    print(f"\nCorrect predictions: {sum(nomov_preds_np == nomov_labels_np)}")
    print(f"High confidence samples: {sum(1 for p in nomov_probs if p >= high_threshold)}")
    print(f"Low confidence samples: {sum(1 for p in nomov_probs if p <= low_threshold)}")
    print(f"Uncertain samples: {sum(1 for p in nomov_probs if low_threshold < p < high_threshold)}")

    cm_nomov = confusion_matrix(nomov_labels, fix_thresholds_preds, labels=[0, 1, 2])

    print(f"Thresholds used: low={low_threshold:.4f}, high={high_threshold:.4f}")
    print(f"Accuracy: {nomov_accuracy:.4f}")
    print(f"Average confidence: {np.mean(nomov_probs):.4f}")
    print("\nOpen-set (3 class) confusion matrix:")
    print("Matrix labels order:", [open_label_to_name(i) for i in [0, 1, 2]])
    print(cm_nomov)

    print("\n3 class confidence stats by predicted class:")
    for class_idx in [0, 1, 2]:
        class_name = open_label_to_name(class_idx)
        class_conf = nomov_probs_np[nomov_preds_np == class_idx]
        if class_conf.size == 0:
            print(f"{class_name:15s} | n=0")
        else:
            print(
                f"{class_name:15s} | n={class_conf.size:4d} | "
                f"min={class_conf.min():.4f} | mean={class_conf.mean():.4f} | max={class_conf.max():.4f}"
            )

    print("\n3 class confidence stats by true class:")
    for class_idx in [0, 1, 2]:
        class_name = open_label_to_name(class_idx)
        mask = (nomov_labels_np == class_idx)
        class_conf = nomov_probs_np[mask]
        if class_conf.size == 0:
            print(f"{class_name:15s} | n=0")
        else:
            class_acc = np.mean(nomov_preds_np[mask] == class_idx)
            print(
                f"{class_name:15s} | n={class_conf.size:4d} | "
                f"mean_prob={class_conf.mean():.4f} | std_prob={class_conf.std():.4f} | recall={class_acc:.4f}"
            )

    # Per-class metrics that are robust to class imbalance
    tp = np.diag(cm_nomov).astype(float)
    support = cm_nomov.sum(axis=1).astype(float)
    pred_count = cm_nomov.sum(axis=0).astype(float)

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
    print(f"\nBalanced accuracy (macro recall): {balanced_accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    # Prior-adjusted accuracy using class recalls (more informative when classes are imbalanced)
    equal_prior = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    deploy_prior = np.array([
        (1.0 - self.noMov_ratio) / 2.0,
        (1.0 - self.noMov_ratio) / 2.0,
        self.noMov_ratio,
    ], dtype=float)
    deploy_prior = np.clip(deploy_prior, 0.0, 1.0)
    deploy_prior = deploy_prior / deploy_prior.sum()

    # Compute adjusted accuracies that reflect expected performance under different class distributions (e.g. deployment scenario vs equal class distribution)
    adjusted_acc_equal = float(np.sum(recall_per_class * equal_prior))
    adjusted_acc_deploy = float(np.sum(recall_per_class * deploy_prior))

    print("\n3 class adjusted metrics:")
    for class_idx in [0, 1, 2]:
        class_name = open_label_to_name(class_idx)
        print(
            f"{class_name:15s} | support={int(support[class_idx]):4d} | "
            f"precision={precision_per_class[class_idx]:.4f} | "
            f"recall={recall_per_class[class_idx]:.4f} | f1={f1_per_class[class_idx]:.4f}"
        )

    print(f"Adjusted accuracy (equal prior 3 class): {adjusted_acc_equal:.4f}")
    print(f"Adjusted accuracy (deployment prior, NOMOV={self.noMov_ratio:.2f}): {adjusted_acc_deploy:.4f}")

    # Print accuracy for only the A/B subset of the nomov_val set to understand how well the model distinguishes A vs B without considering NOMOV.
    ab_mask = nomov_labels_np < 2
    if np.sum(ab_mask) > 0:
        ab_accuracy = np.mean(nomov_preds_np[ab_mask] == nomov_labels_np[ab_mask])
        print(f"\nA/B subset accuracy (ignoring NOMOV): {ab_accuracy:.4f} (n={np.sum(ab_mask)})")

    # -------------------------
    # Save evaluation results to a file for record-keeping and analysis
    # -------------------------
    with open(f"evaluation_results/{self.model_name}_Evaluation_results.txt", "a") as f:
        f.write("-" * 50 + "\n")
        f.write("3 class evaluation summary:\n")
        f.write(f"Datetime: {datetime.now()}\n")
        f.write(f"Model name: {self.model_name}\n")
        f.write(f"Thresholds used: low={low_threshold:.4f}, high={high_threshold:.4f}\n")
        f.write(f"Validation (3 class) confusion matrix:\n")
        f.write(f"Labels order: {[open_label_to_name(i) for i in [0, 1, 2]]}, (rows=true, cols=predicted)\n")
        f.write(np.array2string(cm_nomov, separator=", ") + "\n")
        f.write(f"Total samples: {len(nomov_probs)}\n")
        f.write(f"Correct predictions: {sum(nomov_preds_np == nomov_labels_np)}\n")
        f.write(f"Accuracy: {nomov_accuracy:.4f}\n")
        f.write(f"Average confidence: {np.mean(nomov_probs):.4f}\n")
        f.write("\n3 class confidence stats by predicted class:\n")
        for class_idx in [0, 1, 2]:
            class_name = open_label_to_name(class_idx)
            class_conf = nomov_probs_np[nomov_preds_np == class_idx]
            if class_conf.size == 0:
                f.write(f"{class_name:15s} | n=0\n")
            else:
                f.write(
                    f"{class_name:15s} | n={class_conf.size:4d} | "
                    f"min={class_conf.min():.4f} | mean={class_conf.mean():.4f} | max={class_conf.max():.4f}\n"
                )
        f.write("\n3 class confidence stats by true class:\n")
        for class_idx in [0, 1, 2]:
            class_name = open_label_to_name(class_idx)
            mask = (nomov_labels_np == class_idx)
            class_conf = nomov_probs_np[mask]
            if class_conf.size == 0:
                f.write(f"{class_name:15s} | n=0\n")
            else:
                class_acc = np.mean(nomov_preds_np[mask] == class_idx)
                f.write(
                    f"{class_name:15s} | n={class_conf.size:4d} | mean_prob={class_conf.mean():.4f} | "
                    f"std_prob={class_conf.std():.4f} | recall={class_acc:.4f}\n"
                )
        f.write("\n3 class adjusted metrics:\n")
        for class_idx in [0, 1, 2]:
            class_name = open_label_to_name(class_idx)
            f.write(
                f"{class_name:15s} | support={int(support[class_idx]):4d} | precision={precision_per_class[class_idx]:.4f} | "
                f"recall={recall_per_class[class_idx]:.4f} | f1={f1_per_class[class_idx]:.4f}\n"
            )
        f.write(f"\nBalanced accuracy (macro recall): {balanced_accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"A/B subset accuracy (ignoring NOMOV): {ab_accuracy:.4f} (n={np.sum(ab_mask)})\n")
        f.write("-" * 50 + "\n")

        print("\n3 class evaluation completed and results saved to Evaluation_results.txt")

    # -------------------------
    # Save the trained model
    # -------------------------
    def save_model(self):
        out_path = Path(f"final_models/{self.model_name}_model.pth")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_path)
        with open("final_models/all_thresholds.txt", "w") as f:
            f.write(f"{self.model_name}-\n")
            f.write(f"Low: {low_threshold:.4f}\n")
            f.write(f"High: {high_threshold:.4f}\n")
    save_model(self)

    return low_threshold, high_threshold


if __name__ == '__main__':
    # -------------------------
    # Argument parsing to allow easy configuration from command line
    # -------------------------
    argparser = argparse.ArgumentParser(prog="CNN Training", description="Fine-tune ConvNeXt")
    argparser.add_argument("-d", "--model_name", type=str, default=default_model_name.value, help="se Enums")
    argparser.add_argument("-e", "--num_epochs", type=int, default=default_num_epochs, help="Number of training epochs")
    argparser.add_argument("-t", "--manual_thresholds", nargs='+', type=float, default=[0, 0], help="Manual low and high confidence thresholds for 3 class classification (used when auto_tune_thresholds is False)")
    argparser.add_argument("-r", "--noMov_ratio", type=float, default=default_noMov_ratio, help="Expected ratio of NOMOV samples in the backtesting set (used for auto-tuning thresholds)")
    argparser.add_argument("-s", "--num_stages_to_unfreeze", type=int, default=default_num_stages_to_unfreeze, help="Number of stages to unfreeze for fine-tuning")
    argparser.add_argument("--backbone_lr_scale", type=float, default=backbone_lr_scale, help="Scale factor for backbone LR relative to base LR (e.g. 0.1)")
    args = argparser.parse_args()

    # -------------------------
    # Validate configuration
    # -------------------------
    # Validate manual thresholds if provided (if both are 0, assume auto-tuning is desired)
    if len(args.manual_thresholds) != 2:
        raise ValueError("manual_thresholds argument must contain exactly 2 values: low and high confidence thresholds")
    elif args.manual_thresholds == [0, 0]:
        pass
    else:
        if args.manual_thresholds[0] < 0.0 or args.manual_thresholds[0] > 1.0 or args.manual_thresholds[1] < 0.0 or args.manual_thresholds[1] > 1.0:
            raise ValueError(f"Manual thresholds must be between 0 and 1. Got {args.manual_thresholds}")
        elif args.manual_thresholds[0] >= args.manual_thresholds[1]:
            raise ValueError(f"Manual low confidence threshold must be less than high confidence threshold. Got {args.manual_thresholds}")
        else:
            auto_tune_thresholds = False
            low_threshold, threshold = args.manual_thresholds
            print(f"Using manual thresholds: low={low_threshold}, high={high_threshold:.4f}")

    # Validate noMov_ratio and adjust auto-tuning strategy if needed
    if args.noMov_ratio < 0.0 or args.noMov_ratio > 1.0:
        raise ValueError(f"Expected NOMOV ratio must be between 0 and 1. Got {args.noMov_ratio}")

    # Validate backbone_lr_scale
    if args.backbone_lr_scale < 0.0 or args.backbone_lr_scale > 1.0:
        raise ValueError(f"backbone_lr_scale must be in the interval [0, 1]. Got {args.backbone_lr_scale}")
    else:
        backbone_lr_scale = args.backbone_lr_scale
        print(f"Using backbone LR scale: {backbone_lr_scale:.2f}")

    # Validate num_stages_to_unfreeze
    if args.num_stages_to_unfreeze < 0:
        raise ValueError(f"num_stages_to_unfreeze must be non-negative. Got {args.num_stages_to_unfreeze}")
    elif args.num_stages_to_unfreeze > 4:  # ConvNeXt has 4 stages, so unfreezing more than that doesn't make sense
        raise ValueError(f"num_stages_to_unfreeze cannot be greater than 4 for ConvNeXt. Got {args.num_stages_to_unfreeze}")
    else:
        num_stages_to_unfreeze = args.num_stages_to_unfreeze
        print(f"Number of stages to unfreeze for fine-tuning: {num_stages_to_unfreeze}")

    # Validate epochs
    if args.num_epochs <= 0:
        raise ValueError(f"num_epochs must be a positive integer. Got {args.num_epochs}")
    else:
        num_epochs = args.num_epochs
        print(f"Number of training epochs: {num_epochs}")

    # -------------------------
    # Map model_name argument to names defined in ModelNames class
    # -------------------------
    if (args.model_name == "all_TIs"):
        model_name = ModelNames.ALL_TIs
    elif (args.model_name == "No_TIs"):
        model_name = ModelNames.NO_TIs
    elif (args.model_name == "No_BB"):
        model_name = ModelNames.NO_BB
    elif (args.model_name == "No_BB_No_RSI"):
        model_name = ModelNames.NO_BB_NO_RSI
    elif (args.model_name == "No_BB_No_OBV"):
        model_name = ModelNames.NO_BB_NO_OBV
    elif (args.model_name == "No_RSI"):
        model_name = ModelNames.NO_RSI
    elif (args.model_name == "No_RSI_No_OBV"):
        model_name = ModelNames.NO_RSI_NO_OBV
    elif (args.model_name == "No_OBV"):
        model_name = ModelNames.NO_OBV
    else:
        raise ValueError(f"Invalid model_name argument: {args.model_name}")

    new_model = ModelMaker(model_name=model_name)
    new_model.print_summary()
    new_model.train_model()
    new_model.evaluate_model()
    new_model.save_model()
