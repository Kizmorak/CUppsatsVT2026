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
default_expected_noMov_ratio = 0.0  # Expected ratio of NOMOV samples in the backtesting set (used for auto-tuning)
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
manual_low_threshold = 0.0
manual_high_threshold = 0.0


# Threshold tuning strategy for 3 class mode.
# Options: "prior_quantile" (uses expected_noMov_ratio) or "sweep" (grid-searches thresholds on backtesting set).
threshold_tuning_strategy = "sweep"
# Objective used when threshold_tuning_strategy == "sweep". Options: "macro_f1", "balanced_accuracy".
threshold_sweep_objective = "macro_f1"
# Number of grid points per axis when sweeping [min_prob, max_prob]. Larger = slower but finer search.
threshold_sweep_steps = 61


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
def dataset_setup(batch_size, num_workers, model_name):
    train_transform, eval_transform = transforms_setup()

    train_dataset = ImageFolder(root=f"datasets/{model_name}/train", transform=train_transform)
    val_dataset = ImageFolder(root=f"datasets/{model_name}/val", transform=eval_transform)
    backtest_2_class_dataset = ImageFolder(root=f"datasets/{model_name}/threshold_estimationAB", transform=eval_transform)
    backtest_3_class_dataset = ImageFolder(root=f"datasets/{model_name}/threshold_estimation", transform=eval_transform)

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
    backtest_2_class_loader = DataLoader(
        backtest_2_class_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    backtest_3_class_loader = DataLoader(
        backtest_3_class_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataset, train_loader, val_loader, backtest_2_class_loader, backtest_3_class_dataset, backtest_3_class_loader


# -------------------------
# Model
# -------------------------
def model_setup(device, num_stages_to_unfreeze):
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
    model.to(device)

    # Freeze all layers except the head
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze head
    for p in model.head.parameters():
        p.requires_grad = True

    # Unfreeze the last backbone stages depending on model family.
    stage_prefixes = []
    max_stages = len(model.stages)
    clamped = max(0, min(num_stages_to_unfreeze, max_stages))
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
def optimizer_setup(model, lr=base_lr, backbone_lr_scale=backbone_lr_scale):
    weight_decay = 0.05
    trainable_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable_named_params:
        raise ValueError("No trainable parameters found. Check fine-tuning stage configuration.")

    # Use differential learning rates: lower LR for unfrozen backbone stages, base LR for head.
    head_prefixes = ("head", "fc", "classifier")
    head_params = [p for n, p in trainable_named_params if n.startswith(head_prefixes)]
    backbone_params = [p for n, p in trainable_named_params if not n.startswith(head_prefixes)]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * backbone_lr_scale})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    print(
        f"Optimizer setup: base_lr={lr:.2e}, backbone_lr={lr * backbone_lr_scale:.2e}, "
        f"head_params={len(head_params)}, backbone_params={len(backbone_params)}"
    )
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
def train(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs):

    criterion = nn.BCEWithLogitsLoss() 

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
def backtest_2_class(model, backtest_2_class_loader, device, model_name):
    backtest_2_class_preds = []
    backtest_2_class_labels = []
    backtest_2_class_confidences = []

    model.eval()
    with torch.no_grad():
        for images, labels in backtest_2_class_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

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
# Evaluation on nomov_val (3 class)
# -------------------------
def backtest_3_class(
        model,
        device,
        model_name,
        backtest_3_class_loader,
        eval_transform, expected_noMov_ratio=0.0,
        auto_tune_thresholds=True,
        manual_low_threshold=0.0,
        manual_high_threshold=0.0
        ):
    backtest_3_class_dataset = ImageFolder(root=f"datasets/{model_name}/threshold_estimation", transform=eval_transform)

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
        for idx, name in enumerate(backtest_3_class_dataset.classes)
    }

    def open_label_to_name(label: int) -> str:
        name = {v: k for k, v in name_to_open_label.items()}
        return name.get(label, "Unknown")

    # Get model probabilities on the nomov_val set and map to 3 class labels
    with torch.no_grad():
        for images, labels in backtest_3_class_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1).flatten()

            nomov_probs.extend(probs.cpu().tolist())
            nomov_labels.extend(val_to_open[int(lbl)] for lbl in labels)

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
    low_threshold, high_threshold = None, None
    if manual_high_threshold > 0 or manual_low_threshold > 0:
        high_threshold, low_threshold = manual_high_threshold, manual_low_threshold
        print(f"Manual thresholds applied: low={low_threshold:.4f}, high={high_threshold:.4f}")
    else:
        print("No manual thresholds applied. Thresholds will be auto-tuned based on expected_noMov_ratio and/or backtesting set performance.")
        new_thresholds = threshold_estimator.ThresholdEstimator(
            model, nomov_probs, nomov_labels, expected_noMov_ratio, threshold_sweep_steps=20, threshold_sweep_objective="macro_f1"
        )
        high_threshold, low_threshold = new_thresholds.estimate_thresholds()

    # -------------------------
    # Final evaluation with selected thresholds
    # -------------------------
    # Compute predictions after threshold selection so auto-tuned thresholds are actually applied.
    nomov_preds = [predict_open_set(prob, low_threshold, high_threshold) for prob in nomov_probs]
    nomov_labels_np = np.array(nomov_labels)
    nomov_preds_np = np.array(nomov_preds)
    nomov_probs_np = np.array(nomov_probs)
    nomov_accuracy = sum(nomov_preds_np == nomov_labels_np) / len(nomov_preds)
    print(f"\nCorrect predictions: {sum(nomov_preds_np == nomov_labels_np)}")
    print(f"High confidence samples: {sum(1 for p in nomov_probs if p >= high_threshold)}")
    print(f"Low confidence samples: {sum(1 for p in nomov_probs if p <= low_threshold)}")
    print(f"Uncertain samples: {sum(1 for p in nomov_probs if low_threshold < p < high_threshold)}")

    cm_nomov = confusion_matrix(nomov_labels, nomov_preds, labels=[0, 1, 2])

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
        (1.0 - expected_noMov_ratio) / 2.0,
        (1.0 - expected_noMov_ratio) / 2.0,
        expected_noMov_ratio,
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
    print(f"Adjusted accuracy (deployment prior, NOMOV={expected_noMov_ratio:.2f}): {adjusted_acc_deploy:.4f}")

    # Print accuracy for only the A/B subset of the nomov_val set to understand how well the model distinguishes A vs B without considering NOMOV.
    ab_mask = nomov_labels_np < 2
    if np.sum(ab_mask) > 0:
        ab_accuracy = np.mean(nomov_preds_np[ab_mask] == nomov_labels_np[ab_mask])
        print(f"\nA/B subset accuracy (ignoring NOMOV): {ab_accuracy:.4f} (n={np.sum(ab_mask)})")

    # -------------------------
    # Save predictions CSV file method (for backtesting)
    # -------------------------

    def save_csv_predictions(preds, dataset):
        def extract_datetime_from_path(path):
            stem = Path(path).stem
            try:
                dt = datetime.strptime(stem, "%Y-%m-%d_%H%M")
            except ValueError:
                print(f"Bad filename format: {stem}")
                return None
            plus = dt + timedelta(minutes=31)
            return plus.strftime("%Y-%m-%d %H:%M:00")

        out_path = Path(f"csv_files/{model_name}_Backtesting_predictions.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "target"])

            skipped = 0
            for (path, _), pred in zip(dataset.samples, preds):
                if pred == 2:  # NOMOV
                    skipped += 1
                    continue

                target = 1 if pred == 0 else 0
                t = extract_datetime_from_path(path)
                if t is None:
                    continue

                writer.writerow([t, target])

        print(f"Saved CSV. Skipped {skipped} NOMOV predictions.")

    save_csv_predictions(preds=nomov_preds, dataset=backtest_3_class_dataset)

    # -------------------------
    # Save evaluation results to a file for record-keeping and analysis
    # -------------------------
    with open(f"evaluation_results/{model_name}_Evaluation_results.txt", "a") as f:
        f.write("-" * 50 + "\n")
        f.write("3 class evaluation summary:\n")
        f.write(f"Datetime: {datetime.now()}\n")
        f.write(f"Model name: {model_name}\n")
        f.write(f"Thresholds used: low={low_threshold:.4f}, high={high_threshold:.4f}\n")
        f.write(f"Validation (3 class) confusion matrix:\n")
        f.write(f"Labels order: {[open_label_to_name(i) for i in [0, 1, 2]]}, (rows=true, cols=predicted)\n")
        f.write(np.array2string(cm_nomov, separator=", ") + "\n")
        f.write(f"Total samples: {len(nomov_preds)}\n")
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

        return low_threshold, high_threshold, nomov_preds


def setup_train_and_evaluate(model_name, num_epochs, expected_noMov_ratio=default_expected_noMov_ratio, num_stages_to_unfreeze=default_num_stages_to_unfreeze, manual_thresholds = (manual_low_threshold, manual_high_threshold), auto_tune_thresholds=auto_tune_thresholds, backbone_lr_scale=backbone_lr_scale):
    # --------------------------
    # Print configuration summary
    # --------------------------
    print("-" * 50)
    print("Configuration summary:")
    print(f"Model name: {model_name}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Base learning rate: {base_lr}")
    print(f"Backbone LR scale: {backbone_lr_scale}")
    if auto_tune_thresholds:
        print(f"Auto-tuning NOMOV thresholds")
        if expected_noMov_ratio > 0.0 and expected_noMov_ratio < 1.0:
            print(f"Expected NOMOV ratio for auto-tuning: {expected_noMov_ratio:.4f}")
    else:
        print(f"Using manual NOMOV thresholds: {manual_thresholds}")
    print(f"Number of stages to unfreeze for fine-tuning: {num_stages_to_unfreeze}")

    # --------------------------
    # Save validation results to a file for record-keeping and analysis
    # --------------------------
    # with open(f"{model_name}_Evaluation_results.txt", "a") as f:
    #     f.seek(0)  # Move to the beginning of the file
    #     f.truncate()  # Clear the file before writing new results
    #     f.write("-" * 50 + "\n")
    #     f.write("New training run:\n")
    #     f.write(f"Datetime: {datetime.now()}\n")
    #     f.write(f"Dataset path: {model_name}\n")
    #     f.write(f"Number of epochs: {num_epochs}\n")
    #     f.write(f"Base learning rate: {base_lr}\n")
    #     f.write(f"Backbone LR scale: {backbone_lr_scale}\n")
    #     f.write(f"Number of stages to unfreeze for fine-tuning: {num_stages_to_unfreeze}\n")
    #     if manual_thresholds != (0, 0):
    #         f.write(f"Manual low confidence threshold: {manual_thresholds[0]}\n")
    #         f.write(f"Manual high confidence threshold: {manual_thresholds[1]}\n")
    #     else:
    #         f.write("Auto-tuning of NOMOV thresholds\n")
    #         f.write(f"Expected NOMOV ratio: {expected_noMov_ratio}\n")

    # -------------------------
    # Setup
    # -------------------------
    device, batch_size, num_workers = device_spec_setup()
    _, eval_transform = transforms_setup()
    train_dataset, train_loader, val_loader, backtest_2_class_loader, backtest_3_class_dataset, backtest_3_class_loader = dataset_setup(batch_size=batch_size, num_workers=num_workers, model_name=model_name)
    model = model_setup(device=device, num_stages_to_unfreeze=num_stages_to_unfreeze)
    optimizer = optimizer_setup(model=model, lr=base_lr, backbone_lr_scale=backbone_lr_scale)
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
        num_epochs=num_epochs
    )

    # -------------------------
    # Evaluation
    # -------------------------
    backtest_2_class(
        model=model,
        backtest_2_class_loader=backtest_2_class_loader,
        device=device,
        model_name=model_name)

    low, high, nomov_preds = backtest_3_class(
        model=model,
        device=device,
        model_name=model_name,
        backtest_3_class_loader=backtest_3_class_loader,
        eval_transform=eval_transform,
        expected_noMov_ratio=expected_noMov_ratio,
        auto_tune_thresholds=auto_tune_thresholds,
        manual_low_threshold=manual_thresholds[0],
        manual_high_threshold=manual_thresholds[1],
    )

    # -------------------------
    # Save the trained model
    # -------------------------
    torch.save(model.state_dict(), f"final_models/{model_name}_model.pth")


if __name__ == '__main__':
    # -------------------------
    # Argument parsing to allow easy configuration from command line
    # -------------------------
    argparser = argparse.ArgumentParser(prog="CNN Training", description="Fine-tune ConvNeXt")
    argparser.add_argument("-d", "--model_name", type=str, default=default_model_name.value, help="se Enums")
    argparser.add_argument("-e", "--num_epochs", type=int, default=default_num_epochs, help="Number of training epochs")
    argparser.add_argument("-t", "--manual_thresholds", nargs='+', type=float, default=[0, 0], help="Manual low and high confidence thresholds for 3 class classification (used when auto_tune_thresholds is False)")
    argparser.add_argument("-r", "--expected_noMov_ratio", type=float, default=default_expected_noMov_ratio, help="Expected ratio of NOMOV samples in the backtesting set (used for auto-tuning thresholds)")
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
            print(f"Using manual thresholds: low={args.manual_thresholds[0]:.4f}, high={args.manual_thresholds[1]:.4f}")

    # Validate expected_noMov_ratio and adjust auto-tuning strategy if needed
    if args.expected_noMov_ratio < 0.0 or args.expected_noMov_ratio > 1.0:
        raise ValueError(f"Expected NOMOV ratio must be between 0 and 1. Got {args.expected_noMov_ratio}")
    elif args.expected_noMov_ratio > 0.0 and args.expected_noMov_ratio < 1.0:
        auto_tune_thresholds = True
        threshold_tuning_strategy = "prior_quantile"

    # Validate backbone_lr_scale
    if args.backbone_lr_scale < 0.0 or args.backbone_lr_scale > 1.0:
        raise ValueError(f"backbone_lr_scale must be in the interval [0, 1]. Got {args.backbone_lr_scale}")

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

    setup_train_and_evaluate(
        model_name,
        args.num_epochs,
        args.expected_noMov_ratio,
        args.num_stages_to_unfreeze,
        (args.manual_thresholds[0], args.manual_thresholds[1]),
        auto_tune_thresholds,
        args.backbone_lr_scale,
    )
