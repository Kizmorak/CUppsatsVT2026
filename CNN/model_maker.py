import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, accuracy_score
import numpy as np
import argparse
from enum import Enum
from pathlib import Path
import threshold_estimator
from datetime import datetime
from PIL import Image
import custom_tee
import sys
import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.stdout = custom_tee.CustomTee("night_worker_log.txt")
sys.stdout.flush()
# to avoid loosing too much log info on crashes


class ModelNames(Enum):
    ALL_TIs = "all_TIs"
    NO_TIs = "No_TIs"
    OBV = "OBV"
    RSI = "RSI"
    BB = "BB"
    NO_OBV = "No_OBV"
    NO_TIs_5 = "No_TIs_5"
    M5_RSI = "M5_RSI"

    def __str__(self):
        return str(self.value)


# Dataset wrapper to filter and remap classes, used to transform datasets for 2 class backtesting.
class FilteredRemappedDataset(Dataset):
    def __init__(self, base_dataset, allowed_class_names):
        self.base_dataset = base_dataset
        self.transform = base_dataset.transform
        self.allowed_class_names = list(allowed_class_names)
        self.class_to_new_idx = {
            base_dataset.class_to_idx[class_name]: new_idx
            for new_idx, class_name in enumerate(self.allowed_class_names)
        }
        self.samples = [
            (path, self.class_to_new_idx[target])
            for path, target in base_dataset.samples
            if target in self.class_to_new_idx
        ]
        self.targets = [target for _, target in self.samples]
        self.classes = self.allowed_class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(self.allowed_class_names)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target


# -------------------------
# Variables and hyperparameters
# -------------------------
# Model and training parameters
default_model_name = ModelNames.ALL_TIs  # Contains "train", "val", "threshold_estimation", and "backtesting".

# Training parameters
default_max_epochs = 30
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
default_base_lr = 3e-5
default_backbone_lr_scale = 0.1
weight_decay = 1e-2

# Auto-tuning options for thresholding
auto_tune_thresholds = True
# Manual fallback thresholds (used when auto_tune_thresholds == False)
low_threshold = 0.0
high_threshold = 0.0


class ModelMaker:
    def __init__(
        self,
        model_name=default_model_name,
        max_epochs=default_max_epochs,
        noMov_ratio=0.0,
        num_stages_to_unfreeze=1,
        thresholds=(low_threshold, high_threshold),
        base_lr=default_base_lr,
        backbone_lr_scale=default_backbone_lr_scale,
    ):
        self.model_name = model_name
        self.model_version = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.max_epochs = max_epochs
        self.noMov_ratio = noMov_ratio
        self.num_stages_to_unfreeze = num_stages_to_unfreeze
        self.thresholds = thresholds
        self.auto_tune_thresholds = auto_tune_thresholds
        self.base_lr = base_lr
        self.backbone_lr_scale = backbone_lr_scale
        self.device, self.batch_size, self.num_workers = device_spec_setup()
        self.train_transform, self.eval_transform = transforms_setup()
        (
            self.train_dataset,
            self.train_loader,
            self.val_loader,
            self.fix_thresholds_dataset,
            self.fix_thresholds_loader
        ) = dataset_setup(self)
        self.model = model_setup(self)
        self.optimizer = optimizer_setup(self)
        self.scheduler = scheduler_setup(self)
        print_summary(self)
        visualize_embeddings(self, pre_training=True)
        train_model(self)
        visualize_embeddings(self, pre_training=False)
        tune_and_evaluate_model(self)
        save_model(self)


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
    fix_thresholds_dataset = ImageFolder(
        root=f"datasets/{self.model_name}/threshold_estimation",
        transform=self.eval_transform,
    )

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

    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    fix_thresholds_loader = DataLoader(
        fix_thresholds_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=self.num_workers,
    )
    return (
        train_dataset,
        train_loader,
        val_loader,
        fix_thresholds_dataset,
        fix_thresholds_loader,
    )


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
    trainable_named_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
    if not trainable_named_params:
        raise ValueError("No trainable parameters found. Check fine-tuning stage configuration.")

    # Use differential learning rates: lower LR for unfrozen backbone stages, base LR for head.
    head_prefixes = ("head", "fc", "classifier")
    head_params = [p for n, p in trainable_named_params if n.startswith(head_prefixes)]
    backbone_params = [p for n, p in trainable_named_params if not n.startswith(head_prefixes)]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": self.base_lr * self.backbone_lr_scale})
    if head_params:
        param_groups.append({"params": head_params, "lr": self.base_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    print(
        f"Optimizer setup: base_lr={self.base_lr:.2e}, backbone_lr={self.base_lr * self.backbone_lr_scale:.2e}, "
        f"head_params={len(head_params)}, backbone_params={len(backbone_params)}, weight_decay={weight_decay:.2e}"
    )
    return optimizer


# -------------------------
# Scheduler
# -------------------------
def scheduler_setup(self):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        self.optimizer,
        T_max=self.max_epochs,
        eta_min=1e-6
    )
    return scheduler


# --------------------------
# Print configuration summary
# --------------------------
def print_summary(self):
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    print("Configuration summary:")
    print("-" * 50)
    print(f"Model name: {self.model_name}")
    print(f"Max number of epochs: {self.max_epochs}")
    print(f"Base learning rate: {self.base_lr}")
    print(f"Backbone LR scale: {self.backbone_lr_scale}")
    if self.auto_tune_thresholds:
        print("Auto-tuning NOMOV thresholds")
        if self.noMov_ratio > 0.0 and self.noMov_ratio < 1.0:
            print(f"Expected NOMOV ratio for auto-tuning: {self.noMov_ratio:.4f}")
    else:
        print(f"Using manual NOMOV thresholds: {self.manual_thresholds}")
    print(f"Number of stages to unfreeze for fine-tuning: {self.num_stages_to_unfreeze}")
    print("-" * 50)


# -------------------------
# Print accuracy and loss plot
# -------------------------
def print_2var_plots(self, var_1, var_2, label_1, label_2, title, xlabel, ylabel):
    # After training, plot the training loss curve
    model_name_str = str(self.model_name)
    out_dir = Path("final_models") / model_name_str / self.model_version
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_title = title.replace(" ", "_")
    out_path = out_dir / f"{model_name_str}_{safe_title}_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    plt.figure()
    plt.plot(var_1, label=label_1)
    plt.plot(var_2, label=label_2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(out_path)
    plt.close()


# -------------------------
# Visualize embeddings
# -------------------------
def visualize_embeddings(self, pre_training=False):
    self.model.eval()
    embeddings = []
    labels_list = []
    model_name_str = str(self.model_name)

    with torch.no_grad():
        for x, y in self.train_loader:
            feat = self.model.forward_features(x.to(self.device))
            if feat.ndim == 4:
                feat = feat.mean(dim=[2, 3])
            embeddings.append(feat.cpu())
            labels_list.extend(y.cpu().numpy())

    if not embeddings:
        print("No embeddings available to plot.")
        return

    embedding_matrix = torch.cat(embeddings, dim=0).numpy()
    labels_array = np.array(labels_list)

    if embedding_matrix.shape[0] < 2:
        print("Not enough samples to plot embeddings.")
        return

    pca = PCA(n_components=2)
    embedding_2d = pca.fit_transform(embedding_matrix)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels_array,
        cmap="viridis",
        s=18,
        alpha=0.8,
    )
    plt.title(f"Embedding PCA: {self.model_name}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, ticks=np.unique(labels_array))
    plt.tight_layout()

    if pre_training:
        out_path = Path("final_models") / model_name_str / self.model_version / f"{self.model_version}_preembeddings.png"
    else:
        out_path = Path("final_models") / model_name_str / self.model_version / f"{self.model_version}_postembeddings.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved embedding plot to {out_path}")


# -------------------------
# Training
# -------------------------
def train_model(self):
    criterion = nn.BCEWithLogitsLoss()
    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": [],
        "macro_f1": [],
        "val_macro_f1": []
    }

    # Early stopping variables
    best_macro_f1 = 0.0
    best_model_weights = copy.deepcopy(self.model.state_dict())
    patience = 5
    epochs_no_improve = 0

    # Training loop for binary classification with probability output
    for epoch in range(self.max_epochs):
        # Training
        self.model.train()
        running_loss = 0.0
        train_labels = []
        train_preds = []

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # collect train metrics
            probs = torch.sigmoid(outputs).squeeze(1)
            preds = (probs > 0.5).long()
            train_labels.extend(labels.squeeze(1).cpu().tolist())
            train_preds.extend(preds.cpu().tolist())

        train_loss = running_loss / len(self.train_loader)
        history["loss"].append(train_loss)
        train_acc = accuracy_score(train_labels, train_preds)
        history["accuracy"].append(train_acc)
        train_macro_f1 = f1_score(train_labels, train_preds, average='macro')
        history["macro_f1"].append(train_macro_f1)

        # Validation
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs).squeeze(1)
                preds = (probs > 0.5).long()
                labels_1d = labels.squeeze(1).long()

                correct += (preds == labels_1d).sum().item()
                total += labels_1d.size(0)

                all_labels.extend(labels_1d.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

        val_loss /= len(self.val_loader)
        history["val_loss"].append(val_loss)
        val_acc = correct / total
        history["val_accuracy"].append(val_acc)
        val_macro_f1 = f1_score(all_labels, all_preds, average='macro')
        history["val_macro_f1"].append(val_macro_f1)

        # 🔴 Early stopping logic
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_model_weights = copy.deepcopy(self.model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            self.model.load_state_dict(best_model_weights)
            print(f"Training finished. Best Macro F1: {best_macro_f1:.4f}")
            break

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train Macro F1: {train_macro_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val Macro F1: {val_macro_f1:.4f}")

        self.scheduler.step()
    print_2var_plots(self, history["loss"], history["val_loss"], "Train Loss", "Val Loss", "Loss Curves", "Epoch", "Loss")
    print_2var_plots(self, history["accuracy"], history["val_accuracy"], "Train Accuracy", "Val Accuracy", "Accuracy Curves", "Epoch", "Accuracy")
    print_2var_plots(self, history["macro_f1"], history["val_macro_f1"], "Train Macro F1", "Val Macro F1", "Macro F1 Curves", "Epoch", "Macro F1 Score")


# -------------------------
# Threshold tuning and evaluation
# -------------------------
def tune_and_evaluate_model(self):
    # -------------------------
    # Evaluation on val (A/B)
    # -------------------------
    def backtest_2_class(self):
        backtest_2_class_preds = []
        backtest_2_class_labels = []
        backtest_2_class_confidences = []

        backtest_2_class_dataset = FilteredRemappedDataset(
            self.fix_thresholds_dataset,
            ["downMovement", "upMovement"],
        )
        backtest_2_class_loader = DataLoader(
            backtest_2_class_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.model.eval()
        with torch.no_grad():
            for images, labels in backtest_2_class_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)

                probs = torch.sigmoid(outputs).squeeze(1)
                preds = (probs > 0.5).long()
                confidences = torch.where(preds == 1, probs, 1 - probs)

                backtest_2_class_preds.extend(preds.cpu().numpy())
                backtest_2_class_labels.extend(labels.cpu().numpy())
                backtest_2_class_confidences.extend(confidences.cpu().numpy())

        print("\n")
        print("-" * 50)
        print("2 class validation summary:")
        print("-" * 50)

        cm_val = confusion_matrix(backtest_2_class_labels, backtest_2_class_preds)

        print("Matrix labels order: ['downMovement', 'upMovement']")
        print(cm_val)

        print(f"\nTotal samples: {len(backtest_2_class_preds)}")
        print(f"Correct predictions: {sum(np.array(backtest_2_class_preds) == np.array(backtest_2_class_labels))}")
        print(f"Accuracy: {accuracy_score(backtest_2_class_labels, backtest_2_class_preds):.4f}")
        print(f"Average confidence: {np.mean(backtest_2_class_confidences):.4f}")
        print(f"Macro F1: {f1_score(backtest_2_class_labels, backtest_2_class_preds, average='macro'):.4f}")

    # -------------------------
    # Threshold auto-tuning for 3 class classification
    # -------------------------
    def fix_thresholds(self):
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
            for idx, name in enumerate(self.fix_thresholds_dataset.classes)
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

        print("\n")
        print("-" * 50)
        print("Threshold setting summary:")
        print("-" * 50)

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
        print(f"Thresholds used: low={low_threshold:.4f}, high={high_threshold:.4f}")

        with open(f"final_models/{self.model_name}/{self.model_version}/thresholds_summary.txt", "w") as f:
            f.write(f"{self.model_version}: low={low_threshold:.4f} high={high_threshold:.4f}\n")

        new_thresholds_preds = [predict_open_set(prob, low_threshold, high_threshold) for prob in nomov_probs]

        # -------------------------
        # Final evaluation with selected thresholds
        # -------------------------
        # Compute predictions after threshold selection so auto-tuned thresholds are actually applied.
        nomov_labels_np = np.array(nomov_labels)
        nomov_preds_np = np.array(new_thresholds_preds)
        nomov_probs_np = np.array(nomov_probs)

        print("\n")
        print("-" * 50)
        print("Final evaluation on threshold_estimation set:")
        print("-" * 50)

        cm_nomov = confusion_matrix(nomov_labels, new_thresholds_preds, labels=[0, 1, 2])

        print("Matrix labels order:", [open_label_to_name(i) for i in [0, 1, 2]])
        print(cm_nomov)

        print(f"\nTotal samples: {len(nomov_probs)}")
        print(f"Correct predictions: {sum(nomov_preds_np == nomov_labels_np)}")
        print(f"High confidence samples: {sum(1 for p in nomov_probs if p >= high_threshold)}")
        print(f"Low confidence samples: {sum(1 for p in nomov_probs if p <= low_threshold)}")
        print(f"Average confidence: {np.mean(nomov_probs):.4f}")
        print(f"Uncertain samples: {sum(1 for p in nomov_probs if low_threshold < p < high_threshold)}")

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

        print("\n3 class adjusted metrics:")
        for class_idx in [0, 1, 2]:
            class_name = open_label_to_name(class_idx)
            print(
                f"{class_name:15s} | support={int(support[class_idx]):4d} | "
                f"precision={precision_per_class[class_idx]:.4f} | "
                f"recall={recall_per_class[class_idx]:.4f} | f1={f1_per_class[class_idx]:.4f}"
            )

        # Print accuracy for only the A/B subset of the nomov_val set.
        ab_mask = nomov_labels_np < 2
        if np.sum(ab_mask) > 0:
            ab_accuracy = np.mean(nomov_preds_np[ab_mask] == nomov_labels_np[ab_mask])
            print(f"\nA/B subset accuracy (ignoring NOMOV): {ab_accuracy:.4f} (n={np.sum(ab_mask)})")

    backtest_2_class(self)
    fix_thresholds(self)


# -------------------------
# Save the trained model
# -------------------------
def save_model(self):
    out_path = Path(f"final_models/{self.model_name}/{self.model_version}/{self.model_version}.pth")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(self.model.state_dict(), out_path)


if __name__ == '__main__':
    # -------------------------
    # Argument parsing to allow easy configuration from command line
    # -------------------------
    argparser = argparse.ArgumentParser(prog="CNN Training", description="Fine-tune ConvNeXt")
    argparser.add_argument("-d", "--model_name", type=str, default=default_model_name.value, help="se Enums")
    argparser.add_argument("-e", "--max_epochs", type=int, default=default_max_epochs, help="Number of training epochs")
    argparser.add_argument("-t", "--manual_thresholds", nargs='+', type=float, default=[0, 0], help="Manual low and high confidence thresholds for 3 class classification (used when auto_tune_thresholds is False)")
    argparser.add_argument("-r", "--noMov_ratio", type=float, default=default_noMov_ratio, help="Expected ratio of NOMOV samples in the backtesting set (used for auto-tuning thresholds)")
    argparser.add_argument("-s", "--num_stages_to_unfreeze", type=int, default=default_num_stages_to_unfreeze, help="Number of stages to unfreeze for fine-tuning")
    argparser.add_argument("--backbone_lr_scale", type=float, default=default_backbone_lr_scale, help="Scale factor for backbone LR relative to base LR (e.g. 0.1)")
    args = argparser.parse_args()

    # -------------------------
    # Validate configuration inputs
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
    if args.max_epochs <= 0:
        raise ValueError(f"max_epochs must be a positive integer. Got {args.max_epochs}")
    else:
        max_epochs = args.max_epochs
        print(f"Max number of training epochs: {max_epochs}")

    # -------------------------
    # Map model_name argument to names defined in ModelNames class
    # -------------------------
    if (args.model_name == "all_TIs"):
        model_name = ModelNames.ALL_TIs
    elif (args.model_name == "No_TIs"):
        model_name = ModelNames.NO_TIs
    elif (args.model_name == "OBV"):
        model_name = ModelNames.OBV
    elif (args.model_name == "RSI"):
        model_name = ModelNames.RSI
    elif (args.model_name == "BB"):
        model_name = ModelNames.BB
    elif (args.model_name == "No_OBV"):
        model_name = ModelNames.NO_OBV
    else:
        raise ValueError(f"Invalid model_name argument: {args.model_name}")

    new_model = ModelMaker(model_name=model_name)
    new_model.print_summary()
    new_model.visualize_embeddings()
    new_model.train_model()
    new_model.evaluate_model()
    new_model.save_model()
