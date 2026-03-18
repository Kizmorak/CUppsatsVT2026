import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn.functional as F


# -------------------------
# Variabler och inställningar
# -------------------------
# Model and training parameters
model = "convnextv2_atto"
num_classes = 3
dataset_path = "dataset"  # Path to your dataset folder containing 'train' and 'val' subfolders

# Hyperparameters
workers_cpu = 0
workers_gpu = 4
batch_size_cpu = 8
batch_size_gpu = 16

# Training parameters
epochs = 5
lr = 1e-4
weight_decay = 0.05
#number of layers to unfreeze for fine-tuning (0 = only head, 1 = last stage + head, 2 = last 2 stages + head, etc.)

# Augmentation options
use_random_affine = True


# -------------------------
# Device
# -------------------------
# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#optimize based on hardware
batch_size = batch_size_cpu if device.type == "cpu" else batch_size_gpu
num_workers = workers_cpu if device.type == "cpu" else workers_gpu
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# -------------------------
# Transforms
# -------------------------
# Define the mean and standard deviation for normalization (these are commonly used values for pre-trained models)
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# Limited data augmentation for fine-tuning
if use_random_affine:
    transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=0,
            translate=(0.03, 0.03),
            scale=(0.97, 1.03)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


# -------------------------
# Dataset
# -------------------------
train_dataset = ImageFolder(root=f"{dataset_path}/train", transform=transform)
val_dataset = ImageFolder(root=f"{dataset_path}/val", transform=transform)

# Handle class imbalance
class_counts = np.bincount(train_dataset.targets, minlength=num_classes).tolist()
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)

sample_weights = [class_weights[label] for label in train_dataset.targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# -------------------------
# Model
# -------------------------
model = timm.create_model(
    model,
    pretrained=True
)
model.reset_classifier(num_classes=num_classes)
model.to(device)

# Freeze all layers except the head
for p in model.parameters():
    p.requires_grad = False

for p in model.head.parameters():
    p.requires_grad = True

# Unfreeze the last stage of the model 
for name, p in model.named_parameters():
    if "stages.3" in name:
        p.requires_grad = True



# -------------------------
# Loss + optimizer
# -------------------------
# Define the optimizer (AdamW is commonly used for training vision models)
optimizer = torch.optim.AdamW([
    {"params": model.head.parameters(), "lr": lr},
    {"params": model.stages[3].parameters(), "lr": lr * 0.1},
], weight_decay=weight_decay)

criterion = nn.CrossEntropyLoss()

# -------------------------
# Scheduler
# -------------------------
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-6
)


# -------------------------
# Training
# -------------------------
num_epochs = epochs
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
        for images,labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct/total

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

    scheduler.step()



# -------------------------
# Confusion matrix + Prediction records
# -------------------------

all_preds = []
all_labels = []
all_confidences = []
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        preds = outputs.argmax(1)
        confidences = F.softmax(outputs, dim=1).max(1)[0]

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

print("Confusion Matrix:")
print(cm)

# Save predictions with confidence to file
class_names = train_dataset.classes
with open("predictions.txt", "w", encoding="utf-8") as f:
    f.write("Sample | Predicted | Confidence | Actual | Correct\n")
    f.write("-" * 60 + "\n")
    for i, (pred, conf, label) in enumerate(zip(all_preds, all_confidences, all_labels)):
        is_correct = "✓" if pred == label else "✗"
        f.write(f"{i:4d}   | {class_names[pred]:15s} | {conf:.4f}     | {class_names[label]:15s} | {is_correct}\n")

print(f"\nPrediction records saved to predictions.txt")
print(f"\nPrediction Summary:")
print(f"Total samples: {len(all_preds)}")
print(f"Correct predictions: {sum(np.array(all_preds) == np.array(all_labels))}")
print(f"Accuracy: {sum(np.array(all_preds) == np.array(all_labels)) / len(all_preds):.4f}")
print(f"Average confidence: {np.mean(all_confidences):.4f}")

# -------------------------
# Save model
# -------------------------
torch.save(model.state_dict(),"convnext_atto_finetuned.pth")





""" #Grad-CAM visualization to check which parts of the image the model is focusing on for its predictions
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

target_layer = model.stages[-1].blocks[-1]

cam = GradCAM(model=model, target_layers=[target_layer])
grayscale_cam = cam(input_tensor=input_tensor)
visualization = show_cam_on_image(image, grayscale_cam[0]) """