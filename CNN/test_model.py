import timm
import torch
from datetime import datetime, timedelta
from torchvision import transforms
from PIL import Image
from pathlib import Path
import csv
import os
from torch.utils.data import Dataset
import custom_tee
import sys


# -------------------------
# Configurations
# -------------------------
workers_cpu = 0
workers_gpu = 4
batch_size_cpu = 8
batch_size_gpu = 16

default_low = 0.1
default_high = 0.9

sys.stdout = custom_tee.CustomTee("night_worker_log.txt")


class TestingModel:

    def __init__(self, model_name=None):

        # --------------------------
        # Initialization: set up untrained model, set up device, load pretrained weights
        # --------------------------
        self.model_name = model_name
        self.model_path = f"final_models/{self.model_name}_model.pth"

        # Set the device to GPU if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Optimize based on hardware
        self.batch_size = batch_size_cpu if self.device.type == "cpu" else batch_size_gpu
        self.num_workers = workers_cpu if self.device.type == "cpu" else workers_gpu
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # Set up model
        self.model = timm.create_model("convnextv2_atto", pretrained=False)
        self.model.reset_classifier(num_classes=1)  # because we use sigmoid binary

        # Load the trained model weights
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
        except (FileNotFoundError, OSError):
            print(f"Error: Model file not found at {self.model_path}")
            raise

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from: {self.model_path}")

        # --------------------------
        # Set up image transformations (same as training) and dataloaders for backtesting
        # --------------------------
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # Set up datasets and dataloaders for backtesting
        self.backtest_dataset = BacktestingDataset(
            root_dir=f"datasets/{self.model_name}/backtesting",
            transform=self.transform
        )

        self.backtest_loader = torch.utils.data.DataLoader(
            self.backtest_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        # --------------------------
        # Load thresholds for open-set classification (buy/sell/nomov)
        # --------------------------
        self.low_threshold = default_low
        self.high_threshold = default_high

        # Load thresholds from all_thresholds.txt if available
        with open("final_models/all_thresholds.txt", "r") as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                model_line = lines[i].strip()
                low_line = lines[i + 1].strip()
                high_line = lines[i + 2].strip()

                if model_line.startswith(f"{self.model_name}-"):
                    self.low_threshold = float(low_line.split(":")[1].strip())
                    self.high_threshold = float(high_line.split(":")[1].strip())
                    print(
                        f"Using thresholds for {self.model_name}: "
                        f"Low={self.low_threshold:.4f}, High={self.high_threshold:.4f}"
                    )
                    break
            else:
                print(f"No thresholds found for {self.model_name} in all_thresholds.txt. Using default thresholds.")
                self.low_threshold = default_low
                self.high_threshold = default_high

    # -------------------------
    # Prediction functions
    # -------------------------
    def image_to_prediction(self):

        folder = Path("inputGraph/" + self.model_name + "/")
        img_path = next(folder.glob("*.png"))

        # Load and preprocess the image
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Get the model's output and convert it to a probability
        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.sigmoid(output).item()

        if prob <= self.low_threshold:
            prediction = "sell"
        elif prob >= self.high_threshold:
            prediction = "buy"
        else:
            prediction = ""

        print(f"Prediction : {prediction}")
        print(f"Probability: {prob:.4f}")

        return prediction

    # -------------------------
    # Evaluation on nomov_val (3 class)
    # -------------------------
    def backtesting_dataset_to_predictions(self):
        nomov_probs_and_paths = []

        with torch.no_grad():
            for images, paths in self.backtest_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.sigmoid(outputs).squeeze(1).flatten()

                nomov_probs_and_paths.extend([
                    (float(prob.item()), path)
                    for prob, path in zip(probs, paths)
                ])

        def predict_open_set(prob, low, high):
            if prob <= low:
                return 0
            if prob >= high:
                return 1
            return 2

        def predict_binary(prob, binary_threshold=0.5):
            return 1 if prob >= binary_threshold else 0

        print("\nBacktesting summary:")
        print(f"Total samples: {len(nomov_probs_and_paths)}.")

        backtest_3_class_preds_and_paths = [
            (predict_open_set(prob, self.low_threshold, self.high_threshold), path)
            for prob, path in nomov_probs_and_paths
        ]
        backtest_2_class_preds_and_paths = [
            (predict_binary(prob), path)
            for prob, path in nomov_probs_and_paths
        ]

        # -------------------------
        # Save predictions CSV file methods (for backtesting)
        # -------------------------
        def extract_datetime_from_path(path):
            stem = Path(path).stem
            try:
                dt = datetime.strptime(stem, "%Y-%m-%d_%H%M")
            except ValueError:
                print(f"Bad filename format: {stem}")
                return None
            plus = dt + timedelta(minutes=31)
            return plus.strftime("%Y-%m-%d %H:%M:00")

        def save_3_class_csv_predictions(model_name, backtest_3_class_preds_and_paths):
            out_path = Path(f"csv_files/{model_name}_Backtesting_predictions.csv")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "target"])

                skipped = 0
                for pred, path in backtest_3_class_preds_and_paths:
                    if pred == 2:  # NOMOV
                        skipped += 1
                        continue

                    target = 1 if pred == 0 else 0
                    time = extract_datetime_from_path(path)
                    if time is None:
                        continue

                    writer.writerow([time, target])

            print(f"Saved CSV. Skipped {skipped} NOMOV predictions.")

        def save_2_class_csv_predictions(model_name, backtest_2_class_preds_and_paths):
            out_path = Path(f"csv_files/{model_name}_Backtesting_predictions_2_class.csv")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "target"])

                for pred, path in backtest_2_class_preds_and_paths:
                    target = 1 if pred == 0 else 0
                    time = extract_datetime_from_path(path)
                    if time is None:
                        continue

                    writer.writerow([time, target])

            print("Saved 2-class CSV.")

        save_3_class_csv_predictions(self.model_name, backtest_3_class_preds_and_paths)
        save_2_class_csv_predictions(self.model_name, backtest_2_class_preds_and_paths)


class BacktestingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        self.image_paths.sort()  # keep time order

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, path


if __name__ == '__main__':
    pass
