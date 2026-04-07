import timm
import torch
import model_maker
from torchvision import transforms
from PIL import Image
from pathlib import Path
import csv
from datetime import datetime, timedelta


# -------------------------
# Configurations
# -------------------------
workers_cpu = 0
workers_gpu = 4
batch_size_cpu = 8
batch_size_gpu = 16

default_low = 0.1
default_high = 0.9


class TestingModel:

    def __init__(self, model_name=None):
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

        # Define the mean and std for normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # Set up datasets and dataloaders for backtesting
        self.backtest_dataset = model_maker.ImageFolder(root=f"datasets/{self.model_name}/backtesting", transform=self.transform)
        self.backtest_loader = torch.utils.data.DataLoader(self.backtest_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)  

    # -------------------------
    # Prediction functions
    # -------------------------
    def image_to_prediction(self, new_low=default_low, new_high=default_high):
        low = new_low
        high = new_high

        folder = Path("inputGraph/" + self.model_name + "/")
        img_path = next(folder.glob("*.png"))

        # Load and preprocess the image
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Get the model's output and convert it to a probability
        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.sigmoid(output).item()

        if prob <= low:
            prediction = "sell"
        elif prob >= high:
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
            for idx, name in enumerate(self.backtest_dataset.classes)
        }

        def open_label_to_name(label: int) -> str:
            name = {v: k for k, v in name_to_open_label.items()}
            return name.get(label, "Unknown")

        # Get model probabilities on the nomov_val set and map to 3 class labels
        with torch.no_grad():
            for images, labels in self.backtest_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.sigmoid(outputs).squeeze(1).flatten()

                nomov_probs.extend(probs.cpu().tolist())
                nomov_labels.extend(val_to_open[int(lbl)] for lbl in labels)

        def predict_open_set(prob, low, high):
            if prob <= low:
                return 0
            if prob >= high:
                return 1
            return 2

        def predict_binary(prob, threshold):
            return 1 if prob >= threshold else 0

        print("\nBacktesting summary:")
        print(f"Total samples: {len(nomov_probs)}")

        with open("final_models/all_thresholds.txt", "r") as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                model_line = lines[i].strip()
                low_line = lines[i + 1].strip()
                high_line = lines[i + 2].strip()

                if model_line.startswith(f"{self.model_name}-"):
                    low_threshold = float(low_line.split(":")[1].strip())
                    high_threshold = float(high_line.split(":")[1].strip())
                    print(f"Using thresholds for {self.model_name}: Low={low_threshold:.4f}, High={high_threshold:.4f}")
                    break
            else:
                print(f"No thresholds found for {self.model_name} in all_thresholds.txt. Using default thresholds.")
                low_threshold = default_low
                high_threshold = default_high

        backtest_3_class_preds = [predict_open_set(prob, low_threshold, high_threshold) for prob in nomov_probs]
        backtest_2_class_preds = [predict_binary(prob, 0.5) for prob in nomov_probs]

    # -------------------------
    # Save predictions CSV file method (for backtesting)
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

        def save_csv_predictions(backtest_dataset, model_name, preds):
            out_path = Path(f"csv_files/{model_name}_Backtesting_predictions.csv")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "target"])

                skipped = 0
                for (path, _), pred in zip(backtest_dataset.samples, preds):
                    if pred == 2:  # NOMOV
                        skipped += 1
                        continue

                    target = 1 if pred == 0 else 0
                    t = extract_datetime_from_path(path)
                    if t is None:
                        continue

                    writer.writerow([t, target])

            print(f"Saved CSV. Skipped {skipped} NOMOV predictions.")

        save_csv_predictions(self.backtest_dataset, self.model_name, backtest_3_class_preds)

        def save_2_class_csv_predictions(backtest_dataset, model_name, preds):
            out_path = Path(f"csv_files/{model_name}_Backtesting_predictions_2_class.csv")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "target"])

                for (path, _), pred in zip(backtest_dataset.samples, preds):
                    target = 1 if pred == 0 else 0
                    t = extract_datetime_from_path(path)
                    if t is None:
                        continue

                    writer.writerow([t, target])

            print("Saved 2-class CSV.")

        save_2_class_csv_predictions(self.backtest_dataset, self.model_name, backtest_2_class_preds)


if __name__ == '__main__':
    pass
