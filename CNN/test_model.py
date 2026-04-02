import timm
import torch
import model_maker_clean as model_maker
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from pathlib import Path


# -------------------------
# Configurations
# -------------------------
workers_cpu = 0
workers_gpu = 4
batch_size_cpu = 8
batch_size_gpu = 16

default_low = 0.2
default_high = 0.8

default_model_path = "final_models/all_TIs.pth"     # Path to the trained model weights
default_img_path = "test_image.png"  # Path to the image you want to classify
default_dataset_path = "datasets/all_TIs"  # Path to the backtesting dataset


class TestingModel:

    def __init__(self, model_path=None):
        self.model_path = self._resolve_path(model_path, default_model_path)

        # Set the device to GPU if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

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
            self.model_path = default_model_path
            state_dict = torch.load(self.model_path, map_location=self.device)

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Define the mean and std for normalization 
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # -------------------------
    # Helper functions
    # -------------------------
    @staticmethod
    def _resolve_path(path, default_path):
        if path is None:
            return default_path

        if isinstance(path, str):
            cleaned = path.strip().strip('"').strip("'")
            return cleaned or default_path

        return str(Path(path))

    # -------------------------
    # Prediction functions
    # -------------------------
    def image_to_prediction(self, img_path=default_img_path, new_low=default_low, new_high=default_high):
        low = new_low
        high = new_high

        img_path = self._resolve_path(img_path, default_img_path)

        # Load and preprocess the image
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Get the model's output and convert it to a probability
        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.sigmoid(output).item()

        if prob <= low:
            prediction = "DOWN"
        elif prob >= high:
            prediction = "UP"
        else:
            prediction = "NOMOV"

        print(f"Prediction : {prediction}")
        print(f"Probability: {prob:.4f}")

        return prediction, prob

    def backtesting_dataset_to_predictions(
        self, dataset_path=default_dataset_path, new_low=default_low, new_high=default_high
            ):
        dataset_path = self._resolve_path(dataset_path, default_dataset_path)
        backtest_3_class_dataset = ImageFolder(root=f"{dataset_path}/backtesting", transform=self.transform)
        backtest_3_class_loader = torch.utils.data.DataLoader(
            backtest_3_class_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        low = new_low
        high = new_high

        model_maker.backtest_3_class(
            self.model,
            self.device,
            dataset_path, 
            backtest_3_class_loader,
            self.transform,
            manual_low_confidence_threshold=low,
            manual_high_confidence_threshold=high
            )


if __name__ == '__main__':
    pass
