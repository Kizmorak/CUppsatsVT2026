import timm
import torch
import model_maker
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

    def backtesting_dataset_to_predictions(
        self, new_low=default_low, new_high=default_high
            ):
        backtest_3_class_dataset = ImageFolder(root=f"datasets/{self.model_name}/backtesting", transform=self.transform)
        backtest_3_class_loader = torch.utils.data.DataLoader(
            backtest_3_class_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        low = new_low
        high = new_high

        model_maker.backtest_3_class(
            self.model,
            self.device,
            self.model_name,
            backtest_3_class_loader,
            self.transform,
            manual_low_confidence_threshold=low,
            manual_high_confidence_threshold=high
            )


if __name__ == '__main__':
    pass
