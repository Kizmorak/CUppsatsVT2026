import timm
import torch
import model_maker_clean as model_maker
from torchvision import transforms
from PIL import Image

# -------------------------
# Device and model
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

def run_saved_model(model_path, device):
    # Set up model

    model = timm.create_model("convnextv2_atto", pretrained=False)
    model.reset_classifier(num_classes=1)  # because we use sigmoid binary

    # Load the trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define the mean and std for normalization 
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return model, transform

# -------------------------
# Dataset
# -------------------------
def dataset_setup(batch_size, num_workers, dataset_path):
    train_transform, eval_transform = transforms_setup()
    
    backtest_3_class_dataset = ImageFolder(root=f"{dataset_path}/backtesting", transform=eval_transform)

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
    backtest_3_class_loader = DataLoader(backtest_3_class_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return backtest_3_class_dataset, backtest_3_class_loader

# -------------------------
# Prediction functions
# -------------------------
def image_to_prediction(img_path, model, transform, device):
    # Load and preprocess the image
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    # Get the model's output and convert it to a probability
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    if prob <= low:
        prediction = "DOWN (A)", prob
    elif prob >= high:
        prediction = "UP (B)", prob
    else:
        prediction = "NOMOV", prob

    # ----------------------------------
    # Make a prediction on the image
    # ----------------------------------

    model, transform, device = run_saved_model(model_path)
    prediction, prob = image_to_prediction(img_path, model, transform, device)

    print(f"Probability: {prob:.4f}")
    print(f"Prediction : {prediction}")

    return prediction, prob

def backtesting_dataset_to_predictions(dataset_path, model, transform, device):
    model_maker.backtest_3_class(model, device, dataset_path, backtest_3_class_loader, transform, train_dataset.classes)

if __name__ == '__main__':

    # ----------------------------------
    # Configurations
    # ----------------------------------
    img_path = "test_image.png"  # Path to the image you want to classify
    model_path = "all_TIs.pth"     # Path to the trained model weights
    dataset_path = "path_to_backtesting_dataset"  # Path to the backtesting dataset 

    #----------------------------------
    device, batch_size, num_workers = device_spec_setup()
    model, transform, device =run_saved_model(model_path, device)
    backtest_3_class_dataset, backtest_3_class_loader = dataset_setup(batch_size, num_workers, dataset_path="path_to_backtesting_dataset")

    # ----------------------------------
    # Make a prediction on the image
    
    prediction, prob = image_to_prediction(img_path, model, transform, device)

    # ----------------------------------
    # Make predictions on the backtesting dataset
    
    # backtesting_dataset_to_predictions(model, device, dataset_path, backtest_3_class_loader, transform)


