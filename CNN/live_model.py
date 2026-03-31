import timm
import torch
from torchvision import transforms
from PIL import Image


if __name__ == '__main__':

        
    # ----------------------------------
    # Configurations
    # ----------------------------------

    img_path = "test_image.png"  # Path to the image you want to classify
    model_path = "a_model_to_test_with.pth"     # Path to the trained model weights
    low_threshold = 0.32         # Threshold for classifying as "DOWN (A)"
    high_threshold = 0.81        # Threshold for classifying as "UP (B)"

    #----------------------------------

    # Set up device and model
    device = torch.device("cpu")  # or "cuda"

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

    def image_to_prediction(img_path, model, transform, device, low, high):
        # Load and preprocess the image
        image = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        # Get the model's output and convert it to a probability
        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output).item()

        if prob <= low:
            return "DOWN (A)", prob
        elif prob >= high:
            return "UP (B)", prob
        else:
            return "NOMOV", prob

    # ----------------------------------
    # Make a prediction on the image
    # ----------------------------------

    prediction, prob = image_to_prediction(img_path, model, transform, device, low=0.32, high=0.81)

    print(f"Probability: {prob:.4f}")
    print(f"Prediction : {prediction}")

