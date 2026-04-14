import os
import random

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError as exc:
    raise ImportError(
        "pytorch-grad-cam is required. Install it with: pip install grad-cam"
    ) from exc


class BinaryLogitTarget:
    """Target for single-logit binary classifiers."""

    def __init__(self, positive=True):
        self.sign = 1.0 if positive else -1.0

    def __call__(self, model_output):
        # pytorch-grad-cam may pass scalar, [1], or [B, 1] depending on context.
        if model_output.ndim == 0:
            return self.sign * model_output
        if model_output.ndim == 1:
            return self.sign * model_output[0]
        return self.sign * model_output[:, 0]


def build_model(device, checkpoint_path, model_name="convnextv2_atto"):
    model = timm.create_model(model_name, pretrained=False)
    if not hasattr(model, "norm_pre"):
        model.norm_pre = nn.Identity()
    model.reset_classifier(1)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_target_layer(model):
    # Last ConvNeXt block is usually a good Grad-CAM target layer.
    return model.stages[-1].blocks[-1]


def preprocess_image(image_path):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    tfm = transforms.Compose([transforms.ToTensor(), normalize])

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0
    tensor = tfm(image).unsqueeze(0)
    return image_np, tensor


def render_gradcam(model, target_layer, device, image_path, output_path, negative=False):
    rgb_image, input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    grayscale_cam = None
    try:
        with GradCAM(model=model, target_layers=[target_layer]) as cam:
            targets = [BinaryLogitTarget(positive=not negative)]
            cam_result = cam(input_tensor=input_tensor, targets=targets)
            if cam_result is None or len(cam_result) == 0:
                raise RuntimeError("Grad-CAM returned no activation map.")
            grayscale_cam = cam_result[0]
    except Exception as exc:
        raise RuntimeError(f"Grad-CAM generation failed for image: {image_path}") from exc

    if grayscale_cam is None:
        raise RuntimeError(f"Grad-CAM did not produce an activation map for image: {image_path}")

    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    Image.fromarray(visualization).save(output_path)


def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def pick_random_images_by_class(dataset_dir, per_class=1, seed=None):
    rng = random.Random(seed) if seed is not None else random.Random()
    class_to_images = {}

    for class_name in sorted(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        candidates = [
            os.path.join(class_dir, name)
            for name in os.listdir(class_dir)
            if is_image_file(name)
        ]
        if not candidates:
            continue

        sample_count = min(per_class, len(candidates))
        class_to_images[class_name] = rng.sample(candidates, sample_count)

    return class_to_images


def pick_consecutive_images_by_class(dataset_dir, per_class=10, start_index=None, seed=None):
    rng = random.Random(seed) if seed is not None else random.Random()
    class_to_images = {}

    for class_name in sorted(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        candidates = sorted(
            [
                os.path.join(class_dir, name)
                for name in os.listdir(class_dir)
                if is_image_file(name)
            ],
            key=lambda p: os.path.basename(p),
        )
        if not candidates:
            continue

        window = min(per_class, len(candidates))
        max_start = len(candidates) - window

        if start_index is None:
            idx = rng.randint(0, max_start) if max_start > 0 else 0
        else:
            idx = max(0, min(start_index, max_start))

        class_to_images[class_name] = candidates[idx: idx + window]

    return class_to_images


def main():
    pass


if __name__ == "__main__":
    main()
