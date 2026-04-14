import model_maker
import sys
import custom_tee
from datetime import datetime
from pathlib import Path
import shutil

import torch
import gradcam_visualize


# Run the model training and evaluation for different configurations
# You can adjust the parameters in the loops below to test different configurations as needed.
# Parameters:   max number of epochs    (with early stopping, so it won't necessarily run for all epochs)
#               model name              (remember it's enums)
#               thresholds              (auto-tuned by default, but you can specify manual thresholds for testing))
#               expected nomov ratio    (used for auto-tuning thresholds)
#               num_stages_to_unfreeze  (0 = only head, 1 = last stage + head, 2 = last 2 stages + head, etc.)


sys.stdout = custom_tee.CustomTee("night_worker_log.txt")


def run_gradcam_for_model(model_name):
    model_name_str = str(model_name)
    model_dir = Path("final_models") / model_name_str / "temp"
    checkpoint_candidates = sorted(
        model_dir.glob(f"{model_name_str}*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not checkpoint_candidates:
        print(f"Grad-CAM skipped: no checkpoint starting with '{model_name_str}' found in {model_dir}")
        return
    checkpoint_path = checkpoint_candidates[0]

    image_candidates = []
    for split in ["val", "train", "threshold_estimation"]:
        split_dir = Path("datasets") / model_name_str / split
        if split_dir.exists():
            image_candidates = sorted([
                p for p in split_dir.rglob("*")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            ])
            if image_candidates:
                break

    if not image_candidates:
        print("Grad-CAM skipped: no image found in dataset splits (val/train/threshold_estimation).")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = gradcam_visualize.build_model(device=device, checkpoint_path=str(checkpoint_path))
    target_layer = gradcam_visualize.get_target_layer(model)

    four_random_images = sorted(image_candidates)[:4]
    out_dir = Path("final_models") / model_name_str / "temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    for img in four_random_images:
        image_path = str(img)
        base_name = f"{model_name_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_gradcam"
        output_path = out_dir / f"{base_name}.png"
        suffix = 1
        while output_path.exists():
            output_path = out_dir / f"{base_name}_{suffix}.png"
            suffix += 1
        gradcam_visualize.render_gradcam(
            model=model,
            target_layer=target_layer,
            device=device,
            image_path=image_path,
            output_path=str(output_path),
            negative=False,
        )
    print(f"Saved Grad-CAMs to: {out_dir}")


def copy_log_to_backup():
    sys.stdout.flush()

    with open("night_worker_log.txt", "r", encoding="utf-8") as source:
        last_log_content = source.read()

    if not last_log_content.strip():
        print("No log content to back up.")
        return

    log_lines = last_log_content.splitlines()
    first_line = log_lines[0] if log_lines else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    first_line = log_lines[0].strip()

    prefix = "Starting training for model:"
    if first_line.startswith(prefix):
        payload = first_line[len(prefix):].strip()
        model_name_str, _, timestamp_str = payload.rpartition(" : ")
    else:
        raise ValueError(f"Unexpected first log line: {first_line}")

    with open(f"final_models/{model_name_str}/temp/{model_name_str}_log.txt", "w", encoding="utf-8") as backup:
        backup.write(last_log_content)


def archive_model_outputs(model_name):
    model_name_str = str(model_name)
    source_dir = Path("final_models") / model_name_str / "temp"
    if not source_dir.exists():
        print(f"Archive skipped: source directory not found: {source_dir}")
        return None

    archive_dir = Path("final_models") / model_name_str / f"{model_name_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for item in sorted(source_dir.iterdir()):
        if item.is_file():
            shutil.move(str(item), str(archive_dir / item.name))

    if not any(source_dir.iterdir()):
        try:
            source_dir.rmdir()
        except OSError:
            pass

    print(f"Archived model outputs to: {archive_dir}")
    return archive_dir


def make_night_model(
    model_name,
    num_stages_to_unfreeze=2,
    base_lr=2e-4,
    backbone_lr_scale=0.1,
    max_epochs=30,
):
    # delete temp directory for the model if it exists, to ensure a clean start
    model_name_str = str(model_name)
    temp_dir = Path("final_models") / model_name_str / "temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    print(f"Starting training for model: {model_name} : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    model_maker.ModelMaker(
        model_name=model_name,
        num_stages_to_unfreeze=num_stages_to_unfreeze,
        base_lr=base_lr,
        backbone_lr_scale=backbone_lr_scale,
        max_epochs=max_epochs,
    )

    run_gradcam_for_model(model_name)
    copy_log_to_backup()
    archive_model_outputs(model_name)


if __name__ == '__main__':

    try:
        for i in range(1):
            model_name = model_maker.ModelNames.NO_TIs
            make_night_model(
                model_name=model_name,
                num_stages_to_unfreeze=2,
                base_lr=2e-4,
                backbone_lr_scale=0.1,
                max_epochs=12,
            )

    except Exception as e:
        print(f"Error occurred while training and evaluating model: {model_name}: {e}")

    # -------------------------
    # Testing the models
    # -------------------------

    # Example usage of the testing functions

    model_name = model_maker.ModelNames.ALL_TIs
    low_threshold = 0.1
    high_threshold = 0.9

    # all_TIs_testing_model = test_model.TestingModel(model_maker.ModelNames.ALL_TIs)
    # no_TIs_testing_model = test_model.TestingModel(model_maker.ModelNames.NO_TIs)
    # no_BB_testing_model = test_model.TestingModel(model_maker.ModelNames.NO_BB)
    # no_BB_NO_OBV_testing_model = test_model.TestingModel(model_maker.ModelNames.NO_BB_NO_OBV)
    # no_BB_NO_RSI_testing_model = test_model.TestingModel(model_maker.ModelNames.NO_BB_NO_RSI)
    # no_OBV_testing_model = test_model.TestingModel(model_maker.ModelNames.NO_OBV)
    # no_RSI_testing_model = test_model.TestingModel(model_maker.ModelNames.NO_RSI)
    # no_RSI_NO_OBV_testing_model = test_model.TestingModel(model_maker.ModelNames.NO_RSI_NO_OBV)

    # all_TIs_testing_model.image_to_prediction()

    # all_TIs_testing_model.backtesting_dataset_to_predictions()
    # no_TIs_testing_model.backtesting_dataset_to_predictions()
    # no_BB_testing_model.backtesting_dataset_to_predictions()
    # no_BB_NO_OBV_testing_model.backtesting_dataset_to_predictions()
    # no_BB_NO_RSI_testing_model.backtesting_dataset_to_predictions()
    # no_OBV_testing_model.backtesting_dataset_to_predictions()
    # no_RSI_testing_model.backtesting_dataset_to_predictions()
    # no_RSI_NO_OBV_testing_model.backtesting_dataset_to_predictions()
