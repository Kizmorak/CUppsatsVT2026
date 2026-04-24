import model_maker
import test_model
import sys
import custom_tee
from pathlib import Path

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


def run_gradcam_for_model(new_model):
    model_name_str = str(new_model.model_name)
    model_path = Path("final_models") / model_name_str / new_model.model_version / f"{new_model.model_version}.pth"
    if not model_path.exists():
        print(f"Grad-CAM skipped: model checkpoint not found at {model_path}")
        return

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
    model = gradcam_visualize.build_model(device=device, checkpoint_path=str(model_path))
    target_layer = gradcam_visualize.get_target_layer(model)

    four_random_images = sorted(image_candidates)[:4]
    out_dir = Path("final_models") / model_name_str / new_model.model_version
    out_dir.mkdir(parents=True, exist_ok=True)
    for img in four_random_images:
        image_path = str(img)
        base_name = f"{new_model.model_version}_gradcam"
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


def copy_log_to_backup(new_model):
    sys.stdout.flush()
    model_name_str = str(new_model.model_name)

    with open("night_worker_log.txt", "r", encoding="utf-8") as source:
        last_log_content = source.read()

    if not last_log_content.strip():
        print("No log content to back up.")
        return

    backup_dir = Path("final_models") / model_name_str / new_model.model_version
    backup_dir.mkdir(parents=True, exist_ok=True)
    with open(backup_dir / f"{new_model.model_version}_log.txt", "w", encoding="utf-8") as backup:
        backup.write(last_log_content)

    # Clear the original log file for the next run
    with open("night_worker_log.txt", "w", encoding="utf-8") as source:
        source.write("")


def make_night_model(
    model_name,
    num_stages_to_unfreeze=2,
    base_lr=2e-4,
    backbone_lr_scale=0.1,
    max_epochs=30,
):
    new_model = model_maker.ModelMaker(
        model_name=model_name,
        num_stages_to_unfreeze=num_stages_to_unfreeze,
        base_lr=base_lr,
        backbone_lr_scale=backbone_lr_scale,
        max_epochs=max_epochs,
    )
    print(f"Starting training for model: {new_model.model_version}")

    run_gradcam_for_model(new_model)
    copy_log_to_backup(new_model)
    new_test_model = test_model.TestingModel(new_model.model_version)
    new_test_model.backtesting_dataset_to_predictions()


if __name__ == '__main__':

    # -------------------------
    # Training the models
    # -------------------------

    # model_name = None
    # try:
    #     for i in range(2):
    #         model_name = model_maker.ModelNames.NO_TI_68_17_10_10_20251224_10_15_2
    #         make_night_model(
    #             model_name=model_name,
    #             num_stages_to_unfreeze=2,
    #             base_lr=2e-4,
    #             backbone_lr_scale=0.1,
    #             max_epochs=12,
    #         )
    # except Exception as e:
    #     print(f"Error occurred while training and evaluating model: {model_name}: {e}")

    # model_name = None
    # try:
    #     for i in range(2):
    #         model_name = model_maker.ModelNames.NO_TI_76_19_10_10_20251224_10_1_3
    #         make_night_model(
    #             model_name=model_name,
    #             num_stages_to_unfreeze=2,
    #             base_lr=2e-4,
    #             backbone_lr_scale=0.1,
    #             max_epochs=12,
    #         )
    # except Exception as e:
    #     print(f"Error occurred while training and evaluating model: {model_name}: {e}")

    # model_name = None
    # try:
    #     for i in range(2):
    #         model_name = model_maker.ModelNames.NO_TI_100_25_10_10_20251224_10_1_2
    #         make_night_model(
    #             model_name=model_name,
    #             num_stages_to_unfreeze=2,
    #             base_lr=2e-4,
    #             backbone_lr_scale=0.1,
    #             max_epochs=12,
    #         )
    # except Exception as e:
    #     print(f"Error occurred while training and evaluating model: {model_name}: {e}")

    model_name = None
    try:
        for i in range(2):
            model_name = model_maker.ModelNames.OBV_104_26_10_10_20251224_10_1_4
            make_night_model(
                model_name=model_name,
                num_stages_to_unfreeze=2,
                base_lr=2e-4,
                backbone_lr_scale=0.1,
                max_epochs=12,
            )
    except Exception as e:
        print(f"Error occurred while training and evaluating model: {model_name}: {e}")

    # model_name = None
    # try:
    #     for i in range(2):
    #         model_name = model_maker.ModelNames.OBV_98_14_14_14_20260131_10_1_2
    #         make_night_model(
    #             model_name=model_name,
    #             num_stages_to_unfreeze=2,
    #             base_lr=2e-4,
    #             backbone_lr_scale=0.1,
    #             max_epochs=12,
    #         )
    # except Exception as e:
    #     print(f"Error occurred while training and evaluating model: {model_name}: {e}")

    # model_name = None
    # try:
    #     for i in range(2):
    #         model_name = model_maker.ModelNames.BB_98_14_14_14_20260131_10_1_2
    #         make_night_model(
    #             model_name=model_name,
    #             num_stages_to_unfreeze=2,
    #             base_lr=2e-4,
    #             backbone_lr_scale=0.1,
    #             max_epochs=12,
    #         )
    # except Exception as e:
    #     print(f"Error occurred while training and evaluating model: {model_name}: {e}")

    # model_name = None
    # try:
    #     for i in range(2):
    #         model_name = model_maker.ModelNames.OBV_70_10_10_10_20260131_10_1_3
    #         make_night_model(
    #             model_name=model_name,
    #             num_stages_to_unfreeze=3,
    #             base_lr=2e-4,
    #             backbone_lr_scale=0.1,
    #             max_epochs=12,
    #         )
    # except Exception as e:
    #     print(f"Error occurred while training and evaluating model: {model_name}: {e}")

    # model_name = None
    # try:
    #     for i in range(2):
    #         model_name = model_maker.ModelNames.BB_70_10_10_10_20260131_10_1_3
    #         make_night_model(
    #             model_name=model_name,
    #             num_stages_to_unfreeze=3,
    #             base_lr=2e-4,
    #             backbone_lr_scale=0.1,
    #             max_epochs=12,
    #         )
    # except Exception as e:
    #     print(f"Error occurred while training and evaluating model: {model_name}: {e}")

    # model_name = None
    # try:
    #     for i in range(2):
    #         model_name = model_maker.ModelNames.OBV_84_12_12_12_20260131_10_1_4
    #         make_night_model(
    #             model_name=model_name,
    #             num_stages_to_unfreeze=3,
    #             base_lr=2e-4,
    #             backbone_lr_scale=0.1,
    #             max_epochs=12,
    #         )
    # except Exception as e:
    #     print(f"Error occurred while training and evaluating model: {model_name}: {e}")

    # model_name = None
    # try:
    #     for i in range(2):
    #         model_name = model_maker.ModelNames.BB_84_12_12_12_20260131_10_1_4
    #         make_night_model(
    #             model_name=model_name,
    #             num_stages_to_unfreeze=3,
    #             base_lr=2e-4,
    #             backbone_lr_scale=0.1,
    #             max_epochs=12,
    #         )
    # except Exception as e:
    #     print(f"Error occurred while training and evaluating model: {model_name}: {e}")

    # model_name = None
    # try:
    #     for i in range(2):
    #         model_name = model_maker.ModelNames.RSI_70_10_10_10_20260131_10_15_2
    #         make_night_model(
    #             model_name=model_name,
    #             num_stages_to_unfreeze=3,
    #             base_lr=2e-4,
    #             backbone_lr_scale=0.1,
    #             max_epochs=12,
    #         )
    # except Exception as e:
    #     print(f"Error occurred while training and evaluating model: {model_name}: {e}")

    # model_name = None
    # try:
    #     for i in range(2):
    #         model_name = model_maker.ModelNames.RSI_84_12_12_12_20260131_10_1_4
    #         make_night_model(
    #             model_name=model_name,
    #             num_stages_to_unfreeze=3,
    #             base_lr=2e-4,
    #             backbone_lr_scale=0.1,
    #             max_epochs=12,
    #         )
    # except Exception as e:
    #     print(f"Error occurred while training and evaluating model: {model_name}: {e}")

    # model_name = None
    # try:
    #     for i in range(2):
    #         model_name = model_maker.ModelNames.RSI_98_14_14_14_20260131_10_1_2
    #         make_night_model(
    #             model_name=model_name,
    #             num_stages_to_unfreeze=3,
    #             base_lr=2e-4,
    #             backbone_lr_scale=0.1,
    #             max_epochs=12,
    #         )
    # except Exception as e:
    #     print(f"Error occurred while training and evaluating model: {model_name}: {e}")

    # model_name = None
    # try:
    #     for i in range(2):
    #         model_name = model_maker.ModelNames.RSI_70_10_10_10_20260131_10_1_3
    #         make_night_model(
    #             model_name=model_name,
    #             num_stages_to_unfreeze=3,
    #             base_lr=2e-4,
    #             backbone_lr_scale=0.1,
    #             max_epochs=12,
    #         )
    # except Exception as e:
    #     print(f"Error occurred while training and evaluating model: {model_name}: {e}")

    # -------------------------
    # Testing the models
    # -------------------------

    # model_version = "NO_TI_70_10_10_10_20260131_10_15_2__20260420_153000"  # specify the model version you want to test here

    # try:
    #     new_test_model = test_model.TestingModel(model_version)
    #     new_test_model.backtesting_dataset_to_predictions()
    # except Exception as e:
    #     print(f"Error occurred while testing model: {model_version}: {e}")

    # for i in range(4):
    #     try:
    #         model_version = "NO_TI_70_10_10_10_20260131_10_15_2__20260420_153000"
    #         new_test_model = test_model.TestingModel(model_version)
    #         new_test_model.image_to_prediction()
    #     except Exception as e:
    #         print(f"Error occurred while testing model: {model_version}: {e}")
