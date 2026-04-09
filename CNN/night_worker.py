import model_maker
import test_model
import sys
import custom_tee
from datetime import datetime
import re


# Run the model training and evaluation for different configurations
# You can adjust the parameters in the loops below to test different configurations as needed.
# Parameters:   max number of epochs    (with early stopping, so it won't necessarily run for all epochs)
#               model name              (remember it's enums)
#               thresholds              (auto-tuned by default, but you can specify manual thresholds for testing))
#               expected nomov ratio    (used for auto-tuning thresholds)
#               num_stages_to_unfreeze  (0 = only head, 1 = last stage + head, 2 = last 2 stages + head, etc.)

# copy the last log to a backup file before starting new runs
with open("night_worker_log.txt", "r") as f:
    last_log_content = f.read()
    log_lines = last_log_content.splitlines()
    first_line = log_lines[0] if log_lines else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    first_line = re.sub(r"[^A-Za-z0-9_.-]+", "_", first_line).strip("_") or "backup"
with open(f"night_worker_logs/{first_line}.txt", "w") as f:
    f.write(last_log_content)

sys.stdout = custom_tee.CustomTee("night_worker_log.txt")

if __name__ == '__main__':

    model_name = model_maker.ModelNames.ALL_TIs

    try:

        for i in range(1):
            model_name = model_maker.ModelNames.NO_TIs
            sys.stdout.write(f"{model_name}_{datetime.now().strftime('%d_%H-%M')}\n")
            model_maker.ModelMaker(
                model_name=model_name,
                num_stages_to_unfreeze=3,
                max_epochs=2,
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
