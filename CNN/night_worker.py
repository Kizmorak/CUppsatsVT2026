import model_maker_clean as model_maker
import test_model


def make_model(dataset_path, num_epochs, nomov_ratio=(0.8), num_stages_to_unfreeze=(1), manual_thresholds=(0, 0)):
    # Make models
    try:
        model_maker.setup_train_and_evaluate(
            dataset_path=dataset_path,
            num_epochs=num_epochs,
            expected_nomov_ratio=nomov_ratio,
            num_stages_to_unfreeze=num_stages_to_unfreeze,
            manual_thresholds=manual_thresholds)

    except Exception as e:
        print(f"Error occurred while training and evaluating model: {dataset_path}, {num_epochs}: {e}")

# Add eval results to file
    add_eval_results_to_file()


# Open file and append results
def add_eval_results_to_file():
    try:
        with open("Model_evaluation_results.txt", "a") as f:
            f.write("__" * 25 + "\n")  # Separator between results
            f.write("-." * 25 + "\n")

            # Open Evaluation_results.txt and read results
            try:
                with open("Evaluation_results.txt", "r") as eval_file:
                    eval_results = eval_file.read()
                    f.write(eval_results + "\n\n")
            except FileNotFoundError:
                f.write("Evaluation_results.txt not found.\n\n")

            f.write("-'" * 25 + "\n")
            f.write("__" * 25 + "\n")  # Separator between results
    except Exception as e:
        print(f"Error occurred while writing evaluation results to file: {e}")

# Run the model training and evaluation for different configurations
# You can adjust the parameters in the loops below to test different configurations as needed.
# Parameters:   number of epochs        (10 is a good starting point, but you can try more or less)
#               dataset paths           (remember it's enums)
#               thresholds              (auto-tuned by default, but you can specify manual thresholds for testing))
#               expected nomov ratio    (used for auto-tuning thresholds)
#               num_stages_to_unfreeze  (0 = only head, 1 = last stage + head, 2 = last 2 stages + head, etc.)


if __name__ == '__main__':

    for i in range(1):
        make_model(
            dataset_path=model_maker.DatasetPaths.ALL_TIs,
            num_epochs=8)

    for i in range(1):
        make_model(
            dataset_path=model_maker.DatasetPaths.NO_TIs,
            num_epochs=8
            )

    for i in range(1):
        make_model(
            dataset_path=model_maker.DatasetPaths.NO_BB,
            num_epochs=8)

    for i in range(1):
        make_model(
            dataset_path=model_maker.DatasetPaths.NO_BB_NO_OBV,
            num_epochs=8)

    for i in range(1):
        make_model(
            dataset_path=model_maker.DatasetPaths.NO_BB_NO_RSI,
            num_epochs=8)

    for i in range(1):
        make_model(
            dataset_path=model_maker.DatasetPaths.NO_OBV,
            num_epochs=8)

    for i in range(1):
        make_model(
            dataset_path=model_maker.DatasetPaths.NO_RSI,
            num_epochs=8)
            
    for i in range(1):
        make_model(
            dataset_path=model_maker.DatasetPaths.NO_RSI_NO_OBV,
            num_epochs=8)

# -------------------------
# Testing the models
# -------------------------

# Example usage of the testing functions
image_path = "test_image.png"
dataset_path = "datasets/all_TIs"
low_threshold = 0.2
high_threshold = 0.8

testing_model = test_model.TestingModel()
# testing_model.image_to_prediction(image_path, low_threshold, high_threshold)
# testing_model.backtesting_dataset_to_predictions(dataset_path, low_threshold, high_threshold)
