import model_maker
import test_model


def make_model(model_name, num_epochs=(10), nomov_ratio=(0.0), num_stages_to_unfreeze=(1), thresholds=(0, 0)):
    # Make models
    try:
        new_model = model_maker.ModelMaker(
            model_name=model_name,
            num_epochs=num_epochs,
            noMov_ratio=nomov_ratio,
            num_stages_to_unfreeze=num_stages_to_unfreeze,
            thresholds=thresholds)
        return new_model

    except Exception as e:
        print(f"Error occurred while training and evaluating model: {model_name}, {num_epochs} epochs: {e}")

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

    # for i in range(1):
    #     make_model(
    #         model_name=model_maker.ModelNames.ALL_TIs,
    #         num_epochs=8)

    # for i in range(1):
    #     make_model(
    #         model_name=model_maker.ModelNames.NO_TIs,
    #         num_epochs=8
    #         )

    # for i in range(1):
    #     make_model(
    #         model_name=model_maker.ModelNames.NO_BB,
    #         num_epochs=8)

    # for i in range(1):
    #     make_model(
    #         model_name=model_maker.ModelNames.NO_BB_NO_OBV,
    #         num_epochs=8)

    # for i in range(1):
    #     make_model(
    #         model_name=model_maker.ModelNames.NO_BB_NO_RSI,
    #         num_epochs=8)

    # for i in range(1):
    #     make_model(
    #         model_name=model_maker.ModelNames.NO_OBV,
    #         num_epochs=8)

    # for i in range(1):
    #     make_model(
    #         model_name=model_maker.ModelNames.NO_RSI,
    #         num_epochs=8)

    # for i in range(1):
    #     make_model(
    #         model_name=model_maker.ModelNames.NO_RSI_NO_OBV,
    #         num_epochs=8)

    # -------------------------
    # Testing the models
    # -------------------------

    # Example usage of the testing functions

    model_name = "all_TIs"
    low_threshold = 0.1
    high_threshold = 0.9

    all_TIs_testing_model = test_model.TestingModel("all_TIs")
    no_TIs_testing_model = test_model.TestingModel("No_TIs")
    no_BB_testing_model = test_model.TestingModel("No_BB")
    no_BB_NO_OBV_testing_model = test_model.TestingModel("No_BB_No_OBV")
    no_BB_NO_RSI_testing_model = test_model.TestingModel("No_BB_No_RSI")
    no_OBV_testing_model = test_model.TestingModel("No_OBV")
    no_RSI_testing_model = test_model.TestingModel("No_RSI")
    no_RSI_NO_OBV_testing_model = test_model.TestingModel("No_RSI_No_OBV")

    # all_TIs_testing_model.image_to_prediction()

    all_TIs_testing_model.backtesting_dataset_to_predictions()
    no_TIs_testing_model.backtesting_dataset_to_predictions()
    no_BB_testing_model.backtesting_dataset_to_predictions()
    no_BB_NO_OBV_testing_model.backtesting_dataset_to_predictions()
    no_BB_NO_RSI_testing_model.backtesting_dataset_to_predictions()
    no_OBV_testing_model.backtesting_dataset_to_predictions()
    no_RSI_testing_model.backtesting_dataset_to_predictions()
    no_RSI_NO_OBV_testing_model.backtesting_dataset_to_predictions()
