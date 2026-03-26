import model_maker
from datetime import datetime


def make_model(model_output_classes, dataset_path, num_epochs, nomov_ratio=(0.8), num_stages_to_unfreeze=(1), manual_thresholds=(0, 0)):
# Make models
    try:
        model_maker.setup_train_and_evaluate(model_output_classes=model_output_classes, dataset_path=dataset_path, num_epochs=num_epochs, expected_nomov_ratio=nomov_ratio, num_stages_to_unfreeze=num_stages_to_unfreeze, manual_thresholds=manual_thresholds)

    except Exception as e:
        print(f"Error occurred while training and evaluating model: {model_output_classes}, {dataset_path}, {num_epochs}: {e}")

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
#               model output classes    (A_B_NOMOV for evaluating including NOMOV or A_B for trading every minute)
#               thresholds              (auto-tuned by default, but you can specify manual thresholds for testing))
#               expected nomov ratio    (used for auto-tuning thresholds)
#               num_stages_to_unfreeze  (0 = only head, 1 = last stage + head, 2 = last 2 stages + head, etc. - used for fine-tuning)
if __name__ == '__main__':

    for i in range(0):
        make_model(model_output_classes=model_maker.ModelOutputClasses.A_B_NOMOV, dataset_path=model_maker.DatasetPaths.ALL_TIs, num_epochs=1, manual_thresholds=(0.4, 0.6))

    for i in range(0):
        make_model(model_output_classes=model_maker.ModelOutputClasses.A_B_NOMOV, dataset_path=model_maker.DatasetPaths.ALL_TIs, num_epochs=3)
    
    for i in range(1):
        make_model(model_output_classes=model_maker.ModelOutputClasses.A_B_NOMOV, dataset_path=model_maker.DatasetPaths.ALL_TIs, num_epochs=1, num_stages_to_unfreeze=2)
    
    for i in range(0):
        make_model(model_output_classes=model_maker.ModelOutputClasses.A_B_NOMOV, dataset_path=model_maker.DatasetPaths.ALL_TIs, num_epochs=1, nomov_ratio=0.5)

