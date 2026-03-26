import model_maker
from datetime import datetime


def make_model(model_output_classes, dataset_path, num_epochs, manual_thresholds=(0, 0)):
# Make models
    try:
        model_maker.setup_train_and_evaluate(model_output_classes=model_output_classes, dataset_path=dataset_path, num_epochs=num_epochs)

    except Exception as e:
        print(f"Error occurred while training and evaluating model: {model_output_classes}, {dataset_path}, {num_epochs}: {e}")

# Add eval results to file
    add_eval_results_to_file(model_output_classes, dataset_path, num_epochs, manual_thresholds)


# Open file and append results
def add_eval_results_to_file(model_output_classes, dataset_path, num_epochs, manual_thresholds):
    try:
        with open("Model_evaluation_results.txt", "a") as f:
            f.seek(0)
            f.write("__" * 25 + "\n")  # Separator between results
            f.write("-." * 25 + "\n")  
            f.write(f"\nModel evaluation results for model_output_classes={model_output_classes}, dataset_path={dataset_path}\n")
            f.write(f"Datetime: {datetime.now()}\n")
            f.write(f"Number of epochs: {num_epochs}\n")
            if manual_thresholds != (0, 0):
                f.write(f"Manual low confidence threshold: {manual_thresholds[0]}\n")
                f.write(f"Manual high confidence threshold: {manual_thresholds[1]}\n")
            else:
                f.write("Auto-tuning of NOMOV thresholds\n")
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


if __name__ == '__main__':

    for i in range(1):
        make_model(model_output_classes=model_maker.ModelOutputClasses.A_B_NOMOV, dataset_path=model_maker.DatasetPaths.DATASET_ALL_TIs, num_epochs=1, manual_thresholds=(0.4, 0.6))

    for i in range(1):
        make_model(model_output_classes=model_maker.ModelOutputClasses.A_B_NOMOV, dataset_path=model_maker.DatasetPaths.DATASET_ALL_TIs, num_epochs=3)