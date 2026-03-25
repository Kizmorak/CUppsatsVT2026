import model_maker


def make_model(model_output_classes, dataset_path, num_epochs):
# Make models
    try:
        model_maker.setup_train_and_evaluate(model_output_classes=model_output_classes, dataset_path=dataset_path, num_epochs=num_epochs)

    except Exception as e:
        print(f"Error occurred while training and evaluating model: {model_output_classes}, {dataset_path}, {num_epochs}: {e}")

# Add eval results to file
    add_eval_results_to_file(model_output_classes, dataset_path, num_epochs)


# Open file and append results
def add_eval_results_to_file(model_output_classes, dataset_path, num_epochs):
    try:
        with open("model_evaluation_results.txt", "a") as f:
            f.write(f"Model evaluation results for model_output_classes={model_output_classes}, dataset_path={dataset_path}, num_epochs={num_epochs}:\n")
            # Open A_B_eval_results.txt and read results
            with open("A_B_eval_results.txt", "r") as eval_file:
                eval_results = eval_file.read()
                f.write(eval_results + "\n\n")
                f.write("-" * 50 + "\n\n")  # Separator between results
    except Exception as e:
        print(f"Error occurred while writing evaluation results to file: {e}")

make_model(model_output_classes=model_maker.ModelOutputClasses.A_B_NOMOV, dataset_path=model_maker.DatasetPaths.DATASET_ALL_TIs, num_epochs=1)
make_model(model_output_classes=model_maker.ModelOutputClasses.A_B_NOMOV, dataset_path=model_maker.DatasetPaths.DATASET_ALL_TIs, num_epochs=1)