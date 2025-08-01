import os
import json
import pandas as pd
import numpy as np
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some command-line arguments.")
    parser.add_argument("--model_name", type=str, help="Your model path", required=True)
    parser.add_argument(
        "--model_class", type=str, help="Your model path", required=True
    )
    parser.add_argument("--data_type", type=str, help="Your model path", required=True)
    parser.add_argument(
        "--tasks",
        type=lambda x: x.split(","),
        help="Your tasks, divided by comma. ",
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    DATA_DIR = os.getenv("DATA_DIR")
    root_dir = f"{DATA_DIR}/model/sft/{args.model_name}"
    regressions = ["delaney", "lipo", "freesolv"]
    csv_files = []
    print(args.tasks)
    for task in args.tasks:

        model_name = args.model_name
        if args.data_type != "None":
            model_dir_name = (
                model_name + "_" + args.model_class + "_" + args.data_type + "-" + task
            )
        else:
            model_dir_name = model_name + "_" + args.model_class + "-" + task

        folders = [model_dir_name + "_1", model_dir_name + "_2", model_dir_name + "_3"]

        if task in regressions:
            metrics = ["test_mae", "test_rmse", "test_r2", "test_pearson_corr"]
        else:
            metrics = ["test_accuracy", "test_f1", "test_pr_auc", "test_roc_auc"]

        data = {metric: [] for metric in metrics}
        standard = True

        for folder in folders:
            file_path = os.path.join(root_dir, folder, "all_results.json")

            if not os.path.exists(file_path):
                standard = False
                continue
            with open(file_path, "r") as f:
                results = json.load(f)
                for metric in metrics:
                    data[metric].append(results[metric])

        mean_std_error = {}
        for metric in metrics:
            mean = np.mean(data[metric])
            std_error = np.std(data[metric], ddof=1)
            if not standard:
                std_error = 0
            mean_std_error[metric] = f"{mean:.3f}±{std_error:.3f}"
            # mean_std_error[metric] = f"{mean:.3f}"

        df = pd.DataFrame([mean_std_error])
        df = df.rename(
            columns={
                "test_accuracy": "Accuracy",
                "test_f1": "F1 Score",
                "test_pr_auc": "PR AUC",
                "test_roc_auc": "ROC AUC",
            }
        )
        results_dir = os.path.join(
            DATA_DIR, "results", model_name + "_" + args.model_class
        )
        os.makedirs(results_dir, exist_ok=True)

        csv_file_path = os.path.join(results_dir, f"{model_dir_name}.csv")
        df.to_csv(csv_file_path, index=False)

        csv_files.append(csv_file_path)

        print(f"Generated CSV: {csv_file_path}")

        print(DATA_DIR + "/results" + model_dir_name + ".csv")

    combined_df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files], axis=1)

    final_csv_path = os.path.join(
        DATA_DIR, "results", model_name + "_" + args.model_class, "combined_results.csv"
    )
    combined_df.to_csv(final_csv_path, index=False)
    print(f"Final combined CSV saved at: {final_csv_path}")


if __name__ == "__main__":
    main()
