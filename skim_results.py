import os
import pandas as pd

BASE_DIR = "CSVs"

CASES = {
    "PLAYER_CSVs": {
        "type": "regression",
        "prefixes": ["bet_predictor", "card_predictor"]
    },
    "CLUSTER_CSVs": {
        "type": "regression",
        "prefixes": ["bet_predictor", "card_predictor"]
    },
    "CLASS_CSVs": {
        "type": "classification",
        "prefixes": ["persona_predictor", "persona_predictor_card_info"]
    }
}


def best_rows_from_file(file_path):
    df = pd.read_csv(file_path)

    rows = []

    if "avg_score" in df.columns:
        row = df.loc[df["avg_score"].idxmax()].copy()
        row["score_type"] = "avg_score"
        row["source_file"] = os.path.basename(file_path)
        rows.append(row)

    if "max_score" in df.columns:
        row = df.loc[df["max_score"].idxmax()].copy()
        row["score_type"] = "max_score"
        row["source_file"] = os.path.basename(file_path)
        rows.append(row)

    if len(rows) == 0 and "score" in df.columns:
        row = df.loc[df["score"].idxmax()].copy()
        row["score_type"] = "score"
        row["source_file"] = os.path.basename(file_path)
        rows.append(row)

    return rows


def best_from_files(directory, files):
    best_by_type = {}

    for file in files:
        file_path = os.path.join(directory, file)

        if not os.path.exists(file_path):
            print(f"Missing file, skipped: {file_path}")
            continue

        rows = best_rows_from_file(file_path)

        for row in rows:
            score_type = row["score_type"]
            score_value = row[score_type]

            if score_type not in best_by_type or score_value > best_by_type[score_type][score_type]:
                best_by_type[score_type] = row

    return best_by_type


def analyze_regression_case(directory, prefix):
    svm_files = [
        f"{prefix}_svr_linear_scores.csv",
        f"{prefix}_svr_poly_scores.csv",
        f"{prefix}_svr_rbf_scores.csv",
        f"{prefix}_svr_sigmoid_scores.csv",
        f"{prefix}_RFECV_svr_linear_scores.csv",
        f"{prefix}_KPCA_10_svr_linear_scores.csv",
        f"{prefix}_KPCA_15_svr_linear_scores.csv",
        f"{prefix}_KPCA_20_svr_linear_scores.csv"
    ]

    tree_files = [
        f"{prefix}_random_forest_regressor.csv",
        f"{prefix}_RFECV_random_forest_regressor.csv",
        f"{prefix}_KPCA_10_random_forest_regressor.csv",
        f"{prefix}_KPCA_15_random_forest_regressor.csv",
        f"{prefix}_KPCA_20_random_forest_regressor.csv"
    ]

    mlp_files = [
        f"{prefix}_MLP_scores.csv",
        f"{prefix}_RFECV_MLP_scores.csv",
        f"{prefix}_KPCA_10_MLP_scores.csv",
        f"{prefix}_KPCA_15_MLP_scores.csv",
        f"{prefix}_KPCA_20_MLP_scores.csv"
    ]

    return {
        "SVM/SVR": best_from_files(directory, svm_files),
        "Trees/Random Forest": best_from_files(directory, tree_files),
        "MLP": best_from_files(directory, mlp_files)
    }


def analyze_classification_case(directory, prefix):
    svm_files = [
        f"{prefix}_svc_linear_scores.csv",
        f"{prefix}_svc_poly_scores.csv",
        f"{prefix}_svc_rbf_scores.csv",
        f"{prefix}_svc_sigmoid_scores.csv",
        f"{prefix}_RFECV_svc_linear_scores.csv",
        f"{prefix}_KPCA_10_svc_linear_scores.csv",
        f"{prefix}_KPCA_15_svc_linear_scores.csv",
        f"{prefix}_KPCA_20_svc_linear_scores.csv"
    ]

    tree_files = [
        f"{prefix}_random_forest_classifier.csv",
        f"{prefix}_RFECV_random_forest_classifier.csv",
        f"{prefix}_KPCA_10_random_forest_classifier.csv",
        f"{prefix}_KPCA_15_random_forest_classifier.csv",
        f"{prefix}_KPCA_20_random_forest_classifier.csv"
    ]

    mlp_files = [
        f"{prefix}_MLP_classifier_scores.csv",
        f"{prefix}_RFECV_MLP_classifier_scores.csv",
        f"{prefix}_KPCA_10_MLP_classifier_scores.csv",
        f"{prefix}_KPCA_15_MLP_classifier_scores.csv",
        f"{prefix}_KPCA_20_MLP_classifier_scores.csv"
    ]

    return {
        "SVM/SVC": best_from_files(directory, svm_files),
        "Trees/Random Forest": best_from_files(directory, tree_files),
        "MLP": best_from_files(directory, mlp_files)
    }


def print_result(model_type, rows):
    print("\n--------------------------------------------------")
    print(f"{model_type}")
    print("--------------------------------------------------")

    if not rows:
        print("No valid result found.")
        return

    for score_type, row in rows.items():

        print(f"\nBEST {score_type.upper()}")

        for key, value in row.items():

            if key == score_type:
                print(f"  {key:<20}: {value:.6f}")

            else:
                print(f"  {key:<20}: {value}")


def run_analysis():
    for folder, info in CASES.items():
        directory = os.path.join(BASE_DIR, folder)

        print("\n##################################################")
        print(f"DIRECTORY: {directory}")
        print("##################################################")

        for prefix in info["prefixes"]:
            print("\n====================================")
            print(f"CASE: {prefix}")
            print("====================================")

            if info["type"] == "regression":
                results = analyze_regression_case(directory, prefix)
            else:
                results = analyze_classification_case(directory, prefix)

            for model_type, row in results.items():
                print_result(model_type, row)


if __name__ == "__main__":
    run_analysis()