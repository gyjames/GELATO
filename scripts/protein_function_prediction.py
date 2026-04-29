import argparse
import os
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier


def train_model(train_features, train_labels, test_features, test_labels, param_grid, model, function_name, result_path, model_path, roc_path):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
    grid_search.fit(train_features, train_labels)
    predictions = grid_search.predict(test_features)
    y_scores = grid_search.predict_proba(test_features)[:, 1]
    acc = metrics.accuracy_score(test_labels, predictions)
    pre = metrics.precision_score(test_labels, predictions)
    recall = metrics.recall_score(test_labels, predictions)
    f1 = metrics.f1_score(test_labels, predictions)
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, y_scores)
    auc = metrics.auc(fpr, tpr)
    result_df = pd.DataFrame(
        {
            "Function": [function_name],
            "Accuracy": [acc],
            "Precision": [pre],
            "Recall": [recall],
            "F1": [f1],
            "ROC_AUC": [auc],
        }
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    roc_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(result_path, index=False)
    best_model = grid_search.best_estimator_
    dump(best_model, model_path)
    roc_result = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds})
    roc_result.to_csv(roc_path, index=False)


def combine_metric_files(folder_path, output_path):
    combined_df = pd.DataFrame()
    if folder_path.exists():
        for filename in os.listdir(folder_path):
            if filename.endswith("metrics.csv"):
                file_path = folder_path / filename
                df = pd.read_csv(file_path)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", type=Path, default='../data/function_embedding.csv', required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.embedding)
    output_base = args.output_dir
    genetic_processing = [
        "Transcription",
        "Translation",
        "Folding, sorting and degradation",
        "Replication and repair",
        "Chromosome",
        "Information processing in viruses",
    ]
    data.loc[data["funtion"].isin(genetic_processing), "funtion"] = "Genetic Information Processing"
    function_ls = [
        "Genetic Information Processing",
        "Metabolism of cofactors and vitamins",
        "Glycan biosynthesis and metabolism",
        "Metabolism of terpenoids and polyketides",
        "Xenobiotics biodegradation and metabolism",
        "Biosynthesis of other secondary metabolites",
        "Energy metabolism",
        "Carbohydrate metabolism",
        "Nucleotide metabolism",
        "Membrane transport",
    ]
    data = data[data["funtion"].isin(function_ls)].copy()
    f2l = dict(zip(function_ls, range(len(function_ls))))
    data["label"] = data["funtion"].map(f2l)

    for function_name in tqdm(function_ls):
        function_counts = data[data["funtion"] == function_name].shape[0]
        pos_data = data[data["funtion"] == function_name]
        neg_data = data[data["funtion"] != function_name]
        neg_data = neg_data.sample(n=function_counts, random_state=42)
        train_pos_data = pos_data.sample(frac=0.8, random_state=42)
        train_neg_data = neg_data.sample(frac=0.8, random_state=42)
        test_pos_data = pos_data.drop(train_pos_data.index)
        test_neg_data = neg_data.drop(train_neg_data.index)
        train_data = pd.concat([train_pos_data, train_neg_data])
        test_data = pd.concat([test_pos_data, test_neg_data])
        train_data = train_data.sample(frac=1, random_state=42)
        test_data = test_data.sample(frac=1, random_state=42)
        train_labels = [1 if value == function_name else 0 for value in train_data["funtion"]]
        test_labels = [1 if value == function_name else 0 for value in test_data["funtion"]]
        train_features = train_data.iloc[:, :-3]
        test_features = test_data.iloc[:, :-3]

        rf_param_grid = {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [10, 50, None],
            "min_samples_split": [2, 4, 10],
        }
        rfc = RandomForestClassifier()
        rf_model_path = output_base / "model" / "rf_model" / f"{function_name}_random_forest_model.joblib"
        rf_result_path = output_base / "result" / "rf_result" / f"{function_name}_metrics.csv"
        rf_roc_path = output_base / "result" / "rf_result" / f"{function_name}_roc.csv"
        train_model(train_features, train_labels, test_features, test_labels, rf_param_grid, rfc, function_name, rf_result_path, rf_model_path, rf_roc_path)

        svm_param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
        }
        svm_model = SVC(probability=True)
        svm_roc_path = output_base / "result" / "svm_result" / f"{function_name}_roc.csv"
        svm_result_path = output_base / "result" / "svm_result" / f"{function_name}_metrics.csv"
        svm_model_path = output_base / "model" / "svm_model" / f"{function_name}_svm_model.joblib"
        train_model(train_features, train_labels, test_features, test_labels, svm_param_grid, svm_model, function_name, svm_result_path, svm_model_path, svm_roc_path)

        xgb_param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.3],
        }
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        xgb_roc_path = output_base / "result" / "xgb_result" / f"{function_name}_roc.csv"
        xgb_result_path = output_base / "result" / "xgb_result" / f"{function_name}_metrics.csv"
        xgb_model_path = output_base / "model" / "xgb_model" / f"{function_name}_xgb_model.joblib"
        train_model(train_features, train_labels, test_features, test_labels, xgb_param_grid, xgb_model, function_name, xgb_result_path, xgb_model_path, xgb_roc_path)

    combine_metric_files(output_base / "result" / "rf_result", output_base / "best_rf_result.csv")
    combine_metric_files(output_base / "result" / "svm_result", output_base / "best_svm_result.csv")
    combine_metric_files(output_base / "result" / "xgb_result", output_base / "best_xgb_result.csv")


if __name__ == "__main__":
    main()
