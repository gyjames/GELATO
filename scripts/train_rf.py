import argparse
import json
import os
import re

import pandas as pd
from joblib import dump
from scipy import sparse
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


def tokenization(text):
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '/', ':', ';', '<', '=', '>','\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]
    text = re.sub("<.*?>", " ", text, flags=re.S)
    text = re.sub("|".join(fileters), " ", text, flags=re.S)
    ls = [i.strip() for i in text.split()]
    for i, w in enumerate(ls):
        w = re.sub(r'\.$', '', w)
        ls[i] = w
    return ls


class BuildVocab:
    def __init__(self, data_path=None, min_word_occurences=5, max_word_occurences=None, max_feature=None):
        self.total_file_path = data_path or []
        self.word_counter = {}
        self.w2i = {}
        self.genome_len = []

        for file_path in self.total_file_path:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            genome = [i for i in tokenization(content) if i != "Cluster"]
            self.genome_len.append(len(genome))
            for word in genome:
                self.word_counter[word] = self.word_counter.get(word, 0) + 1

        if max_word_occurences is not None:
            self.word_counter = {k: v for k, v in self.word_counter.items() if v <= max_word_occurences}

        if isinstance(max_feature, int):
            temp = sorted(self.word_counter.items(), key=lambda x: x[1], reverse=True)[:max_feature]
        else:
            temp = sorted(self.word_counter.items(), key=lambda x: x[1], reverse=True)
        self.word_counter = dict(temp)

        for w, c in self.word_counter.items():
            if c >= min_word_occurences:
                self.w2i[w] = len(self.w2i)

        self.i2w = dict(zip(self.w2i.values(), self.w2i.keys()))
        self.seq_len = sum(self.genome_len) / len(self.genome_len) if self.genome_len else 0


def extract_features(paths, word_to_idx_map):
    features_list = []
    for path in paths:
        features = dict.fromkeys(word_to_idx_map.keys(), 0)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        genome = [i for i in tokenization(content) if i != "Cluster"]
        for word in genome:
            if word in features:
                features[word] += 1
        features_list.append([1 if i > 0 else 0 for i in features.values()])
    return features_list


def build_sparse_matrix(features_list):
    if not features_list:
        raise ValueError("No features were extracted from the provided paths.")
    return sparse.vstack([sparse.csr_matrix(arr) for arr in features_list])


def safe_metric(metric_fn, y_true, y_pred):
    try:
        return metric_fn(y_true, y_pred)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest model for compound prediction")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to output folder")
    parser.add_argument("--train", type=str, required=True, help="Path to train data csv")
    parser.add_argument("--valid", type=str, required=True, help="Path to valid data csv")
    parser.add_argument("--test", type=str, required=True, help="Path to test data csv")
    parser.add_argument("--c", type=str, required=True, help="Compound column name")
    args = parser.parse_args()

    param_grid = {
        "n_estimators": [30, 50, 100, 300],
        "max_depth": [100, 300, None],
        "min_samples_split": [2, 3],
    }

    train_data = pd.read_csv(args.train)
    valid_data = pd.read_csv(args.valid)
    test_data = pd.read_csv(args.test)

    train_labels = train_data[args.c].tolist()
    valid_labels = valid_data[args.c].tolist()
    test_labels = test_data[args.c].tolist()

    train_paths = train_data["path"].tolist()
    valid_paths = valid_data["path"].tolist()
    test_paths = test_data["path"].tolist()

    vocab = BuildVocab(train_paths + valid_paths, max_word_occurences=len(train_paths + valid_paths))
    w2i = vocab.w2i

    train_features_list = extract_features(train_paths, w2i)
    valid_features_list = extract_features(valid_paths, w2i)
    test_features_list = extract_features(test_paths, w2i)

    train_X = build_sparse_matrix(train_features_list)
    valid_X = build_sparse_matrix(valid_features_list)
    test_X = build_sparse_matrix(test_features_list)

    best_accuracy = -1
    best_params = {}
    best_rf_model = None

    for n_estimators in param_grid["n_estimators"]:
        for max_depth in param_grid["max_depth"]:
            for min_samples_split in param_grid["min_samples_split"]:
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    n_jobs=-1,
                    max_features=None,
                )
                model.fit(train_X, train_labels)
                valid_predictions = model.predict(valid_X)
                current_accuracy = metrics.accuracy_score(valid_labels, valid_predictions)

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                    }
                    best_rf_model = model

    if best_rf_model is None:
        raise RuntimeError("Failed to train a RandomForest model.")

    predictions = best_rf_model.predict(test_X)
    y_scores = best_rf_model.predict_proba(test_X)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, y_scores)

    roc_output_dir = os.path.join(args.model_dir, "roc")
    result_output_dir = os.path.join(args.model_dir, "result")
    model_output_dir = os.path.join(args.model_dir, "model")
    vocab_output_dir = os.path.join(args.model_dir, "vocab")
    os.makedirs(roc_output_dir, exist_ok=True)
    os.makedirs(result_output_dir, exist_ok=True)
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(vocab_output_dir, exist_ok=True)

    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thresholds})
    roc_df.to_csv(os.path.join(roc_output_dir, f"{args.c}.csv"), index=False)

    metrics_df = pd.DataFrame(
        [
            {
                "Compound": args.c,
                "Accuracy": metrics.accuracy_score(test_labels, predictions),
                "Precision": safe_metric(metrics.precision_score, test_labels, predictions),
                "Recall": safe_metric(metrics.recall_score, test_labels, predictions),
                "F1": safe_metric(metrics.f1_score, test_labels, predictions),
                "ROC_AUC": metrics.auc(fpr, tpr),
                "BestParams": json.dumps(best_params),
            }
        ]
    )
    metrics_df.to_csv(os.path.join(result_output_dir, f"{args.c}_metrics.csv"), index=False)

    dump(best_rf_model, os.path.join(model_output_dir, f"{args.c}_random_forest_model.joblib"))
    with open(os.path.join(vocab_output_dir, f"{args.c}_w2i.json"), "w", encoding="utf-8") as f:
        json.dump(w2i, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
