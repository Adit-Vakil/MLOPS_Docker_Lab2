import json
from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_model(output_dir=Path(".")):
    dataset = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ])
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    metrics = {
        "dataset": "sklearn.datasets.load_wine",
        "model": "StandardScaler + RBF SVC",
        "random_state": 42,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
    }
    artifact = {
        "pipeline": pipeline,
        "feature_names": list(dataset.feature_names),
        "class_labels": list(dataset.target_names),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_dir / "wine_model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    return artifact, metrics


if __name__ == "__main__":
    train_model()
