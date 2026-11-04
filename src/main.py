import os
from math import isfinite
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request


def create_app(artifact):
    app = Flask(__name__, static_folder="statics")
    pipeline = artifact["pipeline"]
    feature_names = artifact["feature_names"]
    class_labels = artifact["class_labels"]

    @app.get("/")
    def home():
        return jsonify(
            application="Wine cultivar classifier",
            prediction_form="/predict",
            feature_schema="/schema",
        )

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.get("/schema")
    def schema():
        return jsonify(
            feature_names=feature_names,
            class_labels=class_labels,
            input_format="JSON object or form with all named features",
        )

    @app.get("/predict")
    def prediction_form():
        return render_template("predict.html", feature_names=feature_names)

    @app.post("/predict")
    def predict():
        data = request.get_json(silent=True) if request.is_json else request.form
        if request.is_json and not isinstance(data, dict):
            return jsonify(error="Request body must be a JSON object"), 400

        unknown = sorted(set(data) - set(feature_names))
        if unknown:
            return jsonify(error="Unknown features", features=unknown), 400

        measurements = []
        for feature in feature_names:
            value = data.get(feature)
            try:
                if isinstance(value, bool):
                    raise ValueError
                measurement = float(value)
                if not isfinite(measurement) or measurement <= 0:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                return jsonify(error=f"{feature} must be a positive finite number"), 400
            measurements.append(measurement)

        inputs = np.asarray([measurements], dtype=np.float64)
        try:
            predicted_index = int(pipeline.predict(inputs)[0])
        except (ValueError, OverflowError):
            app.logger.exception("Prediction failed")
            return jsonify(error="Prediction could not be completed"), 500

        return jsonify(predicted_class=class_labels[predicted_index])

    return app


if __name__ == "__main__":
    model_path = Path(os.environ.get("MODEL_PATH", "wine_model.joblib"))
    artifact = joblib.load(model_path)
    create_app(artifact).run(host="0.0.0.0", port=4000)
