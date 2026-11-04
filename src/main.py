from math import isfinite
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

FEATURES = ("sepal_length", "sepal_width", "petal_length", "petal_width")
CLASS_LABELS = ("Setosa", "Versicolor", "Virginica")
MAX_MEASUREMENT = float(np.finfo(np.float32).max)


def create_app(model):
    app = Flask(__name__, static_folder="statics")

    @app.get("/")
    def home():
        return "Welcome to the Iris Classifier API!"

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.get("/predict")
    def prediction_form():
        return render_template("predict.html")

    @app.post("/predict")
    def predict():
        data = request.get_json(silent=True) if request.is_json else request.form
        if request.is_json and not isinstance(data, dict):
            return jsonify(error="Request body must be a JSON object"), 400

        measurements = []
        for feature in FEATURES:
            value = data.get(feature)
            try:
                if isinstance(value, bool):
                    raise ValueError
                measurement = float(value)
                if not isfinite(measurement) or not 0 < measurement <= MAX_MEASUREMENT:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                return jsonify(error=f"{feature} must be a positive finite number"), 400
            measurements.append(measurement)

        inputs = np.asarray([measurements], dtype=np.float32)
        scores = model(inputs, training=False)
        predicted_class = CLASS_LABELS[int(np.argmax(scores[0]))]
        return jsonify(predicted_class=predicted_class)

    return app


if __name__ == "__main__":
    from tensorflow.keras.models import load_model

    model = load_model(Path("my_model.keras"), compile=False)
    create_app(model).run(host="0.0.0.0", port=4000)
