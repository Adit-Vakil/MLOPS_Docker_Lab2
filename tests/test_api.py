import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import joblib
import numpy as np
from sklearn.datasets import load_wine

from src.main import create_app
from src.model_training import train_model


class ApiTests(unittest.TestCase):
    def setUp(self):
        dataset = load_wine()
        self.features = list(dataset.feature_names)
        self.labels = list(dataset.target_names)
        self.sample = dataset.data[0]
        self.payload = dict(zip(self.features, self.sample.tolist()))
        self.pipeline = Mock()
        self.pipeline.predict.return_value = np.array([0])
        self.client = create_app({
            "pipeline": self.pipeline,
            "feature_names": self.features,
            "class_labels": self.labels,
        }).test_client()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"status": "ok"})

    def test_schema(self):
        response = self.client.get("/schema")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["feature_names"], self.features)
        self.assertEqual(response.json["class_labels"], self.labels)

    def test_form_contains_all_features(self):
        response = self.client.get("/predict")
        self.assertEqual(response.status_code, 200)
        for feature in self.features:
            self.assertIn(f'name="{feature}"'.encode(), response.data)

    def test_json_prediction_preserves_training_order(self):
        payload = dict(reversed(list(self.payload.items())))
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"predicted_class": self.labels[0]})
        self.pipeline.predict.assert_called_once()
        inputs = self.pipeline.predict.call_args.args[0]
        np.testing.assert_array_equal(inputs, self.sample.reshape(1, -1))

    def test_form_prediction(self):
        response = self.client.post("/predict", data=self.payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["predicted_class"], self.labels[0])

    def test_missing_feature(self):
        del self.payload[self.features[0]]
        response = self.client.post("/predict", json=self.payload)
        self.assertEqual(response.status_code, 400)
        self.pipeline.predict.assert_not_called()

    def test_unknown_feature(self):
        self.payload["unexpected"] = 1
        response = self.client.post("/predict", json=self.payload)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["features"], ["unexpected"])
        self.pipeline.predict.assert_not_called()

    def test_invalid_measurements(self):
        for value in (None, True, "abc", "NaN", "inf", 0, -1):
            with self.subTest(value=value):
                payload = dict(self.payload, alcohol=value)
                response = self.client.post("/predict", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn("alcohol", response.json["error"])
        self.pipeline.predict.assert_not_called()

    def test_invalid_json(self):
        for body in ("[]", "{", "null"):
            with self.subTest(body=body):
                response = self.client.post(
                    "/predict", data=body, content_type="application/json"
                )
                self.assertEqual(response.status_code, 400)
        self.pipeline.predict.assert_not_called()

    def test_inference_failure(self):
        self.pipeline.predict.side_effect = ValueError("Inference failed")
        response = self.client.post("/predict", json=self.payload)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json, {"error": "Prediction could not be completed"}
        )


class SavedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.output = Path(cls.directory.name)
        cls.artifact, cls.metrics = train_model(cls.output)
        cls.saved = joblib.load(cls.output / "wine_model.joblib")

    def test_saved_pipeline_preserves_predictions(self):
        samples = load_wine().data
        np.testing.assert_array_equal(
            self.artifact["pipeline"].predict(samples),
            self.saved["pipeline"].predict(samples),
        )

    def test_scaler_fits_only_training_samples(self):
        scaler = self.saved["pipeline"].named_steps["scaler"]
        self.assertEqual(
            int(scaler.n_samples_seen_), self.metrics["train_samples"]
        )
        self.assertEqual(
            self.metrics["train_samples"] + self.metrics["test_samples"], 178
        )

    def test_evaluation_file_matches_returned_metrics(self):
        import json

        recorded = json.loads(
            (self.output / "metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(recorded, self.metrics)
        self.assertTrue(0 <= recorded["accuracy"] <= 1)
        self.assertTrue(0 <= recorded["macro_f1"] <= 1)

    def test_saved_model_serves_a_raw_sample(self):
        dataset = load_wine()
        payload = dict(zip(dataset.feature_names, dataset.data[0].tolist()))
        response = create_app(self.saved).test_client().post(
            "/predict", json=payload
        )
        expected_index = int(
            self.saved["pipeline"].predict(dataset.data[:1])[0]
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json["predicted_class"],
            self.saved["class_labels"][expected_index],
        )


if __name__ == "__main__":
    unittest.main()
