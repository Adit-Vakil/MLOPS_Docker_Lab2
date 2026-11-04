import unittest
from unittest.mock import Mock

import numpy as np

from src.main import create_app


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock(return_value=np.array([[0.9, 0.08, 0.02]]))
        self.client = create_app(self.model).test_client()
        self.measurements = {
            "petal_width": 0.2,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "sepal_length": 5.1,
        }

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_prediction_form_renders(self):
        response = self.client.get("/predict")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"sepal_length", response.data)

    def test_json_prediction_preserves_feature_order(self):
        response = self.client.post("/predict", json=self.measurements)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"predicted_class": "Setosa"})
        self.model.assert_called_once()
        args, kwargs = self.model.call_args
        expected = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
        np.testing.assert_array_equal(args[0], expected)
        self.assertEqual(args[0].dtype, np.float32)
        self.assertEqual(kwargs, {"training": False})

    def test_form_prediction(self):
        response = self.client.post("/predict", data=self.measurements)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"predicted_class": "Setosa"})

    def test_invalid_measurements_are_rejected_before_inference(self):
        for value in (True, None, "abc", "NaN", "inf", "1e100", 0, -1):
            with self.subTest(value=value):
                response = self.client.post(
                    "/predict", json=dict(self.measurements, sepal_length=value)
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("sepal_length", response.get_json()["error"])
        self.model.assert_not_called()

    def test_missing_field_is_rejected_before_inference(self):
        payload = self.measurements.copy()
        del payload["petal_width"]
        response = self.client.post("/predict", data=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("petal_width", response.get_json()["error"])
        self.model.assert_not_called()

    def test_invalid_json_body_is_rejected(self):
        for body in ("[]", "{"):
            with self.subTest(body=body):
                response = self.client.post(
                    "/predict", data=body, content_type="application/json"
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(
                    response.get_json(), {"error": "Request body must be a JSON object"}
                )
        self.model.assert_not_called()
