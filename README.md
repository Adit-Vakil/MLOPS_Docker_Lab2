# Docker Lab 2 — Wine Cultivar Classifier

This project trains a support vector classifier on scikit-learn's Wine dataset and serves predictions through Flask. I used a different dataset and model family from the [instructor's Docker Lab 2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Docker_Labs/Lab2), while preserving its Docker training and serving workflows.

## Dataset and model

The Wine dataset contains 178 samples, 13 chemical measurements, and three cultivar classes. It is bundled with scikit-learn, so training requires no separate dataset download. The labels are `class_0`, `class_1`, and `class_2`.

I split the data into 142 training samples and 36 holdout samples using a stratified split with `random_state=42`. A pipeline applies `StandardScaler` followed by an RBF support vector classifier. Scaling matters because the measurements have different numerical ranges.

The scaler is fitted only on the training data. Saving the complete pipeline keeps preprocessing consistent during prediction. The artifact also includes the feature names and class labels. Holdout data is used for evaluation after fitting.

## Project structure

- `src/model_training.py` trains the pipeline and writes the model and metrics.
- `src/main.py` loads the artifact and creates the Flask app.
- `src/templates/predict.html` provides the Wine measurement form.
- `tests/test_api.py` tests request handling and the saved pipeline.
- `dockerfile` builds the standalone training and serving image.
- `docker-compose.yml` runs training and serving as separate services.
- `requirements.txt` pins direct dependencies.
- `.gitignore` and `.dockerignore` exclude local and generated files.

## Prerequisites

Docker Desktop must be running, with Docker Compose available. Port 4000 must be free. The request example uses `curl`.

Run commands from the repository root. Use one workflow at a time because both publish port 4000.

## Standalone image

```bash
docker build -f dockerfile -t docker-lab2 .
docker run --detach --rm --name docker-lab2-check \
  --publish 127.0.0.1:4000:4000 docker-lab2
docker inspect --format '{{.State.Health.Status}}' docker-lab2-check
```

The image trains during the build and includes `wine_model.joblib` and `metrics.json`. Wait for `healthy` before making requests.

To inspect the build's evaluation metrics:

```bash
docker run --rm docker-lab2 cat /app/metrics.json
```

To stop and remove the running container:

```bash
docker stop docker-lab2-check
```

## Docker Compose

```bash
docker compose up --build --wait --wait-timeout 180
```

The training service writes the model and metrics into a named volume and exits. Flask starts after training succeeds and reads the volume through a read-only mount. Both services use the Dockerfile's dependency stage.

View training results:

```bash
docker compose logs --no-color model-training
```

Remove the lab containers, network, and generated volume:

```bash
docker compose down --volumes
```

## Use the app

Open [the prediction form](http://127.0.0.1:4000/predict) and enter all 13 chemical measurements. The form submits measurements and displays the predicted cultivar.

`GET /schema` lists the required feature names and class labels. `GET /health` returns `{"status":"ok"}` after the app has loaded its model and started.

Example JSON request using the first Wine dataset sample:

```bash
curl --fail-with-body --silent --show-error \
  --json '{"alcohol":14.23,"malic_acid":1.71,"ash":2.43,"alcalinity_of_ash":15.6,"magnesium":127,"total_phenols":2.8,"flavanoids":3.06,"nonflavanoid_phenols":0.28,"proanthocyanins":2.29,"color_intensity":5.64,"hue":1.04,"od280/od315_of_diluted_wines":3.92,"proline":1065}' \
  http://127.0.0.1:4000/predict
```

Observed response from both Docker workflows:

```json
{"predicted_class":"class_0"}
```

Inputs must be positive finite numbers. Missing features, unknown features, invalid measurements, and malformed JSON return HTTP 400. These checks validate request structure and numerical values. They do not establish whether measurements are scientifically plausible.

## Automated tests

Build the standalone image, then run:

```bash
docker run --rm --mount "type=bind,source=$PWD,target=/lab,readonly" \
  --workdir /lab docker-lab2 python -B -m unittest discover -s tests -v
```

The API tests cover the schema, health endpoint, rendered form, JSON and form predictions, feature order, invalid inputs, and inference failures. Saved-pipeline tests train in a temporary directory, reload the artifact, compare predictions, check the scaler's training sample count, verify the metrics file, and exercise Flask with the real pipeline.

## Verified results

| Check | Result |
| --- | --- |
| Clean Wine image build with `--no-cache` | Passed |
| Automated tests in Docker | 14 passed |
| Standalone container health | Healthy |
| Standalone sample prediction | `class_0` |
| Compose training and serving | Trainer completed and serving became healthy |
| Compose sample prediction | `class_0` |
| Compose cleanup | Both containers, network, and model volume removed |

The automated training run produced the following results on the 36-sample holdout:

| Metric | Value |
| --- | --- |
| Accuracy | 0.972222 |
| Macro F1 | 0.970962 |

The request example checks connectivity and inference on one dataset sample. Holdout evaluation provides the separate performance measurement.

## Differences from the instructor's reference

| Component | Reference | This implementation |
| --- | --- | --- |
| Dataset | Iris, four measurements | Wine, 13 chemical measurements |
| Model | TensorFlow dense neural network | RBF support vector classifier |
| Preprocessing | Training scaler separate from served model | Saved scaler and classifier pipeline |
| Artifact | Keras model | Joblib pipeline with feature names and class labels |
| Evaluation | Test split used during neural-network training | Stratified holdout with accuracy and macro F1 recorded to JSON |
| Flask input | Iris form measurements | Wine form and JSON with validation and a schema endpoint |
| Browser interface | Flower classification form and images | Wine measurement form with prediction and error display |
| Docker dependencies | TensorFlow stack | Pinned scikit-learn, Flask, NumPy, and joblib |
| Readiness | No application health check | Health checks for the standalone image and Compose |
| Verification | No test suite in the supplied Lab 2 files | 14 automated API and saved-pipeline tests |

I modified both Python modules, the HTML form, Dockerfile, Compose configuration, and requirements. I added the tests, ignore files, and README. The two-service training-to-serving handoff remains part of the assignment workflow.

## Limitations

Results come from one small holdout split and are not evidence of performance on new laboratory data. Verification was performed using Docker Desktop on ARM64. Browser form rendering is tested, but JavaScript interactions have not been tested with browser automation.

Direct dependencies are pinned. Transitive dependencies and the `python:3.10-slim` tag can change. Flask uses its development server for this local lab. Joblib artifacts must come from a trusted source because loading them can execute code.
