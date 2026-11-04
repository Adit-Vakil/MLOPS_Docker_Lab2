# Docker Lab 2 — Iris Classifier

This lab trains a TensorFlow model on the Iris dataset and serves predictions through Flask. It supports two Docker workflows. The standalone image trains the model during the build and packages it with the app. Docker Compose runs training as a separate service and passes the saved model to Flask through a shared volume.

The starting point was the [instructor's Docker Lab 2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Docker_Labs/Lab2).

## Prerequisites

- Docker Desktop with Docker Compose enabled and the engine running.
- A terminal with `curl` for the request examples.
- Port 4000 available on the host. Run one workflow at a time because both use this port.

Run the commands below from the root of the cloned repository.

## Standalone Docker image

```bash
docker build -f dockerfile -t docker-lab2 .
docker run --detach --rm --name docker-lab2-check \
  --publish 127.0.0.1:4000:4000 docker-lab2
```

Check readiness with `docker inspect --format '{{.State.Health.Status}}' docker-lab2-check`. Once it reports `healthy`, try the prediction example below.

Stop the container with `docker stop docker-lab2-check`. Docker removes it automatically.

## Docker Compose

```bash
docker compose up --build --wait --wait-timeout 180
```

The training service writes `my_model.keras` to the shared volume and exits. The serving service starts after successful training and mounts the volume read-only. View the evaluation metrics with `docker compose logs --no-color model-training`.

To remove the containers, network, and generated model volume, run `docker compose down --volumes`.

## Make a prediction

Open [the prediction form](http://127.0.0.1:4000/predict) in a browser, or send the four measurements in centimeters as JSON:

```bash
curl --fail-with-body --silent --show-error \
  --json '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}' \
  http://127.0.0.1:4000/predict
```

Observed response:

```json
{"predicted_class":"Setosa"}
```

Missing or invalid measurements return HTTP 400 with a JSON error. `GET /health` returns `{"status":"ok"}` after the app starts.

## Automated tests

After building the standalone image, run:

```bash
docker run --rm --mount "type=bind,source=$PWD,target=/lab,readonly" \
  --workdir /lab docker-lab2 python -B -m unittest discover -s tests -v
```

The tests use a stand-in model to check the browser form, form and JSON submissions, feature order and type, the health endpoint, and invalid or missing inputs. The container prediction example above exercises the real model.

## What changed from the reference

| File or directory | Status | Purpose |
| --- | --- | --- |
| `src/model_training.py` | Modified | I moved normalization into the saved Keras model so training and prediction use the same transformation. I fit normalization on the training split, used a seeded stratified split, and reported holdout metrics after training. |
| `src/main.py` | Modified | I kept the browser form, added JSON input and a health endpoint, and made invalid requests return HTTP 400. The app factory lets the tests exercise Flask with a stand-in model. |
| `dockerfile` | Modified | Training and serving reuse a common dependency stage. The serving image includes the trained model, declares the Flask port, and checks app readiness. |
| `docker-compose.yml` | Modified | Both services build from the shared dependency stage. The trainer writes the model to a named volume, and Flask reads it after training succeeds. |
| `requirements.txt` | Modified | Pins the four direct Python dependencies to the versions used in the Docker build. |
| `tests/test_api.py` | Added | Verifies API behavior and checks that rejected requests never reach inference. |
| `.gitignore`, `.dockerignore` | Added | Exclude local environments, caches, and generated models from Git and the build context. |
| `README.md` | Added | Documents setup, execution, testing, and verified results. |
| `src/templates/`, `src/statics/` | Retained | Preserve the instructor's browser interface and static assets. |
| `HOWTO` | Retained | Preserves the original reference notes. Use the commands in this README for this version. |

## Verified results

| Check | Observed result |
| --- | --- |
| Docker build on Linux ARM64 via Docker Desktop | Clean build succeeded with `--no-cache`. |
| Flask test suite inside Docker | 7 tests passed. |
| Saved model check | The model contained the fitted normalization layer and returned three finite probabilities for a raw Iris sample. |
| Standalone container | Docker reported `healthy`, and the sample returned `Setosa`. |
| Compose workflow | Training completed, Flask became healthy, and the sample returned `Setosa`. |
| Compose holdout evaluation | Loss `0.5056`, accuracy `0.7667` on the 30-sample holdout. |
| Compose cleanup | Both containers, the network, and the generated model volume were removed. |

## Limitations

The holdout accuracy comes from one small split and can vary across environments. The prediction example checks one sample. More evaluation across splits would give a better picture of model performance. The direct dependencies are pinned, while the Python image tag and transitive dependencies can change. Flask uses its development server for this local lab.
