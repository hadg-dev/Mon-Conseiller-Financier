# Mon Conseiller Financier

## Installation

```commandline
uv venv --python 3.13.2
source .venv/bin/activate
uv pip install -r requirements.txt
```
## Installation

Set env variable `STEPS` to a comma-separated list of required steps to execute :
- `dataset` : fetch dataset from HF
- `train` : train model
- `score_perplexity` : score perplexity of trained model
- `score_with_llm` : score trained model using LLM-as-judge
- `copy_s3` : copy trained model to S3

Running the training script locally :
```commandline
uv run ./src/main.py
```

Running tests :
```commandline
uv run pytest
```

## Structure 

- **.github/workflows/** contains Github Actions
- **airflow/** contains Airflow Docker compose and DAG
- **configs/**
  - **configs/training_conf.yaml** default configuration used by Python training script; can be overriden by env variables 
- **dataset/**
- **doc/**
- **docker/** Docker file for training image
- **scripts/trigger_airflow.py** triggered by github action to send image hash to Airflow
- **src/** Python project
  - **src/aws/upload_adapter.py** copy lora adapter to S3 
  - **src/dataset/fetch_dataset.py** fetching dataset function
  - **src/training/trainer.py** training function
  - **src/validation/judge.py** score model using LLM-as-judge
  - **src/validation/perplexity.py** score model perplexity
  - **src/main.py** main entrypoint
- **tests/** 

## Configuration of the industrialization pipeline

* [Airflow.md](doc/Airflow.md) : setting up and configuring the local Airflow instance
* [Github_Action.md](doc/Github_Action.md) : env variables for Github Actions
* [Training_script_overrides.md](doc/Training_script_overrides.md) : env variables overrides for the training script
* [Misc.md](doc/Misc.md) : misc env variables

## Running the full pipeline

### Launching it

Prerequisites, described in the configuration docs above :
- `airflow/.env` is filled in and the EC2 private key is saved as `airflow/airflow-training-key.pem`
- Github secrets are set : `DOCKER_USERNAME`, `DOCKER_PASSWORD`, `AIRFLOW_URL`, `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD`
- `AIRFLOW_URL` is the public ngrok URL, without a trailing `/` (it is listed in the [ngrok dashboard](https://dashboard.ngrok.com/endpoints))

Start Airflow and ngrok on your machine (`airflow-init` is only needed the first time) :
```commandline
cd airflow
docker compose up airflow-init
docker compose up -d
```

Start the workflow on your branch, and follow it :
```commandline
gh workflow run build_and_publish.yaml --ref <branch> -f trigger_dag=true
gh run watch
```

The Github run ends once the DAG is triggered. Training continues in the Airflow UI (http://localhost:8080), so keep your machine awake until the DAG finishes : Airflow is what terminates the EC2 instance.

When you are done, close the public access to Airflow :
```commandline
docker compose stop ngrok
```

### How it works

1. **Github Actions** builds the training image, runs the tests inside it, and pushes it to Docker Hub tagged with the commit hash and `latest`.
2. **`scripts/trigger_airflow.py`** logs in to Airflow through the ngrok tunnel and starts the DAG `github_ec2_ml_training`, passing the commit hash.
3. **Airflow** checks the latest CI run on `BRANCH_NAME`, creates an EC2 instance and waits for it to be ready.
4. **Airflow connects to the instance over SSH**, pulls the image for that commit hash and runs the container with the training configuration.
5. **The container** downloads the requested dataset version with DVC, trains the model, then runs the steps listed in `STEPS`. It sends metrics to the MLflow server (`MLFLOW_TRACKING_URI`) and copies the adapter to S3 (`ARTIFACT_STORE_URI`).
6. **MLflow** stores run data in its Postgres database (Neon) and artifacts in S3.
7. **Airflow terminates the instance**, even if training failed.

By default the DAG only tests the pipeline on a small CPU instance : it trains a tiny model (`trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`) for 5 steps, and its scores are not meaningful. For real training, set `MODEL_NAME` (e.g. `Qwen/Qwen3.5-4B`) and `MAX_STEPS=-1` in `airflow/.env`, and set `INSTANCE_TYPE` and `AMI_ID` to a GPU instance.

## Dataset versions

The training set is versioned with [DVC](https://dvc.org). Git keeps a small pointer file (`dataset/combined_train.jsonl.dvc`) with the hash of the data, and S3 keeps the data itself. A `data-v*` git tag marks each version. The validation set stays in Git, so scores stay comparable between versions.

Get the training set locally :
```commandline
dvc pull
```

### Retraining on new data

Update the file, then publish a new version. Pushing the tag retrains the model, without building the image :
```commandline
dvc add dataset/combined_train.jsonl
git add dataset/combined_train.jsonl.dvc
git commit -m "Add new training data"
git tag data-v3
dvc push
git push
git push origin data-v3
```

Run `dvc push` before pushing the tag, otherwise the training can't find the data. Airflow and ngrok must be running.

| Event | Image build | Image used | Dataset used |
|---|---|---|---|
| `gh workflow run` (code change) | Yes | That commit's image | Newest `data-v*` tag |
| `git push origin data-vN` (new data) | No | `latest` | That tag |

Each MLflow run records the base model, the dataset version, its hash and row count, and the S3 path of the adapter.

## Chatting with the trained model

`src/utils/gradio_chat.py` is a chat UI that loads the base model and a trained adapter. It runs on CPU or GPU.

Find the latest training folder on S3 and download its adapter :
```commandline
aws s3 ls s3://<bucket>/
aws s3 sync s3://<bucket>/<timestamp> output/adapter
```

Start the UI, then open http://localhost:7860 :
```commandline
docker run -d --rm --name gradio-chat \
  -p 127.0.0.1:7860:7860 \
  -v "$PWD/output/adapter:/app/output/adapter:ro" \
  <docker-id>/conseiller-financier-training-app:latest \
  python src/utils/gradio_chat.py
```

Stop it :
```commandline
docker stop gradio-chat
```

`MODEL_NAME` must be the model the adapter was trained on. It defaults to the tiny test model, whose answers are gibberish; a real adapter needs `-e MODEL_NAME=Qwen/Qwen3.5-4B` and a GPU.

## Architecture

![Schema](M06-MLOps.png)
