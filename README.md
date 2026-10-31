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
5. **The container** trains the model, then runs the steps listed in `STEPS`. It sends metrics to the MLflow server (`MLFLOW_TRACKING_URI`) and copies the adapter to S3 (`ARTIFACT_STORE_URI`).
6. **MLflow** stores run data in its Postgres database (Neon) and artifacts in S3.
7. **Airflow terminates the instance**, even if training failed.

By default the DAG only tests the pipeline on a small CPU instance : it trains a tiny model (`trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`) for 5 steps, and its scores are not meaningful. For real training, set `MODEL_NAME` (e.g. `Qwen/Qwen3.5-4B`) and `MAX_STEPS=-1` in `airflow/.env`, and set `INSTANCE_TYPE` and `AMI_ID` to a GPU instance.

## Architecture

![Schema](M06-MLOps.png)
