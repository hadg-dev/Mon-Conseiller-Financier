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
* [DVC-Guide.md](doc/DVC-Guide.md) : versioning the training set with DVC

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
2. **`scripts/trigger_airflow.py`** logs in to Airflow through the ngrok tunnel and starts the DAG `github_ec2_ml_training`, passing the image tag and the dataset version.
3. **Airflow** checks the latest CI run on `BRANCH_NAME`, creates an EC2 instance and waits for it to be ready.
4. **Airflow connects to the instance over SSH**, pulls that image and runs the container with the training configuration.
5. **The container** downloads the requested dataset version with DVC, trains the model, then runs the steps listed in `STEPS`. It sends metrics to the MLflow server (`MLFLOW_TRACKING_URI`) and copies the adapter to S3 (`ARTIFACT_STORE_URI`).
6. **MLflow** stores run data in its Postgres database (Neon) and artifacts in S3.
7. **Airflow terminates the instance**, even if training failed.

By default the DAG only tests the pipeline on a small CPU instance : it trains a tiny model (`trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`) for 5 steps, and its scores are not meaningful. For real training, set `MODEL_NAME` (e.g. `Qwen/Qwen3.5-4B`) and `MAX_STEPS=-1` in `airflow/.env`, and set `INSTANCE_TYPE` and `AMI_ID` to a GPU instance.

## Dataset versions

The training set is versioned with [DVC](https://dvc.org). Git keeps a small pointer file (`dataset/combined_train.jsonl.dvc`) with the hash of the data, and S3 keeps the data itself. A `data-v*` git tag marks each version. The validation set stays in Git, so scores stay comparable between versions.

For how DVC works and its commands, read the [DVC guide](doc/DVC-Guide.md). After cloning the repo, run `dvc pull` to get the training set.

### Where the data is

| Location | Content | Role |
|---|---|---|
| Hugging Face (`likhitjuttada/finance-reasoning-sft-dataset`) | Original training and validation sets | Source of D0. The pipeline doesn't read it : the optional `dataset` step is not in the DAG's `STEPS` |
| Git | `combined_val.jsonl`, `additions.jsonl`, `combined_train.jsonl.dvc` | Validation set, test rows, and the pointer to each training set version |
| S3 (`s3://<bucket>/dvc/files/md5/...`) | One full copy of each training set version, named by its hash | Reference storage, read by the training on EC2 |
| Your machine (`dataset/combined_train.jsonl`) | The version you last pulled, ignored by Git | To create new versions and run locally. `.dvc/cache/` keeps a copy of each version you used |
| EC2 container (`/app/dataset/`) | The training set of the requested version, and the validation set from the image | Deleted with the container after each run |

### How the data reaches the model

1. The DAG receives `data_version` and passes it to the container as `DATA_VERSION`.
2. The `dataset_version` step runs `dvc get` : it reads the pointer file at that git tag on GitHub, then downloads the matching file from S3 to `/app/dataset/combined_train.jsonl`.
3. It records the version, md5 hash and row count in MLflow.
4. The `train` step loads the file with pandas, and `load_and_format_dataset` turns each row into one text : a fixed system prompt, then the user question and the assistant answer, formatted with the model's chat template.
5. `SFTTrainer` tokenizes these texts, truncates them to 1536 tokens, and feeds them to the model in batches to train the LoRA adapter.
6. The `score_perplexity` step scores the adapter on `combined_val.jsonl`, which comes from the image.

## Updating the model

Airflow and ngrok must be running in both cases, and `AIRFLOW_URL` must match the current ngrok URL.

| | Code change | Dataset change |
|---|---|---|
| **What starts it** | `gh workflow run` | `git push origin data-vN` |
| **Image build and tests** | Yes | No |
| **Image used** | The one built for that commit | `latest`, the last image built |
| **Dataset used** | The newest `data-v*` tag | The pushed tag |

Each MLflow run records the base model, the dataset version, its hash and row count, and the S3 path of the adapter.

### 1. When you push a code change

A `git push` alone starts nothing : the build is started by hand, and needs at least one `data-v*` tag to train on.

```commandline
# Commit and push the code; this starts nothing
git add <files>
git commit -m "Describe the change"
git push

# Build the image, run the tests, then start the training
gh workflow run build_and_publish.yaml --ref <branch> -f trigger_dag=true

# Follow the Github run
gh run watch
```

Steps :
1. **Github Actions** builds the image, runs the tests inside it, and pushes it to Docker Hub. If the tests fail, the pipeline stops here.
2. **The workflow** finds the newest `data-v*` tag, and `scripts/trigger_airflow.py` starts the DAG through ngrok with the commit hash and that dataset version.
3. **Airflow** creates an EC2 instance, which pulls the image for that commit.
4. **The container** downloads the dataset version with DVC, trains the model, scores it, and copies the adapter to S3.
5. **Airflow** terminates the instance.

To build the image without training, use `-f trigger_dag=false`.

### 2. When you push a dataset change

Update `dataset/combined_train.jsonl`, then publish a new version. DVC needs the AWS keys loaded (see the [DVC guide](doc/DVC-Guide.md#installation)).

```commandline
# Check the number of rows in the updated file
wc -l dataset/combined_train.jsonl

# Take the snapshot and update the pointer file with the new hash
dvc add dataset/combined_train.jsonl

# Commit the pointer file; the data itself doesn't go in Git
git add dataset/combined_train.jsonl.dvc
git commit -m "Describe the new data"

# Name this version
git tag data-vN

# Upload the data to S3, before pushing the tag
dvc push

# Push the commit; this starts nothing
git push

# Push the tag; this starts the training
git push origin data-vN

# Check the Github run started
gh run list --limit 2
```

Replace `data-vN` with the next version, e.g. `data-v3`. Run `dvc push` before pushing the tag, otherwise the training can't find the data.

The test versions were created this way, from the original 651 rows in `data-v0` :

| Version | Rows | Update before `dvc add` |
|---|---|---|
| `data-v0` | 651 | None |
| `data-v1` | 661 | `sed -n '1,10p' dataset/additions.jsonl >> dataset/combined_train.jsonl` |
| `data-v2` | 671 | `sed -n '11,20p' dataset/additions.jsonl >> dataset/combined_train.jsonl` |

Steps :
1. **Pushing the tag** starts the Github workflow, which skips the image build.
2. **`scripts/trigger_airflow.py`** starts the DAG through ngrok with `latest` as the image and the tag as the dataset version.
3. **Airflow** creates an EC2 instance, which pulls the `latest` image.
4. **The container** reads the pointer file at that tag on GitHub, downloads the matching file from S3, trains the model, scores it, and copies the adapter to S3.
5. **Airflow** terminates the instance.

In both cases, the Github run ends once the DAG is triggered. Follow the training in the Airflow UI (http://localhost:8080), and check that `run_training_via_ssh` is green : the DAG run shows success even when training fails, because its last task is the instance termination.

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
