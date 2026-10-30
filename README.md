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

Running the training script :
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
  - **src/dataset/fetch_dataset.py** fetching dataset function
  - **src/training/trainer.py** training function
  - **src/validation/judge.py** score model using LLM-as-judge
  - **src/validation/perplexity.py** score model perplexity
  - **src/main.py** main entrypoint
- **tests/** 

## Environment Variables

|   Variable name                                       |       Description             |
|-------------------------------------------------------|-------------------------------|
| AWS_ACCESS_KEY_ID=your_access_key_id                  |   AWS access key              |
| AWS_SECRET_ACCESS_KEY=your_secret_access_key          | AWS secret access key         |
| ARTIFACT_STORE_URI=s3://mlflow-artifacts-demo-1234    | S3 bucket url                 |
| BACKEND_STORE_URI=postgresql://user:password@hostname/dbname  | Postgres backend url  |
| PORT=7860                                             | MLFlow Hugging Face app port  |  
| MLFLOW_TRACKING_URI                                   | MLFlow tracking               |
| MLFLOW_EXPERIMENT_ID                                  | MLFlow experiment ID          |
| AIRFLOW_URL                                           | Airflow URL                   |
| AIRFLOW_USER                                          | Airflow user                  |
| AIRFLOW_PASS                                          | Airflow password              |
| DAG_ID                                                | Airflow DAG ID                |
| DOCKER_USERNAME                                       | Docker Hub username           |
| DOCKER_PASSWORD                                       | Docker Hub password           |
| PERSONAL_ACCESS_TOKEN                                 | Github access token           |     
| OPENAI_API_KEY                                        | OpenAi key for validation     |
| STEPS | training script steps to run; default 'dataset,train,score_perplexity,score_with_llm'     |

Training script overrides :

|   Variable name          |       Description                   |  Default |
|--------------------------|-------------------------------------|----------|
| STEPS                    | training steps to run               | dataset,train,score_perplexity,score_with_llm     |
| DEBUG                    | Provides additional logging         | True     |
| REPORT_TRAINING          | "mlflow", "none", "all"             | none     |
| NB_EPOCHS                | Number of epochs to train           | 4        |
| MODEL_NAME               | Training model                      | "Qwen/Qwen3.5-4B" |
| LEARNING_RATE            | Traning learning rate               | 2e-4     |
| REPO_ID                  | HF Training dataset                 | "likhitjuttada/finance-reasoning-sft-dataset" |

## Industrialization pipeline

![Schema](M06-MLOps.png)
