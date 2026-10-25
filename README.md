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

```commandline
uv run ./src/main.py
```

## Structure 

- **.github/workflows/** contains Github Actions
- **airflow/** contains Airflow Docker compose and DAG
- **configs/**
  - **configs/training_conf.yaml** default configuration used by Python training script; can be overriden by env variables 
- **dataset/**
- **doc/**
- **docker/** Docker file for training image
- **src/** Python scripts
  - **src/airflow/trigger_airflow.py** triggered by github action to send image hash to Airflow  
  - **src/dataset/fetch_dataset.py** fetching dataset function
  - **src/training/trainer.py** training function
  - **src/validation/model_validation.py** gateway to validation functions 
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


## Industrialization pipeline

![Schema](M06-MLOps.png)
