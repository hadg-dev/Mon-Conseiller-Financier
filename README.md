# Mon Conseiller Financier

## Installation

```commandline
uv venv --python 3.13.2
source .venv/bin/activate
uv pip install -r requirements.txt
```

Copy training dataset locally :
```commandline
uv run ./src/data/fetch_dataset.py
```

Execute training locally :
```commandline
uv run ./src/train/train.py
```

## Structure 

- **.github/workflows/** contains Github Actions
- **airflow/** contains Airflow Docker compose and DAG
- **configs/**
- **dataset/**
- **doc/**
- **docker/** Docker file for training image
- **src/** Python scripts
  - **src/airflow/trigger_airflow.py** triggered by github action to send image hash to Airflow  
  - **src/data/fetch_dataset.py**
  - **src/train/train.py** Training script
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

## Industrialization pipeline

![Schema](M06-MLOps.png)
