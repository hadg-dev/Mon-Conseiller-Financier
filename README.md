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

## Industrialization pipeline

![Schema](M06-MLOps.png)
