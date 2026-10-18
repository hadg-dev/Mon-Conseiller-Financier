# Mon Conseiller Financier

## Installation

```commandline
uv venv --python 3.13.2
source .venv/bin/activate
uv pip install -r requirements.txt
```
Copy training dataset locally :

```commandline
uv run ./scripts/load_dataset.py
```

## Structure

```
./
├── .github/workflows/
│   └── cml.yaml
├── app/
│   └── train.py
├── dataset/
├── doc/
│   └── continuous_training.txt
│   └── projet_v2.md
├── scripts/
│   └── load_dataset.py
│   └── trigger_airflow.py
├── Dockerfile
├── README.md
├── requirements.txt
└── requirements-rocm.txt
```

## Industrialization pipeline

![Schema](M06-MLOps.png)
