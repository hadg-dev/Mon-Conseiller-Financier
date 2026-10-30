## Github Actions

### Repository variables


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


TODO à check

