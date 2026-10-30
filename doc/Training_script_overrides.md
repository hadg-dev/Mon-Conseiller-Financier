## Training script overrides


|   Variable name          |       Description                   |  Default |
|--------------------------|-------------------------------------|----------|
| STEPS                    | training steps to run               | dataset,train,score_perplexity,score_with_llm,copy_s3     |
| DEBUG                    | Provides additional logging         | True     |
| REPORT_TRAINING          | "mlflow", "none", "all"             | none     |
| NB_EPOCHS                | Number of epochs to train           | 4        |
| MODEL_NAME               | Training model                      | "Qwen/Qwen3.5-4B" |
| LEARNING_RATE            | Traning learning rate               | 2e-4     |
| REPO_ID                  | HF Training dataset                 | "likhitjuttada/finance-reasoning-sft-dataset" |
