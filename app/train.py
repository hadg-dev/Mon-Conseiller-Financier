import argparse
import os
import mlflow.sklearn
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import pandas as pd
from datasets import Dataset

#DATA_URL = "https://julie-2-next-resources.s3.eu-west-3.amazonaws.com/full-stack-full-time/linear-regression-ft/californian-housing-market-ft/california_housing_market.csv"
NB_EPOCHS = 10
LEARNING_RATE = 0.01
MODEL_NAME = "Qwen/Qwen3.5-4B"
DATASET_DIR = "../dataset"
TRAINED_OUTPUT_DIR = "../lora/output"
TRAINED_LORA_DIR = "../lora/adapter"
TRAINED_TOKENIZER_DIR = "../lora/tokenizer"
TRAINING_SYSTEM_PROMPT = "You are an expert, empathetic, and highly analytical Personal Financial Advisor. Your mission is to help the user master their money, optimize their budget, eliminate debt, invest wisely, and build long-term wealth."
REPORT_TRAINING_TO = "none" # "mlflow", "none"

def prepare_chat_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer) -> Dataset:
    """
    Prepends the system prompt if missing and applies Qwen's chat template
    to convert the 'messages' list into a single formatted string.
    """

    def format_row(example):
        messages = list(example["messages"])

        # Inject system prompt at index 0 if not already present
        if TRAINING_SYSTEM_PROMPT and (not messages or messages[0].get("role") != "system"):
            messages.insert(0, {"role": "system", "content": TRAINING_SYSTEM_PROMPT})

        # Apply the model's native chat template
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": formatted_text}

    # Convert DataFrame to Hugging Face Dataset and format
    hf_dataset = Dataset.from_pandas(df[["messages"]].reset_index(drop=True))
    hf_dataset_rpw_mapped = hf_dataset.map(format_row, remove_columns=["messages"])
    return hf_dataset_rpw_mapped

def parse_args():
    # 1. Parse ONLY Model Hyperparameters
    parser = argparse.ArgumentParser(description="Financial Advisor Training Script")
    # TODO
    #parser.add_argument("--n_estimators", type=int, default=20)
    #parser.add_argument("--criterion", type=str, default="squared_error")
    #parser.add_argument("--experiment_name", type=str, default="california_housing")
    return parser.parse_args()

def load_model_from_hf():
    compute_dtype = torch.float16  # Use torch.float16 on a T4. Newer cards can use torch.bfloat16.

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
    )
    _model = prepare_model_for_kbit_training(_model)

    memory = torch.cuda.memory_allocated() / 1e9
    logging.info(f"GPU memory in use : {memory:.2f} GB")

    return _model, _tokenizer

def load_dataset():
    target_file = next(f for f in os.listdir(DATASET_DIR) if f.endswith((".jsonl")))
    target_file_path = os.path.join(DATASET_DIR, target_file)
    _train_data = pd.read_json(target_file_path, lines=target_file_path.endswith(".jsonl"))
    return _train_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Arguments
    args = parse_args()

    # setup ML Flow
    if REPORT_TRAINING_TO == "mlflow":
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(args.experiment_name)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logging.warning("Warning: No GPU found.")
    logging.info(f"Device: {device}")

    # Load model and related tokenizer
    model, tokenizer = load_model_from_hf()

    # Load Daatset
    train_data = load_dataset()
    train_dataset = prepare_chat_dataset(train_data, tokenizer)

    # Lora training
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16 and device == "cuda"

    sft_config = SFTConfig(
        output_dir=TRAINED_OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=LEARNING_RATE,
        num_train_epochs=NB_EPOCHS,
        # max_seq_length=1536,
        max_length=1536,
        dataset_text_field="text",
        logging_steps=5,
        save_strategy="epoch",
        fp16=use_fp16,
        bf16=use_bf16,
        report_to=REPORT_TRAINING_TO,
    )

    try:
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            peft_config=lora_config,
            processing_class=tokenizer,
        )
    except TypeError:
        logging.warning("SFTTrainer instanciation failed, retrying with with fixed tokenizer")
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            peft_config=lora_config,
            tokenizer=tokenizer,
        )

    trainer.model.save_pretrained(TRAINED_LORA_DIR)
    tokenizer.save_pretrained(TRAINED_TOKENIZER_DIR)
    logging.info("Training finished. Adapter weights saved to 'financial-model-training/final_adapter'.")
