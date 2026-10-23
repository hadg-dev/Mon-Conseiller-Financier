import argparse
import os
import mlflow.sklearn
import torch
import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
import pandas as pd
from datasets import Dataset

from configuration.config import load_config

#DATA_URL = "https://julie-2-next-resources.s3.eu-west-3.amazonaws.com/full-stack-full-time/linear-regression-ft/californian-housing-market-ft/california_housing_market.csv"

#DEFAULT_NB_EPOCHS = 4
#DEFAULT_LEARNING_RATE = 2e-4
#DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-4B"  # Qwen/Qwen3.5-9B , Qwen/Qwen3.8-27B
#DEFAULT_REPORT_TRAINING_TO = "none" # "mlflow", "none", "all"

DATASET_DIR = "../../dataset"
TRAINED_OUTPUT_DIR = "../../output/trainer"
TRAINED_LORA_DIR = "../../output/adapter"
TRAINED_TOKENIZER_DIR = "../../output/tokenizer"
TRAINING_SYSTEM_PROMPT = "You are an expert and highly analytical Personal Financial Advisor. Your mission is to help the user master their money, optimize their budget, eliminate debt, invest wisely, and build long-term wealth."

def parse_args() -> argparse.Namespace:
    conf = load_config()
    parser = argparse.ArgumentParser(description="Financial Advisor Training Script")
    #parser.add_argument("--n_estimators", type=int, default=20)
    #parser.add_argument("--criterion", type=str, default="squared_error")
    parser.add_argument("--experiment_name", type=str, default="mon_conseiller_financier")
    parser.add_argument("--nb_epochs", type=int, default=conf.nb_epochs)
    parser.add_argument("--model_name", type=str, default=conf.model_name)
    parser.add_argument("--learning_rate", type=float, default=conf.learning_rate)
    parser.add_argument("--report_training", type=str, default=conf.report_training_to)
    _args = parser.parse_args()
    logging.info(_args)
    return _args

def load_and_format_dataset(formatting_tokenizer: PreTrainedTokenizerBase) -> Dataset:
    target_file = next(f for f in os.listdir(DATASET_DIR) if f.endswith((".jsonl")))
    target_file_path = os.path.join(DATASET_DIR, target_file)
    logging.info(f"Loading dataset from {target_file_path}")
    train_data = pd.read_json(target_file_path, lines=target_file_path.endswith(".jsonl"))

    logging.info(f"Loaded {len(train_data):,} rows.")
    columns = train_data.columns.tolist()
    logging.info(f"Columns ({len(columns)}): {', '.join(columns)}")

    def format_row(row):
        if isinstance(row, str):
            try:
                row = json.loads(row)
            except json.JSONDecodeError:
                import ast
                row = ast.literal_eval(row)

        user_content = ""
        assistant_content = ""

        for message in row:
            role = message.get("role")
            content = message.get("content", "")
            if role == "user":
                user_content = content
            elif role == "assistant":
                assistant_content = content  # clean_think_tags(content)

        message = [
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        formatted_message = formatting_tokenizer.apply_chat_template(message, tokenize=False)
        return formatted_message

    _formatted = train_data["messages"].apply(format_row).tolist()
    _formatted_dataset = Dataset.from_dict({"text": _formatted})
    return _formatted_dataset

def load_model_from_hf(model_name: str, device_map: str, use_bf16: bool) -> tuple[PreTrainedModel | PeftModel, PreTrainedTokenizerBase] :
    logging.info(f"Loading model and tokenizer for {model_name}...")

    if device_map == "cuda":
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        logging.info(f"Loading model with bnb quantization and compute dtype {compute_dtype}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
    elif device_map == "cpu":
        logging.info(f"Loading model to cpu")
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        raise Exception(f'Unsupported device {device_map}')

    _model = prepare_model_for_kbit_training(_model)

    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    memory = torch.cuda.memory_allocated() / 1e9
    logging.info(f"GPU memory in use : {memory:.2f} GB")

    return _model, _tokenizer

def train(
        lr: float,
        nb_epochs: int,
        training_model: PreTrainedModel | PeftModel,
        training_tokenizer: PreTrainedTokenizerBase,
        training_dataset: Dataset,
        report_training: str,
        use_bf16: bool,
        use_fp16: bool
) -> None:

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",  # TODO QUESTION_ANS ?
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    sft_config = SFTConfig(
        output_dir=TRAINED_OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=lr,
        num_train_epochs=nb_epochs,
        # max_seq_length=1536,
        max_length=1536,
        dataset_text_field="text",
        logging_steps=5,
        save_strategy="epoch",
        fp16=use_fp16,
        bf16=use_bf16,
        report_to=report_training,
    )

    try:
        trainer = SFTTrainer(
            model=training_model,
            args=sft_config,
            train_dataset=training_dataset,
            peft_config=lora_config,
            processing_class=training_tokenizer,
        )
    except TypeError:
        logging.warning("SFTTrainer instanciation failed, retrying with alternative func signature")
        trainer = SFTTrainer(
            model=training_model,
            args=sft_config,
            train_dataset=training_dataset,
            peft_config=lora_config,
            tokenizer=training_tokenizer,
        )

    trainer.train()

    trainer.model.save_pretrained(TRAINED_LORA_DIR)
    logging.info(f"Training finished. Adapter weights saved to {TRAINED_LORA_DIR}")
    training_tokenizer.save_pretrained(TRAINED_TOKENIZER_DIR)
    logging.info(f"Tokenizer saved to {TRAINED_TOKENIZER_DIR}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Arguments
    args = parse_args()

    # setup ML Flow
    if args.report_training == "mlflow":
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(args.experiment_name)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logging.warning("Warning: No GPU found.")
    logging.info(f"Device: {device}")

    # check bf16 / fp16
    bf16_supported = torch.cuda.is_bf16_supported()
    fp16_supported = not bf16_supported and device == "cuda"

    # Load model and related tokenizer
    model, tokenizer = load_model_from_hf(model_name=args.model_name, device_map=device, use_bf16=bf16_supported)

    # Load Dataset
    formatted_dataset = load_and_format_dataset(tokenizer)

    # Run Lora training
    train(
        lr=args.learning_rate,
        nb_epochs=args.nb_epochs,
        training_model=model,
        training_tokenizer=tokenizer,
        training_dataset=formatted_dataset,
        report_training=args.report_training,
        use_bf16=bf16_supported,
        use_fp16=fp16_supported,
    )
