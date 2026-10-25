import json
import logging
import torch
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from settings import Settings
from utils.file_utils import dataset_root, output_root
from utils.torch_utils import get_device

def score_perplexity(settings: Settings) -> None:
    logging.info(f"\n--- Scoring model on perplexity ---")

    device = get_device()

    path_to_adapter = f"{output_root()}/adapter"

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        settings.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, path_to_adapter)
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    path_to_dataset_eval = f"{dataset_root()}/{settings.validation_set}"

    with open(path_to_dataset_eval, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    with torch.no_grad():
        for item in tqdm(records):
            logging.info(f"Evaluating {item}")

            # read fields from record
            messages = item['messages']

            # 1. Tokenisation du prompt (contexte avant la dernière réponse assistant)
            # On isole la dernière réponse pour ne calculer la loss que dessus
            context_messages = messages[:-1]
            prompt_encoded = tokenizer.apply_chat_template(
                context_messages,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            # Selon la version, apply_chat_template renvoie soit un Tensor direct, soit un BatchEncoding/dict
            prompt_ids = (
                prompt_encoded["input_ids"]
                if isinstance(prompt_encoded, dict) or hasattr(prompt_encoded, "data")
                else prompt_encoded
            ).to(device)

            # 2. Tokenisation du message complet
            full_encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                return_tensors="pt"
            )
            full_ids = (
                full_encoded["input_ids"]
                if isinstance(full_encoded, dict) or hasattr(full_encoded, "data")
                else full_encoded
            ).to(device)

            # 3. Récupération sûre de la longueur du prompt
            prompt_len = prompt_ids.shape[1]

            # 3. Masquage du prompt avec -100
            labels = full_ids.clone()
            labels[:, :prompt_len] = -100

            # 4. Inférence avec les mêmes longueurs de tenseurs
            outputs = model(input_ids=full_ids, labels=labels)
            loss = outputs.loss

            target_tokens = (labels != -100).sum().item()

            if target_tokens > 0:
                total_nll += loss.item() * target_tokens
                total_tokens += target_tokens

    if total_tokens == 0:
        logging.error("Aucun token cible évalué.")
        return

    mean_nll = total_nll / total_tokens
    perplexity = math.exp(mean_nll)

    logging.info(f"\n--- Results ---")
    logging.info(f"evaluated tokens : {total_tokens}")
    logging.info(f"negative mean Log-Likelihood : {mean_nll:.4f}")
    logging.info(f"perplexity (PPL) : {perplexity:.2f}")
