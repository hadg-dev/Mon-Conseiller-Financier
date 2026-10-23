import json
import torch
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from configuration.config import load_config
from utils.utils import dataset_root

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # Adapter selon votre base Qwen
LORA_PATH = "./path_to_your_lora_weights"
VAL_FILE = "combined_val.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Chargement du tokenizer et du modèle...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
# Fusion / chargement de l'adaptateur LoRA
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

total_nll = 0.0
total_tokens = 0

with open(VAL_FILE, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

print(f"Calcul de la perplexité sur {len(records)} exemples...")

with torch.no_grad():
    for item in tqdm(records):
        # Récupération des champs selon le format du JSONL
        instruction = item.get("instruction", "")
        if item.get("input"):
            instruction += f"\n{item['input']}"
        output = item.get("output", "")

        # Application du chat template Qwen
        prompt_messages = [{"role": "user", "content": instruction}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + output + tokenizer.eos_token

        # Tokenisation
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(DEVICE)

        prompt_len = prompt_ids.shape[1]

        # Création des labels : masquer l'instruction avec -100
        labels = full_ids.clone()
        labels[:, :prompt_len] = -100

        outputs = model(input_ids=full_ids, labels=labels)
        loss = outputs.loss

        # Nombre de tokens réels dans la réponse
        target_tokens = (labels != -100).sum().item()

        if target_tokens > 0:
            total_nll += loss.item() * target_tokens
            total_tokens += target_tokens

mean_nll = total_nll / total_tokens
perplexity = math.exp(mean_nll)

print(f"\n--- Résultats ---")
print(f"Tokens évalués : {total_tokens}")
print(f"Negative Log-Likelihood moyen : {mean_nll:.4f}")
print(f"Perplexité (PPL) : {perplexity:.2f}")