import json
import re
from openai import OpenAI
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from configuration.config import load_config
from utils.utils import dataset_root

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "./path_to_your_lora_weights"
VAL_FILE = "combined_val.jsonl"
SAMPLE_SIZE = 50  # Évaluer un échantillon représentatif
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Chargement du modèle local pour génération
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# 2. Chargement des données
with open(VAL_FILE, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f][:SAMPLE_SIZE]

# 3. Génération des réponses du modèle
eval_pairs = []
print("Génération des réponses locales...")
for row in tqdm(dataset):
    user_query = row.get("instruction", "")
    if row.get("input"):
        user_query += f"\n{row['input']}"
    reference = row.get("output", "")

    messages = [{"role": "user", "content": user_query}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    eval_pairs.append({
        "query": user_query,
        "reference": reference,
        "model_output": response,
    })

# 4. Évaluation LLM-as-a-Judge
client = OpenAI()  # Nécessite OPENAI_API_KEY dans l'environnement

JUDGE_PROMPT = """Tu es un expert financier et auditeur de modèles d'IA.
Évalue la pertinence de la réponse générée par un modèle en la comparant à la question posée et à la réponse de référence.

Critères d'évaluation :
1. Rigueur financière (calculs, principes économiques, absence d'hallucinations chiffrées).
2. Clarté du raisonnement (chaîne de pensée logique et justifiable).
3. Pertinence et complétude par rapport au besoin utilisateur.

Donne une note stricte de 1 à 5 :
1 = Dangereux, faux ou hors-sujet.
2 = Erreurs financières notables ou raisonnement incohérent.
3 = Partiellement correct mais incomplet ou calcul imprécis.
4 = Bon raisonnement, financièrement juste, formulation claire.
5 = Excellent, précis, rigoureux et exhaustif.

Réponds STRICTEMENT sous ce format :
Score: <note de 1 à 5>
Justification: <brève explication>
"""

scores = []
print("\nÉvaluation par le LLM juge...")
for item in tqdm(eval_pairs):
    judge_input = f"""### Question de l'utilisateur :
{item['query']}

### Réponse de référence :
{item['reference']}

### Réponse du modèle à évaluer :
{item['model_output']}
"""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # Ou gpt-4o pour une précision maximale
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": judge_input},
        ],
        temperature=0.0,
    )

    verdict = completion.choices[0].message.content
    match = re.search(r"Score:\s*(\d+)", verdict)
    if match:
        score = int(match.group(1))
        scores.append(score)

avg_score = sum(scores) / len(scores) if scores else 0
print(f"\n--- Rapport d'évaluation ---")
print(f"Exemples notés : {len(scores)}/{SAMPLE_SIZE}")
print(f"Note moyenne de pertinence : {avg_score:.2f} / 5")