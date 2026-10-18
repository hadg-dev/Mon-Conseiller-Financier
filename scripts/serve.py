from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

# start : python serve.py
# request :
#   curl -X POST "http://localhost:8000/generate" \
#     -H "Content-Type: application/json" \
#     -d '{
#       "prompt": "A company has EBITDA of $12M, debt of $40M, and cash of $4M. Compute net debt and the leverage ratio, explaining each step.",
#       "temperature": 0.2,
#       "max_new_tokens": 400
#     }'

TRAINED_LORA_DIR = "../lora/adapter"
TRAINED_TOKENIZER_DIR = "../lora/tokenizer"
MODEL_NAME = "Qwen/Qwen3.5-4B"

model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        TRAINED_TOKENIZER_DIR, trust_remote_code=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Applying LoRA weights from {TRAINED_LORA_DIR}...")
    model = PeftModel.from_pretrained(base_model, TRAINED_LORA_DIR)
    model.eval()
    print("Model loaded and ready for inference.")
    yield


app = FastAPI(
    title="Financial Reasoning LLM Service",
    description="FastAPI endpoint serving a QLoRA fine-tuned model for financial reasoning tasks.",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="User query or financial problem requiring reasoning.",
    )
    max_new_tokens: int = Field(
        default=512, ge=16, le=2048, description="Maximum tokens to generate."
    )
    temperature: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Sampling temperature."
    )
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    response: str
    prompt_tokens: int
    completion_tokens: int


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not model or not tokenizer:
        raise HTTPException(
            status_code=503, detail="Model is still initializing."
        )

    messages = [
        {
            "role": "system",
            "content": "You are a financial reasoning assistant. Provide logical, stepwise reasoning before concluding your answer.",
        },
        {"role": "user", "content": req.prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    generation_kwargs = {
        "max_new_tokens": req.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if req.temperature > 0.0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = req.temperature
        generation_kwargs["top_p"] = req.top_p
    else:
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][prompt_length:]
    decoded_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return GenerateResponse(
        response=decoded_output.strip(),
        prompt_tokens=prompt_length,
        completion_tokens=len(new_tokens),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "adapter_loaded": TRAINED_LORA_DIR}


if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)