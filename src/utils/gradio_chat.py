import os
from pathlib import Path
from threading import Thread
import gradio as gr
from peft import PeftModel
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

TRAINED_LORA_DIR = os.getenv("ADAPTER_DIR", str(Path(__file__).resolve().parents[2] / "output" / "adapter"))
# Must be the model the adapter was trained on; real model: Qwen/Qwen3.5-4B
MODEL_NAME = os.getenv("MODEL_NAME", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")

SYSTEM_PROMPT = """
You are an expert, empathetic, and highly analytical Personal Financial Advisor. Your mission is to help the user master their money, optimize their budget, eliminate debt, invest wisely, and build long-term wealth.

Operational Directives:
1. Client-Centric Analysis: Always tailor your advice to the user's specific context, including their income, expenses, geographic location, age, risk tolerance, and financial goals.
2. Educational Clarity: Simplify complex financial concepts (e.g., compound interest, asset allocation, tax-advantaged accounts, inflation) using accessible language and relatable analogies.
3. Actionable & Realistic: Provide concrete, step-by-step strategies instead of overly restrictive plans. Focus on behavioral finance and sustainable financial habits.
4. Information Gathering: If critical data points are missing (such as interest rates, debt balances, tax brackets, or investment horizons), proactively ask clear, targeted questions before providing a definitive strategy.
5. Strict Legal Disclaimer: Include a concise disclaimer reminding the user that you provide educational and informational guidance, not certified financial, legal, or tax advice.
"""


print("Loading tokenizer and base model...")
# The adapter folder on S3 has no tokenizer, the one of the base model is used
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True
)

if torch.cuda.is_available():
    # Load in 4-bit to minimize VRAM footprint during local inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        trust_remote_code=True,
    )

print(f"Attaching LoRA adapter from {TRAINED_LORA_DIR}...")
model = PeftModel.from_pretrained(base_model, TRAINED_LORA_DIR)
model.eval()
print("Model loaded successfully.")


def chat_stream(
    message: str,
    history: list,
    #system_prompt: str,
    temperature: float,
    max_new_tokens: int,
):
    """Formats conversation history using the chat template and yields tokens in real time."""
    #messages = [{"role": "system", "content": system_prompt}]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Convert Gradio history into OpenAI/HF role-content format
    for item in history:
        if isinstance(item, dict):
            content = item["content"]
            # Gradio sends the history as a list of text blocks
            if isinstance(content, list):
                content = "".join(block.get("text", "") for block in content)
            messages.append({"role": item["role"], "content": content})
        elif isinstance(item, (list, tuple)):
            messages.append({"role": "user", "content": item[0]})
            if item[1] is not None:
                messages.append({"role": "assistant", "content": item[1]})

    messages.append({"role": "user", "content": message})

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    # Streamer to capture output tokens incrementally
    streamer = TextIteratorStreamer(
        tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
    )

    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=int(max_new_tokens),
        do_sample=temperature > 0.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0.0:
        generation_kwargs["temperature"] = float(temperature)
        generation_kwargs["top_p"] = 0.9

    # Generation runs on a worker thread so the generator can yield tokens
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    for token in streamer:
        partial_text += token
        yield partial_text

    thread.join()


# Gradio Chat Interface
demo = gr.ChatInterface(
    fn=chat_stream,
    title="Finance Reasoning Assistant",
    description="Interactive chat interface with your fine-tuned financial reasoning LoRA adapter.",
    examples=[
        ["A company has EBITDA of $12M, debt of $40M, and cash of $4M. Compute net debt and leverage ratio.", 0.2, 512],
        ["Explain the difference between Enterprise Value and Equity Value step-by-step.", 0.2, 512],
    ],
    additional_inputs=[
        gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.2,
            step=0.05,
            label="Temperature",
        ),
        gr.Slider(
            minimum=64,
            maximum=2048,
            value=512,
            step=64,
            label="Max Tokens",
        ),
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)