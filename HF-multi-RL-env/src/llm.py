# shared/llm.py
"""
Single point for calling llm for actions (The "harness" components A.T. paper's five-stage spine)

Tasks -> [Harness] -> Reward -> Rollout -> Trainer
The harness is just "how the model gets called".

Three backends:
"hf" 
"ollama_cloud"
"ollama_local" 
  
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BACKEND = "ollama_cloud"   # OR: hf | ollama_local


if BACKEND == "ollama_cloud":
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    DEFAULT_MODEL = "gemma4:31b-cloud"   

elif BACKEND == "ollama_local":
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    DEFAULT_MODEL = "qwen2.5:1.5b"
    
elif BACKEND == "hf":
    _token = os.environ.get("HF_TOKEN")
    if not _token:
        raise RuntimeError("HF_TOKEN not found in .env")
    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=_token)
    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

else:
    raise ValueError(f"Unknown BACKEND: {BACKEND!r}")


def chat(messages, model=None, temperature=0.0, max_tokens=512):
    """Send a list of messages, get back the model's text reply."""
    resp = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content