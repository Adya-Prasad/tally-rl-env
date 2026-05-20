"""
confirm (1) the .env loads, (2) the HF token authenticates, (3) a remote model replies
"""

import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit(
        "HF_TOKEN not found. Check that .env exists in this folder "
        "and contains a line like:  HF_TOKEN=hf_..."
    )
print(f"[ok] Token loaded (starts with {token[:6]}..., length {len(token)})")


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=token,
)


MODEL = "Qwen/Qwen2.5-7B-Instruct"

print(f"[..] Sending a tiny request to {MODEL} ...")

try:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Reply with exactly one word: hello"}
        ],
        max_tokens=10,
        temperature=0.0,
    )
    reply = completion.choices[0].message.content
    print(f"[ok] Model replied: {reply!r}")
    print("\n[SUCCESS] Plumbing works. API is ready for project!")
except Exception as e:
    print(f"\n[FAIL] Request failed: {type(e).__name__}: {e}")
    print("Common causes:")
    print("  - Token missing 'Make calls to Inference Providers' permission")
    print("  - Model not available on free tier today (try another, see below)")
    print("  - No internet / firewall blocking router.huggingface.co")