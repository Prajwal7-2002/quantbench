"""Quantize a real Hugging Face model and measure what each scheme costs and saves.

Reports, per scheme: resting VRAM after conversion, perplexity on a fixed 1,024-token
text, and greedy decode speed (median of interleaved repeats, so laptop clock/thermal
drift hits every scheme equally). Speed is also shown relative to the unquantized model
from the same run, which is more stable than absolute tok/s.

    pip install quantbench-engine transformers accelerate
    python examples/04_real_model_eval.py --model Qwen/Qwen2.5-0.5B-Instruct

Needs a CUDA GPU (about 3 GB for a 0.5B model).
"""

from __future__ import annotations

import argparse
import gc
import inspect
import math
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantbench.nn import convert_model_to_quant

SCHEMES = ["none", "int8", "int4", "ternary"]


def load(model_id: str, scheme: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).cuda().eval()
    if scheme != "none":
        convert_model_to_quant(model, scheme=scheme, group_size=64)
        model.cuda()
    return model


@torch.no_grad()
def decode_tok_s(model, prompt_ids: torch.Tensor, n: int = 64) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(prompt_ids, max_new_tokens=n, min_new_tokens=n, do_sample=False)
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    tok = AutoTokenizer.from_pretrained(args.model)
    # Fixed, license-free evaluation text that ships with every Python install.
    text = inspect.getsource(argparse) + inspect.getsource(statistics)
    ids = tok(text, return_tensors="pt").input_ids[:, :1024].cuda()
    prompt = ids[:, :32]

    rows: dict[str, dict] = {}
    for scheme in SCHEMES:
        model = load(args.model, scheme, dtype)
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl = math.exp(model(ids, labels=ids).loss.item())
        rows[scheme] = {"vram_mb": torch.cuda.memory_allocated() / 2**20, "ppl": ppl, "speed": []}
        del model
        gc.collect()
        torch.cuda.empty_cache()

    for _ in range(args.repeats):  # interleave schemes so drift affects all of them
        for scheme in SCHEMES:
            model = load(args.model, scheme, dtype)
            decode_tok_s(model, prompt, n=8)  # warm-up
            rows[scheme]["speed"].append(decode_tok_s(model, prompt))
            del model
            gc.collect()
            torch.cuda.empty_cache()

    base = statistics.median(rows["none"]["speed"])
    print(f"{args.model} ({args.dtype}) on {torch.cuda.get_device_name(0)}")
    header = f"{'scheme':8s} {'VRAM MB':>8s} {'perplexity':>12s} {'decode tok/s':>13s}"
    print(f"{header} {'vs base':>8s}")
    for scheme, r in rows.items():
        s = statistics.median(r["speed"])
        print(f"{scheme:8s} {r['vram_mb']:8.0f} {r['ppl']:12.2f} {s:13.1f} {s / base:7.2f}x")


if __name__ == "__main__":
    main()
