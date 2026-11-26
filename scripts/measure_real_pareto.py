"""Measure the QuantBench Pareto frontier and layer sensitivity on a real model.

Every number written to hf_dataset/ comes from this run: nothing is typed in.

For each quantization config this script quantizes a real Hugging Face model with
quantbench's QuantLinear and measures:
  * weight SQNR (dB) of every converted layer against its original weights
  * stored bytes of the converted linear layers and the resting VRAM of the whole model
  * perplexity on a fixed, license-free text (Python standard-library source code)
  * greedy decode speed (median of interleaved repeats, so laptop clock and thermal
    drift affect every config equally), also shown relative to the unquantized model

Layer sensitivity quantizes one projection type at a time (q, k, v, o, gate, up, down)
and measures how much perplexity changes, so "sensitive" means measured damage.

    pip install quantbench-engine transformers accelerate
    python scripts/measure_real_pareto.py --model Qwen/Qwen2.5-0.5B-Instruct

Needs a CUDA GPU (about 3 GB for a 0.5B model).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import inspect
import json
import math
import platform
import statistics
import textwrap
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import quantbench
from quantbench.eval.sqnr import compute_sqnr
from quantbench.nn import convert_model_to_quant
from quantbench.nn.quant_linear import QuantLinear

OUT_DIR = Path(__file__).resolve().parent.parent / "hf_dataset"

# (label, scheme, group_size, granularity note)
PARETO_CONFIGS = [
    ("BF16 baseline", "none", None, "unquantized"),
    ("INT8 per-channel", "int8", -1, "one scale per output channel"),
    ("INT4 G=32", "int4", 32, "one scale per 32 weights"),
    ("INT4 G=64", "int4", 64, "one scale per 64 weights"),
    ("INT4 G=128", "int4", 128, "one scale per 128 weights"),
    (
        "2:4 structured sparsity",
        "sparse24",
        -1,
        "2 of every 4 weights kept, values in source dtype",
    ),
    ("Ternary (post-hoc)", "ternary", -1, "{-1, 0, +1} with one scale per tensor, 2-bit storage"),
]

SENSITIVITY_SCHEMES = [
    ("INT8 per-channel", "int8", -1),
    ("INT4 G=128", "int4", 128),
    ("2:4 structured sparsity", "sparse24", -1),
    ("Ternary (post-hoc)", "ternary", -1),
]

PROJECTIONS = [
    ("q_proj", "Attention Query"),
    ("k_proj", "Attention Key"),
    ("v_proj", "Attention Value"),
    ("o_proj", "Attention Output"),
    ("gate_proj", "MLP Gate"),
    ("up_proj", "MLP Up"),
    ("down_proj", "MLP Down"),
]


def eval_text() -> str:
    """Fixed, license-free text that ships with every Python install."""
    return inspect.getsource(argparse) + inspect.getsource(statistics) + inspect.getsource(textwrap)


def load(model_id: str, dtype: torch.dtype):
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).cuda().eval()


def free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def linear_weights(model, targets: list[str] | None) -> dict[str, torch.Tensor]:
    """Copies of the original weights that quantbench will convert (lm_head is never converted)."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
            if targets is None or any(t in name for t in targets):
                out[name] = mod.weight.detach().clone()
    return out


def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def quantize(model, scheme: str, group_size: int | None, targets: list[str] | None) -> dict:
    """Quantize in place and measure weight fidelity and stored bytes of the converted layers."""
    originals = linear_weights(model, targets)
    bytes_before = 0
    for name, mod in model.named_modules():
        if name in originals:
            bytes_before += tensor_bytes(mod.weight) + (
                tensor_bytes(mod.bias) if mod.bias is not None else 0
            )

    convert_model_to_quant(
        model, scheme=scheme, target_modules=targets, group_size=group_size or 64
    )
    model.cuda()

    signal = noise = 0.0
    per_layer = []
    bytes_after = 0
    for name, mod in model.named_modules():
        if isinstance(mod, QuantLinear) and name in originals:
            w = originals[name].float()
            w_hat = mod.dequantize_weight().float().reshape(w.shape)
            signal += float((w**2).sum())
            noise += float(((w - w_hat) ** 2).sum())
            per_layer.append(compute_sqnr(w, w_hat))
            bytes_after += sum(
                tensor_bytes(t) for t in list(mod.parameters()) + list(mod.buffers())
            )
    free(originals)
    return {
        "layers_quantized": len(per_layer),
        "weight_sqnr_db": 10 * math.log10(signal / max(noise, 1e-30)) if per_layer else None,
        "weight_sqnr_db_min_layer": min(per_layer) if per_layer else None,
        "linear_bytes_before": bytes_before,
        "linear_bytes_after": bytes_after,
    }


@torch.no_grad()
def perplexity(model, windows: list[torch.Tensor]) -> float:
    losses = [model(ids, labels=ids).loss.item() for ids in windows]
    return math.exp(statistics.fmean(losses))


@torch.no_grad()
def decode_tok_s(model, prompt_ids: torch.Tensor, n: int = 64) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(prompt_ids, max_new_tokens=n, min_new_tokens=n, do_sample=False)
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def write_table(rows: list[dict], stem: str) -> None:
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with open(OUT_DIR / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def r(x, nd=2):
    return None if x is None else round(x, nd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument(
        "--repeats", type=int, default=3, help="interleaved decode-speed repeats per config"
    )
    ap.add_argument("--window", type=int, default=1024, help="tokens per perplexity window")
    ap.add_argument("--windows", type=int, default=2, help="number of perplexity windows")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    OUT_DIR.mkdir(exist_ok=True)
    started = time.perf_counter()

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(eval_text(), return_tensors="pt").input_ids
    need = args.window * args.windows
    if ids.shape[1] < need:
        raise SystemExit(f"Evaluation text has {ids.shape[1]} tokens; need {need}.")
    windows = [ids[:, i * args.window : (i + 1) * args.window].cuda() for i in range(args.windows)]
    prompt = windows[0][:, :32]

    # ---- Pareto frontier: quality and size per config ----
    pareto: dict[str, dict] = {}
    for label, scheme, group, note in PARETO_CONFIGS:
        print(f"[pareto] {label}: quantize + perplexity", flush=True)
        model = load(args.model, dtype)
        if scheme == "none":
            q = {
                "layers_quantized": 0,
                "weight_sqnr_db": None,
                "weight_sqnr_db_min_layer": None,
                "linear_bytes_before": None,
                "linear_bytes_after": None,
            }
        else:
            q = quantize(model, scheme, group, None)
        free()
        pareto[label] = {
            **q,
            "scheme": scheme,
            "group_size": group,
            "granularity": note,
            "resting_vram_mb": torch.cuda.memory_allocated() / 2**20,
            "perplexity": perplexity(model, windows),
            "speeds": [],
        }
        free(model)

    # ---- Decode speed, interleaved so drift affects every config equally ----
    for rep in range(args.repeats):
        for label, scheme, group, _ in PARETO_CONFIGS:
            print(f"[speed] repeat {rep + 1}/{args.repeats}: {label}", flush=True)
            model = load(args.model, dtype)
            if scheme != "none":
                convert_model_to_quant(model, scheme=scheme, group_size=group or 64)
                model.cuda()
            decode_tok_s(model, prompt, n=8)  # warm-up
            pareto[label]["speeds"].append(decode_tok_s(model, prompt))
            free(model)

    base = pareto["BF16 baseline"]
    base_speed = statistics.median(base["speeds"])
    pareto_rows = []
    for label, row in pareto.items():
        speed = statistics.median(row["speeds"])
        linear_ratio = (
            (row["linear_bytes_before"] / row["linear_bytes_after"])
            if row["linear_bytes_after"]
            else 1.0
        )
        pareto_rows.append(
            {
                "config": label,
                "scheme": row["scheme"],
                "group_size": row["group_size"],
                "granularity": row["granularity"],
                "layers_quantized": row["layers_quantized"],
                "weight_sqnr_db": r(row["weight_sqnr_db"], 1),
                "weight_sqnr_db_min_layer": r(row["weight_sqnr_db_min_layer"], 1),
                "linear_layer_compression": round(linear_ratio, 2),
                "resting_vram_mb": round(row["resting_vram_mb"]),
                "model_compression": round(base["resting_vram_mb"] / row["resting_vram_mb"], 2),
                "perplexity": round(row["perplexity"], 3),
                "perplexity_delta_pct": round(
                    (row["perplexity"] / base["perplexity"] - 1) * 100, 2
                ),
                "decode_tok_s": round(speed, 1),
                "decode_speed_vs_bf16": round(speed / base_speed, 2),
            }
        )

    # ---- Layer sensitivity: quantize one projection type at a time ----
    sens_rows = []
    for proj, component in PROJECTIONS:
        for label, scheme, group in SENSITIVITY_SCHEMES:
            print(f"[sensitivity] {component} ({proj}) with {label}", flush=True)
            model = load(args.model, dtype)
            q = quantize(model, scheme, group, [proj])
            free()
            ppl = perplexity(model, windows)
            free(model)
            sens_rows.append(
                {
                    "layer_component": f"{component} ({proj})",
                    "quantization_scheme": label,
                    "layers_quantized": q["layers_quantized"],
                    "weight_sqnr_db": r(q["weight_sqnr_db"], 1),
                    "linear_layer_compression": round(
                        q["linear_bytes_before"] / q["linear_bytes_after"], 2
                    ),
                    "perplexity": round(ppl, 3),
                    "perplexity_delta_pct": round((ppl / base["perplexity"] - 1) * 100, 2),
                }
            )

    meta = {
        "model": args.model,
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "quantbench": getattr(quantbench, "__version__", "unknown"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "measured_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "eval_text": "Python standard-library source of argparse, statistics and textwrap",
        "perplexity_windows": args.windows,
        "tokens_per_window": args.window,
        "decode_tokens": 64,
        "decode_repeats": args.repeats,
        "runtime_s": round(time.perf_counter() - started),
    }

    write_table(pareto_rows, "pareto_frontier")
    write_table(sens_rows, "layer_sensitivity")
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(dataset_card(pareto_rows, sens_rows, meta), encoding="utf-8")
    print(
        f"Done in {meta['runtime_s']} s. Wrote {len(pareto_rows)} Pareto rows and {len(sens_rows)} sensitivity rows to {OUT_DIR}"
    )


def md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = (
        "| "
        + " | ".join(h for _, h in cols)
        + " |\n|"
        + "|".join(":---" if i == 0 else "---:" for i in range(len(cols)))
        + "|\n"
    )
    body = "".join(
        "| " + " | ".join("" if row[k] is None else str(row[k]) for k, _ in cols) + " |\n"
        for row in rows
    )
    return head + body


def dataset_card(pareto: list[dict], sens: list[dict], meta: dict) -> str:
    worst = max(
        sens,
        key=lambda s: (
            s["perplexity_delta_pct"] if s["quantization_scheme"] == "INT4 G=128" else -1e9
        ),
    )
    best = min(
        sens,
        key=lambda s: (
            s["perplexity_delta_pct"] if s["quantization_scheme"] == "INT4 G=128" else 1e9
        ),
    )
    return f"""---
language:
- en
license: apache-2.0
size_categories:
- n<1K
task_categories:
- text-generation
tags:
- quantization
- pareto-frontier
- int4
- int8
- ternary
- sparsity
- llm-compression
- benchmark
pretty_name: "QuantBench: measured quantization quality vs compression on {meta["model"]}"
configs:
  - config_name: pareto_frontier
    data_files:
      - split: train
        path: pareto_frontier.json
  - config_name: layer_sensitivity
    data_files:
      - split: train
        path: layer_sensitivity.json
---

# QuantBench: measured quantization quality vs compression

Every number in this dataset was **measured** on a real model in a single run by
[`scripts/measure_real_pareto.py`](https://github.com/Prajwal7-2002/quantbench/blob/main/scripts/measure_real_pareto.py).
Nothing is estimated or typed in. Re-run the script to reproduce it on your own GPU.

| | |
|:---|:---|
| Model | `{meta["model"]}` ({meta["dtype"]}) |
| Hardware | {meta["device"]} |
| Software | torch {meta["torch"]}, transformers {meta["transformers"]}, quantbench {meta["quantbench"]} |
| Measured | {meta["measured_at"]} |
| Perplexity text | {meta["eval_text"]} ({meta["perplexity_windows"]} windows of {meta["tokens_per_window"]} tokens) |
| Decode speed | {meta["decode_tokens"]} greedy tokens, median of {meta["decode_repeats"]} interleaved repeats |

## Pareto frontier

{md_table(pareto, [("config", "Config"), ("weight_sqnr_db", "Weight SQNR (dB)"), ("linear_layer_compression", "Linear-layer compression"), ("resting_vram_mb", "Model VRAM (MB)"), ("model_compression", "Whole-model compression"), ("perplexity", "Perplexity"), ("perplexity_delta_pct", "Perplexity change (%)"), ("decode_speed_vs_bf16", "Decode speed vs BF16")])}
## Layer sensitivity

One projection type is quantized at a time; everything else stays in {meta["dtype"]}.
With INT4 G=128, quantizing **{worst["layer_component"]}** hurt perplexity most
({worst["perplexity_delta_pct"]:+.2f}%) and **{best["layer_component"]}** least ({best["perplexity_delta_pct"]:+.2f}%).

{md_table(sens, [("layer_component", "Layer"), ("quantization_scheme", "Scheme"), ("layers_quantized", "Layers"), ("weight_sqnr_db", "Weight SQNR (dB)"), ("perplexity_delta_pct", "Perplexity change (%)")])}
## How to read this honestly

- **One model, one GPU, one run.** Results describe `{meta["model"]}` on this hardware; other models differ.
- **Perplexity is measured on Python source code**, chosen because it is license-free and ships with Python.
  Use it to compare configs with each other, not as a general language-modeling score.
- **Decode speed reflects quantbench's reference runtime**, which dequantizes full weights on every forward
  pass. Quantization saves memory here but does not speed decoding up; that needs fused dequant-matmul kernels.
- **Whole-model compression** includes the embedding and `lm_head`, which are not quantized, so it is lower
  than the linear-layer compression.
- **2:4 sparsity does not save memory here.** quantbench stores the kept values in the source dtype plus a
  mask, so the model gets slightly larger. Savings need a sparse storage format and sparse kernels.
- **Ternary is post-hoc** quantization of a model not trained for it, unlike BitNet models trained ternary
  from the start.

## Files

- `pareto_frontier.json` / `.csv`: one row per config
- `layer_sensitivity.json` / `.csv`: one row per (layer type, scheme)
- `run_metadata.json`: model, hardware, software versions, and evaluation settings
"""


if __name__ == "__main__":
    main()
