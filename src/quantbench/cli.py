"""Command Line Interface for quantbench."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import quantbench
from quantbench.quant.packing import (
    pack_int4,
    pack_ternary,
    prune_and_compress_2_4,
    unpack_int4,
    unpack_ternary,
)

app = typer.Typer(
    name="quantbench",
    help="First-principles LLM quantization engine and Pareto benchmark.",
    add_completion=False,
)
console = Console()


def version_callback(value: bool):
    if value:
        ver = quantbench.__version__
        console.print(f"[bold cyan]quantbench[/bold cyan] version [bold green]{ver}[/bold green]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """quantbench CLI root."""
    pass


@app.command()
def info():
    """Display system and supported quantization capabilities."""
    table = Table(title="quantbench Capabilities & Formats", border_style="cyan")
    table.add_column("Category", style="bold yellow")
    table.add_column("Supported Schemes", style="green")
    table.add_column("Target Hardware", style="magenta")

    table.add_row(
        "Integer Math",
        "INT8 AbsMax, INT8 Affine, INT4 Groupwise",
        "CPU / Tensor Core / Metal",
    )
    table.add_row(
        "Bit-Packing",
        "INT4 (2 vals/byte), BitNet Ternary (4 or 5 vals/byte), 2:4 Sparsity",
        "Vectorized Sub-Byte Runtimes",
    )
    table.add_row(
        "Modern FP8",
        "OCP E4M3FN (GEMM), OCP E5M2 (KV/Gradients)",
        "Ada / Hopper / CPU Emulation",
    )
    table.add_row(
        "Optimization",
        "SmoothQuant, AWQ Grid Search, GPTQ Hessian",
        "Zero-compile PyTorch",
    )
    table.add_row(
        "Hardware Sparsity",
        "2:4 Structured Sparsity (50% compression)",
        "Ampere / Ada / Hopper",
    )
    table.add_row(
        "Benchmarking",
        "Sliding WikiText-2 Perplexity, SQNR (dB)",
        "Local CPU & GPU",
    )

    console.print(table)


@app.command("pack-inspect")
def pack_inspect(
    fmt: str = typer.Option(
        "int4",
        "--format",
        "-f",
        help="Format to inspect: int4, ternary, sparse24",
    ),
):
    """Inspect bit-packing layout, sub-byte representation, and theoretical memory savings."""
    import torch

    fmt_lower = fmt.lower()
    if fmt_lower == "int4":
        sample = torch.tensor([5, 12], dtype=torch.int32)
        packed, orig_len = pack_int4(sample, signed=False)
        unpacked = unpack_int4(packed, original_len=orig_len, signed=False)
        byte_val = packed.item()

        panel = Panel(
            f"[bold green]INT4 Sub-Byte Packing (2 values per uint8 byte)[/bold green]\n\n"
            f"Input values: [cyan]{sample.tolist()}[/cyan]\n"
            f"• Low nibble [0:3]: {sample[0].item()} -> [yellow]{sample[0].item():04b}[/yellow]\n"
            f"• High nibble [4:7]: {sample[1].item()} -> [yellow]{sample[1].item():04b}[/yellow]\n"
            f"Packed uint8 byte: [bold magenta]{byte_val}[/bold magenta] "
            f"(binary: [yellow]{byte_val:08b}[/yellow], hex: [yellow]0x{byte_val:02X}[/yellow])\n"
            f"Unpacked roundtrip: [cyan]{unpacked.tolist()}[/cyan]\n"
            f"Memory compression: [bold green]4.0x vs FP16[/bold green] (2.0x vs INT8)",
            title="INT4 Bit Layout",
            border_style="cyan",
        )
        console.print(panel)

    elif fmt_lower == "ternary":
        sample = torch.tensor([-1, 0, 1, -1])
        packed, orig_len = pack_ternary(sample, mode="2bit")
        unpacked = unpack_ternary(packed, original_len=orig_len, mode="2bit")
        byte_val = packed.item()

        panel = Panel(
            f"[bold green]BitNet 1.58b Ternary Packing (4 weights per uint8 byte)[/bold green]\n\n"
            f"Input weights: [cyan]{sample.tolist()}[/cyan]\n"
            f"Offset mapping: u = v + 1 in {{0, 1, 2}}\n"
            f"Packed uint8 byte: [bold magenta]{byte_val}[/bold magenta] "
            f"(binary: [yellow]{byte_val:08b}[/yellow], hex: [yellow]0x{byte_val:02X}[/yellow])\n"
            f"Unpacked roundtrip: [cyan]{unpacked.tolist()}[/cyan]\n"
            f"Memory compression: [bold green]8.0x vs FP16[/bold green] (Multiplication-free GEMM)",
            title="BitNet 1.58b Layout",
            border_style="green",
        )
        console.print(panel)

    elif fmt_lower in ("sparse24", "2:4", "sparse"):
        sample = torch.tensor([[0.0, 3.5, 0.0, -1.2, 4.0, 0.0, 0.0, -2.1]])
        condensed, metadata = prune_and_compress_2_4(sample)
        meta_byte = metadata.item()

        panel = Panel(
            f"[bold green]2:4 Structured Sparsity (NVIDIA Sparse Tensor Cores)[/bold green]\n\n"
            f"Dense input row (8 values): [cyan]{sample.tolist()[0]}[/cyan]\n"
            f"Condensed non-zeros (4 values): [cyan]{condensed.tolist()[0]}[/cyan]\n"
            f"Packed metadata byte: [bold magenta]{meta_byte}[/bold magenta] "
            f"(binary: [yellow]{meta_byte:08b}[/yellow], hex: [yellow]0x{meta_byte:02X}[/yellow])\n"
            f"Storage reduction: [bold green]50% weight + 2-bit metadata/val[/bold green]\n"
            f"Theoretical math throughput: [bold green]2.0x Tensor Core TFLOPS[/bold green]",
            title="2:4 Sparse Tensor Layout",
            border_style="magenta",
        )
        console.print(panel)

    else:
        err_msg = f"Unknown format '{fmt}'. Choose 'int4', 'ternary', or 'sparse24'."
        console.print(f"[bold red]{err_msg}[/bold red]")
        raise typer.Exit(code=1)


@app.command("eval")
def eval_cmd(
    shape: str = typer.Option("1024x1024", "--shape", help="Weight matrix shape (MxN)."),
):
    """Evaluate tensor-level SQNR (dB) and compression factor across quantization schemes."""
    import torch

    from quantbench.eval.sqnr import compute_sqnr
    from quantbench.quant.affine import dequantize_absmax, quantize_absmax
    from quantbench.quant.blockwise import dequantize_blockwise, quantize_blockwise

    try:
        m_str, n_str = shape.lower().split("x")
        m, n = int(m_str), int(n_str)
    except Exception:
        console.print(
            f"[bold red]Invalid shape '{shape}'. Expected format e.g. '1024x1024'[/bold red]"
        )
        raise typer.Exit(code=1) from None

    torch.manual_seed(42)
    w = torch.randn(m, n)

    # 1. INT8 AbsMax
    q8, s8 = quantize_absmax(w, num_bits=8, dim=1)
    w8 = dequantize_absmax(q8, s8)
    sqnr8 = compute_sqnr(w, w8)

    # 2. INT4 Blockwise (G=128)
    q4_128, s4_128, zp4_128, meta_128 = quantize_blockwise(w, group_size=128, num_bits=4)
    w4_128 = dequantize_blockwise(q4_128, s4_128, zp4_128, meta_128)
    sqnr4_128 = compute_sqnr(w, w4_128)

    # 3. INT4 Blockwise (G=64)
    q4_64, s4_64, zp4_64, meta_64 = quantize_blockwise(w, group_size=64, num_bits=4)
    w4_64 = dequantize_blockwise(q4_64, s4_64, zp4_64, meta_64)
    sqnr4_64 = compute_sqnr(w, w4_64)

    table = Table(
        title=f"Quantization Fidelity Benchmark ({m}x{n} Weight Matrix)",
        border_style="cyan",
    )
    table.add_column("Scheme", style="bold yellow")
    table.add_column("Group Size", style="cyan")
    table.add_column("Bit-Width", style="green")
    table.add_column("SQNR (dB)", style="bold magenta")
    table.add_column("Compression", style="bold green")

    table.add_row("FP16 Baseline", "None", "16-bit", "100.0 dB", "1.00x")
    table.add_row("INT8 AbsMax", "Per-Channel", "8-bit", f"{sqnr8:.2f} dB", "2.00x")
    table.add_row("INT4 Blockwise", "128", "4-bit", f"{sqnr4_128:.2f} dB", "3.55x")
    table.add_row("INT4 Blockwise", "64", "4-bit", f"{sqnr4_64:.2f} dB", "3.20x")

    console.print(table)


@app.command("pareto")
def pareto_cmd(
    vocab_size: int = typer.Option(128, "--vocab-size", help="Vocabulary cardinality."),
    hidden_dim: int = typer.Option(128, "--hidden-dim", help="Model hidden dimension."),
    seq_len: int = typer.Option(300, "--seq-len", help="Number of calibration tokens."),
):
    """Run an end-to-end Pareto quality-vs-compression benchmark across all quantization schemes."""
    import torch
    import torch.nn as nn

    from quantbench.bench.pareto import ParetoBenchmark
    from quantbench.eval.perplexity import generate_synthetic_corpus

    class DemoTransformer(nn.Module):
        def __init__(self, vocab: int, dim: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab, dim)
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.o_proj = nn.Linear(dim, dim, bias=False)
            self.gate_proj = nn.Linear(dim, dim * 2, bias=False)
            self.down_proj = nn.Linear(dim * 2, dim, bias=False)
            self.lm_head = nn.Linear(dim, vocab, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.embedding(x)
            attn = self.o_proj(torch.relu(self.q_proj(h) + self.k_proj(h) + self.v_proj(h)))
            mlp = self.down_proj(torch.relu(self.gate_proj(attn)))
            return self.lm_head(mlp)

    console.print(
        f"[bold cyan]Initializing Demo Transformer[/bold cyan] "
        f"(Vocab={vocab_size}, Dim={hidden_dim})..."
    )
    torch.manual_seed(42)
    model = DemoTransformer(vocab=vocab_size, dim=hidden_dim)
    tokens = generate_synthetic_corpus(vocab_size=vocab_size, seq_length=seq_len, seed=1)

    bench = ParetoBenchmark(
        base_model=model, test_tokens=tokens, window_size=64, stride=32
    )
    console.print(
        "[bold yellow]Running Pareto sweep across all precision runtimes...[/bold yellow]"
    )

    points = bench.run_sweep(exclude_modules=["lm_head"])

    table = Table(title="Pareto Frontier (Quality vs Compression)", border_style="cyan")
    table.add_column("Configuration", style="bold yellow")
    table.add_column("Scheme", style="cyan")
    table.add_column("Group", style="magenta")
    table.add_column("Memory (MB)", style="green")
    table.add_column("Compression", style="bold green")
    table.add_column("Perplexity", style="bold white")
    table.add_column("Delta PPL", style="bold red")
    table.add_column("SQNR (dB)", style="bold magenta")

    for p in points:
        grp = str(p.group_size) if p.group_size > 0 else "None"
        delta_str = (
            f"+{p.perplexity_delta:.3f}"
            if p.perplexity_delta > 0
            else f"{p.perplexity_delta:.3f}"
        )
        table.add_row(
            p.name,
            p.scheme,
            grp,
            f"{p.memory_mb:.2f}",
            f"{p.compression_ratio:.2f}x",
            f"{p.perplexity:.3f}",
            delta_str,
            f"{p.avg_sqnr_db:.1f}",
        )

    console.print(table)


if __name__ == "__main__":
    app()
