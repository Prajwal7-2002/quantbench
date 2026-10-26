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


if __name__ == "__main__":
    app()
