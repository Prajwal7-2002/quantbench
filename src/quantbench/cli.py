"""Command Line Interface for quantbench."""

import typer
from rich.console import Console
from rich.table import Table

import quantbench

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


if __name__ == "__main__":
    app()
