"""
QUMO CLI - Command-line interface for the Quantum Unicode Machine Optimizer.
"""

import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.syntax import Syntax
from . import __version__
from .transformer import optimize_code

app = typer.Typer(help="Quantum Unicode Machine Optimizer")
console = Console()

@app.command()
def optimize(
    input_file: Path = typer.Argument(..., help="Python file to optimize"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file (default: stdout)"),
    show_diff: bool = typer.Option(False, "--diff", "-d", help="Show diff of changes"),
) -> None:
    """Optimize Python code using quantum patterns and Unicode optimization."""
    try:
        # Read input file
        source = input_file.read_text()
        
        # Optimize code
        optimized = optimize_code(source)
        
        if show_diff:
            # Show diff using rich
            console.print("Original:", style="bold green")
            console.print(Syntax(source, "python", theme="monokai"))
            console.print("\nOptimized:", style="bold blue")
            console.print(Syntax(optimized, "python", theme="monokai"))
        
        # Write output
        if output_file:
            output_file.write_text(optimized)
            console.print(f"Optimized code written to {output_file}", style="bold green")
        else:
            console.print(optimized)
            
    except Exception as e:
        console.print(f"Error: {e}", style="bold red")
        sys.exit(1)

@app.command()
def version() -> None:
    """Show QUMO version."""
    console.print(f"QUMO version {__version__}", style="bold blue")

def main() -> None:
    """Main entry point for the CLI."""
    app() 