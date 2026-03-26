"""Clinical AI Agent — CLI entry point.

Run: python cli.py assess "65-year-old male, EF 30%, creatinine 1.9"
Run: python cli.py demo  (runs predefined scenarios)
"""

import sys

# PyTorch + sentence-transformers + LangChain together create a very deep
# import chain that exceeds Python's default recursion limit (1000).
# Must be set before any imports that trigger torch/transformers loading.
sys.setrecursionlimit(10000)

# Pre-load heavy libraries to avoid recursion during Pydantic validation
# when ChatBedrockConverse initializes. Order matters.
import torch  # noqa: F401, E402
import truststore  # noqa: E402

truststore.inject_into_ssl()

import typer  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402

from agent.agent import run_assessment  # noqa: E402

app = typer.Typer(help="Clinical AI Agent for Heart Failure Risk Assessment")
console = Console()


@app.command()
def assess(
    scenario: str = typer.Argument(
        ...,
        help="Patient scenario description, e.g. '65yo male, EF 30%, creatinine 1.9'",
    ),
) -> None:
    """Run clinical AI assessment for a patient scenario."""
    console.print(Panel(f"[bold]Patient Scenario:[/bold] {scenario}", title="Input"))
    console.print()

    with console.status("[bold green]Running clinical assessment..."):
        result = run_assessment(scenario)

    console.print(Panel(result, title="Clinical Assessment", border_style="green"))


@app.command()
def demo() -> None:
    """Run demo with predefined patient scenarios."""
    scenarios = [
        "65-year-old male, ejection fraction 30%, serum creatinine 1.9, "
        "NYHA Class III, currently on lisinopril and metoprolol",
        "45-year-old female, ejection fraction 55%, serum creatinine 0.8, "
        "no symptoms, routine screening",
    ]

    for i, scenario in enumerate(scenarios, 1):
        console.print(f"\n[bold cyan]Demo Scenario {i}:[/bold cyan]")
        console.print(Panel(scenario, title=f"Scenario {i}"))
        console.print()

        with console.status("[bold green]Running clinical assessment..."):
            result = run_assessment(scenario)

        console.print(Panel(result, title=f"Assessment {i}", border_style="green"))
        console.print("-" * 60)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
