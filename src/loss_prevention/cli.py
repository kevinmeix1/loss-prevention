"""CLI for training, recommendation, evaluation, API, and UI."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Autonomous insurance loss prevention recommendation agent")
console = Console()


@app.command("train")
def train(
    n_customers: int = 4000,
    n_historical: int = 10000,
    seed: int = 42,
    out: Path = Path("artifacts/models"),
) -> None:
    """Train predictive + uplift + causal forest models."""
    from loss_prevention.models.bundle import train_all_models

    bundle = train_all_models(
        n_customers=n_customers, n_historical=n_historical, seed=seed
    )
    bundle.save(out)
    table = Table(title="Training metrics")
    table.add_column("Section")
    table.add_column("Metrics")
    table.add_row("predictive", json.dumps(bundle.train_metrics.get("predictive", {}), indent=0))
    table.add_row(
        "method_compare",
        json.dumps(bundle.train_metrics.get("method_compare", {}), indent=0)[:800],
    )
    console.print(table)
    console.print(f"[green]Saved bundle → {out}/model_bundle.joblib[/green]")


@app.command("recommend")
def recommend(
    customer_index: int = 0,
    method: str = "x_learner",
    seed: int = 42,
) -> None:
    """Recommend an intervention for a synthetic customer."""
    from loss_prevention.pipeline import LossPreventionPipeline

    pipe = LossPreventionPipeline()
    pipe.ensure_ready(seed=seed, n_customers=2000, n_historical=6000)
    customer_id = pipe.customers[customer_index].customer_id
    rec = pipe.recommend(customer_id, method=method)
    console.print(
        f"[bold]{rec.customer_id}[/bold] · risk={rec.risk.p_loss:.1%} "
        f"({rec.risk.risk_segment.value})"
    )
    console.print(f"Recommended → [cyan]{rec.recommended_intervention.value}[/cyan]")
    for b in rec.explanation_bullets:
        console.print(f" • {b}")
    for t in rec.tradeoffs:
        console.print(f" ⇄ {t}")


@app.command("counterfactual")
def counterfactual(
    intervention: str = "inspection",
    customer_index: int = 0,
    method: str = "x_learner",
) -> None:
    """What-if simulator for a single intervention."""
    from loss_prevention.pipeline import LossPreventionPipeline
    from loss_prevention.schemas import InterventionType

    pipe = LossPreventionPipeline()
    pipe.ensure_ready(n_customers=2000, n_historical=6000)
    cid = pipe.customers[customer_index].customer_id
    result = pipe.counterfactual(cid, InterventionType(intervention), method=method)
    console.print_json(result.model_dump_json(indent=2))


@app.command("evaluate")
def evaluate(seed: int = 42) -> None:
    """Run deployment simulation metrics."""
    from loss_prevention.evaluation.deploy import simulate_deployment

    metrics = simulate_deployment(n_customers=1500, n_historical=5000, seed=seed)
    console.print_json(metrics.model_dump_json(indent=2))


@app.command("research")
def research(seed: int = 42) -> None:
    """Run research experiments (bias, heterogeneity, policy, explore/exploit)."""
    from loss_prevention.evaluation.experiments import run_research_suite

    for report in run_research_suite(seed=seed):
        console.rule(report.name)
        console.print(report.summary)
        for t in report.takeaways:
            console.print(f" • {t}")
        console.print(json.dumps(report.metrics, indent=2)[:1200])


@app.command("serve-api")
def serve_api(host: str = "127.0.0.1", port: int = 8010) -> None:
    """Start FastAPI server."""
    import uvicorn

    uvicorn.run("loss_prevention.api.app:app", host=host, port=port, reload=False)


@app.command("serve-ui")
def serve_ui(host: str = "127.0.0.1", port: int = 8010) -> None:
    """Start the web UI (FastAPI-served dashboard; Streamlit optional)."""
    import uvicorn

    console.print(f"[green]UI + API → http://{host}:{port}/[/green]")
    uvicorn.run("loss_prevention.api.app:app", host=host, port=port, reload=False)


@app.command("serve-streamlit")
def serve_streamlit(port: int = 8510) -> None:
    """Optional Streamlit UI (requires: pip install streamlit)."""
    import subprocess
    from pathlib import Path

    ui = Path(__file__).resolve().parent / "ui" / "app.py"
    subprocess.run(
        ["streamlit", "run", str(ui), "--server.port", str(port)],
        check=False,
    )


if __name__ == "__main__":
    app()
