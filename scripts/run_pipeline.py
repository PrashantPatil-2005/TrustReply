"""Main pipeline: download → preprocess → build golden set → evaluate."""
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()


def cmd_download():
    from scripts.download_data import main
    main()


def cmd_preprocess():
    from src.preprocess import extract_brand_pairs, build_retrieval_index
    from src.config import ROOT, load_config
    cfg = load_config()
    out = ROOT / "data" / "processed" / f"{cfg['brand']}_pairs.csv"
    df = extract_brand_pairs(
        brand=cfg["brand"],
        max_pairs=cfg["data"]["max_train_samples"],
        output_path=out,
    )
    build_retrieval_index(df, ROOT / "data" / "processed" / f"{cfg['brand']}_retrieval.jsonl")


def cmd_golden():
    from scripts.build_golden_set import main
    main()


def cmd_evaluate():
    from src.evaluate import run_full_evaluation
    results = run_full_evaluation()
    _print_summary(results)


def cmd_demo(message: str):
    from src.agent import SupportAgent
    agent = SupportAgent()
    resp = agent.handle(message)
    console.print(f"\n[bold]Customer:[/bold] {message}")
    console.print(f"[bold]Intent:[/bold] {resp.intent} ({resp.intent_confidence:.2f})")
    console.print(f"[bold]Escalation:[/bold] {resp.escalation_action}")
    console.print(f"[bold]Reason:[/bold] {resp.escalation_reason}")
    console.print(f"[bold]Draft reply:[/bold] {resp.draft_reply}")
    console.print(f"[dim]Similar examples used: {resp.similar_examples_used}[/dim]")


def _print_summary(results: dict):
    table = Table(title="Evaluation Results")
    table.add_column("Model")
    table.add_column("Intent Acc")
    table.add_column("Intent F1")
    table.add_column("Escalation Acc")
    table.add_column("Escalate F1")

    for name in ["trivial_baseline", "simple_baseline", "agent"]:
        if name not in results:
            continue
        r = results[name]
        table.add_row(
            name,
            str(r["intent"]["accuracy"]),
            str(r["intent"]["macro_f1"]),
            str(r["escalation"]["accuracy"]),
            str(r["escalation"]["f1_escalate"]),
        )
    console.print(table)

    if "llm_judge" in results:
        j = results["llm_judge"]
        console.print(f"\nLLM Judge mean overall: {j['mean_judge_overall']}/5")
        console.print(f"Human-LLM agreement (within 1): {j['agreement_with_human']['agreement_within_1']}")


def cmd_all():
    console.print("[bold green]Step 1/4: Download data[/bold green]")
    cmd_download()
    console.print("[bold green]Step 2/4: Preprocess[/bold green]")
    cmd_preprocess()
    console.print("[bold green]Step 3/4: Build golden set[/bold green]")
    cmd_golden()
    console.print("[bold green]Step 4/4: Evaluate[/bold green]")
    cmd_evaluate()


def main():
    parser = argparse.ArgumentParser(description="AmazonHelp AI Support Agent Pipeline")
    parser.add_argument("command", choices=["all", "download", "preprocess", "golden", "evaluate", "demo"])
    parser.add_argument("--message", "-m", default="@AmazonHelp my package still hasn't arrived after 2 weeks")
    args = parser.parse_args()

    commands = {
        "all": cmd_all,
        "download": cmd_download,
        "preprocess": cmd_preprocess,
        "golden": cmd_golden,
        "evaluate": cmd_evaluate,
        "demo": lambda: cmd_demo(args.message),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
