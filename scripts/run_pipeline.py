"""ThreadVault pipeline: download → preprocess → split → train → evaluate."""
import argparse
import sys
from pathlib import Path

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
    df = extract_brand_pairs(brand=cfg["brand"], max_pairs=cfg["data"]["max_train_samples"], output_path=out)
    build_retrieval_index(df, ROOT / "data" / "processed" / f"{cfg['brand']}_retrieval.jsonl")


def cmd_split():
    from scripts.split_golden_set import main
    main()


def cmd_train():
    from scripts.train_models import main
    main()


def cmd_golden():
    from scripts.build_golden_set import main
    main()


def cmd_evaluate():
    from src.evaluate import run_full_evaluation
    results = run_full_evaluation()
    _print_summary(results)


def cmd_demo(message: str):
    from src.agent import ThreadVaultAgent
    agent = ThreadVaultAgent()
    trace = agent.handle(message)
    console.print(f"\n[bold]Customer:[/bold] {message}")
    console.print(f"[bold]Intent:[/bold] {trace.intent} ({trace.intent_confidence:.2f})")
    console.print(f"[bold]Retrieval top score:[/bold] {trace.retrieval_top_score:.3f}")
    console.print(f"[bold]Evidence gate:[/bold] {'PASS' if trace.evidence_gate_passed else 'FAIL'}")
    console.print(f"[bold]Groundedness:[/bold] {trace.groundedness_score:.2f}")
    console.print(f"[bold]Decision:[/bold] {trace.escalation_action.upper()}")
    console.print(f"[bold]Reason:[/bold] {trace.escalation_reason}")
    console.print(f"[bold]Reply:[/bold] {trace.draft_reply}")
    console.print(f"[bold]Strict trust:[/bold] {'PASS' if trace.strict_trust_pass else 'FAIL'}")


def _print_summary(results: dict):
    table = Table(title="ThreadVault Evaluation")
    table.add_column("Model")
    table.add_column("Intent Acc")
    table.add_column("Esc Acc")
    table.add_column("Esc F1")
    table.add_column("STS")

    for name in ["trivial_baseline", "simple_baseline", "threadvault_agent"]:
        if name not in results:
            continue
        r = results[name]
        table.add_row(
            name,
            str(r["intent"]["accuracy"]),
            str(r["escalation"]["accuracy"]),
            str(r["escalation"]["f1_escalate"]),
            str(r["strict_trust_score"]["sts"]),
        )
    console.print(table)


def cmd_all():
    steps = [
        ("Download data", cmd_download),
        ("Preprocess", cmd_preprocess),
        ("Build golden set", cmd_golden),
        ("Split golden set", cmd_split),
        ("Train models + index", cmd_train),
        ("Evaluate", cmd_evaluate),
    ]
    for i, (label, fn) in enumerate(steps, 1):
        console.print(f"[bold green]Step {i}/{len(steps)}: {label}[/bold green]")
        fn()


def main():
    parser = argparse.ArgumentParser(description="ThreadVault Pipeline")
    parser.add_argument(
        "command",
        choices=["all", "download", "preprocess", "golden", "split", "train", "evaluate", "demo"],
    )
    parser.add_argument("--message", "-m", default="@AmazonHelp I was charged twice and want a refund!")
    args = parser.parse_args()

    cmds = {
        "all": cmd_all,
        "download": cmd_download,
        "preprocess": cmd_preprocess,
        "golden": cmd_golden,
        "split": cmd_split,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "demo": lambda: cmd_demo(args.message),
    }
    cmds[args.command]()


if __name__ == "__main__":
    main()
