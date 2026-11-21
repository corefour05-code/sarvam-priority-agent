"""
Master Entry Point for FinVoice Collections Hackathon (Problem 2).
Offers one-click commands for:
  1. Live Dynamic Re-ranking Showcase ('The Moment We Want To See')
  2. Judge Curveball & Benchmark Evaluation (FinVoice vs Baseline)
  3. Regulatory Fairness & Red-Line Audit
  4. Real-Time FastAPI REST & Webhook Server
"""

import sys
import os
import argparse
from rich.console import Console
from rich.panel import Panel

console = Console()

def main():
    parser = argparse.ArgumentParser(description="FinVoice Autonomous Recovery Agent")
    parser.add_argument(
        "--mode",
        choices=["live", "curveball", "fairness", "server"],
        default="live",
        help="Execution mode: live (default), curveball, fairness, or server"
    )
    parser.add_argument("--csv", type=str, default=None, help="Path to unseen test CSV for curveball")
    parser.add_argument("--capacity", type=int, default=60, help="Calling capacity (default: 60)")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server (default: 8000)")
    args = parser.parse_args()

    if args.mode == "live":
        from demo.live_dashboard import run_live_demonstration
        run_live_demonstration()
        
    elif args.mode == "curveball":
        from demo.curveball_eval import evaluate_curveball
        evaluate_curveball(args.csv, args.capacity)
        
    elif args.mode == "fairness":
        from ml.fairness_audit import FairnessAuditor
        import json
        auditor = FairnessAuditor()
        res = auditor.run_full_audit()
        console.print(Panel.fit(
            f"[bold green]Regulatory Compliance Status:[/bold green] {res['status']}\n"
            f"• Demographics Leaked: {len(res['feature_leakage_check']['leaked_columns'])}\n"
            f"• Demographic Swap Max Delta: {res['counterfactual_permutation_test']['max_score_delta_under_demographic_swap']:.6f}\n"
            f"• Gender DI Ratio: {res['disparate_impact_ratios']['gender_di_ratio']}\n"
            f"• Caste DI Ratio: {res['disparate_impact_ratios']['caste_di_ratio']}\n"
            f"• Religion DI Ratio: {res['disparate_impact_ratios']['religion_di_ratio']}\n\n"
            f"[italic]{res['regulatory_statement']}[/italic]",
            title="[bold yellow]Fairness & Compliance Report[/bold yellow]"
        ))
        
    elif args.mode == "server":
        import uvicorn
        console.print(f"[bold green]Starting FinVoice FastAPI server on http://0.0.0.0:{args.port}...[/bold green]")
        uvicorn.run("api.server:app", host="0.0.0.0", port=args.port, reload=False)

if __name__ == "__main__":
    main()
