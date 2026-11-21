"""
Live Real-Time Re-ranking Demonstration for Judges.
Directly implements 'THE MOMENT WE WANT TO SEE':
1. Initial priority queue displayed with auditable reasons.
2. A Cashfree payment webhook lands -> Account settled, queue re-forms in front of judges!
3. A Sarvam voice signal arrives ('Salary 5th ko aayega') -> Deferred, next borrower moves up!
4. A promise lapses -> Escalation penalty applied, moves to top of queue!
5. A dispute is flagged -> Frozen from calling and routed to grievance desk.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from database.connection import init_db, SessionLocal
from database.models import Loan, Borrower
from engine.monitor import monitor
from integrations.cashfree_client import cashfree_client
from integrations.sarvam_adapter import sarvam_adapter

console = Console()

def render_queue_table(queue, title="Current Priority Calling Queue (Top 5)"):
    table = Table(title=f"[bold cyan]{title}[/bold cyan]", show_lines=True)
    table.add_column("Rank", style="bold yellow", width=6)
    table.add_column("Loan & Borrower", style="cyan", width=22)
    table.add_column("Overdue (₹)", style="bold", width=12)
    table.add_column("P(Pay) / P(Reach)", style="magenta", width=16)
    table.add_column("Expected Rec.", style="bold green", width=14)
    table.add_column("Status / Action", style="bold yellow", width=22)
    table.add_column("Auditable Reason (Read Aloud)", style="white")
    
    for item in queue[:5]:
        table.add_row(
            f"#{item['queue_rank']}",
            f"{item['borrower_name']}\n({item['loan_id']}) - Lang: {item['preferred_language']}",
            f"₹{item['amount_overdue']:,.0f}",
            f"{item['repayment_probability']*100:.0f}% / {item['contact_probability']*100:.0f}%",
            f"₹{item['expected_recovery']:,.0f}",
            item['recommended_action'],
            item['manager_explanation']
        )
    console.print(table)

def run_live_demonstration():
    console.clear()
    console.print(Panel.fit(
        "[bold green]FinVoice - Autonomous Collections & Replanning Engine[/bold green]\n"
        "[italic]Problem 2 Demonstration: Real-Time Dynamic Queue Re-ranking & Regulatory Compliance[/italic]",
        border_style="green"
    ))
    
    init_db()
    
    # -------------------------------------------------------------
    # Step 1: Initial Priority Queue
    # -------------------------------------------------------------
    console.print("\n[bold yellow]► STEP 1: Cold Start Portfolio Ranking (Day Start)[/bold yellow]")
    console.print("Meena's team has 4,200 accounts. Standard callers reach under 10%.")
    console.print("FinVoice ranks all accounts by [bold green]Expected Recovery / Cost[/bold green] + compliance gatekeeper:")
    
    initial_res = monitor.re_rank_and_persist(capacity=60)
    top_queue = initial_res["ranked_queue"]
    render_queue_table(top_queue, "Initial Priority Queue - 09:00 AM")
    
    top_loan_1 = top_queue[0]
    top_loan_2 = top_queue[1]
    top_loan_3 = top_queue[2]
    
    time.sleep(1.5)
    
    # -------------------------------------------------------------
    # Step 2: Cashfree Payment Webhook Lands!
    # -------------------------------------------------------------
    console.print("\n" + "="*80)
    console.print(f"[bold green]► STEP 2: EVENT - Cashfree Payment Lands for Rank #1 ({top_loan_1['borrower_name']})[/bold green]")
    console.print(f"Borrower [bold]{top_loan_1['borrower_name']}[/bold] clicks payment link and completes UPI payment of [bold green]₹{top_loan_1['amount_overdue']:,.0f}[/bold green].")
    console.print("[dim]Cashfree Webhook Payload arriving: PAYMENT_SUCCESS_WEBHOOK...[/dim]")
    
    payment_result = monitor.handle_payment_event(
        loan_id=top_loan_1["loan_id"],
        amount=top_loan_1["amount_overdue"],
        payment_method="UPI_AUTOPAY"
    )
    
    console.print(f"[bold green]✓ ACTION TAKEN:[/bold green] {payment_result['action_summary']}")
    console.print("[bold yellow]► WATCH THE QUEUE RE-FORM LIVE:[/bold yellow]")
    render_queue_table(payment_result["top_5_queue"], "Updated Queue after Cashfree Payment (Rank #1 Settled & Removed)")
    
    time.sleep(1.5)
    
    # -------------------------------------------------------------
    # Step 3: Sarvam Post-Call Intent - Future Promise ("Salary 5th ko aayega")
    # -------------------------------------------------------------
    console.print("\n" + "="*80)
    current_rank_1 = payment_result["top_5_queue"][0]
    console.print(f"[bold blue]► STEP 3: EVENT - Sarvam Voice Call Completes with {current_rank_1['borrower_name']}[/bold blue]")
    raw_speech = "Bhaiya meri salary 5th ko aayegi, 5th ko main pura overdue jama kar dunga. Abhi mat call karo."
    console.print(f"[italic cyan]Borrower Audio Transcript (Saaras STT):[/italic cyan] \"{raw_speech}\"")
    
    # Sarvam adapter parses intent
    parsed = sarvam_adapter.parse_speech_or_ocr_transcript(raw_speech)
    console.print(f"[bold]Detected Intent:[/bold] [yellow]{parsed['detected_intent']}[/yellow] | [bold]Extracted Date:[/bold] {parsed['extracted_promise_date']}")
    
    intent_result = monitor.handle_intent_signal(
        loan_id=current_rank_1["loan_id"],
        intent_data=parsed
    )
    
    console.print(f"[bold green]✓ ACTION TAKEN:[/bold green] {intent_result['action_taken']}")
    console.print("[bold yellow]► WATCH THE QUEUE RE-FORM LIVE:[/bold yellow]")
    console.print(f"[dim]{current_rank_1['borrower_name']} is deferred to commitment date. Next borrower automatically shifts into calling slot.[/dim]")
    render_queue_table(intent_result["top_5_queue"], f"Updated Queue after Promise Rescheduled ({current_rank_1['borrower_name']} Deferred)")
    
    time.sleep(1.5)
    
    # -------------------------------------------------------------
    # Step 4: Dispute Detected from Speech / OCR
    # -------------------------------------------------------------
    console.print("\n" + "="*80)
    current_rank_2 = intent_result["top_5_queue"][1]
    console.print(f"[bold red]► STEP 4: EVENT - Borrower {current_rank_2['borrower_name']} Declares a Dispute[/bold red]")
    dispute_speech = "Maine loan settle kar liya tha pehle hi! Ye extra charges galat lagaye hain, I dispute this interest."
    console.print(f"[italic cyan]Borrower Audio Transcript:[/italic cyan] \"{dispute_speech}\"")
    
    parsed_dispute = sarvam_adapter.parse_speech_or_ocr_transcript(dispute_speech)
    dispute_result = monitor.handle_intent_signal(
        loan_id=current_rank_2["loan_id"],
        intent_data=parsed_dispute
    )
    
    console.print(f"[bold red]✓ COMPLIANCE SHIELD ACTIVATED:[/bold red] {dispute_result['action_taken']}")
    render_queue_table(dispute_result["top_5_queue"], f"Updated Queue after Dispute Freeze ({current_rank_2['borrower_name']} Routed to Grievance)")
    
    time.sleep(1.5)
    
    # -------------------------------------------------------------
    # Step 5: Immediate Payment Request & Cashfree Link Generation
    # -------------------------------------------------------------
    console.print("\n" + "="*80)
    current_lead = dispute_result["top_5_queue"][0]
    console.print(f"[bold magenta]► STEP 5: EVENT - Borrower {current_lead['borrower_name']} Wants to Pay ₹5,000 Right Now[/bold magenta]")
    pay_now_speech = "Salary credit ho gayi hai, main abhi 5000 pay kar sakta hu. Link bhejo turant."
    console.print(f"[italic cyan]Borrower WhatsApp / Speech:[/italic cyan] \"{pay_now_speech}\"")
    
    parsed_pay_now = sarvam_adapter.parse_speech_or_ocr_transcript(pay_now_speech)
    pay_now_result = monitor.handle_intent_signal(
        loan_id=current_lead["loan_id"],
        intent_data=parsed_pay_now
    )
    
    console.print(f"[bold green]✓ ACTION TAKEN:[/bold green] {pay_now_result['action_taken']}")
    if pay_now_result.get("payment_link"):
        console.print(f"[bold cyan]Generated Cashfree Link:[/bold cyan] {pay_now_result['payment_link']['payment_url']}")
        
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold green]✓ LIVE DEMONSTRATION COMPLETE[/bold green]\n"
        "• Queue dynamically re-ranked on every event\n"
        "• Manager auditable explanation available for every single decision\n"
        "• Regulatory compliance enforced: calling hours, dispute pauses, and hardship support\n"
        "• Seamlessly connects Cashfree Webhooks + Sarvam Voice/OCR signals",
        border_style="green"
    ))

if __name__ == "__main__":
    run_live_demonstration()
