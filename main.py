"""
main.py

CLI entry point for InboxMind.

Usage:
    python main.py
    python main.py --query "has:attachment from:boss@company.com" --max 10
    python main.py --query "has:attachment subject:invoice" --max 5 --no-hitl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="InboxMind — AI Gmail Analysis Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query", "-q",
        default=os.environ.get("GMAIL_QUERY", "has:attachment"),
        help="Gmail search query (default: 'has:attachment')",
    )
    parser.add_argument(
        "--max", "-m",
        type=int,
        default=int(os.environ.get("GMAIL_MAX_EMAILS", "10")),  # fetches 10 by default
        help="Maximum number of emails to fetch (default: 10)",
    )
    parser.add_argument(
        "--no-hitl",
        action="store_true",
        help="Skip the human-in-the-loop checkpoint and auto-approve",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="LangGraph thread ID for checkpoint persistence (auto-generated per run by default)",
    )
    return parser.parse_args()


def run_with_hitl(graph, initial_state: dict, config: dict) -> None:
    """
    Run the graph with human-in-the-loop support.

    LangGraph pauses at the human_review node and returns an interrupt.
    We read the human's decision from stdin, then resume the graph.
    """
    print("\n[main] Starting InboxMind agent with HITL enabled...\n")

    # Phase 1: Run until the interrupt
    for event in graph.stream(initial_state, config=config, stream_mode="values"):
        # Print progress as state keys are updated
        if "classified" in event and event["classified"]:
            print(f"  → Classified {len(event['classified'])} emails")
        if "draft_results" in event and event["draft_results"]:
            print(f"  → Created {len(event['draft_results'])} drafts")
        if "attachment_results" in event and event["attachment_results"]:
            print(f"  → Processed {len(event['attachment_results'])} attachments")

    # Check if we hit an interrupt (human_review node)
    state_snapshot = graph.get_state(config)
    if state_snapshot.next:
        print(f"\n[main] Graph paused at: {state_snapshot.next}")

        # Prompt the human for a decision
        print("\nOptions: [a]pprove / [r]etry with feedback / [s]kip review")
        choice = input("Your choice [a/r/s]: ").strip().lower()

        if choice == "r":
            feedback = input("Feedback for the agent: ").strip()
            decision = {"approved": False, "feedback": feedback}
        else:
            decision = {"approved": True, "feedback": ""}

        # Resume the graph by providing the human's decision as a Command
        print("\n[main] Resuming graph with your decision...")
        final_state = graph.invoke(
            Command(resume=decision),
            config=config,
        )
        print_final_results(final_state)

    else:
        # Graph completed without interrupt
        print("\n[main] Graph completed.")


def run_auto(graph, initial_state: dict, config: dict) -> None:
    """
    Run the graph without HITL (auto-approve at human_review).
    Useful for testing or scheduled/automated runs.
    """
    print("\n[main] Starting InboxMind agent (auto-approve mode)...\n")

    final_state = graph.invoke(initial_state, config=config)
    print_final_results(final_state)


def print_final_results(final_state: dict) -> None:
    output_path = os.environ.get("REPORT_OUTPUT_PATH", "./report.md")
    print(f"\n{'='*60}")
    print("  InboxMind — Run Complete")
    print(f"{'='*60}")
    print(f"  Emails processed : {len(final_state.get('emails', []))}")
    print(f"  Drafts saved     : {len(final_state.get('draft_results', []))}")
    print(f"  Attachments read : {len(final_state.get('attachment_results', []))}")
    print(f"  Errors           : {len(final_state.get('errors', []))}")
    print(f"  Report saved to  : {output_path}")

    if final_state.get("errors"):
        print("\n  Errors encountered:")
        for err in final_state["errors"]:
            print(f"    - {err}")

    print(f"\n{'='*60}\n")


def main() -> None:
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    # Import here to avoid slow startup when --help is used
    from agent.graph import build_graph_with_sqlite

    graph = build_graph_with_sqlite()

    initial_state = {
        "messages":           [],
        "gmail_query":        args.query,
        "max_emails":         args.max,
        "emails":             [],
        "classified":         [],
        "current_batch":      [],
        "attachment_results": [],
        "draft_results":      [],
        "summaries":          [],
        "aggregated_output":  {},
        "human_decision":     None,
        "final_report":       None,
        "errors":             [],
        "retry_count":        0,
        "processing_complete": False,
    }

    # Generate a unique thread ID per run by default so stale checkpoint
    # state from previous runs never bleeds into a new one.
    # Pass --thread-id explicitly only if you want to resume a specific run.
    import uuid as _uuid
    thread_id = args.thread_id or f"run-{_uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}
    print(f"[main] Thread ID: {thread_id}")

    if args.no_hitl:
        run_auto(graph, initial_state, config)
    else:
        run_with_hitl(graph, initial_state, config)


if __name__ == "__main__":
    main()