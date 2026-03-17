"""
agent/graph.py

Defines and compiles the InboxMind LangGraph StateGraph.

Graph structure:
    START
      └─▶ fetch_emails
            └─▶ triage_classifier
                  └─▶ [conditional_router]
                        ├─▶ attachment_processor ─┐
                        ├─▶ reply_drafter         ├─▶ result_aggregator
                        └─▶ summarizer ───────────┘
                                                       └─▶ human_review
                                                             ├─▶ [approved] ─▶ apply_gmail_actions ─▶ report_writer ─▶ END
                                                             └─▶ [retry]    ─▶ triage_classifier (loop)
"""

from __future__ import annotations

import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.state import InboxMindState
from agent.nodes import (
    fetch_emails_node,
    triage_classifier_node,
    attachment_processor_node,
    reply_drafter_node,
    summarizer_node,
    result_aggregator_node,
    human_review_node,
    apply_gmail_actions_node,
    report_writer_node,
)


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def route_after_triage(state: InboxMindState) -> list[str]:
    """
    Fan-out routing: after classification, we send ALL emails to ALL
    three processing branches in parallel (LangGraph will run them
    in the same super-step). Each branch filters internally to only
    act on emails relevant to its intent.

    Returns a list of node names for LangGraph's Send API.
    """
    return ["attachment_processor", "reply_drafter", "summarizer"]


def route_after_review(state: InboxMindState) -> str:
    """
    After the human-in-the-loop checkpoint:
    - If approved → write the final report
    - If rejected and under retry limit → go back to triage
    - If rejected and over retry limit → write report anyway (with errors noted)
    """
    decision = state.get("human_decision")
    retry_count = state.get("retry_count", 0)

    if decision and not decision.get("approved") and retry_count < 3:
        print(f"[router] Human rejected — retry #{retry_count}. Re-running triage.")
        return "triage_classifier"

    return "apply_gmail_actions"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None) -> CompiledStateGraph:
    """
    Constructs and compiles the InboxMind StateGraph.

    Args:
        checkpointer: A LangGraph checkpointer (e.g. SqliteSaver) for
                      durable state persistence. If None, state is in-memory only.

    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    builder = StateGraph(InboxMindState)

    # ── Register nodes ──────────────────────────────────────────────────────
    builder.add_node("fetch_emails",           fetch_emails_node)
    builder.add_node("triage_classifier",      triage_classifier_node)
    builder.add_node("attachment_processor",   attachment_processor_node)
    builder.add_node("reply_drafter",          reply_drafter_node)
    builder.add_node("summarizer",             summarizer_node)
    builder.add_node("result_aggregator",      result_aggregator_node)
    builder.add_node("human_review",           human_review_node)
    builder.add_node("apply_gmail_actions",    apply_gmail_actions_node)
    builder.add_node("report_writer",          report_writer_node)

    # ── Fixed edges ──────────────────────────────────────────────────────────
    builder.add_edge(START, "fetch_emails")
    builder.add_edge("fetch_emails", "triage_classifier")

    # After triage → parallel fan-out to all three processing branches
    builder.add_conditional_edges(
        "triage_classifier",
        route_after_triage,
        {
            "attachment_processor": "attachment_processor",
            "reply_drafter":        "reply_drafter",
            "summarizer":           "summarizer",
        },
    )

    # All three processing branches converge at result_aggregator
    builder.add_edge("attachment_processor", "result_aggregator")
    builder.add_edge("reply_drafter",        "result_aggregator")
    builder.add_edge("summarizer",           "result_aggregator")

    # Aggregator → human checkpoint
    builder.add_edge("result_aggregator", "human_review")

    # Human checkpoint → conditional: approved or retry
    builder.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "apply_gmail_actions": "apply_gmail_actions",
            "triage_classifier":   "triage_classifier",
        },
    )

    # Gmail actions → report writer → end
    builder.add_edge("apply_gmail_actions", "report_writer")
    builder.add_edge("report_writer", END)

    # ── Compile ──────────────────────────────────────────────────────────────
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    return builder.compile(**compile_kwargs)


def build_graph_with_sqlite(db_path: str = "inboxmind.db") -> CompiledStateGraph:
    """
    Convenience wrapper that wires in a SqliteSaver checkpointer.
    Enables durable execution — the graph can survive process restarts.

    LangGraph 1.0: pass a live sqlite3.Connection directly to SqliteSaver.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return build_graph(checkpointer=checkpointer)