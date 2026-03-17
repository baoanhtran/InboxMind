"""
agent/state.py

Defines the single shared state TypedDict that flows through every
node in the LangGraph StateGraph. All nodes read from and write
partial updates to this structure.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# ---------------------------------------------------------------------------
# Sub-schemas (used as values inside the main state)
# ---------------------------------------------------------------------------

class EmailRecord(TypedDict):
    """Raw email data fetched from Gmail."""
    id: str
    thread_id: str
    subject: str
    sender: str
    date: str
    body: str
    snippet: str
    has_attachments: bool
    attachments: list[dict]     # [{"id": str, "filename": str, "mime_type": str, "size": int}]


class ClassifiedEmail(TypedDict):
    """Output of the triage_classifier node for a single email."""
    email_id: str
    intent: str                 # "has_attachment" | "needs_reply" | "info_only"
    priority: str               # "high" | "medium" | "low"
    summary: str                # one-sentence GPT-4o summary
    reasoning: str              # why this intent was chosen


class AttachmentResult(TypedDict):
    """Extracted content from one attachment."""
    email_id: str
    filename: str
    mime_type: str
    extracted_text: str
    key_findings: list[str]


class DraftResult(TypedDict):
    """Result of creating a Gmail draft."""
    email_id: str
    draft_id: str
    draft_subject: str
    draft_body: str
    recipient: str


class EmailSummary(TypedDict):
    """Structured summary produced by the summarizer node."""
    email_id: str
    key_points: list[str]
    action_items: list[str]
    deadline: Optional[str]


class GmailActionResult(TypedDict):
    """Result of applying a label and marking an email as read."""
    email_id: str
    label_applied: str          # the label name that was applied
    marked_read: bool           # whether mark-as-read succeeded


class HumanDecision(TypedDict):
    """Decision recorded at the human_review checkpoint."""
    approved: bool
    feedback: Optional[str]     # free-text feedback if not approved


# ---------------------------------------------------------------------------
# Main state
# ---------------------------------------------------------------------------

class InboxMindState(TypedDict):
    """
    Central state that flows through the entire LangGraph workflow.

    LangGraph applies each node's return dict as a partial update —
    keys not returned by a node are left unchanged.

    The `messages` field uses the add_messages reducer so that LLM
    conversation turns accumulate rather than overwrite.
    """

    # Conversation history (for LLM turns inside nodes)
    messages: Annotated[list[BaseMessage], add_messages]

    # --- Input ---
    gmail_query: str                        # e.g. "is:unread newer_than:1d"
    max_emails: int                         # fetch limit

    # --- Fetched data ---
    emails: list[EmailRecord]               # raw emails from Gmail

    # --- Classification ---
    classified: list[ClassifiedEmail]       # one entry per email
    current_batch: list[str]               # email IDs queued for processing

    # --- Branch outputs ---
    attachment_results: Annotated[list[AttachmentResult], operator.add]
    draft_results: Annotated[list[DraftResult], operator.add]
    summaries: Annotated[list[EmailSummary], operator.add]

    # --- Gmail actions ---
    gmail_action_results: Annotated[list[GmailActionResult], operator.add]

    # --- Aggregated & reviewed ---
    aggregated_output: dict[str, Any]       # merged view after aggregator
    human_decision: Optional[HumanDecision]

    # --- Final ---
    final_report: Optional[str]             # Markdown report
    errors: Annotated[list[str], operator.add]  # non-fatal errors collected

    # --- Control ---
    retry_count: int                        # guard against infinite retry loops
    processing_complete: bool